//! Minimal HTTP/1.1 on `std::net`: request parsing with keep-alive, fixed and chunked responses.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub const MAX_HEADER: usize = 64 * 1024;
pub const MAX_BODY: usize = 64 * 1024 * 1024;

#[derive(Debug)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// Reads one request; `Ok(None)` on a clean close between requests.
pub fn read_request(r: &mut BufReader<TcpStream>) -> Result<Option<HttpRequest>, String> {
    let mut line = String::new();
    let n = r.read_line(&mut line).map_err(|e| e.to_string())?;
    if n == 0 {
        return Ok(None);
    }
    let line = line.trim_end();
    if line.is_empty() {
        return Ok(None);
    }
    let mut parts = line.split(' ');
    let (method, path) = (parts.next().ok_or("bad request line")?.to_string(), parts.next().ok_or("bad request line")?.to_string());
    let mut headers = Vec::new();
    let mut total = 0usize;
    loop {
        let mut h = String::new();
        let n = r.read_line(&mut h).map_err(|e| e.to_string())?;
        total += n;
        if n == 0 || total > MAX_HEADER {
            return Err("headers too large or truncated".into());
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let mut req = HttpRequest { method, path, headers, body: Vec::new() };
    if req.header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
        loop {
            let mut size = String::new();
            r.read_line(&mut size).map_err(|e| e.to_string())?;
            let n = usize::from_str_radix(size.trim().split(';').next().unwrap_or("0"), 16).map_err(|_| "bad chunk size")?;
            if n == 0 {
                let mut end = String::new();
                r.read_line(&mut end).map_err(|e| e.to_string())?;
                break;
            }
            if req.body.len() + n > MAX_BODY {
                return Err("body too large".into());
            }
            let at = req.body.len();
            req.body.resize(at + n, 0);
            r.read_exact(&mut req.body[at..]).map_err(|e| e.to_string())?;
            let mut crlf = [0u8; 2];
            r.read_exact(&mut crlf).map_err(|e| e.to_string())?;
        }
    } else if let Some(len) = req.header("content-length") {
        let len: usize = len.parse().map_err(|_| "bad content-length")?;
        if len > MAX_BODY {
            return Err("body too large".into());
        }
        req.body.resize(len, 0);
        r.read_exact(&mut req.body).map_err(|e| e.to_string())?;
    }
    Ok(Some(req))
}

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

pub fn write_response(w: &mut impl Write, code: u16, content_type: &str, body: &[u8], keep_alive: bool) -> std::io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {code} {}\r\nContent-Type: {content_type}; charset=utf-8\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
        status_text(code),
        body.len(),
        if keep_alive { "keep-alive" } else { "close" }
    )?;
    w.write_all(body)?;
    w.flush()
}

/// Headers of a streamed (server-sent events) response; the body follows as chunks.
pub fn write_stream_head(w: &mut impl Write) -> std::io::Result<()> {
    write!(w, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n")?;
    w.flush()
}

pub fn write_chunk(w: &mut impl Write, data: &[u8]) -> std::io::Result<()> {
    write!(w, "{:X}\r\n", data.len())?;
    w.write_all(data)?;
    w.write_all(b"\r\n")?;
    w.flush()
}

pub fn write_last_chunk(w: &mut impl Write) -> std::io::Result<()> {
    w.write_all(b"0\r\n\r\n")?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};

    fn pair() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let c = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (s, _) = l.accept().unwrap();
        (c, s)
    }

    #[test]
    fn parses_a_request_with_a_body_and_keeps_the_connection_for_the_next_one() {
        let (mut c, s) = pair();
        c.write_all(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nContent-Type: application/json\r\n\r\nhelloGET /health HTTP/1.1\r\n\r\n").unwrap();
        let mut r = BufReader::new(s);
        let a = read_request(&mut r).unwrap().unwrap();
        assert_eq!((a.method.as_str(), a.path.as_str()), ("POST", "/v1/completions"));
        assert_eq!(a.body, b"hello");
        assert_eq!(a.header("CONTENT-TYPE"), Some("application/json"));
        let b = read_request(&mut r).unwrap().unwrap();
        assert_eq!((b.method.as_str(), b.path.as_str()), ("GET", "/health"));
        drop(c);
        assert!(read_request(&mut r).unwrap().is_none());
    }

    #[test]
    fn parses_a_chunked_body() {
        let (mut c, s) = pair();
        c.write_all(b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n").unwrap();
        let mut r = BufReader::new(s);
        assert_eq!(read_request(&mut r).unwrap().unwrap().body, b"abcde");
    }

    #[test]
    fn writes_chunks() {
        let mut out = Vec::new();
        write_chunk(&mut out, b"data: x\n\n").unwrap();
        write_last_chunk(&mut out).unwrap();
        assert_eq!(out, b"9\r\ndata: x\n\n\r\n0\r\n\r\n");
    }
}