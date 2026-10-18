//! The HTTP server: one engine thread owns the GPU and runs the scheduler; every connection has a
//! thread that parses the request, tokenizes, submits it and streams the events back.

use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::Duration;

use lev_model::gpu::GpuModel;
use lev_tokenizer::{Tokenizer, Utf8Stream};

use crate::api::{self, Prompt};
use crate::gpu_engine::GpuEngine;
use crate::http::{self, HttpRequest};
use crate::sched::{Event, FinishReason, Request, Scheduler};

pub struct ServerConfig {
    pub model_path: String,
    pub model_name: String,
    pub host: String,
    pub port: u16,
    /// Context length per sequence.
    pub ctx: usize,
    /// Concurrent sequences (recurrent-state slots).
    pub slots: usize,
    /// KV blocks of 16 positions (default: as many as `slots` full contexts).
    pub kv_blocks: Option<usize>,
}

enum Command {
    Submit(Request),
    Stats(Sender<String>),
}

struct Shared {
    tk: Tokenizer,
    model: String,
    next_id: AtomicU64,
    ctx: usize,
}

/// Runs until the process is stopped.
pub fn serve(cfg: ServerConfig) -> Result<(), String> {
    let tk = Tokenizer::from_gguf(&cfg.model_path)?;
    let shared = Arc::new(Shared { tk, model: cfg.model_name.clone(), next_id: AtomicU64::new(1), ctx: cfg.ctx });
    let (cmd_tx, cmd_rx) = channel::<Command>();
    let (ready_tx, ready_rx) = channel::<Result<String, String>>();
    {
        let (path, ctx, slots, blocks) = (cfg.model_path.clone(), cfg.ctx, cfg.slots, cfg.kv_blocks);
        // The GPU model has large frames: give its thread a big stack.
        std::thread::Builder::new()
            .name("engine".into())
            .stack_size(256 << 20)
            .spawn(move || engine_thread(&path, ctx, slots, blocks, cmd_rx, ready_tx))
            .map_err(|e| e.to_string())?;
    }
    let info = ready_rx.recv().map_err(|_| "the engine thread died while loading".to_string())??;
    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port)).map_err(|e| format!("cannot listen on {}:{}: {e}", cfg.host, cfg.port))?;
    eprintln!("lev-server: {info}; listening on http://{}:{}", cfg.host, cfg.port);
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let (tx, sh) = (cmd_tx.clone(), shared.clone());
        let _ = stream.set_nodelay(true);
        std::thread::spawn(move || {
            let _ = handle_connection(stream, tx, sh);
        });
    }
    Ok(())
}

fn engine_thread(path: &str, ctx: usize, slots: usize, blocks: Option<usize>, rx: Receiver<Command>, ready: Sender<Result<String, String>>) {
    std::env::set_var("LEV_BATCH_SLOTS", slots.to_string());
    if let Some(b) = blocks {
        std::env::set_var("LEV_KV_BLOCKS", b.to_string());
    }
    let gpu = match GpuModel::open(path, ctx, false) {
        Ok(g) => g,
        Err(e) => {
            let _ = ready.send(Err(format!("cannot open the model: {e}")));
            return;
        }
    };
    let kv = gpu.kv_blocks().map_or("slab KV".to_string(), |(_, t)| format!("{t} KV blocks"));
    let mut sched = Scheduler::new(GpuEngine::new(gpu));
    // Measured on 32 concurrent 256-token prompts: 1 chunk per step gives 47 tokens/s, 3 gives 60,
    // 8 gives 63 (and a shorter time to first token); 4 keeps decoding latency moderate.
    sched.chunks_per_step = std::env::var("LEV_PREFILL_CHUNKS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let _ = ready.send(Ok(format!("{} slots, context {ctx}, {kv}", slots)));
    loop {
        // Take every command that is waiting; block only when there is nothing to run.
        loop {
            let cmd = if sched.is_idle() {
                match rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(c) => Some(c),
                    Err(RecvTimeoutError::Timeout) => None,
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            } else {
                match rx.try_recv() {
                    Ok(c) => Some(c),
                    Err(_) => None,
                }
            };
            match cmd {
                Some(Command::Submit(r)) => sched.submit(r),
                Some(Command::Stats(reply)) => {
                    let s = &sched.stats;
                    let _ = reply.send(format!(
                        "{{\"running\":{},\"waiting\":{},\"steps\":{},\"step_columns\":{},\"prefill_chunks\":{},\"preemptions\":{},\"finished\":{},\"free_kv_blocks\":{}}}",
                        sched.running_count(),
                        sched.waiting_count(),
                        s.steps,
                        s.step_columns,
                        s.prefill_chunks,
                        s.preemptions,
                        s.finished,
                        crate::sched::Engine::free_blocks(&sched.engine)
                    ));
                }
                None => break,
            }
        }
        if !sched.is_idle() {
            sched.tick();
        }
    }
}

fn handle_connection(stream: TcpStream, tx: Sender<Command>, sh: Arc<Shared>) -> Result<(), String> {
    let mut writer = stream.try_clone().map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream);
    loop {
        let req = match http::read_request(&mut reader) {
            Ok(Some(r)) => r,
            Ok(None) => return Ok(()),
            Err(e) => {
                let _ = http::write_response(&mut writer, 400, "application/json", api::error_json(&e).as_bytes(), false);
                return Err(e);
            }
        };
        let keep = !req.header("connection").is_some_and(|v| v.eq_ignore_ascii_case("close"));
        let path = req.path.split('?').next().unwrap_or("").to_string();
        let result = match (req.method.as_str(), path.as_str()) {
            ("GET", "/health") | ("GET", "/v1/health") => http::write_response(&mut writer, 200, "application/json", b"{\"status\":\"ok\"}", keep).map_err(|e| e.to_string()),
            ("GET", "/v1/models") | ("GET", "/models") => {
                let body = format!("{{\"object\":\"list\",\"data\":[{{\"id\":{},\"object\":\"model\",\"created\":0,\"owned_by\":\"leviathan\",\"max_model_len\":{}}}]}}", api::escape(&sh.model), sh.ctx);
                http::write_response(&mut writer, 200, "application/json", body.as_bytes(), keep).map_err(|e| e.to_string())
            }
            ("GET", "/v1/stats") => {
                let (rtx, rrx) = channel();
                let _ = tx.send(Command::Stats(rtx));
                let body = rrx.recv_timeout(Duration::from_secs(5)).unwrap_or_else(|_| "{}".into());
                http::write_response(&mut writer, 200, "application/json", body.as_bytes(), keep).map_err(|e| e.to_string())
            }
            ("POST", "/v1/completions") | ("POST", "/completions") => complete(&mut writer, &req, false, &tx, &sh, keep),
            ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => complete(&mut writer, &req, true, &tx, &sh, keep),
            _ => http::write_response(&mut writer, 404, "application/json", api::error_json("not found").as_bytes(), keep).map_err(|e| e.to_string()),
        };
        result?;
        if !keep {
            return Ok(());
        }
    }
}

/// Withholds the end of the text that could still turn into a stop string.
struct StopFilter {
    stops: Vec<String>,
    pending: String,
}

impl StopFilter {
    /// Text that is safe to send, and whether a stop string was found (text before it is returned).
    fn push(&mut self, text: &str) -> (String, bool) {
        self.pending.push_str(text);
        if let Some(at) = self.stops.iter().filter(|s| !s.is_empty()).filter_map(|s| self.pending.find(s.as_str())).min() {
            let out = self.pending[..at].to_string();
            self.pending.clear();
            return (out, true);
        }
        // Keep the longest suffix of `pending` that is a prefix of some stop string.
        let mut keep = 0;
        for s in self.stops.iter().filter(|s| !s.is_empty()) {
            for k in (1..s.len().min(self.pending.len() + 1)).rev() {
                if s.is_char_boundary(k) && self.pending.is_char_boundary(self.pending.len() - k) && self.pending.ends_with(&s[..k]) {
                    keep = keep.max(k);
                    break;
                }
            }
        }
        let cut = self.pending.len() - keep;
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (out, false)
    }

    fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn complete(w: &mut TcpStream, req: &HttpRequest, chat: bool, tx: &Sender<Command>, sh: &Shared, keep: bool) -> Result<(), String> {
    let send_error = |w: &mut TcpStream, code: u16, msg: &str| http::write_response(w, code, "application/json", api::error_json(msg).as_bytes(), keep).map_err(|e| e.to_string());
    let params = match api::parse_request(&req.body, chat, &sh.model) {
        Ok(p) => p,
        Err(e) => return send_error(w, 400, &e),
    };
    let prompt: Vec<u32> = match &params.prompt {
        Prompt::Text(t) => sh.tk.encode(t, true),
        Prompt::Ids(ids) => ids.clone(),
    };
    if prompt.is_empty() {
        return send_error(w, 400, "the prompt is empty");
    }
    if prompt.iter().any(|&t| t as usize >= sh.tk.vocab_size()) {
        return send_error(w, 400, "a token id is outside the vocabulary");
    }
    let id = format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, sh.next_id.fetch_add(1, Ordering::Relaxed));
    let (etx, erx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let request = Request { id: sh.next_id.load(Ordering::Relaxed), prompt, max_tokens: params.max_tokens, eos: sh.tk.eos, ignore_eos: params.ignore_eos, cancel: cancel.clone(), tx: etx };
    if tx.send(Command::Submit(request)).is_err() {
        return send_error(w, 503, "the engine is not running");
    }
    let created = unix_secs();
    let mut utf8 = Utf8Stream::default();
    let mut stops = StopFilter { stops: params.stop.clone(), pending: String::new() };
    let mut full = String::new();
    let mut head_sent = false;
    let mut stopped_by_text = false;
    loop {
        let ev = match erx.recv() {
            Ok(e) => e,
            Err(_) => return send_error(w, 500, "the engine stopped"),
        };
        match ev {
            Event::Token(t) => {
                if stopped_by_text {
                    continue; // draining until the scheduler confirms the cancel
                }
                let text = utf8.push(sh.tk.token_bytes(t));
                let (out, hit) = stops.push(&text);
                if hit {
                    stopped_by_text = true;
                    cancel.store(true, Ordering::Relaxed);
                }
                if out.is_empty() {
                    continue;
                }
                if params.stream {
                    if !head_sent {
                        if http::write_stream_head(w).is_err() {
                            cancel.store(true, Ordering::Relaxed);
                            return Err("client went away".into());
                        }
                        head_sent = true;
                    }
                    let data = format!("data: {}\n\n", api::chunk_json(&id, created, &params, &out, None, None));
                    if http::write_chunk(w, data.as_bytes()).is_err() {
                        cancel.store(true, Ordering::Relaxed);
                        return Err("client went away".into());
                    }
                } else {
                    full.push_str(&out);
                }
            }
            Event::Finished { reason, prompt_tokens, completion_tokens } => {
                if let FinishReason::Error(msg) = &reason {
                    if !head_sent {
                        return send_error(w, 400, msg);
                    }
                }
                let tail = if stopped_by_text { String::new() } else { format!("{}{}", utf8.push(&[]), stops.flush()) };
                let reason_text = if stopped_by_text { "stop" } else { api::finish_reason_text(&reason) };
                if params.stream {
                    if !head_sent {
                        http::write_stream_head(w).map_err(|e| e.to_string())?;
                    }
                    let mut data = String::new();
                    if !tail.is_empty() {
                        data.push_str(&format!("data: {}\n\n", api::chunk_json(&id, created, &params, &tail, None, None)));
                    }
                    let usage = params.include_usage.then_some((prompt_tokens, completion_tokens));
                    data.push_str(&format!("data: {}\n\ndata: [DONE]\n\n", api::chunk_json(&id, created, &params, "", Some(reason_text), usage)));
                    http::write_chunk(w, data.as_bytes()).map_err(|e| e.to_string())?;
                    http::write_last_chunk(w).map_err(|e| e.to_string())?;
                    return Ok(());
                }
                full.push_str(&tail);
                let body = api::completion_json(&id, created, &params, &full, reason_text, prompt_tokens, completion_tokens);
                return http::write_response(w, 200, "application/json", body.as_bytes(), keep).map_err(|e| e.to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StopFilter;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut f = StopFilter { stops: stops.iter().map(|s| s.to_string()).collect(), pending: String::new() };
        let mut out = String::new();
        for p in pieces {
            let (o, hit) = f.push(p);
            out.push_str(&o);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&f.flush());
        (out, false)
    }

    #[test]
    fn a_stop_string_split_across_pieces_is_found_and_cut() {
        assert_eq!(run(&["END"], &["hello E", "N", "D world"]), ("hello ".to_string(), true));
        assert_eq!(run(&["\n\n"], &["a\n", "\nb"]), ("a".to_string(), true));
    }

    #[test]
    fn partial_matches_that_do_not_complete_are_released() {
        assert_eq!(run(&["END"], &["hello E", "N x"]), ("hello EN x".to_string(), false));
        assert_eq!(run(&["END"], &["abc"]), ("abc".to_string(), false));
    }

    #[test]
    fn multibyte_stop_strings() {
        assert_eq!(run(&["끝"], &["안녕 ", "끝", "!"]), ("안녕 ".to_string(), true));
    }
}