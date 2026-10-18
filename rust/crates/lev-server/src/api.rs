//! The OpenAI-compatible request and response shapes (completions and chat completions).

use lev_ir::json::{self, Json};

pub fn escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

#[derive(Debug, Clone, PartialEq)]
pub enum Prompt {
    Text(String),
    Ids(Vec<u32>),
}

#[derive(Debug, Clone)]
pub struct Params {
    pub prompt: Prompt,
    pub max_tokens: usize,
    pub stream: bool,
    pub include_usage: bool,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    pub temperature: f64,
    pub chat: bool,
    pub model: String,
}

fn string_list(j: Option<&Json>) -> Vec<String> {
    match j {
        Some(Json::Str(s)) => vec![s.clone()],
        Some(Json::Arr(a)) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    }
}

/// The ChatML prompt of a chat request (the Qwen family format): every message as
/// `<|im_start|>role\ncontent<|im_end|>\n`, then the opening of the assistant turn.
pub fn chat_prompt(messages: &[Json]) -> Result<String, String> {
    let mut p = String::new();
    for m in messages {
        let role = m.get("role").and_then(Json::as_str).ok_or("message without a role")?;
        let content = match m.get("content") {
            Some(Json::Str(s)) => s.clone(),
            // Content as a list of parts: concatenate the text parts.
            Some(Json::Arr(parts)) => parts.iter().filter_map(|x| x.get("text").and_then(Json::as_str)).collect::<Vec<_>>().join(""),
            _ => String::new(),
        };
        p.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    p.push_str("<|im_start|>assistant\n");
    Ok(p)
}

pub fn parse_request(body: &[u8], chat: bool, default_model: &str) -> Result<Params, String> {
    let text = std::str::from_utf8(body).map_err(|_| "the body is not UTF-8")?;
    let j = json::parse(text).map_err(|e| format!("invalid JSON: {e}"))?;
    let prompt = if chat {
        let msgs = j.get("messages").and_then(Json::as_arr).ok_or("missing messages")?;
        Prompt::Text(chat_prompt(msgs)?)
    } else {
        match j.get("prompt") {
            Some(Json::Str(s)) => Prompt::Text(s.clone()),
            Some(Json::Arr(a)) if a.iter().all(|x| matches!(x, Json::Int(_))) => Prompt::Ids(a.iter().filter_map(|x| x.as_i64()).map(|i| i as u32).collect()),
            Some(Json::Arr(a)) if a.len() == 1 => match &a[0] {
                Json::Str(s) => Prompt::Text(s.clone()),
                Json::Arr(ids) => Prompt::Ids(ids.iter().filter_map(|x| x.as_i64()).map(|i| i as u32).collect()),
                _ => return Err("unsupported prompt".into()),
            },
            Some(Json::Arr(_)) => return Err("a batch of prompts in one request is not supported (send one request per prompt)".into()),
            _ => return Err("missing prompt".into()),
        }
    };
    if let Some(n) = j.get("n").and_then(Json::as_i64) {
        if n != 1 {
            return Err("only n = 1 is supported".into());
        }
    }
    let max_tokens = j
        .get("max_tokens")
        .or_else(|| j.get("max_completion_tokens"))
        .and_then(Json::as_i64)
        .map_or(if chat { 4096 } else { 16 }, |v| v.max(1) as usize);
    Ok(Params {
        prompt,
        max_tokens,
        stream: j.get("stream").and_then(Json::as_bool).unwrap_or(false),
        include_usage: j.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Json::as_bool).unwrap_or(false),
        stop: string_list(j.get("stop")),
        ignore_eos: j.get("ignore_eos").and_then(Json::as_bool).unwrap_or(false),
        temperature: j.get("temperature").and_then(Json::as_f64).unwrap_or(0.0),
        chat,
        model: j.get("model").and_then(Json::as_str).unwrap_or(default_model).to_string(),
    })
}

fn finish_json(reason: &str) -> String {
    format!("\"{reason}\"")
}

pub fn finish_reason_text(r: &crate::sched::FinishReason) -> &'static str {
    match r {
        crate::sched::FinishReason::Length => "length",
        crate::sched::FinishReason::Stop => "stop",
        _ => "stop",
    }
}

/// A completion (or chat completion) object; `text` is the whole generated text.
pub fn completion_json(id: &str, created: u64, p: &Params, text: &str, reason: &str, prompt_tokens: usize, completion_tokens: usize) -> String {
    let usage = format!("{{\"prompt_tokens\":{prompt_tokens},\"completion_tokens\":{completion_tokens},\"total_tokens\":{}}}", prompt_tokens + completion_tokens);
    if p.chat {
        format!(
            "{{\"id\":{},\"object\":\"chat.completion\",\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":{}}}],\"usage\":{usage}}}",
            escape(id),
            escape(&p.model),
            escape(text),
            finish_json(reason)
        )
    } else {
        format!(
            "{{\"id\":{},\"object\":\"text_completion\",\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,\"text\":{},\"logprobs\":null,\"finish_reason\":{}}}],\"usage\":{usage}}}",
            escape(id),
            escape(&p.model),
            escape(text),
            finish_json(reason)
        )
    }
}

/// One streamed chunk: a piece of text, or (with `reason`) the closing chunk.
pub fn chunk_json(id: &str, created: u64, p: &Params, text: &str, reason: Option<&str>, usage: Option<(usize, usize)>) -> String {
    let fin = reason.map_or("null".to_string(), finish_json);
    let usage = usage.map_or(String::new(), |(pt, ct)| format!(",\"usage\":{{\"prompt_tokens\":{pt},\"completion_tokens\":{ct},\"total_tokens\":{}}}", pt + ct));
    if p.chat {
        let delta = if text.is_empty() { "{}".to_string() } else { format!("{{\"content\":{}}}", escape(text)) };
        format!(
            "{{\"id\":{},\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{fin}}}]{usage}}}",
            escape(id),
            escape(&p.model)
        )
    } else {
        format!(
            "{{\"id\":{},\"object\":\"text_completion\",\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,\"text\":{},\"logprobs\":null,\"finish_reason\":{fin}}}]{usage}}}",
            escape(id),
            escape(&p.model),
            escape(text)
        )
    }
}

pub fn error_json(message: &str) -> String {
    format!("{{\"error\":{{\"message\":{},\"type\":\"invalid_request_error\"}}}}", escape(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_json_strings() {
        assert_eq!(escape("a\"b\\c\n\t\u{1}é"), "\"a\\\"b\\\\c\\n\\t\\u0001é\"");
    }

    #[test]
    fn parses_completion_requests() {
        let p = parse_request(br#"{"model":"m","prompt":"hi","max_tokens":5,"stream":true,"stop":["x","y"],"ignore_eos":true}"#, false, "d").unwrap();
        assert_eq!(p.prompt, Prompt::Text("hi".into()));
        assert_eq!((p.max_tokens, p.stream, p.ignore_eos), (5, true, true));
        assert_eq!(p.stop, ["x", "y"]);
        let ids = parse_request(br#"{"prompt":[1,2,3]}"#, false, "d").unwrap();
        assert_eq!(ids.prompt, Prompt::Ids(vec![1, 2, 3]));
        assert_eq!(ids.model, "d");
        assert!(parse_request(br#"{"prompt":["a","b"]}"#, false, "d").is_err());
        assert!(parse_request(br#"{"prompt":"a","n":2}"#, false, "d").is_err());
    }

    #[test]
    fn builds_the_chat_prompt() {
        let p = parse_request(br#"{"messages":[{"role":"system","content":"S"},{"role":"user","content":"Hi"}]}"#, true, "d").unwrap();
        assert_eq!(p.prompt, Prompt::Text("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n".into()));
    }

    #[test]
    fn response_objects_are_valid_json_of_the_expected_shape() {
        let p = parse_request(br#"{"prompt":"hi"}"#, false, "m").unwrap();
        let j = json::parse(&completion_json("cmpl-1", 5, &p, "a \"b\"", "length", 3, 4)).unwrap();
        assert_eq!(j.get("choices").unwrap().as_arr().unwrap()[0].get("text").unwrap().as_str(), Some("a \"b\""));
        assert_eq!(j.get("usage").unwrap().get("total_tokens").unwrap().as_i64(), Some(7));
        let c = json::parse(&chunk_json("cmpl-1", 5, &p, "x", None, None)).unwrap();
        assert!(matches!(c.get("choices").unwrap().as_arr().unwrap()[0].get("finish_reason"), Some(Json::Null)));
    }
}