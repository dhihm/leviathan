//! Continuous batching: a queue of requests, a pool of sequence slots, one prefill at a time and a
//! decode step over every running sequence.
//!
//! The scheduler talks to the GPU through the [`Engine`] trait, so its decisions (who is admitted,
//! what is batched, who is preempted when the KV blocks run out, who is cancelled) are tested with a
//! mock engine without a GPU.
//!
//! One `tick` does at most one prefill chunk and one decode step over all running sequences, so a
//! long prompt delays the running sequences by one chunk per tick instead of its whole prefill.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

#[derive(Debug, PartialEq, Eq)]
pub enum EngineError {
    /// No free KV blocks for the step: the scheduler preempts a sequence and retries.
    OutOfBlocks,
    Other(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::OutOfBlocks => write!(f, "out of KV blocks"),
            EngineError::Other(s) => write!(f, "{s}"),
        }
    }
}

/// One token of one running sequence in a decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepItem {
    pub slot: usize,
    pub token: u32,
    pub pos: usize,
}

pub trait Engine {
    /// Sequence slots (recurrent state and KV table rows).
    fn slots(&self) -> usize;
    /// Positions the context can hold per sequence.
    fn ctx_len(&self) -> usize;
    /// Free KV blocks and positions per block.
    fn free_blocks(&self) -> usize;
    fn block_size(&self) -> usize;
    /// Starts prefilling `prompt` into the engine's prefill state (discarding any earlier one).
    fn prefill_begin(&mut self, prompt: &[u32]) -> Result<(), EngineError>;
    /// Runs the next prefill chunk; `Some(token)` is the first generated token once the prompt is done.
    fn prefill_advance(&mut self) -> Result<Option<u32>, EngineError>;
    /// Copies the finished prefill into `slot` (allocating its KV blocks).
    fn import(&mut self, slot: usize, prompt_len: usize) -> Result<(), EngineError>;
    /// One decode step: the greedy next token of every item.
    fn step(&mut self, items: &[StepItem]) -> Result<Vec<u32>, EngineError>;
    /// Ends the sequence in `slot` and frees its KV blocks.
    fn release(&mut self, slot: usize);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Length,
    Stop,
    Cancelled,
    Error(String),
}

/// What the scheduler tells a request's owner.
#[derive(Debug, PartialEq, Eq)]
pub enum Event {
    Token(u32),
    Finished { reason: FinishReason, prompt_tokens: usize, completion_tokens: usize },
}

pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub eos: Option<u32>,
    /// Keep generating after the end-of-sequence token (benchmarks fix the output length).
    pub ignore_eos: bool,
    pub cancel: Arc<AtomicBool>,
    pub tx: Sender<Event>,
}

struct Running {
    req: Request,
    slot: usize,
    /// Prompt length of the original request (usage reporting survives preemption).
    prompt_len: usize,
    /// Tokens generated so far (all of them, also across a preemption).
    generated: Vec<u32>,
    /// The token to feed next and its position.
    next_token: u32,
    pos: usize,
    admitted: u64,
}

struct Prefilling {
    req: Request,
    slot: usize,
    prompt_len: usize,
    generated: Vec<u32>,
}

pub struct Scheduler<E: Engine> {
    pub engine: E,
    waiting: VecDeque<(Request, Vec<u32>, usize)>, // request, tokens already generated, original prompt length
    prefilling: Option<Prefilling>,
    running: Vec<Running>,
    free_slots: Vec<usize>,
    admitted_counter: u64,
    /// Prefill chunks per decode step while prompts are waiting: a decode step has a high fixed
    /// cost (every weight is read once, however few sequences run), so several chunks per step
    /// raise throughput at the price of the running sequences' latency.
    pub chunks_per_step: usize,
    pub stats: Stats,
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub steps: u64,
    pub step_columns: u64,
    pub prefill_chunks: u64,
    pub preemptions: u64,
    pub finished: u64,
}

impl<E: Engine> Scheduler<E> {
    pub fn new(engine: E) -> Scheduler<E> {
        let slots = engine.slots();
        Scheduler { engine, waiting: VecDeque::new(), prefilling: None, running: Vec::new(), free_slots: (0..slots).rev().collect(), admitted_counter: 0, chunks_per_step: 1, stats: Stats::default() }
    }

    pub fn submit(&mut self, req: Request) {
        let len = req.prompt.len();
        self.waiting.push_back((req, Vec::new(), len));
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.prefilling.is_none() && self.running.is_empty()
    }

    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    fn finish(&mut self, req: &Request, reason: FinishReason, prompt_tokens: usize, completion_tokens: usize) {
        let _ = req.tx.send(Event::Finished { reason, prompt_tokens, completion_tokens });
        self.stats.finished += 1;
    }

    /// Admits the next waiting request as the prefill job when a slot and enough blocks are free.
    fn start_prefill(&mut self) {
        if self.prefilling.is_some() || self.free_slots.is_empty() {
            return;
        }
        let Some((req, generated, orig)) = self.waiting.pop_front() else { return };
        if req.cancel.load(Ordering::Relaxed) {
            self.finish(&req, FinishReason::Cancelled, orig, generated.len());
            return;
        }
        let tokens = req.prompt.len() + generated.len();
        if tokens + 1 >= self.engine.ctx_len() {
            let g = generated.len();
            self.finish(&req, FinishReason::Error(format!("the prompt of {tokens} tokens does not fit the context of {}", self.engine.ctx_len())), orig, g);
            return;
        }
        let need = tokens.div_ceil(self.engine.block_size()) + 1;
        if need > self.engine.free_blocks() && !self.running.is_empty() {
            // Wait for running sequences to finish and return their blocks.
            self.waiting.push_front((req, generated, orig));
            return;
        }
        // Re-prefill of a preempted sequence covers its prompt and the tokens generated before.
        let mut full = req.prompt.clone();
        full.extend_from_slice(&generated);
        match self.engine.prefill_begin(&full) {
            Ok(()) => {
                let slot = self.free_slots.pop().unwrap();
                self.prefilling = Some(Prefilling { req, slot, prompt_len: orig, generated });
            }
            Err(e) => {
                let g = generated.len();
                self.finish(&req, FinishReason::Error(e.to_string()), orig, g);
            }
        }
    }

    fn advance_prefill(&mut self) {
        let Some(job) = self.prefilling.as_mut() else { return };
        if job.req.cancel.load(Ordering::Relaxed) {
            let job = self.prefilling.take().unwrap();
            self.free_slots.push(job.slot);
            self.finish(&job.req, FinishReason::Cancelled, job.prompt_len, job.generated.len());
            return;
        }
        self.stats.prefill_chunks += 1;
        match self.engine.prefill_advance() {
            Ok(None) => {}
            Ok(Some(first)) => {
                let job = self.prefilling.take().unwrap();
                self.complete_prefill(job, first);
            }
            Err(e) => {
                let job = self.prefilling.take().unwrap();
                self.free_slots.push(job.slot);
                self.finish(&job.req, FinishReason::Error(e.to_string()), job.prompt_len, job.generated.len());
            }
        }
    }

    fn complete_prefill(&mut self, job: Prefilling, first: u32) {
        let Prefilling { req, slot, prompt_len, mut generated } = job;
        let pos = req.prompt.len() + generated.len();
        // The first token is produced by the prefill itself.
        if !self.deliver(&req, first, &mut generated) {
            self.free_slots.push(slot);
            let reason = self.reason_after(&req, first, generated.len());
            self.finish(&req, reason, prompt_len, generated.len());
            return;
        }
        if let Err(e) = self.engine.import(slot, pos) {
            self.engine.release(slot);
            self.free_slots.push(slot);
            self.finish(&req, FinishReason::Error(e.to_string()), prompt_len, generated.len());
            return;
        }
        self.admitted_counter += 1;
        self.running.push(Running { req, slot, prompt_len, generated, next_token: first, pos, admitted: self.admitted_counter });
    }

    /// Sends a generated token (unless it is the stop token) and reports whether the sequence goes on.
    fn deliver(&mut self, req: &Request, token: u32, generated: &mut Vec<u32>) -> bool {
        let is_eos = !req.ignore_eos && req.eos == Some(token);
        if is_eos {
            return false;
        }
        generated.push(token);
        if req.tx.send(Event::Token(token)).is_err() {
            req.cancel.store(true, Ordering::Relaxed);
            return false;
        }
        generated.len() < req.max_tokens
    }

    fn reason_after(&self, req: &Request, token: u32, generated: usize) -> FinishReason {
        if req.cancel.load(Ordering::Relaxed) {
            FinishReason::Cancelled
        } else if !req.ignore_eos && req.eos == Some(token) {
            FinishReason::Stop
        } else {
            let _ = generated;
            FinishReason::Length
        }
    }

    /// One decode step over all running sequences; preempts when the KV blocks run out.
    fn decode_step(&mut self) {
        // Cancelled sequences leave first.
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].req.cancel.load(Ordering::Relaxed) {
                let r = self.running.swap_remove(i);
                self.engine.release(r.slot);
                self.free_slots.push(r.slot);
                self.finish(&r.req, FinishReason::Cancelled, r.prompt_len, r.generated.len());
            } else {
                i += 1;
            }
        }
        let ctx = self.engine.ctx_len();
        // A sequence at the end of the context finishes with "length".
        let mut k = 0;
        while k < self.running.len() {
            if self.running[k].pos + 1 >= ctx {
                let r = self.running.swap_remove(k);
                self.engine.release(r.slot);
                self.free_slots.push(r.slot);
                self.finish(&r.req, FinishReason::Length, r.prompt_len, r.generated.len());
            } else {
                k += 1;
            }
        }
        if self.running.is_empty() {
            return;
        }
        let tokens = loop {
            let items: Vec<StepItem> = self.running.iter().map(|r| StepItem { slot: r.slot, token: r.next_token, pos: r.pos }).collect();
            match self.engine.step(&items) {
                Ok(t) => break t,
                Err(EngineError::OutOfBlocks) => {
                    // Recompute preemption: the newest sequence gives its blocks back and starts over later.
                    let Some(victim) = (0..self.running.len()).max_by_key(|&i| self.running[i].admitted) else { return };
                    let r = self.running.swap_remove(victim);
                    self.engine.release(r.slot);
                    self.free_slots.push(r.slot);
                    self.stats.preemptions += 1;
                    // The tokens generated so far are prefilled again with the prompt when it restarts.
                    self.waiting.push_front((r.req, r.generated, r.prompt_len));
                    if self.running.is_empty() {
                        return;
                    }
                }
                Err(e) => {
                    // A failing step ends every running sequence.
                    let msg = e.to_string();
                    let all: Vec<Running> = self.running.drain(..).collect();
                    for r in all {
                        self.engine.release(r.slot);
                        self.free_slots.push(r.slot);
                        self.finish(&r.req, FinishReason::Error(msg.clone()), r.prompt_len, r.generated.len());
                    }
                    return;
                }
            }
        };
        self.stats.steps += 1;
        self.stats.step_columns += tokens.len() as u64;
        let all: Vec<Running> = self.running.drain(..).collect();
        for (mut r, token) in all.into_iter().zip(tokens) {
            r.pos += 1;
            let mut generated = std::mem::take(&mut r.generated);
            let keep = self.deliver(&r.req, token, &mut generated);
            r.generated = generated;
            if keep {
                r.next_token = token;
                self.running.push(r);
            } else {
                self.engine.release(r.slot);
                self.free_slots.push(r.slot);
                let reason = self.reason_after(&r.req, token, r.generated.len());
                self.finish(&r.req, reason, r.prompt_len, r.generated.len());
            }
        }
    }

    /// One scheduling round: advance the prefill job by up to `chunks_per_step` chunks (while
    /// prompts are waiting), then decode all running sequences once.
    pub fn tick(&mut self) {
        for _ in 0..self.chunks_per_step.max(1) {
            self.start_prefill();
            self.advance_prefill();
            if self.prefilling.is_none() && self.waiting.is_empty() {
                break;
            }
        }
        self.decode_step();
    }
}

#[cfg(test)]
mod tests;