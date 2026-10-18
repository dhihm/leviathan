use super::*;
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver};

/// Deterministic engine: the next token depends only on (input token, position), so any batching
/// must produce the same sequence as running the rule alone.
fn rule(token: u32, pos: usize) -> u32 {
    (token * 7 + pos as u32) % 1000 + 1
}

struct Mock {
    slots: usize,
    ctx: usize,
    total_blocks: usize,
    held: HashMap<usize, usize>, // slot -> positions held
    prefill: Option<(Vec<u32>, usize)>,
    chunk: usize,
    pub steps: Vec<Vec<StepItem>>,
    pub order: Vec<&'static str>,
}

impl Mock {
    fn new(slots: usize, total_blocks: usize) -> Mock {
        Mock { slots, ctx: 4096, total_blocks, held: HashMap::new(), prefill: None, chunk: 8, steps: Vec::new(), order: Vec::new() }
    }
    fn used_blocks(&self) -> usize {
        self.held.values().map(|&n| n.div_ceil(16)).sum()
    }
}

impl Engine for Mock {
    fn slots(&self) -> usize {
        self.slots
    }
    fn ctx_len(&self) -> usize {
        self.ctx
    }
    fn free_blocks(&self) -> usize {
        self.total_blocks - self.used_blocks()
    }
    fn block_size(&self) -> usize {
        16
    }
    fn prefill_begin(&mut self, prompt: &[u32]) -> Result<(), EngineError> {
        self.prefill = Some((prompt.to_vec(), 0));
        Ok(())
    }
    fn prefill_advance(&mut self) -> Result<Option<u32>, EngineError> {
        self.order.push("prefill");
        let (p, done) = self.prefill.as_mut().unwrap();
        *done = (*done + self.chunk).min(p.len());
        if *done == p.len() {
            let first = rule(*p.last().unwrap(), p.len() - 1);
            return Ok(Some(first));
        }
        Ok(None)
    }
    fn import(&mut self, slot: usize, prompt_len: usize) -> Result<(), EngineError> {
        if self.used_blocks() + prompt_len.div_ceil(16) > self.total_blocks {
            return Err(EngineError::OutOfBlocks);
        }
        self.held.insert(slot, prompt_len);
        Ok(())
    }
    fn step(&mut self, items: &[StepItem]) -> Result<Vec<u32>, EngineError> {
        self.order.push("step");
        // Blocks needed after this step.
        let mut need = 0;
        for it in items {
            let have = self.held[&it.slot];
            need += (it.pos + 1).div_ceil(16) - have.div_ceil(16);
        }
        if self.used_blocks() + need > self.total_blocks {
            return Err(EngineError::OutOfBlocks);
        }
        self.steps.push(items.to_vec());
        for it in items {
            self.held.insert(it.slot, it.pos + 1);
        }
        Ok(items.iter().map(|it| rule(it.token, it.pos)).collect())
    }
    fn release(&mut self, slot: usize) {
        self.held.remove(&slot);
    }
}

fn request(id: u64, prompt: Vec<u32>, max_tokens: usize) -> (Request, Receiver<Event>, Arc<AtomicBool>) {
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    (Request { id, prompt, max_tokens, eos: None, ignore_eos: false, cancel: cancel.clone(), tx }, rx, cancel)
}

/// The sequence the rule produces alone for `prompt`.
fn expected(prompt: &[u32], n: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut t = rule(*prompt.last().unwrap(), prompt.len() - 1);
    let mut pos = prompt.len();
    for _ in 0..n {
        out.push(t);
        t = rule(t, pos);
        pos += 1;
    }
    out
}

fn drain(rx: &Receiver<Event>) -> (Vec<u32>, Option<Event>) {
    let (mut toks, mut fin) = (Vec::new(), None);
    while let Ok(e) = rx.try_recv() {
        match e {
            Event::Token(t) => toks.push(t),
            f => fin = Some(f),
        }
    }
    (toks, fin)
}

fn run_to_idle<E: Engine>(s: &mut Scheduler<E>) {
    for _ in 0..10_000 {
        if s.is_idle() {
            return;
        }
        s.tick();
    }
    panic!("the scheduler did not become idle");
}

#[test]
fn a_single_request_runs_to_its_length() {
    let mut s = Scheduler::new(Mock::new(4, 64));
    let prompt: Vec<u32> = (1..=20).collect();
    let (req, rx, _) = request(1, prompt.clone(), 10);
    s.submit(req);
    run_to_idle(&mut s);
    let (toks, fin) = drain(&rx);
    assert_eq!(toks, expected(&prompt, 10));
    assert_eq!(fin, Some(Event::Finished { reason: FinishReason::Length, prompt_tokens: 20, completion_tokens: 10 }));
    assert_eq!(s.engine.free_blocks(), 64, "all blocks are returned");
}

#[test]
fn concurrent_requests_share_decode_steps_and_keep_their_own_sequences() {
    let mut s = Scheduler::new(Mock::new(4, 64));
    let prompts: Vec<Vec<u32>> = vec![(1..=5).collect(), (10..=30).collect(), (100..=103).collect()];
    let rxs: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (req, rx, _) = request(i as u64, p.clone(), 12);
            s.submit(req);
            rx
        })
        .collect();
    run_to_idle(&mut s);
    for (p, rx) in prompts.iter().zip(&rxs) {
        let (toks, _) = drain(rx);
        assert_eq!(toks, expected(p, 12));
    }
    // At some step all three ran together.
    assert!(s.engine.steps.iter().any(|st| st.len() == 3), "no step held all three sequences");
    assert!(s.stats.step_columns > s.stats.steps, "steps carried more than one column on average");
}

#[test]
fn requests_wait_for_a_free_slot_in_arrival_order() {
    let mut s = Scheduler::new(Mock::new(1, 64));
    let mut rxs = Vec::new();
    for i in 0..3u32 {
        let (req, rx, _) = request(i as u64, vec![i + 1, i + 2], 4);
        s.submit(req);
        rxs.push(rx);
    }
    s.tick();
    assert_eq!(s.running_count() + usize::from(s.prefilling.is_some()), 1);
    assert_eq!(s.waiting_count(), 2);
    run_to_idle(&mut s);
    for (i, rx) in rxs.iter().enumerate() {
        let (toks, fin) = drain(rx);
        assert_eq!(toks, expected(&[i as u32 + 1, i as u32 + 2], 4));
        assert!(matches!(fin, Some(Event::Finished { reason: FinishReason::Length, .. })));
    }
}

#[test]
fn the_end_of_sequence_token_stops_unless_ignored() {
    let prompt = vec![3u32, 4, 5];
    let seq = expected(&prompt, 8);
    let eos = seq[3];
    for ignore in [false, true] {
        let mut s = Scheduler::new(Mock::new(2, 64));
        let (mut req, rx, _) = request(1, prompt.clone(), 8);
        req.eos = Some(eos);
        req.ignore_eos = ignore;
        s.submit(req);
        run_to_idle(&mut s);
        let (toks, fin) = drain(&rx);
        if ignore {
            assert_eq!(toks, seq);
            assert!(matches!(fin, Some(Event::Finished { reason: FinishReason::Length, completion_tokens: 8, .. })));
        } else {
            assert_eq!(toks, seq[..3].to_vec(), "the stop token itself is not delivered");
            assert!(matches!(fin, Some(Event::Finished { reason: FinishReason::Stop, completion_tokens: 3, .. })));
        }
    }
}

#[test]
fn cancelled_requests_free_their_slot_and_blocks() {
    let mut s = Scheduler::new(Mock::new(2, 64));
    let (req, rx, cancel) = request(1, (1..=40).collect(), 1000);
    s.submit(req);
    for _ in 0..8 {
        s.tick();
    }
    assert_eq!(s.running_count(), 1);
    cancel.store(true, Ordering::Relaxed);
    s.tick();
    assert!(s.is_idle());
    assert_eq!(s.engine.free_blocks(), 64);
    let (_, fin) = drain(&rx);
    assert!(matches!(fin, Some(Event::Finished { reason: FinishReason::Cancelled, .. })));
}

#[test]
fn a_dropped_receiver_cancels_the_request() {
    let mut s = Scheduler::new(Mock::new(2, 64));
    let (req, rx, _) = request(1, vec![1, 2, 3], 1000);
    s.submit(req);
    drop(rx);
    for _ in 0..50 {
        s.tick();
    }
    assert!(s.is_idle());
}

#[test]
fn running_out_of_blocks_preempts_the_newest_and_everything_still_completes() {
    // Two sequences that together outgrow the pool: positions up to ~100 need 7 blocks each.
    let mut s = Scheduler::new(Mock::new(2, 9));
    let p1: Vec<u32> = (1..=20).collect();
    let p2: Vec<u32> = (50..=70).collect();
    let (r1, rx1, _) = request(1, p1.clone(), 80);
    let (r2, rx2, _) = request(2, p2.clone(), 80);
    s.submit(r1);
    s.submit(r2);
    run_to_idle(&mut s);
    assert!(s.stats.preemptions >= 1, "the pool was big enough to avoid preemption");
    let (t1, f1) = drain(&rx1);
    let (t2, f2) = drain(&rx2);
    // The rule only depends on (token, position), so a restarted sequence continues identically.
    assert_eq!(t1, expected(&p1, 80));
    assert_eq!(t2, expected(&p2, 80));
    assert!(matches!(f1, Some(Event::Finished { completion_tokens: 80, .. })));
    assert!(matches!(f2, Some(Event::Finished { completion_tokens: 80, .. })));
    assert_eq!(s.engine.free_blocks(), 9);
}

#[test]
fn a_long_prefill_does_not_stall_decoding() {
    let mut s = Scheduler::new(Mock::new(4, 400));
    let (short, rx_short, _) = request(1, vec![1, 2, 3], 200);
    s.submit(short);
    for _ in 0..6 {
        s.tick();
    }
    let before = s.stats.steps;
    // A long prompt arrives: its prefill takes many chunks.
    let long: Vec<u32> = (0..800).map(|i| i % 90 + 1).collect();
    let (req, rx_long, _) = request(2, long.clone(), 5);
    s.submit(req);
    let chunks_before = s.stats.prefill_chunks;
    for _ in 0..20 {
        s.tick();
    }
    assert!(s.stats.prefill_chunks > chunks_before + 5, "the prefill advanced chunk by chunk");
    assert!(s.stats.steps >= before + 20, "decode steps continued during the prefill");
    run_to_idle(&mut s);
    assert_eq!(drain(&rx_short).0, expected(&[1, 2, 3], 200));
    assert_eq!(drain(&rx_long).0, expected(&long, 5));
}

#[test]
fn a_prompt_longer_than_the_context_is_rejected() {
    let mut s = Scheduler::new(Mock::new(2, 64));
    s.engine.ctx = 64;
    let (req, rx, _) = request(1, (0..100).collect(), 5);
    s.submit(req);
    s.tick();
    let (_, fin) = drain(&rx);
    assert!(matches!(fin, Some(Event::Finished { reason: FinishReason::Error(_), .. })));
    assert!(s.is_idle());
}