//! The Vulkan engine behind the scheduler's `Engine` trait.

use lev_model::gpu::{BatchItem, GpuModel, PrefillState};
use lev_model::kv::OutOfBlocks;

use crate::sched::{Engine, EngineError, StepItem};

pub struct GpuEngine {
    pub gpu: GpuModel,
    prefill: Option<PrefillState>,
}

impl GpuEngine {
    pub fn new(gpu: GpuModel) -> GpuEngine {
        GpuEngine { gpu, prefill: None }
    }
}

fn map(e: Box<dyn std::error::Error>) -> EngineError {
    if e.downcast_ref::<OutOfBlocks>().is_some() {
        EngineError::OutOfBlocks
    } else {
        EngineError::Other(e.to_string())
    }
}

impl Engine for GpuEngine {
    fn slots(&self) -> usize {
        self.gpu.batch_slots()
    }

    fn ctx_len(&self) -> usize {
        self.gpu.context_len()
    }

    fn free_blocks(&self) -> usize {
        // Without a paged pool (slab mode) the slabs are preallocated: no block shortage.
        self.gpu.kv_blocks().map_or(usize::MAX / 2, |(free, _)| free)
    }

    fn block_size(&self) -> usize {
        lev_model::kv::PAGE_SIZE
    }

    fn prefill_begin(&mut self, prompt: &[u32]) -> Result<(), EngineError> {
        let p: Vec<usize> = prompt.iter().map(|&t| t as usize).collect();
        self.prefill = Some(self.gpu.prefill_begin(&p).map_err(map)?);
        Ok(())
    }

    fn prefill_advance(&mut self) -> Result<Option<u32>, EngineError> {
        let st = self.prefill.as_mut().ok_or_else(|| EngineError::Other("no prefill in progress".into()))?;
        let r = self.gpu.prefill_advance(st).map_err(map)?;
        if r.is_some() {
            self.prefill = None;
        }
        Ok(r.map(|t| t as u32))
    }

    fn import(&mut self, slot: usize, _prompt_len: usize) -> Result<(), EngineError> {
        self.gpu.batch_import(slot).map_err(map)
    }

    fn step(&mut self, items: &[StepItem]) -> Result<Vec<u32>, EngineError> {
        let mut out = Vec::with_capacity(items.len());
        // The widest program holds 64 columns; a larger batch is split.
        for part in items.chunks(64) {
            let b: Vec<BatchItem> = part.iter().map(|i| BatchItem { slot: i.slot, token: i.token as usize, pos: i.pos }).collect();
            out.extend(self.gpu.batch_step(&b).map_err(map)?.into_iter().map(|(t, _)| t as u32));
        }
        Ok(out)
    }

    fn release(&mut self, slot: usize) {
        self.gpu.batch_release(slot);
    }
}