//! Paged KV cache bookkeeping: which physical blocks hold which positions of which sequence.
//!
//! Block 0 is reserved as the scratch block that padding columns write to, so a real sequence never
//! owns it. A block holds `PAGE_SIZE` consecutive positions of one sequence in every attention
//! layer (the layers share the block ids, each has its own pool).

pub use lev_forge::ops::PAGE_SIZE;

#[derive(Debug)]
pub struct BlockManager {
    free: Vec<u32>,
    owned: Vec<Vec<u32>>,
    max_blocks: usize,
    total: usize,
}

/// The error `GpuModel::batch_step` and `batch_import` return when the pool has no free block
/// (downcast the boxed error to this type to tell it from other failures).
#[derive(Debug, PartialEq, Eq)]
pub struct OutOfBlocks;

impl std::fmt::Display for OutOfBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KV cache: no free blocks")
    }
}

impl std::error::Error for OutOfBlocks {}

#[derive(Debug, PartialEq, Eq)]
pub enum KvError {
    OutOfBlocks,
    BeyondContext,
}

impl BlockManager {
    /// `blocks` physical blocks (block 0 is the scratch block), `slots` sequences, at most
    /// `max_blocks` blocks per sequence.
    pub fn new(blocks: usize, slots: usize, max_blocks: usize) -> BlockManager {
        assert!(blocks >= 2, "need the scratch block and at least one more");
        // Hand out low ids first (pop from the end of a descending list).
        BlockManager { free: (1..blocks as u32).rev().collect(), owned: vec![Vec::new(); slots], max_blocks, total: blocks }
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    pub fn total_blocks(&self) -> usize {
        self.total
    }

    pub fn blocks_of(&self, slot: usize) -> &[u32] {
        &self.owned[slot]
    }

    /// Makes sure the sequence in `slot` owns the block of position `pos` and returns the flat pool
    /// position (block * PAGE_SIZE + offset) of that position.
    pub fn ensure(&mut self, slot: usize, pos: usize) -> Result<u32, KvError> {
        let need = pos / PAGE_SIZE + 1;
        if need > self.max_blocks {
            return Err(KvError::BeyondContext);
        }
        while self.owned[slot].len() < need {
            let b = self.free.pop().ok_or(KvError::OutOfBlocks)?;
            self.owned[slot].push(b);
        }
        Ok(self.owned[slot][pos / PAGE_SIZE] * PAGE_SIZE as u32 + (pos % PAGE_SIZE) as u32)
    }

    /// Returns all blocks of `slot` to the free list.
    pub fn release(&mut self, slot: usize) {
        while let Some(b) = self.owned[slot].pop() {
            self.free.push(b);
        }
    }

    /// The block table as the kernels read it: `[slots + 1][max_blocks]`, the last row (the scratch
    /// slot) and unused entries pointing to the scratch block 0.
    pub fn table(&self) -> Vec<u32> {
        let mut t = vec![0u32; (self.owned.len() + 1) * self.max_blocks];
        for (s, blocks) in self.owned.iter().enumerate() {
            t[s * self.max_blocks..s * self.max_blocks + blocks.len()].copy_from_slice(blocks);
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_are_allocated_on_demand_and_returned() {
        let mut m = BlockManager::new(6, 2, 4);
        assert_eq!(m.free_blocks(), 5);
        assert_eq!(m.ensure(0, 0), Ok(PAGE_SIZE as u32)); // first block is id 1
        assert_eq!(m.ensure(0, 15), Ok(2 * PAGE_SIZE as u32 - 1));
        assert_eq!(m.blocks_of(0).len(), 1);
        m.ensure(1, 16).unwrap(); // two blocks at once
        assert_eq!(m.blocks_of(1).len(), 2);
        assert_eq!(m.free_blocks(), 2);
        m.release(1);
        assert_eq!(m.free_blocks(), 4);
        assert!(m.blocks_of(1).is_empty());
    }

    #[test]
    fn running_out_and_exceeding_the_context_are_errors() {
        let mut m = BlockManager::new(3, 2, 2);
        m.ensure(0, 0).unwrap();
        m.ensure(1, 0).unwrap();
        assert_eq!(m.ensure(0, 16), Err(KvError::OutOfBlocks));
        assert_eq!(m.ensure(0, 32), Err(KvError::BeyondContext));
    }

    #[test]
    fn the_table_never_contains_a_scratch_block_for_a_real_position() {
        let mut m = BlockManager::new(8, 2, 3);
        for p in 0..40 {
            m.ensure(0, p).unwrap();
        }
        let t = m.table();
        assert_eq!(t.len(), 3 * 3);
        assert!(t[..3].iter().all(|&b| b != 0));
        assert!(t[6..].iter().all(|&b| b == 0)); // scratch slot row
    }
}