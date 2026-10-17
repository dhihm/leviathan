//! Schedule derivation for the Q4_K matvec: a pure function of the operation
//! shape and the measured device capabilities.

/// Measured device capabilities that influence legality and ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCaps {
    pub subgroup_size: u32,
    pub max_workgroup_invocations: u32,
}

/// Schedule parameters with a real effect on the emitted code.
/// `t`: threads cooperating on one row (1 = one thread per row).
/// `r`: rows per workgroup.
/// `xv`: width of the input-vector loads in floats (1 = scalar, 4 = vec4).
/// `m`: thread-data mapping. 0 = a thread owns whole blocks; 1 = a block is
/// split into 16 units of 8 contiguous bytes shared by the row's threads.
/// `sg`: reduction kind (see field). `rr`: rows per thread (1 for mapping 0); a workgroup covers `r * rr` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Q4kSchedule {
    pub t: u32,
    pub r: u32,
    pub xv: u32,
    pub m: u32,
    /// Rows each thread accumulates for the same input values (mapping 1 only).
    pub rr: u32,
    /// Row reduction: 0 = shared-memory tree, 1 = clustered subgroup add (mapping 1, t <= subgroup size).
    pub sg: u32,
    /// Units per thread per loop iteration (mapping 1); the row must divide evenly.
    pub u: u32,
    /// Input columns processed per weight read (1 or 2; 2 needs mapping 1, rr = 1).
    pub nb: u32,
}

impl Q4kSchedule {
    /// The stable fallback, always candidate 0.
    pub const BASELINE: Q4kSchedule = Q4kSchedule { t: 1, r: 1, xv: 1, m: 0, rr: 1, sg: 0, u: 1, nb: 1 };

    pub fn invocations(&self) -> u32 {
        self.t * self.r
    }
}

pub const MAX_CANDIDATES: usize = 96;
/// Each cooperating thread must get at least this many blocks per row (mapping 0).
pub const MIN_BLOCKS_PER_THREAD: usize = 4;
/// A split block is 16 units of 8 bytes (two uints).
pub const UNITS_PER_BLOCK: usize = 16;

const T_CHOICES: [u32; 5] = [1, 2, 4, 8, 16];
const R_CHOICES: [u32; 6] = [1, 2, 4, 8, 16, 32];
const XV_CHOICES: [u32; 2] = [1, 4];
const T_SPLIT_CHOICES: [u32; 5] = [8, 16, 32, 64, 128];
const RR_CHOICES: [u32; 3] = [1, 2, 4];

/// Whether `s` is legal for a row of `blocks_per_row` Q4_K blocks.
pub fn is_legal(s: &Q4kSchedule, blocks_per_row: usize, caps: &DeviceCaps) -> bool {
    if s.invocations() > caps.max_workgroup_invocations || !R_CHOICES.contains(&s.r) {
        return false;
    }
    let ok = match s.m {
        0 => {
            s.rr == 1
                && s.nb == 1
                && T_CHOICES.contains(&s.t)
                && XV_CHOICES.contains(&s.xv)
                && (s.t == 1 || blocks_per_row / s.t as usize >= MIN_BLOCKS_PER_THREAD)
        }
        1 => RR_CHOICES.contains(&s.rr)
            && matches!(s.nb, 1 | 2)
            && (s.nb == 1 || s.rr == 1)
            && matches!(s.u, 1 | 2 | 4)
            && (blocks_per_row * UNITS_PER_BLOCK) % (s.t * s.u) as usize == 0
            && (s.sg == 0 || (s.sg == 1 && s.t <= caps.subgroup_size && caps.subgroup_size % s.t == 0))
            && T_SPLIT_CHOICES.contains(&s.t) && s.xv == 4 && blocks_per_row * UNITS_PER_BLOCK >= s.t as usize,
        _ => false,
    };
    ok && (s.m == 1 || (s.sg == 0 && s.u == 1))
}

/// Bounded, deterministic candidate list: the baseline first, then the
/// whole-block mapping with small thread counts (the shapes that won in
/// EXP-02/03), then the split-block mapping.
pub fn derive(blocks_per_row: usize, caps: &DeviceCaps) -> Vec<Q4kSchedule> {
    let mut out = vec![Q4kSchedule::BASELINE];
    for &t in &[1u32, 2, 4] {
        for &r in &[1u32, 2, 4, 8] {
            out.push(Q4kSchedule { t, r, xv: 1, m: 0, rr: 1, sg: 0, u: 1, nb: 1 });
        }
    }
    for &t in &T_SPLIT_CHOICES {
        for &r in &[1u32, 2, 4, 8] {
            if (32..=256).contains(&(t * r)) {
                for &sg in &[0u32, 1] {
                    for &u in &[1u32, 2, 4] {
                        out.push(Q4kSchedule { t, r, xv: 4, m: 1, rr: 1, sg, u, nb: 1 });
                    }
                }
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|s| is_legal(s, blocks_per_row, caps) && seen.insert(*s));
    out.truncate(MAX_CANDIDATES);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAPS: DeviceCaps = DeviceCaps { subgroup_size: 64, max_workgroup_invocations: 1024 };

    #[test]
    fn baseline_is_first_and_list_is_bounded() {
        for blocks in [16usize, 48, 96] {
            let c = derive(blocks, &CAPS);
            assert_eq!(c[0], Q4kSchedule::BASELINE);
            assert!(c.len() <= MAX_CANDIDATES);
            assert!(c.len() >= 2);
        }
    }

    #[test]
    fn every_candidate_is_legal_and_unique() {
        for blocks in [16usize, 48, 96] {
            let c = derive(blocks, &CAPS);
            for s in &c {
                assert!(is_legal(s, blocks, &CAPS), "{s:?} at {blocks} blocks");
            }
            let mut seen = std::collections::HashSet::new();
            assert!(c.iter().all(|s| seen.insert(*s)));
        }
    }

    #[test]
    fn small_rows_admit_only_small_cooperation() {
        // 16 blocks per row: T=8 would leave 2 blocks per thread, below the minimum.
        assert!(!is_legal(&Q4kSchedule { t: 8, r: 1, xv: 1, m: 0, rr: 1, sg: 0, u: 1, nb: 1 }, 16, &CAPS));
        assert!(is_legal(&Q4kSchedule { t: 4, r: 1, xv: 1, m: 0, rr: 1, sg: 0, u: 1, nb: 1 }, 16, &CAPS));
    }

    #[test]
    fn split_mapping_needs_vec4_and_enough_units() {
        assert!(is_legal(&Q4kSchedule { t: 64, r: 1, xv: 4, m: 1, rr: 1, sg: 0, u: 1, nb: 1 }, 16, &CAPS));
        assert!(!is_legal(&Q4kSchedule { t: 64, r: 1, xv: 1, m: 1, rr: 1, sg: 0, u: 1, nb: 1 }, 16, &CAPS));
        // 2 blocks = 32 units: T=64 would leave threads without work.
        assert!(!is_legal(&Q4kSchedule { t: 64, r: 1, xv: 4, m: 1, rr: 1, sg: 0, u: 1, nb: 1 }, 2, &CAPS));
    }

    #[test]
    fn workgroup_limit_is_respected() {
        let tiny = DeviceCaps { subgroup_size: 32, max_workgroup_invocations: 16 };
        for s in derive(96, &tiny) {
            assert!(s.invocations() <= 16);
        }
    }

    #[test]
    fn derivation_is_deterministic() {
        assert_eq!(derive(48, &CAPS), derive(48, &CAPS));
    }
}
