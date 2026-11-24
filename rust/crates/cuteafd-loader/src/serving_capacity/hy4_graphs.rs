//! Lazy rank-local decode graphs and per-rank C1 working-set admission.

use crate::placement::graph_reserve::{self, Error, Resolved, WorkingSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hy4GraphLayout {
    Exl3Min,
    Exl3Max,
    Exl3MinSpark,
    Exl3MaxSpark,
}

impl Hy4GraphLayout {
    pub fn resolve(ranks: usize, sparks: usize) -> Option<Self> {
        match (ranks, sparks) {
            (1, 0) => Some(Self::Exl3Min),
            (2, 0) => Some(Self::Exl3Max),
            (1, s) if s > 0 => Some(Self::Exl3MinSpark),
            (2, s) if s > 0 => Some(Self::Exl3MaxSpark),
            _ => None,
        }
    }

}

/// Short and 32K C1 physical graph-arena censuses on the reference cards.
/// Local cards use DFly2; the 1-Spark census uses the stock Spark setting
/// (DFly2 off) and supplies the single-GPU Spark layouts.
fn working_set(layout: Option<Hy4GraphLayout>, rank: usize) -> Option<WorkingSet> {
    let (w1, step_growth, capture_growth) = match (layout, rank) {
        (Some(Hy4GraphLayout::Exl3Min), 0) => (88, 18, 4),
        (Some(Hy4GraphLayout::Exl3Max), 0) => (100, 20, 4),
        (Some(Hy4GraphLayout::Exl3Max), 1) => (98, 18, 4),
        (Some(Hy4GraphLayout::Exl3MinSpark), 0) => (12, 12, 2),
        _ => return None,
    };
    Some(WorkingSet {
        w1: w1 << 20,
        step_growth: step_growth << 20,
        capture_growth: capture_growth << 20,
    })
}

/// Use identical inputs for admission, provenance, and runtime. Unmeasured
/// layouts do not borrow a census; explicit reserves still work there.
pub fn reserve_policy(layout: Option<Hy4GraphLayout>, rank: usize) -> Result<Resolved, Error> {
    match working_set(layout, rank) {
        Some(measured) => graph_reserve::current_working_set(rank, measured),
        None => graph_reserve::current(rank, false, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c1_census_sets_rank_local_soft_and_hard_caps() {
        use Hy4GraphLayout::*;
        for (layout, rank, w1, soft, hard) in [
            (Exl3Min, 0, 88, 184, 192),
            (Exl3Max, 0, 100, 196, 204),
            (Exl3Max, 1, 98, 194, 202),
            (Exl3MinSpark, 0, 12, 108, 112),
        ] {
            let census = working_set(Some(layout), rank).unwrap();
            assert_eq!(census.w1, w1 << 20);
            let policy = graph_reserve::resolve_working_set(None, None, None, rank, census).unwrap();
            assert_eq!(policy.base_bytes, soft << 20);
            assert_eq!(policy.bytes, hard << 20);
        }
    }

    #[test]
    fn unmeasured_ranks_do_not_borrow_graph_working_sets() {
        assert_eq!(working_set(Hy4GraphLayout::resolve(1, 0), 1), None);
        assert_eq!(working_set(Hy4GraphLayout::resolve(2, 0), 2), None);
        assert_eq!(working_set(Hy4GraphLayout::resolve(2, 6), 1), None);
        assert_eq!(working_set(Hy4GraphLayout::resolve(3, 0), 0), None);
        assert_eq!(working_set(Hy4GraphLayout::resolve(1, 1), 0),
                   working_set(Hy4GraphLayout::resolve(1, 4), 0));
    }
}
