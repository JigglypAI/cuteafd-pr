//! Admission and runtime share the same Inkling graph limit.
use crate::placement::graph_reserve::{self, Error, Resolved};

pub fn policy(rank: usize, ranks: usize) -> Result<Resolved, Error> {
    if std::env::var_os("CUTEAFD_INKLING2_GRAPHS").is_some_and(|on| on == "0") {
        return graph_reserve::resolve(Some("0"), rank, false, 0, 0);
    }
    // Provisional first-census bounds, not trace-derived working sets. TP2
    // keeps its prior 256 MiB base; TP1 starts with the same bounded allowance.
    // An explicit GPU_GRAPH_RESERVE can be used for a larger census.
    debug_assert!(rank < ranks && matches!(ranks, 1 | 2));
    graph_reserve::current(rank, false, 256 << 20, 0)
}
