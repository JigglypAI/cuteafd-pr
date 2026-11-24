//! Shared #2697 working-set soft-cap graph reserve for Inkling 2 Mini (TP1).
//!
//! Soft cap = `W1 + max(20% W1, 100 MB, 2 * step_growth)`, rounded up to 2 MiB;
//! hard cap = `soft + 2 * capture_growth` (clamped to <= soft).
use crate::placement::graph_reserve::{self, Error, Resolved, WorkingSet};

const MIB: u64 = 1 << 20;

/// Working-set seed for 1x RTX PRO 6000 (48 exact-row visits, rank-local TP1).
/// Updated to measured W1/step/capture once live graph census is collected.
pub const WORKING_SET: WorkingSet = WorkingSet {
    w1: 220 * MIB,
    step_growth: 16 * MIB,
    capture_growth: 16 * MIB,
};

fn resolve(
    rank: usize,
    enabled: bool,
    total: Option<&str>,
    extra: Option<&str>,
    cap: Option<&str>,
) -> Result<Resolved, Error> {
    if !enabled {
        return graph_reserve::resolve(Some("0"), rank, false, 0, 0);
    }
    graph_reserve::resolve_working_set(total, extra, cap, rank, WORKING_SET)
}

pub fn policy(rank: usize) -> Result<Resolved, Error> {
    let enabled = !std::env::var_os("CUTEAFD_INKLING2_MINI_GRAPHS").is_some_and(|on| on == "0");
    if !enabled {
        return resolve(rank, false, None, None, None);
    }
    let read = |key: &'static str| match std::env::var(format!("CUTEAFD_{key}")) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(Error::Environment { key }),
    };
    resolve(
        rank,
        true,
        read("GPU_GRAPH_RESERVE")?.as_deref(),
        read("GPU_GRAPH_RESERVE_EXTRA")?.as_deref(),
        read("GPU_GRAPH_RESERVE_EXTRA_CAP")?.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_working_set_uses_shared_2697_soft_cap_and_headroom() {
        let p = resolve(0, true, None, None, None).unwrap();
        // W1 = 220 MiB; max(20% W1 = 44 MiB, 100 MB, 2*16 MiB = 32 MiB) = 100 MB -> soft = 316 MiB, headroom = 32 MiB
        assert_eq!(p.base_bytes, 316 * MIB);
        assert_eq!(p.growth_headroom, Some(32 * MIB));
        assert_eq!(p.bytes, 348 * MIB);
        assert_eq!(resolve(0, false, Some("2GiB"), None, None).unwrap().bytes, 0);
        assert_eq!(resolve(0, true, Some("0"), None, None).unwrap().bytes, 0);
        assert_eq!(resolve(0, true, Some("gpu0=256MiB"), None, None).unwrap().bytes, 256 * MIB);
        assert!(resolve(0, true, None, Some("10MB"), None).is_err());
    }
}
