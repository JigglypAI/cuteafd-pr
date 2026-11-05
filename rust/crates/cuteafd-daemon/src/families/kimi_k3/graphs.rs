//! Exact startup-only rank-local arithmetic graphs. Transport and event joins
//! remain in the shared layer executor, outside these captured segments.
use super::block::MixSite;
use crate::shared::{decode_graph::{GraphBank, GraphDecision, GraphOwner, GraphPolicy},
    memory::device::Stream};
use anyhow::{ensure, Result};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config},
    placement::inventory::StartupGraphs};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Op {
    Residual(MixSite),
    Attention { layer: usize, sequences: usize, speculative: bool },
    Context { layer: usize, speculative: bool, phase: cuteafd_loader::families::kimi_k3::context::GraphPhase },
    Ffn(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key { pub lane: usize, pub rank: usize, pub rows: usize, pub op: Op }

pub(crate) use cuteafd_loader::families::kimi_k3::allocation::{GRAPH_RESERVE as RESERVE,
    graph_demand as demand};
use cuteafd_loader::families::kimi_k3::allocation::{graph_rows, verify_graph_rows};

/// An estimated reserve until the first real serving census qualifies its bytes.
/// The exact same key set is charged by admission and sealed by each binding.
/// `verify` (the drafter's block, 0 without one) adds every exact verify width
/// with its speculative single-sequence attention, the C1 verify shape.
pub(crate) fn inventory(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize, verify: usize,
    resident: impl Fn(usize, usize) -> bool) -> Result<StartupGraphs<Key>> {
    inventory_for_ranks(cfg, lanes, capacity, slots, verify, 2, resident)
}

pub(crate) fn inventory_for_ranks(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize, verify: usize,
    ranks: usize, resident: impl Fn(usize, usize) -> bool) -> Result<StartupGraphs<Key>> {
    inventory_with_context(cfg, lanes, capacity, slots, verify, ranks, resident, |_| false, None)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn inventory_with_context(cfg: &KimiK3Config, lanes: usize, capacity: usize, slots: usize,
    verify: usize, ranks: usize, resident: impl Fn(usize, usize) -> bool, context: impl Fn(usize) -> bool,
    decode: Option<&cuteafd_loader::families::kimi_k3::context::DecodeBuckets>)
    -> Result<StartupGraphs<Key>> {
    ensure!(matches!(ranks, 1 | 2) && (1..=2).contains(&lanes) && (1..=4096).contains(&capacity) && (1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&slots), "Kimi graph lane/row capacity");
    let mut keys = Vec::new();
    for lane in 0..lanes {
        for rank in 0..ranks {
            for rows in graph_rows(verify, capacity) {
                for layer in 0..cfg.layers {
                    keys.push(Key { lane, rank, rows, op: Op::Residual(MixSite::Attention(layer)) });
                    keys.push(Key { lane, rank, rows, op: Op::Residual(MixSite::Ffn(layer)) });
                    keys.push(Key { lane, rank, rows, op: Op::Ffn(layer) });
                    if resident(rank, layer) && !context(layer) {
                        let mut attention = |speculative| {
                            keys.push(Key { lane, rank, rows, op: Op::Attention { layer, sequences: 1, speculative } });
                        };
                        if cuteafd_loader::families::kimi_k3::allocation::GRAPH_ROWS.contains(&rows) {
                            attention(false);
                            if cfg.attention[layer] == KimiAttention::Kda && (2..=slots).contains(&rows) {
                                keys.push(Key { lane, rank, rows, op: Op::Attention { layer, sequences: rows, speculative: false } });
                            }
                        }
                        if verify_graph_rows(verify, capacity).contains(&rows) {
                            // Keep ordinary and speculative keys distinct even
                            // when they share a plain row bucket.
                            keys.push(Key { lane, rank, rows, op: Op::Attention { layer, sequences: 1, speculative: true } });
                        }
                    }
                }
                if rank == ranks - 1 { keys.push(Key { lane, rank, rows, op: Op::Residual(MixSite::Output) }); }
            }
            if let Some(decode) = decode {
                for rows in decode.graph_rows(verify, capacity) {
                    for layer in (0..cfg.layers).filter(|&layer| context(layer) && resident(rank, layer)) {
                        for speculative in [false, true] {
                            if if speculative { verify_graph_rows(verify, capacity).contains(&rows) }
                                else { decode.plain().contains(&rows) } {
                                for phase in cuteafd_loader::families::kimi_k3::context::graph_phases(rows) {
                                    keys.push(Key { lane, rank, rows, op: Op::Context { layer, speculative, phase } });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    cuteafd_loader::families::kimi_k3::allocation::graphs_with_context(cfg, lanes, capacity, slots, verify, ranks, resident, context, decode)?
        .startup(keys).map_err(anyhow::Error::msg)
}

/// Use the solver's alternating selected-layer helper, not layer parity: KDA
/// gaps must not alter the MLA owner's ordinal.
pub(crate) fn owned_inventory(cfg: &KimiK3Config, lanes: usize, rows: usize, slots: usize, verify: usize)
    -> Result<StartupGraphs<Key>> {
    let owners = cuteafd_loader::placement::attention::alternating_owners(
        cfg.attention.iter().map(|kind| (*kind == KimiAttention::Mla, None)), 0)?;
    inventory(cfg, lanes, rows, slots, verify, |rank, layer|
        cfg.attention[layer] == KimiAttention::Kda || rank == usize::from(owners[layer]))
}

pub(crate) fn check_placement(cfg: &KimiK3Config, placement: &cuteafd_loader::placement::Placement,
    keys: &[Key]) -> Result<()> {
    check_placement_for_ranks(cfg, placement, keys, 2)
}

pub(crate) fn check_placement_for_ranks(cfg: &KimiK3Config, placement: &cuteafd_loader::placement::Placement,
    keys: &[Key], ranks: usize) -> Result<()> {
    use cuteafd_loader::placement::LayerMode;
    ensure!(matches!(ranks, 1 | 2) && keys.iter().all(|key| key.rank < ranks),
        "Kimi graph physical rank count");
    ensure!(placement.layers.len() == cfg.layers, "Kimi graph placement layer count");
    let inventory: std::collections::HashSet<_> = keys.iter().copied().collect();
    for key in keys {
        if let Op::Context { layer, speculative, phase } = key.op {
            ensure!(ranks == 2 && cfg.attention.get(layer) == Some(&KimiAttention::Mla)
                && placement.layers.get(layer).is_some_and(|a| a.mode == LayerMode::ContextSplit)
                && (!speculative || (2..=64).contains(&key.rows))
                && cuteafd_loader::families::kimi_k3::context::graph_phases(key.rows).contains(&phase),
                "Kimi context graph differs from accepted placement/phase");
        }
        if let Op::Attention { layer, .. } = key.op {
            let mode = placement.layers.get(layer).ok_or_else(|| anyhow::anyhow!("Kimi graph layer"))?.mode;
            ensure!(matches!(mode, LayerMode::HeadSplit)
                || matches!(mode, LayerMode::Whole { gpu, .. } if usize::from(gpu) == key.rank),
                "Kimi attention graph is absent from accepted placement");
        }
    }
    for (layer, assignment) in placement.layers.iter().enumerate() {
        ensure!(ranks == 2 || matches!(assignment.mode, LayerMode::Whole { gpu: 0, .. }),
            "Kimi single-owner graphs require whole attention residency");
        for rank in 0..ranks {
            let resident = matches!(assignment.mode, LayerMode::HeadSplit | LayerMode::ContextSplit)
                || matches!(assignment.mode, LayerMode::Whole { gpu, .. } if usize::from(gpu) == rank);
            ensure!(keys.iter().any(|key| key.rank == rank && matches!(key.op,
                Op::Attention { layer: captured, .. } | Op::Context { layer: captured, .. } if captured == layer)) == resident,
                "Kimi graph inventory differs from accepted attention residency");
            if assignment.mode == LayerMode::ContextSplit {
                for key in keys.iter().filter(|key| key.rank == rank && matches!(key.op,
                    Op::Context { layer: captured, .. } if captured == layer)) {
                    let Op::Context { speculative, .. } = key.op else { unreachable!() };
                    for phase in cuteafd_loader::families::kimi_k3::context::graph_phases(key.rows) {
                        ensure!(inventory.contains(&Key { op: Op::Context { layer, speculative, phase }, ..*key }),
                            "Kimi context graph phase inventory is incomplete");
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) struct Graphs<'a> { bank: GraphBank<Key, GraphOwner<'a, ()>>, replays: u64 }

impl<'a> Graphs<'a> {
    pub(crate) fn capture(stream: &Stream<'a>, keys: &[Key],
        mut arithmetic: impl FnMut(Key) -> Result<()>) -> Result<Self> {
        let mut bank = GraphBank::with_policy(GraphPolicy::FixedStartup);
        bank.warm(keys, |&key| stream.device.run(|| {
            // SAFETY: startup is exclusive; only rank-local arithmetic enters
            // capture. The containing LocalLane pins every operand and drains
            // its exact stream before releasing this bank or its allocations.
            unsafe { stream.device.library.cuda_graph_begin_capture(stream.raw)?; }
            let submitted = arithmetic(key);
            // End even a failed submission, so startup cannot leave capture active.
            // SAFETY: this ends the matching capture on the same retained stream.
            let ended = unsafe { stream.device.library.cuda_graph_end_capture(stream.raw) };
            match (submitted, ended) {
                (Ok(()), Ok(raw)) => {
                    // SAFETY: LocalLane retains all captured pointers and drains before drop.
                    let owner = unsafe { GraphOwner::new(stream.device.library, stream.device.id, raw, ())? };
                    Ok((owner, None))
                }
                (Err(error), Ok(raw)) => {
                    // SAFETY: the failed startup executable was never launched.
                    unsafe { stream.device.library.cuda_graph_exec_destroy(raw)?; }
                    Err(error)
                }
                (Err(error), Err(ended)) => Err(error.context(format!("ending failed capture: {ended:#}"))),
                (Ok(()), Err(error)) => Err(error),
            }
        }))?;
        Ok(Self { bank, replays: 0 })
    }

    pub(crate) fn replays(&self) -> u64 { self.replays }

    pub(crate) fn launch(&mut self, key: Key, stream: &Stream<'a>) -> Result<bool> {
        let (decision, owner) = self.bank.enqueue(&key);
        match decision {
            GraphDecision::Replay => {
                let owner = owner.expect("replay has executable");
                ensure!(owner.device() == stream.device.id, "Kimi graph replay device changed");
                stream.device.run(|| {
                    // SAFETY: binding owns immutable capture pointers and this exact stream.
                    unsafe { stream.device.library.cuda_graph_launch(owner.raw, stream.raw) }
                })?;
                self.replays += 1;
                Ok(true)
            }
            GraphDecision::Eager if matches!(key.op, Op::Context { .. }) =>
                anyhow::bail!("Kimi context requires a startup-captured phase graph"),
            GraphDecision::Eager => Ok(false),
            GraphDecision::Capture => anyhow::bail!("Kimi live submission attempted graph capture"),
        }
    }
}

pub(crate) fn check_captures(startup: u64, current: u64) -> Result<()> {
    ensure!(current == startup, "Kimi zero-steady-capture contract: startup={startup}, current={current}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_phases_match_admission_and_capture_every_live_split_bucket() -> Result<()> {
        use cuteafd_loader::{families::kimi_k3::context::{graph_phases, GraphPhase, SPLIT_BUCKETS},
            placement::LayerMode};
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let inputs = super::super::admission::tests::inputs(&cfg);
        let mut placement = super::super::admission::admit(inputs)?.placement;
        for (kind, assignment) in cfg.attention.iter().zip(&mut placement.layers) {
            if *kind == KimiAttention::Mla { assignment.mode = LayerMode::ContextSplit; }
        }
        let decode = cuteafd_loader::families::kimi_k3::context::DecodeBuckets::from_manifest(
            &serde_json::json!({"programs":[{"name":"dense_trellis_k6_trellis_mla_q_a_mcg",
                "geometry":{"bits":6,"decode_route_thresholds":[8,16,32,48]}}]}))?;
        for capacity in [64, 2752, 4096] { for verify in [0, 8, 64] {
            let set = inventory_with_context(&cfg, 1, capacity, 16, verify, 2, |_, _| true,
                |layer| cfg.attention[layer] == KimiAttention::Mla, Some(&decode))?;
            let unique = set.keys.iter().collect::<std::collections::HashSet<_>>();
            assert_eq!(unique.len(), set.keys.len());
            check_placement(&cfg, &placement, &set.keys)?;
            for rank in 0..2 {
                assert_eq!(set.inventory.ranks[rank].executables,
                    set.keys.iter().filter(|key| key.rank == rank).count() as u64);
            }
            let layer = cfg.attention.iter().position(|kind| *kind == KimiAttention::Mla).unwrap();
            for heads_rank in 0..2 { for splits in SPLIT_BUCKETS {
                assert!(unique.contains(&Key { lane: 0, rank: 0, rows: 16,
                    op: Op::Context { layer, speculative: false, phase: GraphPhase::Partial { heads_rank, splits } } }));
            }}
            assert!(!set.keys.iter().any(|key| matches!(key.op,
                Op::Attention { layer, .. } if cfg.attention[layer] == KimiAttention::Mla)));
            for rank in 0..2 {
                let plain = set.keys.iter().filter(|key| key.rank == rank && matches!(key.op,
                    Op::Context { layer: captured, speculative: false, .. } if captured == layer)).count();
                assert_eq!(plain, 7 * 10);
                for rows in [8, 32, 48] {
                    assert!(!set.keys.iter().any(|key| key.rank == rank && key.rows == rows
                        && matches!(key.op, Op::Ffn(_) | Op::Residual(_)))
                        || verify_graph_rows(verify, capacity).contains(&rows));
                }
            }
            assert!(!set.keys.iter().any(|key| key.rows > 64 && matches!(key.op, Op::Context { .. })));
            assert_eq!(graph_phases(64).len(), 10);
            assert_eq!(graph_phases(4096), [GraphPhase::Produce, GraphPhase::GatheredAttention, GraphPhase::Output]);
            let missing = set.keys.iter().copied().filter(|key| !matches!(key.op,
                Op::Context { phase: GraphPhase::Partial { splits: 8, .. }, .. })).collect::<Vec<_>>();
            assert!(check_placement(&cfg, &placement, &missing).is_err());
        }}
        assert!(inventory_with_context(&cfg, 1, 64, 16, 0, 1, |_, _| true,
            |layer| cfg.attention[layer] == KimiAttention::Mla, Some(&decode)).is_err());
        Ok(())
    }

    #[test]
    fn single_owner_keys_capture_output_without_a_shadow_rank() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        for capacity in [64, 2752, 4096] {
            let set = inventory_for_ranks(&cfg, 2, capacity, 16, 8, 1, |rank, _| rank == 0)?;
            assert_eq!(set.inventory.ranks.len(), 1);
            assert!(set.keys.iter().all(|key| key.rank == 0));
            assert_eq!(set.inventory.ranks[0].executables, set.keys.len() as u64);
            let unique = set.keys.iter().collect::<std::collections::HashSet<_>>();
            assert_eq!(unique.len(), set.keys.len());
            for lane in 0..2 { for rows in graph_rows(8, capacity) {
                assert!(unique.contains(&Key { lane, rank: 0, rows, op: Op::Residual(MixSite::Output) }));
            }}
            let pair = inventory(&cfg, 2, capacity, 16, 8, |_, _| true)?;
            let output = set.keys.iter().filter(|key| key.op == Op::Residual(MixSite::Output)).count();
            assert_eq!(pair.keys.len(), 2 * set.keys.len() - output);
        }
        assert!(inventory_for_ranks(&cfg, 1, 64, 16, 8, 0, |_, _| true).is_err());
        Ok(())
    }

    #[test]
    fn owned_keys_match_shared_admission_and_reject_wrong_residency() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let mut inputs = super::super::admission::tests::inputs(&cfg);
        let provisional = owned_inventory(&cfg, inputs.lanes, inputs.rows, inputs.state_slots, inputs.verify_rows)?;
        inputs.startup.extend((0..2).map(|rank| demand(&provisional.inventory, rank)));
        let admitted = super::super::admission::admit(inputs)?;
        let inventory = admitted.graph_set.as_ref().expect("winning owner graph inventory");
        check_placement(&cfg, &admitted.placement, &inventory.keys)?;
        let mut wrong = inventory.keys.clone();
        let layer = cfg.attention.iter().position(|kind| *kind == KimiAttention::Mla).unwrap();
        let key = wrong.iter_mut().find(|key| matches!(key.op,
            Op::Attention { layer: captured, .. } if captured == layer)).unwrap();
        key.rank = 1 - key.rank;
        assert!(check_placement(&cfg, &admitted.placement, &wrong).is_err());
        let missing = inventory.keys.iter().copied().filter(|key| !matches!(key.op,
            Op::Attention { layer: captured, .. } if captured == layer)).collect::<Vec<_>>();
        assert!(check_placement(&cfg, &admitted.placement, &missing).is_err());
        Ok(())
    }

    #[test]
    fn startup_inventory_is_exact_and_partition_sensitive() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let admitted = inventory(&cfg, 2, 4096, 16, 0, |rank, layer|
            cfg.attention[layer] == KimiAttention::Kda || rank == (layer % 2))?;
        let unique = admitted.keys.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), admitted.keys.len());
        for rank in 0..2 {
            assert_eq!(admitted.inventory.ranks[rank].executables,
                admitted.keys.iter().filter(|key| key.rank == rank).count() as u64);
            assert!(admitted.inventory.bytes(rank) > 0);
            let reserve = demand(&admitted.inventory, rank);
            assert_eq!(reserve.group, RESERVE);
            assert_eq!(reserve.basis, cuteafd_core::memory_layout::Basis::Estimated);
            assert_eq!(reserve.bytes, admitted.inventory.bytes(rank));
        }
        let layer = cfg.attention.iter().position(|kind| *kind == KimiAttention::Kda).unwrap();
        for sequences in [1, 16] {
            assert!(unique.contains(&Key { lane: 1, rank: 1, rows: 16, op: Op::Attention { layer, sequences, speculative: false } }));
        }
        let small = inventory(&cfg, 1, 4096, 4, 0, |_, _| true)?;
        assert!(!small.keys.iter().any(|key| matches!(key.op,
            Op::Attention { sequences, .. } if sequences > 4)));
        for (lanes, rows, slots) in [(0, 1, 1), (1, 0, 1), (1, 4097, 1), (1, 1, 0), (1, 1, 161)] {
            assert!(inventory(&cfg, lanes, rows, slots, 0, |_, _| true).is_err());
        }
        // Verify widths add exact rows and speculative single-sequence attention only.
        let verify = inventory(&cfg, 1, 4096, 4, 8, |_, _| true)?;
        let unique = verify.keys.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), verify.keys.len());
        for rows in 2..=8 {
            assert!(unique.contains(&Key { lane: 0, rank: 0, rows, op: Op::Ffn(0) }));
            assert!(unique.contains(&Key { lane: 0, rank: 1, rows, op: Op::Attention { layer, sequences: 1, speculative: true } }));
        }
        assert!(!verify.keys.iter().any(|key| matches!(key.op, Op::Attention { speculative: true, sequences, .. } if sequences > 1)));
        assert!(!verify.keys.iter().any(|key| key.rows > 8 && matches!(key.op, Op::Attention { speculative: true, .. })));
        for rank in 0..2 {
            assert_eq!(verify.inventory.ranks[rank].executables,
                verify.keys.iter().filter(|key| key.rank == rank).count() as u64);
        }
        assert!(verify.keys.len() > small.keys.len());
        check_captures(19, 19)?;
        assert!(check_captures(19, 20).is_err());
        Ok(())
    }
}
