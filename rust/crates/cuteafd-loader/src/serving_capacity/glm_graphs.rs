//! GLM owned-attention captures: one census for warm-up, replay and admission.
use super::{glm_decode_row_buckets, glm_decode_table_widths};
use crate::placement::inventory::{GraphSet, Lifetime, StartupGraphs, GLM_OWNED_SM120_GRAPH_BYTES};

/// These values are the persistent graph-key phase, not a launch count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlmOwnedPhase {
    Attention = 0,
    FfnFront = 1,
    FfnJoin = 2,
    RemoteNorm = 3,
}

impl GlmOwnedPhase {
    pub fn key(self, layer: usize) -> usize { 4 * layer + self as usize }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlmOwnedSegment {
    pub rank: usize,
    pub layer: usize,
    pub phase: GlmOwnedPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlmGraphShape {
    pub rows: usize,
    pub table_width: usize,
}

/// Every layer captures owner attention, peer normalization and two joins.
/// Only dense FFN fronts are captured: runtime-width MoE work and reductions
/// stay outside bucket graphs. Keep this census shared with the bindings.
pub fn glm_owned_segments(owners: &[u8], dense_layers: usize) -> Result<Vec<GlmOwnedSegment>, String> {
    if owners.is_empty() || dense_layers > owners.len() || owners.iter().any(|&rank| rank > 1) {
        return Err("invalid GLM owned graph geometry".into());
    }
    let mut segments = Vec::new();
    for (layer, &owner) in owners.iter().enumerate() {
        for rank in 0..2 {
            segments.push(GlmOwnedSegment { rank, layer, phase: if rank == owner as usize {
                GlmOwnedPhase::Attention
            } else { GlmOwnedPhase::RemoteNorm } });
            if layer < dense_layers { segments.push(GlmOwnedSegment { rank, layer, phase: GlmOwnedPhase::FfnFront }); }
            segments.push(GlmOwnedSegment { rank, layer, phase: GlmOwnedPhase::FfnJoin });
        }
    }
    Ok(segments)
}

pub fn glm_graph_shapes(context: usize, pool_pages: usize) -> Vec<GlmGraphShape> {
    glm_decode_table_widths(context, pool_pages).into_iter().flat_map(|table_width| {
        glm_decode_row_buckets(table_width).iter().map(move |&rows| GlmGraphShape { rows, table_width })
    }).collect()
}

/// Measured SM120 owned-phase calibration with a separate 10% growth margin
/// (64 MiB minimum). Planner and startup use the same executable census.
pub fn glm_owned_startup_graphs(context: usize, pool_pages: usize, owners: &[u8], dense_layers: usize)
    -> Result<StartupGraphs<GlmGraphShape>, String> {
    let segments = glm_owned_segments(owners, dense_layers)?;
    let keys = glm_graph_shapes(context, pool_pages);
    let mut counts = [0u64; 2];
    for segment in segments {
        counts[segment.rank] = counts[segment.rank].checked_add(keys.len() as u64)
            .ok_or("GLM graph count overflow")?;
    }
    GraphSet::new(&counts, GLM_OWNED_SM120_GRAPH_BYTES, 10, 64 << 20, keys.len() as u64, Lifetime::Startup).startup(keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn owned_census_counts_phases_not_whole_engine_allowances() {
        for owners in [vec![0, 0, 1, 1], vec![1, 1, 0, 0]] {
            let dense_layers = 2;
            let segments = glm_owned_segments(&owners, dense_layers).unwrap();
            assert_eq!(segments.len(), 20);
            assert_eq!(segments.iter().copied().collect::<HashSet<_>>().len(), segments.len());
            for (layer, &owner) in owners.iter().enumerate() {
                assert!(segments.contains(&GlmOwnedSegment { rank: owner as usize, layer, phase: GlmOwnedPhase::Attention }));
                assert!(segments.contains(&GlmOwnedSegment { rank: 1 - owner as usize, layer, phase: GlmOwnedPhase::RemoteNorm }));
                for rank in 0..2 {
                    assert_eq!(segments.contains(&GlmOwnedSegment { rank, layer, phase: GlmOwnedPhase::FfnFront }), layer < dense_layers);
                }
            }
            let startup = glm_owned_startup_graphs(1 << 20, 32768, &owners, dense_layers).unwrap();
            for rank in 0..2 {
                assert_eq!(startup.inventory.ranks[rank].executables, startup.keys.len() as u64 * 10);
                assert_eq!(startup.inventory.at_ready(rank), startup.keys.len() as u64 * 10 * GLM_OWNED_SM120_GRAPH_BYTES);
                assert_eq!(startup.inventory.at_ready(rank) + startup.inventory.growth(rank), startup.inventory.bytes(rank));
            }
        }
        assert!(glm_owned_segments(&[2], 0).is_err());
        assert!(glm_owned_segments(&[0], 2).is_err());
    }

    #[test]
    fn owned_sm120_allowance_uses_only_dense_fronts() {
        // Retain the conservative per-executable SM120 allowance; MoE fronts no
        // longer contribute executables. Hardware records the new actual bytes.
        let owners: Vec<_> = (0..78).map(|layer| (layer % 2) as u8).collect();
        let startup = glm_owned_startup_graphs(1 << 20, 32768, &owners, 3).unwrap();
        assert_eq!(startup.keys.len(), 213);
        for rank in 0..2 {
            let count = (2 * 78 + 3) * startup.keys.len() as u64;
            assert_eq!(startup.inventory.ranks[rank].executables, count);
            assert_eq!(startup.inventory.at_ready(rank), count * GLM_OWNED_SM120_GRAPH_BYTES);
            assert_eq!(startup.inventory.growth(rank), startup.inventory.at_ready(rank) / 10);
        }
    }

    #[test]
    fn warm_shapes_match_every_capped_table_and_row_bucket() {
        for context in [64, 131072, 1 << 20] {
            for pages in [1, 1024, 32768] {
                let shapes = glm_graph_shapes(context, pages);
                assert_eq!(shapes.iter().copied().collect::<HashSet<_>>().len(), shapes.len());
                for width in glm_decode_table_widths(context, pages) {
                    for &rows in glm_decode_row_buckets(width) {
                        assert!(shapes.contains(&GlmGraphShape { rows, table_width: width }));
                    }
                }
            }
        }
    }
}
