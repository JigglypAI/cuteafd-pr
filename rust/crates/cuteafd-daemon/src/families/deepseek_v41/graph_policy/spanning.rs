//! Capture-site classification, not a GPU-count heuristic.
//!
//! Evidence line numbers refer to b23fa2b0d. Local captures contain only rank
//! operands; copies/joins outside them are issued before their consumers. The
//! distributed target runs whole-layer owners in order (v41_target_pass/
//! distributed.rs:610), with ownership hops outside graphs (shared/peer_split/
//! hop.rs:1). A distributed drafter wrapper may conservatively promote any
//! child to spanning: never erase that caller-provided dependency information.

use crate::shared::decode_graph::sites::{Scope, Table};

// (visitor site, intrinsic scope, capture/dependency evidence).
pub(super) const SITES: &Table = &[
    ("window", Scope::Local, "v41_window.rs:659 local norm/project/pack; peer publication is outside capture at :397"),
    ("compressor", Scope::Local, "v41_compressor.rs:903 local project/pool/index; capture :1172 excludes peer copy :587"),
    ("index_selection", Scope::Local, "v41_index_selection.rs:455 score/top-k; :634 requires local query; proposal replica binds local keys"),
    ("target_head", Scope::Local, "v41_target_head.rs:202 local mHC/norm/vocabulary projection"),
    ("v41.attention.query", Scope::Local, "v41_attention_query.rs:587 local graph dispatch; input staging and chain joins outside capture"),
    ("v41.attention.query.tp2_prefix", Scope::Local, "v41_attention_query.rs:452 captures enqueue_rank only; :659-668 waits projection completion before capture"),
    ("v41.attention.output", Scope::Local, "v41_attention_output.rs:349 local grouped projection/rope; :372 local output-B"),
    ("v41.backbone.shared", Scope::Local, "v41_backbone_shared.rs:277 local FFN enqueue, :315 only enqueue captured"),
    ("v41.router.remote", Scope::Local, "v41_backbone_router.rs:596 local route/quantize; Spark exchange consumes outputs outside graph"),
    ("v41.router.local", Scope::Local, "v41_backbone_router.rs:596 local route/quantize; no expert exchange in capture"),
    ("v41.index.query", Scope::Local, "v41_index_query.rs:307 local frequency/projection/query preparation"),
    ("v41.engram.gate", Scope::Local, "v41_engram/layer.rs:331 local projection/gate; external gathered inputs staged before :396 capture"),
    ("sparse_attention", Scope::Local, "v41_sparse_attention.rs:930 peer query copy precedes capture; :490 drains warmup; dual.rs:307 submits both halves before either wait and only gathers afterward"),
    ("dual_attention_tail", Scope::Local, "v41_sparse_attention/dual.rs:178 awaits gather and tail completion before :200 capture; v41_backbone_lane.rs:257 tail is local projection/block only"),
    ("tp2_ffn", Scope::Local, "v41_backbone_lane.rs:307 awaits finish_tp2_then before :319 capture; v41_block.rs:550 only local attention finish/copy/FFN begin, not expert exchange"),
    ("v41.projection.query.rank", Scope::Local, "v41_projection_tp2.rs:149 local GEMM only; :395 waits gathered completion before :397 capture"),
    ("v41.projection.output.rank", Scope::Local, "v41_projection_tp2.rs:149 local GEMM only; input copy/wait outside graph at :369"),
    ("v41.projection.query.gather", Scope::Spanning, "v41_projection_tp2.rs:300 reads both rank outputs with peer-copy kernel"),
    ("v41.projection.output.gather", Scope::Spanning, "v41_projection_tp2.rs:300 reads both rank outputs with peer-copy kernel"),
    ("dspark_chain", Scope::Declared, "v41_experts/dspark/chain.rs:736 peer stage/embedding detection; chain/distributed.rs:224 promotes children"),
    ("dspark_distributed_terminal", Scope::Local, "v41_experts/dspark/terminal/distributed.rs:180 queries completed sampling stream before :190 capture; :205 completes peer logit assembly first; terminal.rs:286 local selector/confidence only"),
    ("dspark_stage", Scope::Declared, "v41_experts/dspark/stage.rs:203 peer_stream determines TP2 FFN dependency"),
    ("dspark_ffn", Scope::Declared, "v41_experts/dspark/ffn.rs:155 peer_stream determines split FFN participation"),
    ("dspark_terminal", Scope::Local, "v41_experts/dspark/terminal.rs:235 local projection/sampling; peer head/assembly is outside capture"),
    ("dspark_main_context", Scope::Local, "v41_experts/dspark/main_context.rs:172 local projection/norm/KV preparation"),
    ("dspark_projection", Scope::Local, "v41_experts/dspark/projection.rs:218 local operands GEMM"),
    ("dspark_attention_wave", Scope::Local, "v41_experts/dspark/attention_wave.rs:196 local query/window kernels; composite wrapper retains peer hazard"),
    ("dspark_attention_output", Scope::Local, "v41_experts/dspark/attention_output.rs:92 local grouped/output projections"),
    ("dspark_markov", Scope::Local, "v41_experts/dspark/markov.rs:77 local token/Markov projection"),
    ("dspark_confidence", Scope::Local, "v41_experts/dspark/confidence.rs:73 local confidence projection"),
];

/// An unclassified site is spanning: it stays prewarmed and sealed (#2303).
pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_v41_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("deepseek_v41", super::SITES);
        assert!(super::spans("new-unreviewed-site", false));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| crate::shared::decode_graph::sites::testing::assert_off_transitions(super::SITES, name, declared);
        // Two-GPU layout: gathers always span; rank-local children stay lazy;
        // a distributed drafter promotes its declared children.
        for &(name, _, _) in super::SITES {
            let spanning = off(name, false);
            assert_eq!(spanning, name.ends_with(".gather"), "V4.1 {name}");
        }
        assert!(off("dspark_ffn", true) && off("dspark_chain", true));
        assert!(!super::super::spanning_layout(1) && super::super::spanning_layout(2));
    }
}
