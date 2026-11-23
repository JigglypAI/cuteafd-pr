//! Native V4 operand and residency contract shared by header-only admission and
//! the CUDA loader. Attention ownership never moves the split FFN operands.
use super::DeepseekV4Config;
use anyhow::{ensure, Context, Result};
use crate::{placement::{FfnMode, LayerMode}, SafetensorsTensorMetadata};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub enum Prep {
    Raw,
    /// Block-FP8 scales for a weight [groups*n, k].
    Scale { n: usize, k: usize, groups: usize },
}

pub struct Source {
    pub operand: &'static str,
    pub tensors: &'static [&'static str],
    pub prep: Prep,
}

pub fn layer_sources(cfg: &DeepseekV4Config, ratio: usize) -> Vec<Source> {
    let (h, q, heads, g, r) = (cfg.dim, cfg.q_lora_rank, cfg.n_heads, cfg.o_groups, cfg.o_lora_rank);
    let w = heads * 512 / g;
    let inter = cfg.moe_inter_dim;
    let mut sources = vec![
        Source { operand: "attn.fn", tensors: &["hc_attn_fn"], prep: Prep::Raw },
        Source { operand: "attn.scale", tensors: &["hc_attn_scale"], prep: Prep::Raw },
        Source { operand: "attn.base", tensors: &["hc_attn_base"], prep: Prep::Raw },
        Source { operand: "attn.norm", tensors: &["attn_norm.weight"], prep: Prep::Raw },
        Source { operand: "ffn.fn", tensors: &["hc_ffn_fn"], prep: Prep::Raw },
        Source { operand: "ffn.scale", tensors: &["hc_ffn_scale"], prep: Prep::Raw },
        Source { operand: "ffn.base", tensors: &["hc_ffn_base"], prep: Prep::Raw },
        Source { operand: "ffn.norm", tensors: &["ffn_norm.weight"], prep: Prep::Raw },
        Source { operand: "w_qkv", tensors: &["attn.wq_a.weight", "attn.wkv.weight"], prep: Prep::Raw },
        Source { operand: "w_qkv_scale", tensors: &["attn.wq_a.scale", "attn.wkv.scale"],
            prep: Prep::Scale { n: q + 512, k: h, groups: 1 } },
        Source { operand: "w_q", tensors: &["attn.wq_b.weight"], prep: Prep::Raw },
        Source { operand: "w_q_scale", tensors: &["attn.wq_b.scale"], prep: Prep::Scale { n: heads * 512, k: q, groups: 1 } },
        Source { operand: "q_norm", tensors: &["attn.q_norm.weight"], prep: Prep::Raw },
        Source { operand: "kv_norm", tensors: &["attn.kv_norm.weight"], prep: Prep::Raw },
        Source { operand: "attn_sink", tensors: &["attn.attn_sink"], prep: Prep::Raw },
        Source { operand: "wo_a", tensors: &["attn.wo_a.weight"], prep: Prep::Raw },
        Source { operand: "wo_a_scale", tensors: &["attn.wo_a.scale"], prep: Prep::Scale { n: r, k: w, groups: g } },
        Source { operand: "wo_b", tensors: &["attn.wo_b.weight"], prep: Prep::Raw },
        Source { operand: "wo_b_scale", tensors: &["attn.wo_b.scale"], prep: Prep::Scale { n: h, k: g * r, groups: 1 } },
        Source { operand: "w13", tensors: &["ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight"], prep: Prep::Raw },
        Source { operand: "w13_scale", tensors: &["ffn.shared_experts.w1.scale", "ffn.shared_experts.w3.scale"],
            prep: Prep::Scale { n: 2 * inter, k: h, groups: 1 } },
        Source { operand: "w2", tensors: &["ffn.shared_experts.w2.weight"], prep: Prep::Raw },
        Source { operand: "w2_scale", tensors: &["ffn.shared_experts.w2.scale"], prep: Prep::Scale { n: h, k: inter, groups: 1 } },
        Source { operand: "gate", tensors: &["ffn.gate.weight"], prep: Prep::Raw },
    ];
    if ratio == 4 {
        sources.extend([
            Source { operand: "joint_projection", tensors: &["attn.compressor.wkv.weight", "attn.compressor.wgate.weight",
                "attn.indexer.compressor.wkv.weight", "attn.indexer.compressor.wgate.weight"], prep: Prep::Raw },
            Source { operand: "index_ape", tensors: &["attn.indexer.compressor.ape"], prep: Prep::Raw },
            Source { operand: "index_norm", tensors: &["attn.indexer.compressor.norm.weight"], prep: Prep::Raw },
            Source { operand: "index_w_q", tensors: &["attn.indexer.wq_b.weight"], prep: Prep::Raw },
            Source { operand: "index_w_q_scale", tensors: &["attn.indexer.wq_b.scale"],
                prep: Prep::Scale { n: cfg.index_n_heads * cfg.index_head_dim, k: q, groups: 1 } },
            Source { operand: "index_w_proj", tensors: &["attn.indexer.weights_proj.weight"], prep: Prep::Raw },
        ]);
    } else if ratio == 128 {
        sources.push(Source { operand: "joint_projection",
            tensors: &["attn.compressor.wkv.weight", "attn.compressor.wgate.weight"], prep: Prep::Raw });
    }
    if ratio != 0 {
        sources.extend([
            Source { operand: "main_ape", tensors: &["attn.compressor.ape"], prep: Prep::Raw },
            Source { operand: "main_norm", tensors: &["attn.compressor.norm.weight"], prep: Prep::Raw },
        ]);
    }
    sources
}

/// How a head split shares an operand's checkpoint tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Share {
    /// Every rank holds the whole operand.
    All,
    /// Rows split evenly (per tensor, concatenated): heads, wo groups, shared-expert intermediate.
    Rows,
    /// Columns split evenly: wo_b over the groups, the shared expert's w2 over the intermediate.
    Cols,
}

/// Attention and FFN ownership are independent for whole-attention layers.
pub fn share(operand: &str, attention_split: bool, ffn_split: bool) -> Share {
    match operand {
        "w_q" | "w_q_scale" | "attn_sink" | "wo_a" | "wo_a_scale" if attention_split => Share::Rows,
        "wo_b" | "wo_b_scale" if attention_split => Share::Cols,
        "w13" | "w13_scale" if ffn_split => Share::Rows,
        "w2" | "w2_scale" if ffn_split => Share::Cols,
        _ => Share::All,
    }
}

/// A mode's checkpoint layout, resolved before uploading any operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightSplit {
    pub attention: bool,
    pub ffn: bool,
    pub owner: Option<usize>,
}

impl WeightSplit {
    pub fn for_mode(mode: crate::placement::LayerMode, ratio: usize, ranks: usize) -> Result<Self> {
        use crate::placement::{FfnMode, LayerMode};
        ensure!(ranks == 1 || ranks == 2, "V4 weights need one or two ranks");
        Ok(match mode {
            LayerMode::HeadSplit => {
                ensure!(ranks == 2, "V4 head split needs two ranks");
                Self { attention: true, ffn: true, owner: None }
            }
            LayerMode::ContextSplit => {
                ensure!(ranks == 2 && ratio == 4, "V4 context weights need two ranks and C4");
                Self { attention: true, ffn: true, owner: None }
            }
            LayerMode::Whole { gpu, ffn } => {
                let owner = usize::from(gpu);
                ensure!(owner < ranks && (ranks == 1 || ratio == 4), "V4 attention owner or ratio unsupported");
                ensure!((ranks == 1 && ffn == FfnMode::Owner) || (ranks == 2 && ffn == FfnMode::Split),
                    "V4 owned attention keeps the existing FFN placement");
                Self { attention: false, ffn: ffn == FfnMode::Split, owner: Some(owner) }
            }
        })
    }

    pub fn operand(self, operand: &str, rank: usize) -> Option<Share> {
        let ffn = ffn_operand(operand);
        if !ffn && self.owner.is_some_and(|owner| owner != rank) {
            None
        } else {
            Some(share(operand, self.attention, self.ffn))
        }
    }
}

pub fn scale_extents(operand: &str, kind: Share, n: usize, k: usize, groups: usize, ranks: usize)
    -> (usize, usize, usize) {
    match (operand, kind) {
        ("wo_a_scale", Share::Rows) => (n, k, groups / ranks),
        (_, Share::Rows) => (n / ranks, k, groups),
        (_, Share::Cols) => (n, k / ranks, groups),
        _ => (n, k, groups),
    }
}


/// Operands needed by both FFN halves, independent of attention placement.
pub fn ffn_operand(operand: &str) -> bool {
    operand.starts_with("ffn.") || operand.starts_with("gate")
        || matches!(operand, "w13" | "w13_scale" | "w2" | "w2_scale")
}

/// One final device allocation, after concatenation, slicing and scale prep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentOperand {
    pub rank: usize,
    pub layer: Option<usize>,
    pub draft_stage: Option<usize>,
    pub operand: String,
    pub group: String,
    pub format: String,
    pub bytes: u64,
    pub attention: bool,
}

fn operand_bytes(source: &Source, prefix: &str, kind: Share, ranks: usize,
    headers: &BTreeMap<&str, &SafetensorsTensorMetadata>) -> Result<(u64, String)> {
    let mut raw = 0u64;
    let mut dtype = None;
    for suffix in source.tensors {
        let name = format!("{prefix}.{suffix}");
        let tensor = headers.get(name.as_str()).with_context(|| format!("V4 resident tensor missing: {name}"))?;
        if !matches!(kind, Share::All) {
            ensure!(tensor.byte_length % ranks as u64 == 0, "{name}: uneven byte split");
            match kind {
                Share::Rows => ensure!(tensor.shape.first().is_some_and(|n| n % ranks == 0),
                    "{name}: uneven row split"),
                Share::Cols => ensure!(tensor.shape.len() == 2 && tensor.shape[1] % ranks == 0
                    && tensor.byte_length == (tensor.shape[0] as u64).checked_mul(tensor.shape[1] as u64)
                        .context("V4 column extent overflow")?, "{name}: invalid byte-column split"),
                Share::All => unreachable!(),
            }
        }
        let bytes = tensor.byte_length / if kind == Share::All { 1 } else { ranks as u64 };
        raw = raw.checked_add(bytes).context("V4 resident operand overflow")?;
        let format = format!("{:?}", tensor.dtype).to_ascii_lowercase();
        ensure!(dtype.as_ref().is_none_or(|previous| previous == &format), "{prefix}: mixed operand dtypes");
        dtype = Some(format);
    }
    Ok(match source.prep {
        Prep::Raw => (raw.max(16), dtype.context("empty V4 operand")?),
        Prep::Scale { n, k, groups } => {
            let (n, k, groups) = scale_extents(source.operand, kind, n, k, groups, ranks);
            ensure!(n % 128 == 0 && k % 128 == 0, "{prefix}: invalid MMA scale extents");
            let expected = (groups as u64).checked_mul(n as u64 / 128)
                .and_then(|n| n.checked_mul(k as u64 / 128)).context("V4 scale grid overflow")?;
            ensure!(raw == expected, "{prefix}.{}: scale bytes {raw} do not cover {expected} blocks", source.operand);
            (raw.checked_mul(512).context("V4 scale expansion overflow")?, "fp8-mma-scale".into())
        }
    })
}

/// Exact uploaded operands. Extra checkpoint heads (confidence heads, etc.)
/// are not resident merely because they are classified as speculator tensors.
pub fn resident_operands<'a>(cfg: &DeepseekV4Config, modes: &[LayerMode], ranks: usize, dspark: bool,
    tensors: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>) -> Result<Vec<ResidentOperand>> {
    ensure!(modes.len() == cfg.n_layers, "V4 weight mode count differs from config");
    ensure!(cfg.compress_ratios.len() == cfg.n_layers, "V4 ratio count differs from config");
    let layouts = modes.iter().zip(&cfg.compress_ratios).map(|(&mode, &ratio)|
        WeightSplit::for_mode(mode, ratio, ranks)).collect::<Result<Vec<_>>>()?;
    let headers: BTreeMap<_, _> = tensors.into_iter().map(|t| (t.name.as_str(), t)).collect();
    let mut out = Vec::new();
    let mut block = |prefix: &str, layer: Option<usize>, draft_stage: Option<usize>, ratio, hash, layout: WeightSplit, count| -> Result<()> {
        for source in layer_sources(cfg, ratio) {
            for rank in 0..count {
                let Some(kind) = layout.operand(source.operand, rank) else { continue };
                let (bytes, format) = operand_bytes(&source, prefix, kind, count, &headers)?;
                out.push(ResidentOperand { rank, layer, draft_stage, operand: source.operand.into(),
                    group: if layer.is_some() { "target" } else { "speculator" }.into(),
                    format, bytes, attention: !ffn_operand(source.operand) });
            }
        }
        let suffix = if hash { "ffn.gate.tid2eid" } else { "ffn.gate.bias" };
        let name = format!("{prefix}.{suffix}");
        let tensor = headers.get(name.as_str()).with_context(|| format!("V4 resident tensor missing: {name}"))?;
        if hash { ensure!(tensor.byte_length % 8 == 0, "{name}: incomplete I64 expert ids"); }
        for rank in 0..count {
            out.push(ResidentOperand { rank, layer, draft_stage, operand: if hash { "gate.tid2eid" } else { "gate.bias" }.into(),
                group: if layer.is_some() { "target" } else { "speculator" }.into(),
                format: if hash { "i32".into() } else { format!("{:?}", tensor.dtype).to_ascii_lowercase() },
                bytes: (tensor.byte_length / if hash { 2 } else { 1 }).max(16), attention: false });
        }
        Ok(())
    };
    for (layer, &layout) in layouts.iter().enumerate() {
        block(&format!("layers.{layer}"), Some(layer), None, cfg.compress_ratios[layer], cfg.is_hash_layer(layer), layout, ranks)?;
    }
    let mut names = vec![("embed.weight", "embedding"), ("head.weight", "lm_head"),
        ("hc_head_fn", "hyper_connection"), ("hc_head_scale", "hyper_connection"),
        ("hc_head_base", "hyper_connection"), ("norm.weight", "norm")];
    if dspark && cfg.dspark_block_size > 0 && headers.contains_key("mtp.0.main_proj.weight") {
        let whole = WeightSplit::for_mode(LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }, 0, 1)?;
        for stage in 0..3 { block(&format!("mtp.{stage}"), None, Some(stage), 0, false, whole, 1)?; }
        names.extend([("mtp.0.main_proj.weight", "speculator"), ("mtp.0.main_proj.scale", "speculator"),
            ("mtp.0.main_norm.weight", "speculator"), ("mtp.2.hc_head_fn", "speculator"),
            ("mtp.2.hc_head_scale", "speculator"), ("mtp.2.hc_head_base", "speculator"),
            ("mtp.2.norm.weight", "speculator"), ("mtp.2.markov_head.markov_w1.weight", "speculator"),
            ("mtp.2.markov_head.markov_w2.weight", "speculator")]);
    }
    for (name, group) in names {
        let tensor = headers.get(name).with_context(|| format!("V4 resident tensor missing: {name}"))?;
        out.push(ResidentOperand { rank: 0, layer: None, draft_stage: None, operand: name.into(), group: group.into(),
            format: format!("{:?}", tensor.dtype).to_ascii_lowercase(), bytes: tensor.byte_length.max(16), attention: false });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::DType;

    fn fixture() -> (DeepseekV4Config, Vec<SafetensorsTensorMetadata>) {
        let cfg = DeepseekV4Config::from_model_args(&serde_json::json!({
            "vocab_size":129280,"dim":4096,"moe_inter_dim":2048,"n_layers":4,"n_hash_layers":1,
            "n_heads":64,"n_routed_experts":256,"n_shared_experts":1,"n_activated_experts":6,
            "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,"q_lora_rank":1024,
            "head_dim":512,"rope_head_dim":64,"o_groups":8,"o_lora_rank":1024,"window_size":128,
            "original_seq_len":65536,"rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
            "index_n_heads":64,"index_head_dim":128,"index_topk":512,"hc_mult":4,"hc_sinkhorn_iters":20,
            "compress_rope_theta":160000,"compress_ratios":[0,4,128,4],"dspark_block_size":5
        }), 3).unwrap();
        let mut tensors = Vec::new();
        let mut add = |name: String, dtype, shape: Vec<usize>, byte_length| {
            tensors.push(SafetensorsTensorMetadata { name, dtype, shape, byte_offset: 0, byte_length });
        };
        for (prefix, ratio, hash) in (0..cfg.n_layers).map(|layer|
            (format!("layers.{layer}"), cfg.compress_ratios[layer], cfg.is_hash_layer(layer)))
            .chain((0..3).map(|stage| (format!("mtp.{stage}"), 0, false))) {
            for source in layer_sources(&cfg, ratio) {
                for &suffix in source.tensors {
                    let (dtype, shape, bytes) = match source.prep {
                        Prep::Raw => (DType::F8E4M3, vec![256, 256], 256 * 256),
                        Prep::Scale { n, k, groups } => {
                            let rows = groups * n / 128 / source.tensors.len();
                            (DType::F8E8M0, vec![rows, k / 128], (rows * k / 128) as u64)
                        }
                    };
                    add(format!("{prefix}.{suffix}"), dtype, shape, bytes);
                }
            }
            if hash { add(format!("{prefix}.ffn.gate.tid2eid"), DType::I64, vec![64, 6], 64 * 6 * 8); }
            else { add(format!("{prefix}.ffn.gate.bias"), DType::F32, vec![256], 1024); }
        }
        for name in ["embed.weight", "head.weight", "hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight",
            "mtp.0.main_proj.weight", "mtp.0.main_proj.scale", "mtp.0.main_norm.weight",
            "mtp.2.hc_head_fn", "mtp.2.hc_head_scale", "mtp.2.hc_head_base", "mtp.2.norm.weight",
            "mtp.2.markov_head.markov_w1.weight", "mtp.2.markov_head.markov_w2.weight",
            "mtp.2.confidence_head.weight"] {
            add(name.into(), DType::Bf16, vec![256], 512);
        }
        (cfg, tensors)
    }

    #[test]
    fn arenas_cover_selected_tp1_tp2_modes_and_final_formats() -> Result<()> {
        use super::super::weight_arena::{coordinator_arena, operand_name};
        let (cfg, tensors) = fixture();
        for ranks in [1, 2] {
            for owner in 0..ranks {
                let mut modes = vec![if ranks == 1 {
                    LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }
                } else { LayerMode::HeadSplit }; cfg.n_layers];
                if ranks == 2 { modes[1] = LayerMode::Whole { gpu: owner as u8, ffn: FfnMode::Split }; }
                for draft in [false, true] {
                    let operands = resident_operands(&cfg, &modes, ranks, draft, &tensors)?;
                    for rank in 0..ranks {
                        for scope in [false, true] {
                            let arena = coordinator_arena(&operands, rank, scope, true)?;
                            let items = operands.iter().filter(|i| i.rank == rank && (i.group == "speculator") == scope)
                                .collect::<Vec<_>>();
                            assert_eq!(arena.is_some(), !items.is_empty());
                            if let Some(arena) = arena {
                                assert_eq!(arena.views().len(), items.len());
                                for item in items {
                                    assert_eq!(arena.view(&operand_name(item)).unwrap().bytes() as u64, item.bytes);
                                }
                                assert!(!arena.views().keys().any(|k| k.contains("confidence_head")));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "header-only installed V4 CPU allocation inventory; set CUTEAFD_V4_ARENA_SNAPSHOT"]
    fn installed_v4_arena_cpu_inventory() -> Result<()> {
        use super::super::weight_arena::coordinator_arena;
        let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_V4_ARENA_SNAPSHOT")?);
        let cfg = DeepseekV4Config::read(&snapshot, 1)?;
        let catalog = crate::read_expert_catalog(&snapshot)?;
        for ranks in [1, 2] {
            let modes = vec![if ranks == 1 { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } }
                else { LayerMode::HeadSplit }; cfg.n_layers];
            let operands = resident_operands(&cfg, &modes, ranks, true, catalog.tensors().iter().map(|t| &t.metadata))?;
            for rank in 0..ranks {
                for draft in [false, true] {
                    let Some(arena) = coordinator_arena(&operands, rank, draft, true)? else { continue; };
                    let extents = arena.views().values().map(|v| v.bytes()).collect::<Vec<_>>();
                    let large = extents.iter().copied().filter(|&b| b > 1 << 20).collect::<Vec<_>>();
                    let tail = |bytes: usize| if bytes > 1 << 20 { bytes.next_multiple_of(2 << 20) - bytes } else { 0 };
                    let charged = crate::placement::planner::allocation_granularity_forecast(&[arena.bytes() as u64])
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    println!("V4_ARENA ranks={ranks} rank={rank} draft={draft} count={} payload={} large_count={} large_bytes={} old_tail={} arena={} padding={} final_tail={} charged_tail={charged}",
                        extents.len(), arena.payload_bytes(), large.len(), large.iter().sum::<usize>(),
                        large.iter().map(|&b| tail(b)).sum::<usize>(), arena.bytes(), arena.padding_bytes(), tail(arena.bytes()));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn whole_c4_keeps_ffn_halves_and_removes_every_nonowner_attention_operand() -> Result<()> {
        let (cfg, tensors) = fixture();
        let split_modes = vec![LayerMode::HeadSplit; cfg.n_layers];
        let split = resident_operands(&cfg, &split_modes, 2, true, &tensors)?;
        for owner in 0..2 {
            let mut modes = split_modes.clone();
            modes[1] = LayerMode::Whole { gpu: owner, ffn: FfnMode::Split };
            let whole = resident_operands(&cfg, &modes, 2, true, &tensors)?;
            assert_eq!(whole.iter().filter(|i| i.layer != Some(1) || !i.attention).collect::<Vec<_>>(),
                split.iter().filter(|i| i.layer != Some(1) || !i.attention).collect::<Vec<_>>());
            assert!(whole.iter().filter(|i| i.layer == Some(1) && i.attention).all(|i| i.rank == owner as usize));
            for item in whole.iter().filter(|i| i.layer == Some(1) && i.attention) {
                let previous = split.iter().find(|i| i.rank == item.rank && i.layer == item.layer
                    && i.operand == item.operand).unwrap();
                let share = share(&item.operand, true, true);
                assert_eq!(item.bytes, previous.bytes * if share == Share::All { 1 } else { 2 });
            }
        }
        Ok(())
    }

    #[test]
    fn header_inventory_matches_native_scale_and_hash_conversion_and_exact_drafter_subset() -> Result<()> {
        let (cfg, tensors) = fixture();
        let inventory = resident_operands(&cfg, &vec![LayerMode::HeadSplit; cfg.n_layers], 2, true, &tensors)?;
        let scale = inventory.iter().find(|i| i.layer == Some(1) && i.rank == 0 && i.operand == "w_q_scale").unwrap();
        let header = tensors.iter().find(|t| t.name == "layers.1.attn.wq_b.scale").unwrap();
        assert_eq!(scale.bytes, header.byte_length / 2 * 512);
        let ids = inventory.iter().find(|i| i.layer == Some(0) && i.operand == "gate.tid2eid").unwrap();
        assert_eq!(ids.bytes, 64 * 6 * 4);
        assert_eq!(ids.format, "i32");
        assert!(inventory.iter().filter(|i| i.group == "speculator").all(|i| i.rank == 0));
        assert!(!inventory.iter().any(|i| i.operand.contains("confidence_head")));
        let raw_scale = inventory.iter().find(|i| i.operand == "mtp.0.main_proj.scale").unwrap();
        assert_eq!(raw_scale.bytes, 512);
        let off = resident_operands(&cfg, &vec![LayerMode::HeadSplit; cfg.n_layers], 2, false, &tensors)?;
        assert!(!off.iter().any(|i| i.group == "speculator"));
        Ok(())
    }

    #[test]
    fn admitted_arenas_equal_every_mode_the_runtime_can_select() -> Result<()> {
        use super::super::weight_arena::coordinator_arena;
        let (cfg, tensors) = fixture();
        for ranks in [1, 2] {
            for dspark in [false, true] {
                for embedding in [false, true] {
                    let weights = WeightInventory::from_headers(&cfg, ranks, dspark, embedding, &tensors)?;
                    assert_eq!(weights.target.len(), ranks);
                    assert_eq!(weights.draft.is_some(), dspark);
                    let mut selected = vec![admitted_modes(&cfg, ranks)];
                    if ranks == 2 {
                        selected.push(cfg.compress_ratios.iter().map(|&ratio| if ratio == 4 {
                            LayerMode::ContextSplit } else { LayerMode::HeadSplit }).collect());
                    }
                    for modes in selected {
                        let operands = resident_operands(&cfg, &modes, ranks, dspark, &tensors)?;
                        for (rank, admitted) in weights.target.iter().enumerate() {
                            assert_eq!(coordinator_arena(&operands, rank, false, embedding)?.as_ref(), Some(admitted));
                            assert_eq!(admitted.id(), format!("v4/target/{rank}"));
                        }
                        assert_eq!(coordinator_arena(&operands, 0, true, false)?, weights.draft);
                    }
                    let totals = (0..ranks).map(|rank| weights.target[rank].bytes() as u64
                        + if rank == 0 { weights.draft.as_ref().map_or(0, |d| d.bytes() as u64) } else { 0 })
                        .collect::<Vec<_>>();
                    assert_eq!(weights.loaded_bytes()?, totals);
                    assert!(weights.target.iter().enumerate().all(|(rank, arena)|
                        arena.views().contains_key("embedding.table") == (embedding && rank == 0)));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn startup_charges_only_device_scratch() {
        use crate::shared::dense_read::STEP_BYTES;
        let (cfg, tensors) = fixture();
        for ranks in [1, 2] {
            for dspark in [false, true] {
                let weights = WeightInventory::from_headers(&cfg, ranks, dspark, true, &tensors).unwrap();
                let phases = weights.startup.device_phases(ranks).unwrap();
                assert_eq!(phases.len(), ranks);
                for rank in 0..ranks {
                    assert_eq!(phases[rank].bytes, STEP_BYTES as u64);
                }
            }
        }
    }

    #[test]
    fn inventory_rejects_arena_extent_overflow() {
        let (cfg, mut tensors) = fixture();
        tensors.iter_mut().find(|t| t.name == "layers.0.hc_ffn_fn").unwrap().byte_length = u64::MAX - 1024;
        let error = WeightInventory::from_headers(&cfg, 2, false, true, &tensors).unwrap_err();
        assert!(format!("{error:#}").contains("overflow"), "{error:#}");
    }

    #[test]
    fn inventory_refuses_missing_headers_bad_scales_modes_and_column_shapes() {
        let (cfg, tensors) = fixture();
        let modes = vec![LayerMode::HeadSplit; cfg.n_layers];
        assert!(resident_operands(&cfg, &modes[..3], 2, false, &tensors).is_err());
        let mut missing = tensors.clone();
        missing.retain(|t| t.name != "layers.1.attn.indexer.weights_proj.weight");
        assert!(resident_operands(&cfg, &modes, 2, false, &missing).is_err());
        let mut malformed = tensors.clone();
        malformed.iter_mut().find(|t| t.name == "layers.1.attn.wq_b.scale").unwrap().byte_length += 2;
        assert!(resident_operands(&cfg, &modes, 2, false, &malformed).is_err());
        let mut malformed = tensors;
        malformed.iter_mut().find(|t| t.name == "layers.1.attn.wo_b.weight").unwrap().shape[1] = 255;
        assert!(resident_operands(&cfg, &modes, 2, false, &malformed).is_err());
        let mut modes = modes;
        modes[2] = LayerMode::Whole { gpu: 1, ffn: FfnMode::Split };
        assert!(resident_operands(&cfg, &modes, 2, false, &missing).is_err());
    }
}

/// The retained coordinator weights admission charges: one target arena per
/// rank and the optional lead-GPU dSpark arena, each exactly the layout the
/// loader allocates. C4 attention ownership (whole vs split) never changes
/// these extents: every admitted mode slices the same final operands.
#[derive(Debug, Clone, Default)]
pub struct WeightInventory {
    /// Target arena per rank: core, globals and the GPU embedding when selected.
    pub target: Vec<crate::serving_capacity::WeightArenaLayout>,
    /// The separately admitted and retired dSpark dense arena (lead GPU).
    pub draft: Option<crate::serving_capacity::WeightArenaLayout>,
    pub startup: StartupInventory,
}

/// Device scratch used only during checkpoint loading.
#[derive(Debug, Clone, Default)]
pub struct StartupInventory {
    pub device: Vec<StartupItem>,
}

#[derive(Debug, Clone)]
pub struct StartupItem {
    pub rank: usize,
    pub name: &'static str,
    pub bytes: u64,
}

impl StartupInventory {
    fn from_operands(operands: &[ResidentOperand], ranks: usize) -> Result<Self> {
        let mut inventory = Self::default();
        for rank in 0..ranks {
            let scratch = if operands.iter().any(|item| item.rank == rank && item.format == "fp8-mma-scale") {
                crate::shared::dense_read::STEP_BYTES as u64
            } else { 0 };
            // Retained operands are views of the RESIDENT arena admitted beside
            // this group; only the device-side scale scratch is Load-only.
            inventory.device.push(StartupItem { rank, name: "staging/coordinator-dense-device", bytes: scratch });
        }
        Ok(inventory)
    }

    pub fn device_bytes(&self, rank: usize) -> Result<u64> {
        self.device.iter().filter(|item| item.rank == rank).try_fold(0u64, |sum, item|
            sum.checked_add(item.bytes).context("V4 startup device overflow"))
    }

    pub fn device_phases(&self, ranks: usize) -> Result<Vec<crate::placement::Demand>> {
        use cuteafd_core::memory_layout::{Basis, Category};
        (0..ranks).map(|rank| Ok(crate::placement::Demand::new(rank as u8, Category::Weights,
            "V4 coordinator dense loading", self.device_bytes(rank)?, Basis::Exact))).collect()
    }

    pub fn notes(&self) -> Vec<String> {
        self.device.iter().map(|item| format!("V4 startup device rtx{} {}: {} bytes (loading only)",
            item.rank, item.name, item.bytes)).collect()
    }
}

impl WeightInventory {
    /// The admitted arenas and startup owners. Every V4 mode on `ranks` GPUs
    /// uploads the same final operands (head and context splits share one
    /// slicing; TP1 is whole), so the layouts admission charges are the ones
    /// the runtime allocates after selection.
    pub fn from_headers<'a>(cfg: &DeepseekV4Config, ranks: usize, dspark: bool, device_embedding: bool,
        tensors: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>) -> Result<Self> {
        let operands = resident_operands(cfg, &admitted_modes(cfg, ranks), ranks, dspark, tensors)?;
        let target = (0..ranks).map(|rank| super::weight_arena::coordinator_arena(&operands, rank, false, device_embedding)?
            .context("V4 target weight arena absent")).collect::<Result<Vec<_>>>()?;
        let draft = super::weight_arena::coordinator_arena(&operands, 0, true, false)?;
        let startup = StartupInventory::from_operands(&operands, ranks)?;
        Ok(Self { target, draft, startup })
    }

    /// Retained arena bytes per rank, embedding included. A post-load CUDA
    /// sample is rebased by these exact bytes to avoid charging twice.
    pub fn loaded_bytes(&self) -> Result<Vec<u64>> {
        let mut bytes = self.target.iter().map(|arena| arena.bytes() as u64).collect::<Vec<_>>();
        if let Some(draft) = &self.draft {
            bytes[0] = bytes[0].checked_add(draft.bytes() as u64).context("V4 drafter arena overflow")?;
        }
        Ok(bytes)
    }
}

/// The one mode set every V4 placement on `ranks` GPUs shares for weights.
pub fn admitted_modes(cfg: &DeepseekV4Config, ranks: usize) -> Vec<LayerMode> {
    vec![if ranks == 1 { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } } else { LayerMode::HeadSplit }; cfg.n_layers]
}
