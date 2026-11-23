use super::*;
use crate::placement::families::hy4_flash::HY4_FLASH as HY4_EXECUTOR;
use crate::placement::AttentionPlacement;
use crate::plan::{
    checkpoint::{CheckpointView, CoreView},
    layout::LayoutOptions,
    ExpertPlacement, PlanInput, PlanOptions,
};

fn hy4_snapshot() -> Option<std::path::PathBuf> {
    let hf = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME")
        .map(std::path::PathBuf::from)?;
    let snap = hf.join(
        "hub/models--wrldsuksgo2mars--Hy4-flash-EXL3-K4-MC/snapshots/cc72179113573d751ed3245e3e58d3640694378d",
    );
    snap.is_dir().then_some(snap)
}

#[test]
fn hy4_flash_family_descriptor_invariants() {
    assert_eq!(HY4_FLASH.id(), "hy4_flash");
    assert_eq!(HY4_FLASH.runtime(), RuntimeStatus::Serving);
    assert!(HY4_FLASH.expert_catalog());
    assert!(HY4_FLASH.optional(Component::Vision));
    assert!(HY4_FLASH.optional(Component::Speculator));
    assert!(!HY4_FLASH.optional(Component::Attention));
    assert!(!HY4_FLASH.optional(Component::RoutedExpert));
    assert!(!HY4_FLASH.optional(Component::MappedTable));
    assert_ne!(
        HY4_EXECUTOR.attention,
        AttentionPlacement::Heads,
        "2-GPU default must not replicate DSA KV"
    );
}

#[test]
fn hy4_flash_activation_clamps_routed_experts_only() {
    let mut cfg: serde_json::Value = serde_json::from_str(r#"{
        "model_type": "hy_v4_flash",
        "architectures": ["HYV4FlashForConditionalGeneration"],
        "image_token_id": 120001,
        "video_token_id": 120002,
        "eos_token_id": 2,
        "text_config": {
            "vocab_size": 120832,
            "hidden_size": 3072,
            "num_hidden_layers": 42,
            "intermediate_size": 13824,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "attention_bias": false,
            "enable_ihc": true,
            "use_mla": true,
            "gated_mla": true,
            "enable_lm_head_fp32": true,
            "moe_router_enable_expert_bias": true,
            "moe_router_use_sigmoid": true,
            "compressed_indexer_key_pooling": "gated",
            "compressed_indexer_skip_fine_select": true,
            "learnable_sink": true,
            "sandwich_norm": true,
            "use_compressed_indexer": true,
            "use_dsa": true,
            "dsa_skip_attn_post_norm": false,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_magnitude": 2.0,
            "q_lora_rank": 1024,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "num_attention_heads": 32,
            "swa_num_attention_heads": 64,
            "swa_qk_nope_head_dim": 192,
            "swa_qk_rope_head_dim": 64,
            "swa_v_head_dim": 256,
            "sliding_window": 512,
            "compressed_indexer_head_dim": 128,
            "compressed_indexer_num_attention_heads": 16,
            "compressed_indexer_compress_ratio": 4,
            "compressed_indexer_group_topk": 512,
            "compressed_indexer_candidate_topk": 2048,
            "n_routed_experts": 512,
            "num_experts_per_tok": 16,
            "moe_intermediate_size": 1536,
            "routed_expert_hidden_size": 1536,
            "n_shared_experts": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 3.998,
            "swiglu_limit": 10.0,
            "engram_layer_ids": [2, 12, 22, 32],
            "engram_max_ngram_size": 3,
            "engram_n_embed_per_ngram": 768,
            "engram_n_head_per_ngram": 8,
            "engram_pad_id": 0,
            "engram_seed": 10007,
            "engram_fp8_embedding": true,
            "engram_lookup_table_size": 120818,
            "engram_layer_head_vocab_sizes": {
                "2": [[10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10]],
                "12": [[10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10]],
                "22": [[10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10]],
                "32": [[10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10]]
            },
            "rope_parameters": {
                "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 10000000.0, "mrope_interleaved": true},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "mrope_interleaved": true}
            }
        }
    }"#).expect("valid json");
    cfg["text_config"]["layer_types"] = serde_json::json!(
        (0..42).map(|i| if i % 4 == 0 { "deepseek_sparse_attention" } else { "sliding_attention" }).collect::<Vec<_>>()
    );
    cfg["text_config"]["mlp_layer_types"] = serde_json::json!(
        (0..42).map(|i| if i == 0 { "dense" } else { "sparse" }).collect::<Vec<_>>()
    );
    let parsed = Hy4FlashConfig::from_hf(&cfg).expect("valid Hy4 config");
    assert_eq!(parsed.activation(0, false), cuteafd_core::FfnActivation::SiluBf16Product { limit: None });
    assert_eq!(parsed.activation(1, true), cuteafd_core::FfnActivation::SiluBf16Product { limit: None });
    assert_eq!(parsed.activation(1, false), cuteafd_core::FfnActivation::SiluBf16Product { limit: Some(10.0) });
    assert_eq!(
        crate::families::hy4_flash::weight_layout::core_split("model.layers.0.mlp.gate_proj"),
        crate::formats::trellis_core::CoreSplit::Output
    );
    assert_eq!(
        crate::families::hy4_flash::weight_layout::core_split("model.layers.1.mlp.gate"),
        crate::formats::trellis_core::CoreSplit::Whole
    );

    cfg["text_config"]["engram_fp8_embedding"] = serde_json::json!(false);
    assert!(Hy4FlashConfig::from_hf(&cfg).is_err(), "engram_fp8_embedding=false must be rejected");
}

/// Synthetic header-only 42-layer Hy4-flash EXL3 K4 MUL1 snapshot + sibling
/// `exl3-hy4f-k45-mul1/rtx-tp{1,2}` manifests for CPU planner matrix tests.
pub(crate) fn concurrency_matrix_snapshot() -> (tempfile::TempDir, std::path::PathBuf) {
    use crate::plan::testing::{t, write_snapshot, Tensor};

    let mut cfg: serde_json::Value = serde_json::from_str(r#"{
        "model_type": "hy_v4_flash",
        "architectures": ["HYV4FlashForConditionalGeneration"],
        "image_token_id": 120120,
        "video_token_id": 120124,
        "eos_token_id": 120025,
        "quantization_config": {
            "quant_method": "exl3",
            "format": "exl3",
            "checkpoint_format": "exl3",
            "bits": 4.0,
            "codebook": "mul1",
            "mul1_multiplier": 2212286765,
            "expert_bits": 4.0
        },
        "text_config": {
            "vocab_size": 120832,
            "hidden_size": 3072,
            "num_hidden_layers": 42,
            "intermediate_size": 13824,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "attention_bias": false,
            "enable_ihc": true,
            "use_mla": true,
            "gated_mla": true,
            "enable_lm_head_fp32": true,
            "moe_router_enable_expert_bias": true,
            "moe_router_use_sigmoid": true,
            "compressed_indexer_key_pooling": "gated",
            "compressed_indexer_skip_fine_select": true,
            "learnable_sink": true,
            "sandwich_norm": true,
            "use_compressed_indexer": true,
            "use_dsa": true,
            "dsa_skip_attn_post_norm": false,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_magnitude": 2.0,
            "q_lora_rank": 1024,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "num_attention_heads": 32,
            "swa_num_attention_heads": 64,
            "swa_qk_nope_head_dim": 192,
            "swa_qk_rope_head_dim": 64,
            "swa_v_head_dim": 256,
            "sliding_window": 512,
            "compressed_indexer_head_dim": 128,
            "compressed_indexer_num_attention_heads": 16,
            "compressed_indexer_compress_ratio": 4,
            "compressed_indexer_group_topk": 512,
            "compressed_indexer_candidate_topk": 2048,
            "n_routed_experts": 512,
            "num_experts_per_tok": 16,
            "moe_intermediate_size": 1536,
            "routed_expert_hidden_size": 1536,
            "n_shared_experts": 1,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": true,
            "routed_scaling_factor": 3.998,
            "swiglu_limit": 10.0,
            "engram_layer_ids": [2, 12, 22, 32],
            "engram_max_ngram_size": 3,
            "engram_n_embed_per_ngram": 768,
            "engram_n_head_per_ngram": 8,
            "engram_pad_id": 120002,
            "engram_seed": 10007,
            "engram_fp8_embedding": true,
            "engram_lookup_table_size": 120818,
            "engram_layer_head_vocab_sizes": {
                "2": [[10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10]],
                "12": [[10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10]],
                "22": [[10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10]],
                "32": [[10,10,10,10,10,10,10,10],[10,10,10,10,10,10,10,10]]
            },
            "rope_parameters": {
                "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 10000000.0, "mrope_interleaved": true},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "mrope_interleaved": true}
            }
        }
    }"#).unwrap();
    cfg["text_config"]["layer_types"] = serde_json::json!(
        (0..42).map(|i| if i % 4 == 0 { "deepseek_sparse_attention" } else { "sliding_attention" }).collect::<Vec<_>>()
    );
    cfg["text_config"]["mlp_layer_types"] = serde_json::json!(
        (0..42).map(|i| if i == 0 { "dense" } else { "sparse" }).collect::<Vec<_>>()
    );
    let parsed = Hy4FlashConfig::from_hf(&cfg).unwrap();
    let (h, v, hc) = (parsed.hidden, parsed.vocab_size, parsed.hc_mult);
    let mut tensors: Vec<Tensor> = vec![
        t("model.embed_tokens.weight", "BF16", &[v, h]),
        t("lm_head.weight", "BF16", &[v, h]),
        t("model.norm.weight", "BF16", &[h]),
        t("model.hc_head.hc_head_fn", "F32", &[hc, hc * h]),
        t("model.hc_head.hc_head_base", "F32", &[hc]),
        t("model.hc_head.hc_head_scale", "F32", &[1]),
    ];
    for layer in 0..parsed.layers {
        let base = format!("model.layers.{layer}");
        let attn = parsed.attention(layer);
        tensors.push(t(format!("{base}.input_layernorm.weight"), "BF16", &[h]));
        tensors.push(t(format!("{base}.post_attention_layernorm.weight"), "BF16", &[h]));
        for tag in ["hc_attn_layer", "hc_mlp_layer"] {
            let p = format!("{base}.{tag}.hc_pre");
            tensors.push(t(format!("{p}.hc_fn"), "F32", &[2 * hc, hc * h]));
            tensors.push(t(format!("{p}.hc_base"), "F32", &[2 * hc]));
            tensors.push(t(format!("{p}.hc_scale"), "F32", &[2]));
        }
        let a = format!("{base}.self_attn");
        tensors.push(t(format!("{a}.q_a_proj.weight"), "BF16", &[parsed.q_lora_rank, h]));
        tensors.push(t(format!("{a}.q_a_layernorm.weight"), "BF16", &[parsed.q_lora_rank]));
        tensors.push(t(format!("{a}.q_b_proj.weight"), "BF16", &[attn.heads * attn.qk_head_dim(), parsed.q_lora_rank]));
        tensors.push(t(format!("{a}.kv_a_proj_with_mqa.weight"), "BF16", &[parsed.kv_lora_rank + attn.qk_rope_dim, h]));
        tensors.push(t(format!("{a}.kv_a_layernorm.weight"), "BF16", &[parsed.kv_lora_rank]));
        tensors.push(t(format!("{a}.kv_b_proj.weight"), "BF16", &[attn.heads * (attn.qk_nope_dim + attn.v_head_dim), parsed.kv_lora_rank]));
        tensors.push(t(format!("{a}.linear_gate.weight"), "BF16", &[attn.heads * attn.v_head_dim, h]));
        tensors.push(t(format!("{a}.o_proj.weight"), "BF16", &[h, attn.heads * attn.v_head_dim]));
        tensors.push(t(format!("{a}.post_proj_norm.weight"), "BF16", &[h]));
        tensors.push(t(format!("{a}.learnable_sink_param"), "F32", &[attn.heads]));
        if attn.sparse {
            let (d, n) = (parsed.indexer_head_dim, parsed.indexer_heads);
            let ip = format!("{a}.compressed_indexer");
            tensors.push(t(format!("{ip}.query_projection.weight"), "BF16", &[n * d, parsed.q_lora_rank]));
            tensors.push(t(format!("{ip}.head_weight_projection.weight"), "BF16", &[n, h]));
            tensors.push(t(format!("{ip}.token_key_projection.weight"), "BF16", &[d, h]));
            tensors.push(t(format!("{ip}.token_key_norm.weight"), "BF16", &[d]));
            tensors.push(t(format!("{ip}.key_norm.weight"), "F32", &[d]));
            tensors.push(t(format!("{ip}.key_value_projection.weight"), "F32", &[d, d]));
            tensors.push(t(format!("{ip}.key_gate_projection.weight"), "F32", &[d, d]));
            tensors.push(t(format!("{ip}.intra_group_ape"), "F32", &[parsed.indexer_compress_ratio, d]));
        }
        if parsed.dense_layer[layer] {
            tensors.push(t(format!("{base}.mlp.gate_proj.weight"), "BF16", &[parsed.intermediate, h]));
            tensors.push(t(format!("{base}.mlp.up_proj.weight"), "BF16", &[parsed.intermediate, h]));
            tensors.push(t(format!("{base}.mlp.down_proj.weight"), "BF16", &[h, parsed.intermediate]));
            tensors.push(t(format!("{base}.mlp.post_fc2_norm.weight"), "BF16", &[h]));
        } else {
            let shared = parsed.moe_intermediate * parsed.shared_experts;
            tensors.push(t(format!("{base}.mlp.gate.weight"), "BF16", &[parsed.experts, h]));
            tensors.push(t(format!("{base}.mlp.gate.e_score_correction_bias"), "F32", &[parsed.experts]));
            tensors.push(t(format!("{base}.mlp.routed_expert_down_proj.weight"), "BF16", &[parsed.routed_expert_hidden, h]));
            tensors.push(t(format!("{base}.mlp.routed_expert_up_proj.weight"), "BF16", &[h, parsed.routed_expert_hidden]));
            tensors.push(t(format!("{base}.mlp.combine_output_norm.weight"), "BF16", &[h]));
            tensors.push(t(format!("{base}.mlp.shared_experts.gate_proj.weight"), "BF16", &[shared, h]));
            tensors.push(t(format!("{base}.mlp.shared_experts.up_proj.weight"), "BF16", &[shared, h]));
            tensors.push(t(format!("{base}.mlp.shared_experts.down_proj.weight"), "BF16", &[h, shared]));
            let (eh, ei, bits) = (parsed.routed_expert_hidden, parsed.moe_intermediate, 4usize);
            for expert in 0..parsed.experts {
                for (proj, n, k) in [("gate_proj", ei, eh), ("up_proj", ei, eh), ("down_proj", eh, ei)] {
                    let stem = format!("{base}.mlp.experts.{expert}.{proj}");
                    tensors.push(t(format!("{stem}.trellis"), "I16", &[k / 16, n / 16, 16 * bits]));
                    tensors.push(t(format!("{stem}.suh"), "F16", &[k]));
                    tensors.push(t(format!("{stem}.svh"), "F16", &[n]));
                    tensors.push(t(format!("{stem}.mul1"), "I32", &[]));
                }
            }
        }
        if let Some(rows) = parsed.engram_rows(layer) {
            let ep = format!("{base}.engram_layer");
            let w = parsed.engram_head_width();
            let concat = parsed.engram_concat_width();
            tensors.push(t(format!("{ep}.multi_head_embedding.embedding.weight"), "BF16", &[rows as usize, w]));
            tensors.push(t(format!("{ep}.lookup_table"), "I32", &[parsed.engram_lookup_table_size]));
            tensors.push(t(format!("{ep}.unified_pre_gamma"), "BF16", &[h]));
            tensors.push(t(format!("{ep}.value_proj.weight"), "BF16", &[h, concat]));
            tensors.push(t(format!("{ep}.value_proj.bias"), "BF16", &[h]));
            for j in 0..parsed.engram_total_heads() {
                tensors.push(t(format!("{ep}.lookup_per_head_norms.{j}.weight"), "BF16", &[w]));
            }
            for j in 0..hc {
                tensors.push(t(format!("{ep}.key_projs.{j}.weight"), "BF16", &[h, concat]));
                tensors.push(t(format!("{ep}.key_projs.{j}.bias"), "BF16", &[h]));
                tensors.push(t(format!("{ep}.norm_key.{j}.weight"), "BF16", &[h]));
                tensors.push(t(format!("{ep}.norm_query.{j}.weight"), "BF16", &[h]));
                tensors.push(t(format!("{ep}.unified_per_hc_scales.{j}"), "BF16", &[h]));
            }
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let mut non_engram = Vec::with_capacity(tensors.len());
    let mut weight_map = serde_json::Map::new();
    for tensor in tensors {
        if tensor.0.ends_with(".engram_layer.multi_head_embedding.embedding.weight") {
            let layer: usize = tensor
                .0
                .strip_prefix("model.layers.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse().ok())
                .unwrap();
            let shard = format!("engram-bf16-layer-{layer:02}.safetensors");
            crate::plan::testing::write_safetensors(&dir.path().join(&shard), std::slice::from_ref(&tensor));
            weight_map.insert(tensor.0, serde_json::json!(shard));
        } else {
            weight_map.insert(tensor.0.clone(), serde_json::json!("model-00001-of-00001.safetensors"));
            non_engram.push(tensor);
        }
    }
    write_snapshot(dir.path(), &cfg, &non_engram, None);
    let idx_bytes = serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap();
    std::fs::write(dir.path().join("model.safetensors.index.json"), &idx_bytes).unwrap();
    let mut tensor_storage = serde_json::Map::new();
    for layer in parsed.routed_layers() {
        let base = format!("model.layers.{layer}");
        for expert in 0..parsed.experts {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let stem = format!("{base}.mlp.experts.{expert}.{proj}");
                tensor_storage.insert(
                    stem.clone(),
                    serde_json::json!({
                        "quant_format": "exl3",
                        "bits_per_weight": 4,
                        "mul1_multiplier": 2212286765u64,
                        "stored_tensors": {
                            format!("{stem}.trellis"): {"dtype": "int16", "shape": [96, 96, 64]},
                            format!("{stem}.suh"): {"dtype": "float16", "shape": [1536]},
                            format!("{stem}.svh"): {"dtype": "float16", "shape": [1536]},
                            format!("{stem}.mul1"): {"dtype": "int32", "shape": []},
                        }
                    }),
                );
            }
        }
    }
    let mut quant_cfg = cfg["quantization_config"].as_object().unwrap().clone();
    quant_cfg.insert("method".to_string(), serde_json::json!("exl3"));
    quant_cfg.insert("out_scales".to_string(), serde_json::json!("auto"));
    quant_cfg.insert("group_size".to_string(), serde_json::json!(-1));
    quant_cfg.insert("desc_act".to_string(), serde_json::json!(false));
    quant_cfg.insert("lm_head".to_string(), serde_json::json!(false));
    quant_cfg.insert("tensor_storage".to_string(), serde_json::Value::Object(tensor_storage));
    let quant_bytes = serde_json::to_vec(&serde_json::Value::Object(quant_cfg)).unwrap();
    std::fs::write(dir.path().join("quantize_config.json"), &quant_bytes).unwrap();
    let cfg_bytes = std::fs::read(dir.path().join("config.json")).unwrap();
    {
        use sha2::{Digest, Sha256};
        let core_tensor = "model.layers.0.self_attn.q_a_proj.weight";
        let mut sorted: std::collections::BTreeMap<String, String> = weight_map
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
            .collect();
        sorted.remove(core_tensor);
        let mut sh = Sha256::new();
        for (k, v) in &sorted {
            sh.update(format!("{k}\0{v}\n"));
        }
        let shared_digest = format!("{:x}", sh.finalize());
        let mut metas = std::collections::BTreeMap::new();
        metas.insert("config.json".to_string(), format!("{:x}", Sha256::digest(&cfg_bytes)));
        metas.insert(
            "model.safetensors.index.json".to_string(),
            format!("{:x}", Sha256::digest(&idx_bytes)),
        );
        metas.insert("quantize_config.json".to_string(), format!("{:x}", Sha256::digest(&quant_bytes)));
        let mut fp = Sha256::new();
        fp.update(format!("cuteafd-mc-view-v1\0original\0{shared_digest}\0"));
        for (k, v) in &metas {
            fp.update(format!("{k}\0{v}\n"));
        }
        let fingerprint = format!("{:x}", fp.finalize());
        let k8_idx_bytes = idx_bytes.clone();
        let k8_cfg_bytes = cfg_bytes.clone();
        let k8_quant_bytes = quant_bytes.clone();
        std::fs::write(dir.path().join("model.k8.safetensors.index.json"), &k8_idx_bytes).unwrap();
        std::fs::write(dir.path().join("config.k8.json"), &k8_cfg_bytes).unwrap();
        std::fs::write(dir.path().join("quantize_config.k8.json"), &k8_quant_bytes).unwrap();
        let mut k8_metas = std::collections::BTreeMap::new();
        k8_metas.insert("config.k8.json".to_string(), format!("{:x}", Sha256::digest(&k8_cfg_bytes)));
        k8_metas.insert(
            "model.k8.safetensors.index.json".to_string(),
            format!("{:x}", Sha256::digest(&k8_idx_bytes)),
        );
        k8_metas.insert(
            "quantize_config.k8.json".to_string(),
            format!("{:x}", Sha256::digest(&k8_quant_bytes)),
        );
        let mut k8_fp = Sha256::new();
        k8_fp.update(format!("cuteafd-mc-view-v1\0k8-compact\0{shared_digest}\0"));
        for (k, v) in &k8_metas {
            k8_fp.update(format!("{k}\0{v}\n"));
        }
        let k8_fingerprint = format!("{:x}", k8_fp.finalize());
        let engram_layers: Vec<serde_json::Value> = parsed
            .engram_layers
            .iter()
            .map(|&layer| {
                let rows = parsed.engram_rows(layer).unwrap();
                serde_json::json!({
                    "tensor": format!("model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight"),
                    "sha256": "b".repeat(64),
                    "parts": [{
                        "tensor": format!("model.layers.{layer}.engram_layer.multi_head_embedding.embedding.strided_00000"),
                        "file": format!("engram-fp8-layer-{layer:02}-part-00000.safetensors"),
                        "rows": rows,
                        "sha256": "c".repeat(64),
                    }]
                })
            })
            .collect();
        let manifest_mc = serde_json::json!({
            "schema": "cuteafd-mc-v1",
            "required_features": [],
            "source": {"repo": "synthetic/hy4-flash", "revision": "a".repeat(40)},
            "original_core_dtype": "BF16",
            "shared_inventory_sha256": shared_digest,
            "variants": [
                {
                    "name": "original",
                    "index": "model.safetensors.index.json",
                    "config": "config.json",
                    "quantize": "quantize_config.json",
                    "core_bits": 16,
                    "core_tensors": [core_tensor],
                    "metadata_sha256": metas,
                    "fingerprint": fingerprint,
                },
                {
                    "name": "k8-compact",
                    "index": "model.k8.safetensors.index.json",
                    "config": "config.k8.json",
                    "quantize": "quantize_config.k8.json",
                    "core_bits": 8,
                    "core_tensors": [core_tensor],
                    "metadata_sha256": k8_metas,
                    "fingerprint": k8_fingerprint,
                }
            ],
            "engram_fp8": {
                "layout": crate::plan::checkpoint::mc::ENGRAM_FP8_LAYOUT,
                "row_bytes": crate::plan::checkpoint::mc::ENGRAM_FP8_ROW_BYTES,
                "code_dtype": "F8_E4M3",
                "scale_dtype": "F32",
                "recipe": format!(
                    "vendor BF16 per-row E4M3 round trip, BF16 scale held in F32, decode = BF16(F32 code*scale); reference={}",
                    "a".repeat(40)
                ),
                "layers": engram_layers,
            },
            "exl3_codebook": {"name": "mul1", "multiplier": 2212286765u64},
        });
        std::fs::write(
            dir.path().join(crate::plan::checkpoint::mc::MC_MANIFEST),
            serde_json::to_vec(&manifest_mc).unwrap(),
        )
        .unwrap();
    }
    let manifest = dir.path().join("share/PROGRAMS.json");
    std::fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    std::fs::write(&manifest, r#"{"programs":[]}"#).unwrap();
    for (tp, intermediate) in [(1usize, 1536usize), (2, 768)] {
        let root = dir.path().join(format!("lib/exl3/exl3-hy4f-k45-mul1/rtx-tp{tp}"));
        for capacity in crate::placement::inventory::exl3_capacities(4096) {
            let path = root.join(format!("m{capacity}"));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&serde_json::json!({
                "capacity": capacity, "hidden": 1536, "experts": 512, "intermediate": intermediate, "top_k": 16,
                "bits": [4, 5], "codebook": "mul1", "activation": "silu_post_clamp", "swiglu_limit": 10.0,
                "input_format": "e4m3_k32", "trellis_lut": {"bytes": 4096}, "buffers": {
                    "scratch": {"allocation": "scratch", "dtype": "f32", "bytes": capacity * 64, "zero_on_create": false}}
            })).unwrap()).unwrap();
        }
    }
    (dir, manifest)
}

#[test]
#[ignore = "reads installed wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC snapshot via CUTEAFD_DEPLOYMENT_TEST_HF_HOME"]
fn installed_hy4_flash_plan_across_all_mc_views_and_two_cards() {
    let snap = hy4_snapshot().expect("CUTEAFD_DEPLOYMENT_TEST_HF_HOME missing Hy4-flash snapshot");

    // 1. Verify all 4 MC views classify every single tensor with zero unclassified and zero rejections,
    //    and wire cleanly through the real EXL3 MUL1 expert catalog.
    for view in [CoreView::Original, CoreView::K8Compact, CoreView::K6Compact, CoreView::Fp8Compact] {
        let selected = CheckpointView::select(&snap, view).expect("select Hy4 view");
        let input = PlanInput::open_view(&selected).expect("open Hy4 view");
        let catalog = input.expert_catalog().expect("real hy4f EXL3 MUL1 expert catalog");
        let exl3 = catalog.exl3().expect("hy4f catalog must be EXL3");
        assert_eq!(exl3.package_name("hy4f"), "exl3-hy4f-k45-mul1");
        let report = input.plan(&PlanOptions::default()).expect("plan Hy4 view");
        assert_eq!(report.family.as_deref(), Some("hy4_flash"), "view={view}");
        assert!(report.unclassified.is_empty(), "view={view} unclassified: {:?}", report.unclassified);
        for comp in &report.components {
            assert_eq!(
                comp.rejected, 0,
                "view={view} component {:?} had rejections: {:?}",
                comp.component, comp.rejections
            );
        }
    }

    // 2. Verify honest refusal when no exl3-hy4f-k45-mul1 package is installed beside PROGRAMS.json,
    //    and verify real Spark placement with measured SM121 census + MappedPlacement host Engram.
    let selected = CheckpointView::select(&snap, CoreView::Original).expect("select original");
    let input = PlanInput::open_view(&selected).expect("open original");
    let mut missing_layout = LayoutOptions::default();
    missing_layout.rtx_bytes = vec![96u64 << 30, 96u64 << 30];
    missing_layout.host_memory = Some((512 << 30, 400 << 30));
    missing_layout.workspace_manifest = Some(std::path::PathBuf::from("/nonexistent/PROGRAMS.json"));
    let mut missing_opts = PlanOptions::default();
    missing_opts.placement = ExpertPlacement::Local;
    missing_opts.layout = Some(missing_layout);
    let missing_report = input.plan(&missing_opts).expect("plan missing local package");
    assert!(
        missing_report.solver_placement.is_none(),
        "0-Spark must refuse honestly when exl3-hy4f-k45-mul1 is not installed"
    );

    // 3. Verify 2x RTX + 2 Sparks plans cleanly with real SM121 Spark census, non-replicated 2-GPU
    //    default attention, and real MappedPlacement host Engram (61.47 GiB on host, 0 on GPU/Spark).
    let mut layout = LayoutOptions::default();
    layout.rtx_bytes = vec![96u64 << 30, 96u64 << 30];
    layout.host_memory = Some((512 << 30, 400 << 30));
    layout.workspace_manifest = Some(std::path::PathBuf::from("/nonexistent/PROGRAMS.json"));
    let mut opts = PlanOptions::default();
    opts.placement = ExpertPlacement::Sparks { ranks: 2 };
    opts.layout = Some(layout);
    let report = input.plan(&opts).expect("plan 2x+2sparks");
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert_ne!(report.attention_placement, AttentionPlacement::Heads);
    let mem = report.memory_layout.as_ref().expect("memory layout");
    assert!(!mem.devices.is_empty());
    assert!(mem.pool_tokens >= 2_000_000, "pool {} < 2M", mem.pool_tokens);
    let host = report.host_ledger.as_ref().expect("host ledger");
    let engram_host: u64 = host.items.iter()
        .filter(|item| item.group == "mapped-tables/preloaded (coordinator local)")
        .map(|item| item.bytes)
        .sum();
    assert_eq!(engram_host, 66_000_641_600, "expected exact 61.47 GiB engram_fp8 via MappedPlacement");
    for dev in &mem.devices {
        assert!(
            !dev.items.iter().any(|item| item.group.contains("mapped-tables/preloaded")),
            "Engram tables must not be charged to GPU/Spark {}",
            dev.name()
        );
    }
}

#[test]
fn hy4_flash_plan_refuses_checkpoint_without_engram_fp8() {
    let (dir, _manifest) = concurrency_matrix_snapshot();
    // Remove cuteafd-mc.json so no FP8 Engram shards are declared; planning must refuse by name.
    std::fs::remove_file(dir.path().join(crate::plan::checkpoint::mc::MC_MANIFEST)).unwrap();
    let input = PlanInput::open(dir.path()).expect("open plain snapshot");
    let report = input.plan(&PlanOptions::default()).expect("plan plain snapshot");
    let err = report
        .config_error
        .as_ref()
        .expect("Hy4-flash without engram_fp8 must refuse");
    assert!(
        err.contains("engram-fp8-layer-"),
        "expected refusal naming engram-fp8-layer-* shards, got: {err}"
    );
}
