use super::*;
use crate::plan::exlr8::dense::{
    DenseAxis, DenseComponent, DenseEncoding, DenseExecution, DenseOwnership, DenseRange,
    DenseRecord, KvBGeometry, ShardSide,
};
use crate::plan::testing::{self, Tensor, exl3, t, write_snapshot};

fn ordinary(
    config: &serde_json::Value,
    tensors: &[Tensor],
    ranks: usize,
    owners: Option<&[u8]>,
    bf16: bool,
    embedding: EmbeddingHome,
) -> WeightInventory {
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(dir.path(), config, tensors, None);
    let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
    WeightInventory::from_checkpoint(&checkpoint, ranks, 2, owners, bf16, false, embedding).unwrap()
}

fn bytes(inventory: &WeightInventory, suffix: &str) -> u64 {
    inventory
        .owners
        .iter()
        .find(|owner| owner.allocation.id.ends_with(suffix))
        .unwrap_or_else(|| panic!("missing {suffix}"))
        .allocation
        .bytes
}

#[test]
fn checkpoint_owners_match_loader_floors_splits_and_pinned_embedding() {
    let (config, tensors) = testing::glm5_compact_weights();
    let inventory = ordinary(&config, &tensors, 2, None, false, EmbeddingHome::HostPinned);
    assert_eq!(
        bytes(&inventory, "gpu.0.layer.0.operand.w_q_b_fp8"),
        128 * 256
    );
    assert_eq!(bytes(&inventory, "gpu.0.layer.0.operand.w_q_b_scale"), 256);
    assert_eq!(bytes(&inventory, "gpu.0.layer.1.operand.gate.bias"), 256);
    assert_eq!(
        bytes(&inventory, "gpu.0.layer.0.operand.w_qkv_a_kscale"),
        (256 + 192) * 2 * 4
    );
    let embedding = inventory
        .owners
        .iter()
        .find(|o| o.allocation.id == "glm.weights.embedding")
        .unwrap();
    assert_eq!(embedding.allocation.location, Location::PINNED);
    assert_eq!(embedding.allocation.bytes, 64 * 256 * 2);
    let groups = inventory.resident_groups().unwrap();
    assert_eq!(
        groups.iter().flatten().map(|g| g.2).sum::<u64>(),
        inventory
            .owners
            .iter()
            .filter(|o| o.allocation.location != Location::PINNED)
            .map(|o| o.allocation.bytes)
            .sum::<u64>()
    );
    assert!(inventory.components(0).is_err());
    assert_eq!(
        inventory.components(16).unwrap().len(),
        inventory.owners.len()
    );

    // ModelOpt conversion splits request exact small grids, not landing's floor.
    let converted: Vec<_> = tensors
        .iter()
        .flat_map(|(name, dtype, shape)| {
            if name.ends_with("_scale_inv") {
                return vec![];
            }
            let mut out = vec![(name.clone(), *dtype, shape.clone())];
            if *dtype == "F8_E4M3" {
                out.push(t(format!("{name}_scale"), "F32", &[]));
                out.push(t(
                    format!("{}input_scale", name.strip_suffix("weight").unwrap()),
                    "F32",
                    &[],
                ));
            }
            out
        })
        .collect();
    let modelopt = ordinary(
        &config,
        &converted,
        2,
        None,
        false,
        EmbeddingHome::Device(0),
    );
    assert_eq!(bytes(&modelopt, "gpu.0.layer.0.operand.w_q_b_scale"), 8);
    assert_eq!(bytes(&modelopt, "gpu.0.layer.0.operand.w_o_scale"), 256);
    for rank in 0..2 {
        for name in ["w_gate_up_tscale", "w_down_tscale"] {
            assert_eq!(
                bytes(&modelopt, &format!("gpu.{rank}.layer.0.operand.{name}")),
                256
            );
        }
    }

    let owned = ordinary(
        &config,
        &tensors,
        2,
        Some(&[1, 0]),
        false,
        EmbeddingHome::Device(0),
    );
    assert!(!owned.owners.iter().any(|o| {
        o.allocation
            .id
            .starts_with("glm.weights.gpu.0.layer.0.operand.w_q")
    }));
    assert_eq!(bytes(&owned, "gpu.1.layer.0.operand.w_q_b_fp8"), 256 * 256);
    for rank in 0..2 {
        assert_eq!(
            bytes(&owned, &format!("gpu.{rank}.layer.0.operand.w_gate_up_fp8")),
            512 * 256
        );
    }
}

#[test]
fn bf16_and_k6_owners_keep_distinct_payload_and_vector_requests() {
    let (mut config, tensors) = testing::glm5_compact_weights();
    let bf16: Vec<_> = tensors
        .into_iter()
        .filter(|t| !t.0.ends_with("_scale_inv"))
        .map(|(name, dtype, shape)| (name, if dtype == "F8_E4M3" { "BF16" } else { dtype }, shape))
        .collect();
    let inventory = ordinary(&config, &bf16, 2, None, true, EmbeddingHome::Device(0));
    assert_eq!(
        bytes(&inventory, "gpu.0.layer.0.operand.w_q_b"),
        128 * 256 * 2
    );
    assert_eq!(
        bytes(&inventory, "gpu.1.layer.1.operand.w_down"),
        256 * 128 * 2
    );
    assert!(
        !inventory
            .owners
            .iter()
            .any(|o| o.allocation.id.ends_with("w_qkv_a_kscale"))
    );

    config["vocab_size"] = serde_json::json!(256);
    let cfg = GlmDsaConfig::from_hf(&config).unwrap();
    let core: Vec<_> = bf16
        .into_iter()
        .flat_map(|(name, dtype, mut shape)| {
            if ["lm_head.weight", "model.embed_tokens.weight"].contains(&name.as_str()) {
                shape[0] = 256;
            }
            let Some(stem) = name
                .strip_suffix(".weight")
                .filter(|stem| crate::families::glm5::core::split(&cfg, stem).is_some())
            else {
                return vec![(name, dtype, shape)];
            };
            let n = if stem.ends_with("kv_a_proj_with_mqa") {
                shape[0].next_multiple_of(128)
            } else {
                shape[0]
            };
            exl3(stem, n, shape[1], 6)
        })
        .collect();
    let inventory = ordinary(
        &config,
        &core,
        2,
        Some(&[0, 1]),
        false,
        EmbeddingHome::Device(0),
    );
    assert_eq!(
        bytes(&inventory, "gpu.0.layer.0.core.kv_a.payload"),
        256 * 256 * 6 / 8
    );
    assert_eq!(bytes(&inventory, "gpu.0.layer.0.core.q_b.suh"), 512);
    assert_eq!(bytes(&inventory, "gpu.1.layer.0.core.down.suh"), 512);
    assert_eq!(bytes(&inventory, "gpu.0.core.head.svh"), 512);
    assert!(
        !inventory
            .owners
            .iter()
            .any(|o| o.allocation.id.contains(".operand.w_gate_up"))
    );
}

#[test]
fn checkpoint_owners_refuse_missing_sources_and_invalid_block_splits() {
    let (config, tensors) = testing::glm5_compact_weights();
    let dir = tempfile::tempdir().unwrap();
    let check = |tensors: &[Tensor], ranks, owners: Option<&[u8]>, home| {
        write_snapshot(dir.path(), &config, tensors, None);
        let checkpoint = Checkpoint::inventory(dir.path()).unwrap();
        WeightInventory::from_checkpoint(&checkpoint, ranks, 2, owners, false, false, home)
    };
    assert!(check(&tensors, 0, None, EmbeddingHome::Device(0)).is_err());
    assert!(check(&tensors, 2, Some(&[0]), EmbeddingHome::Device(0)).is_err());
    assert!(check(&tensors, 2, None, EmbeddingHome::Device(2)).is_err());
    let missing: Vec<_> = tensors
        .iter()
        .filter(|t| t.0 != "model.layers.0.self_attn.q_b_proj.weight_scale_inv")
        .cloned()
        .collect();
    assert!(check(&missing, 2, None, EmbeddingHome::Device(0)).is_err());
    let mut broken = tensors.clone();
    broken
        .iter_mut()
        .find(|t| t.0 == "model.layers.0.self_attn.q_b_proj.weight")
        .unwrap()
        .2[0] = 128;
    assert!(check(&broken, 2, None, EmbeddingHome::Device(0)).is_err());
    broken = tensors.clone();
    broken
        .iter_mut()
        .find(|t| t.0 == "model.layers.0.self_attn.q_b_proj.weight_scale_inv")
        .unwrap()
        .1 = "BF16";
    assert!(check(&broken, 2, None, EmbeddingHome::Device(0)).is_err());
    assert!(product(&[u64::MAX, 2]).is_err());
    assert!(sum([u64::MAX, 1]).is_err());
    let mut inventory = check(&tensors, 2, None, EmbeddingHome::Device(0)).unwrap();
    inventory.add_peer_routers(&[1]).unwrap();
    assert_eq!(bytes(&inventory, "gpu.1.layer.1.operand.gate.bias"), 256);
    assert!(inventory.add_peer_routers(&[1]).is_err());
    assert!(inventory.add_peer_routers(&[0]).is_err());
}

#[test]
fn native_owners_consume_shared_slices_absorption_and_vocabulary_once() {
    let (config, _) = testing::glm5_compact_weights();
    let selection = DenseSelection::from_hf(&config).unwrap();
    let component = |suffix: &str, dtype: &str, shape: Vec<usize>, bytes| DenseComponent {
        suffix: suffix.into(),
        dtype: dtype.into(),
        shape,
        source: DenseRange {
            path: "not-read".into(),
            offset: 0,
            bytes,
        },
        window: None,
    };
    let record = |tensor: &str, encoding, shape: Vec<usize>, execution, components| DenseRecord {
        tensor: tensor.into(),
        encoding,
        shape,
        side: ShardSide::Replicated,
        ownership: DenseOwnership::new(execution),
        channels_per_shard: 0,
        rotation_id: 0,
        rotation: None,
        hessian_solved: false,
        components,
        native_blocks: 0,
    };
    let records = vec![
        record(
            "model.layers.0.self_attn.q_b_proj.weight",
            DenseEncoding::Bf16,
            vec![256, 256],
            DenseExecution::Split(DenseAxis::Output),
            vec![component("bf16", "BF16", vec![256, 256], 256 * 256 * 2)],
        ),
        record(
            "model.layers.0.input_layernorm.weight",
            DenseEncoding::Bf16,
            vec![256],
            DenseExecution::Replicated,
            vec![component("plain", "BF16", vec![256], 512)],
        ),
        record(
            "model.layers.0.self_attn.kv_b_proj.weight",
            DenseEncoding::BlockFp8,
            vec![384, 128],
            DenseExecution::AbsorbHeads(KvBGeometry {
                heads: 2,
                nope: 64,
                values: 128,
                latent: 128,
            }),
            vec![
                component("fp8", "U8", vec![384, 128], 384 * 128),
                component("scale", "F32", vec![3, 1], 12),
            ],
        ),
        record(
            "model.norm.weight",
            DenseEncoding::Bf16,
            vec![256],
            DenseExecution::RankZero,
            vec![component("plain", "BF16", vec![256], 512)],
        ),
        record(
            "model.layers.1.mlp.gate.weight",
            DenseEncoding::Bf16,
            vec![2, 256],
            DenseExecution::RankZero,
            vec![component("bf16", "BF16", vec![2, 256], 1024)],
        ),
        record(
            "model.embed_tokens.weight",
            DenseEncoding::RowFp8,
            vec![64, 256],
            DenseExecution::RankZero,
            vec![
                component("fp8", "U8", vec![64, 256], 64 * 256),
                component("scale", "F32", vec![64], 256),
            ],
        ),
        record(
            "lm_head.weight",
            DenseEncoding::RowFp8,
            vec![64, 256],
            DenseExecution::RankZero,
            vec![
                component("fp8", "U8", vec![64, 256], 64 * 256),
                component("scale", "F32", vec![64], 256),
            ],
        ),
    ];
    let plans: Vec<_> = (0..2)
        .map(|rank| DensePlan::for_rank(&records, 2, rank).unwrap())
        .collect();
    let inventory = WeightInventory::from_native(&selection, 2, &plans).unwrap();
    assert_eq!(
        bytes(&inventory, "gpu.1.layer.0.native.n.q_b.bf16"),
        128 * 256 * 2
    );
    assert_eq!(
        bytes(&inventory, "gpu.1.layer.0.operand.w_uk_scale"),
        128 * 4
    );
    assert_eq!(bytes(&inventory, "gpu.0.layer.1.operand.gate"), 1024);
    assert_eq!(bytes(&inventory, "glm.weights.embedding"), 256);
    assert_eq!(
        inventory
            .owners
            .iter()
            .map(|o| o.allocation.bytes)
            .sum::<u64>(),
        plans.iter().map(|p| p.resident_bytes as u64).sum::<u64>() + 256
    );
    assert_eq!(
        inventory
            .owners
            .iter()
            .filter(|o| o.allocation.id.contains("vocabulary."))
            .count(),
        4
    );
    let mut invalid = plans.clone();
    invalid[0].resident_bytes += 1;
    assert!(WeightInventory::from_native(&selection, 2, &invalid).is_err());
    invalid = plans.clone();
    invalid[1].rank = 0;
    assert!(WeightInventory::from_native(&selection, 2, &invalid).is_err());
    invalid = plans;
    let duplicate = invalid[0].tensors[0].clone();
    invalid[0].tensors.push(duplicate);
    assert!(WeightInventory::from_native(&selection, 2, &invalid).is_err());
}
