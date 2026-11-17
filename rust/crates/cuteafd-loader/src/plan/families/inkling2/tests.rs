use super::*;
use crate::plan::{self, testing, PlanOptions};

/// Which routed-expert storage the fixture writes.
#[derive(Clone, Copy, PartialEq)]
enum Fixture { Mxfp8, Mixed, Exl3 }

/// A four-layer release-shaped snapshot (layers 0..3 local, layer 3 global so
/// both attention kinds, both conv kinds and a three-vector residual query
/// appear). Expert storage follows `fixture` per layer, as the three target
/// checkpoints store them.
fn fixture(kind: Fixture) -> (tempfile::TempDir, serde_json::Value, Vec<testing::Tensor>) {
    fixture_with_experts(kind, 4, 2)
}
fn fixture_with_experts(kind: Fixture, experts: usize, topk: usize) -> (tempfile::TempDir, serde_json::Value, Vec<testing::Tensor>) {
    let dir = tempfile::tempdir().unwrap();
    let mut small = crate::families::inkling2::config::tests::release_config();
    small["text_config"]["num_hidden_layers"] = 4.into();
    small["text_config"]["local_layer_ids"] = serde_json::json!([0, 1, 2]);
    small["text_config"]["n_routed_experts"] = experts.into();
    small["text_config"]["num_experts_per_tok"] = topk.into();
    small["text_config"]["n_layers_per_attn_res_block"] = 2.into();
    small["mtp_config"]["target_layer_ids"] = serde_json::json!([1, 4]);
    let c = Inkling2Config::from_hf(&small).unwrap();
    let (h, d, v, e, i, k) = (c.hidden, c.head_dim, c.vocab_size, c.experts, c.intermediate, c.sconv_kernel);
    let mut tensors = Vec::new();
    let mut plain = |name: String, dtype: &'static str, shape: Vec<usize>| tensors.push((name, dtype, shape));
    plain(format!("{TEXT}embed.weight"), "BF16", vec![v, h]);
    plain(format!("{TEXT}unembed.weight"), "BF16", vec![v, h]);
    plain(format!("{TEXT}embed_norm.weight"), "BF16", vec![h]);
    plain(format!("{TEXT}norm.weight"), "BF16", vec![h]);
    plain(format!("{TEXT}attn_res.output_query"), "F32", vec![h]);
    for layer in 0..4 {
        let local = c.local[layer];
        let p = format!("{TEXT}layers.{layer}.");
        for (name, n, kk) in [("wq_du", c.heads * d, h), ("wr_du", c.heads * c.d_rel, h), ("wk_dv", c.kv_heads * d, h),
            ("wv_dv", c.kv_heads * d, h), ("gate_attn", c.heads * d, h), ("attn_sink_proj", c.heads, h), ("wo_ud", h, c.heads * d)] {
            plain(format!("{p}attn.{name}.weight"), "BF16", vec![n, kk]);
        }
        plain(format!("{p}attn.q_l2_clip.scale"), "BF16", vec![c.heads, d]);
        plain(format!("{p}attn.rel_l2_clip.scale"), "BF16", vec![c.heads, c.d_rel]);
        plain(format!("{p}attn.rel_logits_proj.proj"), "BF16", vec![c.d_rel, if local { c.window } else { c.rel_extent }]);
        for name in ["k", "v"] {
            if local { plain(format!("{p}attn.{name}_sconv.weight"), "BF16", vec![c.kv_heads * d, 1, k]); }
            else {
                plain(format!("{p}attn.{name}_sconv.static_w"), "BF16", vec![k, c.kv_heads * d]);
                plain(format!("{p}attn.{name}_sconv.weight_proj"), "BF16", vec![c.kv_heads * k, h]);
            }
        }
        for name in ["attn_sconv", "mlp_sconv"] { plain(format!("{p}{name}.weight"), "BF16", vec![h, 1, k]); }
        let norms: &[&str] = if local { &["attn_norm"] } else { &["q_input_norm", "k_input_norm", "v_input_norm"] };
        for norm in norms.iter().chain(["mlp_norm"].iter()) { plain(format!("{p}{norm}.weight"), "BF16", vec![h]); }
        plain(format!("{p}mlp.gate.weight"), "BF16", vec![e, h]);
        plain(format!("{p}mlp.gate.bias"), "F32", vec![e]);
        plain(format!("{TEXT}attn_res.mlp_queries.{layer}"), "F32", vec![h]);
        if layer > 0 { plain(format!("{TEXT}attn_res.attn_queries.{}", layer - 1), "F32", if local { vec![h] } else { vec![3, h] }); }
        let storage = match kind {
            Fixture::Mxfp8 => ExpertStorage::Mxfp8,
            Fixture::Mixed => if layer < 3 { ExpertStorage::Nvfp4 } else { ExpertStorage::Mxfp8 },
            Fixture::Exl3 => ExpertStorage::Exl3 { bits: 4 },
        };
        match storage {
            ExpertStorage::Mxfp8 => {
                plain(format!("{p}mlp.experts.w13_weight"), "F8_E4M3", vec![e, 2 * i, h]);
                plain(format!("{p}mlp.experts.w13_weight_scale"), "U8", vec![e, 2 * i, h / 32]);
                plain(format!("{p}mlp.experts.w2_weight"), "F8_E4M3", vec![e, h, i]);
                plain(format!("{p}mlp.experts.w2_weight_scale"), "U8", vec![e, h, i / 32]);
            }
            ExpertStorage::Nvfp4 => {
                for (proj, n, kk) in [("w13", 2 * i, h), ("w2", h, i)] {
                    plain(format!("{p}mlp.experts.{proj}_weight"), "U8", vec![e, n, kk / 2]);
                    plain(format!("{p}mlp.experts.{proj}_weight_scale"), "F8_E4M3", vec![e, n, kk / 16]);
                    plain(format!("{p}mlp.experts.{proj}_weight_amax"), "F32", vec![e]);
                    plain(format!("{p}mlp.experts.{proj}_weight_outer_scale_denominator"), "F32", vec![1]);
                }
            }
            ExpertStorage::Exl3 { bits } => {
                for expert in 0..e {
                    for (proj, n, kk) in [("gate_proj", i, h), ("up_proj", i, h), ("down_proj", h, i)] {
                        for (name, dtype, shape) in testing::exl3(&format!("{p}mlp.experts.{expert}.{proj}"), n, kk, bits) {
                            plain(name, dtype, shape);
                        }
                    }
                }
            }
        }
    }
    if kind == Fixture::Exl3 {
        small["quantization_config"] = testing::exl3_compact(4);
    }
    testing::write_snapshot(dir.path(), &small, &tensors, None);
    // The fixtures carry the checkpoint's quantization metadata beside the
    // headers, as the release does; the loader cross-checks and never requires it.
    std::fs::write(dir.path().join("hf_quant_config.json"),
        serde_json::to_vec(&crate::families::inkling2::config::tests::release_quant_config()).unwrap()).unwrap();
    (dir, small, tensors)
}

#[test]
fn layout_reuses_one_immutable_catalog_across_candidates() -> anyhow::Result<()> {
    let (dir, _, _) = fixture_with_experts(Fixture::Exl3, 384, 8);
    let input = plan::PlanInput::open(dir.path())?;
    let catalog = input.expert_catalog()? as *const crate::OfficialV41Catalog;
    for entry in std::fs::read_dir(dir.path())? {
        let path = entry?.path();
        if path.extension().is_some_and(|extension| extension == "safetensors") {
            std::fs::remove_file(path)?;
        }
    }
    for ranks in [3, 6, 3] {
        let report = input.plan(&PlanOptions { placement: plan::ExpertPlacement::from_spark_ranks(ranks),
            vision: plan::MediaMode::Off, audio: plan::MediaMode::Off,
            layout: Some(plan::layout::LayoutOptions::default()), ..Default::default() })?;
        assert!(report.memory_layout.as_ref().is_some_and(|layout| !layout.devices.is_empty()), "{:?}", report.hints);
        assert!(!report.hints.iter().any(|hint| hint.what.contains("catalog refused")), "{:?}", report.hints);
        assert_eq!(catalog, input.expert_catalog()? as *const crate::OfficialV41Catalog);
    }
    Ok(())
}

#[test]
fn inkling_mxfp8_inventory_is_planned_not_executable() {
    let (dir, _, _) = fixture(Fixture::Mxfp8);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert_eq!(report.family.as_deref(), Some("inkling2"));
    assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    assert!(!report.executable());
    let spec = report.spec.unwrap();
    assert_eq!(spec.moe_layers(), 4);
    assert!(matches!(spec.layers[0].attention, AttentionKind::SlidingGqa { window: 1024, sinks: true, .. }));
    assert!(matches!(spec.layers[3].attention, AttentionKind::Gqa { heads: 48, kv_heads: 8, head_dim: 128 }));
    assert!(spec.notes.iter().any(|note| note.contains("MXFP8 W8A8 layers 0..=3")), "{:?}", spec.notes);
    let experts = report.experts.expect("routed expert contract");
    assert_eq!(experts.package, "inkling2:mxfp8");
    assert!(experts.spark_worlds.contains(&3) && experts.spark_worlds.contains(&6),
        "registered native width-class recipes cover TP3/TP6: {:?}", experts.spark_worlds);
}

#[test]
fn mixed_four_rank_assignment_uses_each_layers_native_unit() -> anyhow::Result<()> {
    use crate::placement::{SparkAssignment, SparkResidentLayout, SparkWorkingSet};
    use crate::formats::fp8_experts::Slicing;
    let (dir, _, _) = fixture_with_experts(Fixture::Mixed, 384, 8);
    let catalog = crate::read_expert_catalog(dir.path())?;
    let working = SparkWorkingSet { spark_layers: (0..4).collect(), spark_ranks: 4,
        prefill_lanes: 1, ..Default::default() };
    for unit in [16, 32, 64] {
        for rank in 0..4 {
            let assignment = SparkAssignment::resident_from_catalog(&catalog, &working, rank,
                SparkResidentLayout::Fp8 { world: 4, rank, slicing: Slicing::Blocks(unit) })?;
            assert_eq!(assignment.slices.len(), 4);
            for slice in assignment.slices {
                assert_eq!((slice.first, slice.width, slice.stored_width), ((rank * 576) as u32, 576, 576));
            }
        }
    }
    Ok(())
}

#[test]
fn mixed_native_checkpoint_reads_each_layers_storage_from_its_headers() {
    let (dir, _, _) = fixture(Fixture::Mixed);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    assert_eq!((0..4).map(|layer| expert_storage(&checkpoint, layer).unwrap()).collect::<Vec<_>>(),
        [ExpertStorage::Nvfp4, ExpertStorage::Nvfp4, ExpertStorage::Nvfp4, ExpertStorage::Mxfp8]);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    let spec = report.spec.unwrap();
    assert!(spec.notes.iter().any(|note| note.contains("native NVFP4 W4A4 (per-token amax, fouroversix) layers 0..=2")
        && note.contains("MXFP8 W8A8 layers 3..=3")), "{:?}", spec.notes);
    // Two packages, one contract: the plan names both and keeps the common worlds.
    let contract = report.experts.unwrap();
    assert_eq!(contract.package, "inkling2:mxfp8 + inkling2:nvfp4a4");
    assert!(contract.spark_worlds.contains(&6), "compatible K32/K16 ownership must remain admitted");
    assert!(!contract.spark_worlds.contains(&5), "unequal mixed row ranges must remain refused");
    let cfg = Inkling2Config::from_hf(&checkpoint.config).unwrap();
    let experts = crate::families::inkling2::experts::layer_experts(&checkpoint, &cfg).unwrap();
    // NVFP4 holds half the codes plus twice the scales of MXFP8; both halves cover the whole.
    assert!(experts[0].whole < experts[3].whole, "{:?} vs {:?}", experts[0], experts[3]);
    for layer in &experts {
        assert!(layer.half[0] + layer.half[1] >= layer.whole - 2 * 4 * 2, "{layer:?}");
        assert!(layer.half[0] >= layer.half[1], "{layer:?}");
    }
    assert_eq!(experts[3].whole, 4 * (2 * 2304 * 4608 + 4608 * 2304) + 4 * (2 * 2304 * 144 + 4608 * 72));
    let catalog = crate::read_expert_catalog(dir.path()).unwrap();
    let tensors = catalog.fp8().unwrap();
    let groups = tensors.homogeneous_layers(&[0, 1, 2, 3]).unwrap();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].1, [0, 1, 2]);
    assert!(groups[0].0.dynamic_nvfp4());
    assert_eq!(groups[1].1, [3]);
    assert_eq!(groups[1].0.format(), crate::formats::fp8_experts::ExpertFormat::Mxfp8);
    use crate::formats::fp8_experts::Slicing;
    for (layer, expert) in experts.iter().enumerate() {
        let tensors = tensors.for_layer_metadata(layer).unwrap().for_coordinator();
        for world in [1, 2] {
            for rank in 0..world {
                let regions = crate::serving_capacity::fp8_expert_regions(&tensors, &[layer], world,
                    rank, Slicing::Blocks(128)).unwrap();
                let arena = crate::serving_capacity::WeightArenaLayout::new("native-fixture", regions).unwrap();
                assert_eq!(arena.bytes() as u64, if world == 1 { expert.whole } else { expert.half[rank] });
            }
        }
    }
}

#[test]
fn exl3_checkpoint_accepts_per_expert_k4_modules_split_by_row_parity() {
    let (dir, _, _) = fixture(Fixture::Exl3);
    let report = plan::plan(dir.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    assert!(report.unclassified.is_empty(), "{:?}", report.unclassified);
    assert!(report.components.iter().all(|c| c.rejected == 0), "{:?}", report.components);
    assert_eq!(report.experts.unwrap().package, "inkling2:exl3-k45");
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    assert_eq!(expert_storage(&checkpoint, 0).unwrap(), ExpertStorage::Exl3 { bits: 4 });
    let catalog = crate::read_expert_catalog(dir.path()).unwrap();
    assert_eq!(catalog.exl3().unwrap().decoder_tiers(), &[4, 5]);
}

#[test]
fn installed_exl3_onboarding_keeps_descriptor_arenas_and_each_workspace_request() -> anyhow::Result<()> {
    use crate::families::inkling2::admission::local_package;
    use crate::placement::families::inkling2::LocalPackage;
    let (dir, _, _) = fixture_with_experts(Fixture::Exl3, 384, 8);
    let catalog = crate::read_expert_catalog(dir.path())?;
    let install = tempfile::tempdir()?;
    let share = install.path().join("share");
    std::fs::create_dir_all(&share)?;
    let manifest = share.join("PROGRAMS.json");
    for tp in [1, 2] {
        let root = install.path().join(format!("lib/exl3/exl3-inkling2-k45/rtx-tp{tp}/m1"));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("v41_exl3.json"), serde_json::to_vec(&serde_json::json!({
            "capacity": 1, "hidden": 4608, "intermediate":2304/tp, "experts":384,
            "input_format":"e4m3_k32", "trellis_lut":{"bytes":1234},
            "buffers":{"scratch":{"bytes":2345678,"allocation":"scratch","dtype":"f32","zero_on_create":false},
                "state":{"bytes":64,"allocation":"state","dtype":"i32","zero_on_create":true}}
        }))?)?;
        let package = local_package(&catalog, Some(&manifest), tp, 1);
        let LocalPackage::Installed { backings } = package
            else { anyhow::bail!("EXL3 installed requests missing: {package:?}") };
        for (count, groups) in backings.iter().enumerate().skip(1) {
            assert_eq!(groups.len(), tp);
            for backing in groups {
                let runtime_regions = (0..count).map(|layer| {
                    let resident = catalog.exl3().unwrap().residency(crate::V41Exl3Layer::Backbone(layer), tp, backing.gpu)?;
                    let name = if tp == 1 { format!("BackboneFull {{ layer: {layer} }}.0") }
                        else { format!("BackboneTp2 {{ layer: {layer}, rank: {} }}.0", backing.gpu) };
                    Ok((name, resident.device_arena_layout()?.1))
                }).collect::<anyhow::Result<Vec<_>>>()?;
                let id = crate::serving_capacity::expert_weight_arena_id("exl3", backing.gpu as i32, &runtime_regions)?;
                assert_eq!(backing.arena, crate::serving_capacity::WeightArenaLayout::new(id, runtime_regions)?);
                let expected = crate::placement::rtx_onboard::expert_workspace_requests(&catalog, Some(&manifest), 1, tp, "inkling2")?;
                assert_eq!(backing.workspace.iter().map(|(_, bytes)| *bytes).collect::<Vec<_>>(), expected);
                assert_eq!(backing.workspace.last().unwrap(), &("output".into(), 4608*4));
            }
        }
    }
    Ok(())
}

#[test]
fn installed_local_native_owners_use_shared_workspace_requests_once() -> anyhow::Result<()> {
    use crate::families::inkling2::admission::local_package;
    use crate::placement::families::inkling2::LocalPackage;
    let (dir, _, _) = fixture(Fixture::Mixed);
    let catalog = crate::read_expert_catalog(dir.path())?;
    let install = tempfile::tempdir()?;
    let share = install.path().join("share");
    std::fs::create_dir_all(&share)?;
    let manifest = share.join("PROGRAMS.json");
    let rows = 80u64;
    assert!(matches!(local_package(&catalog, Some(&manifest), 2, rows), LocalPackage::Missing(_)));
    for (name, weights, input, scratch) in [("fp8-inkling2-nvfp4a4", "nvfp4", "bf16", 12345u64),
        ("fp8-inkling2-mxfp8", "mxfp8", "wire", 23456)] {
        let root = install.path().join("lib/fp8").join(name);
        std::fs::create_dir_all(root.join("tp2"))?;
        std::fs::write(root.join("manifest.json"), serde_json::to_vec(&serde_json::json!({
            "layouts":{"tp2":{"abi_version":6,"weights":weights,"input":input,
                "activation":"situ","projection":"bf16","swiglu_alpha":1.0,"swiglu_beta":0.0,
                "swiglu_limit":0.0,"capacities":[{"capacity":80,"scratch_bytes":scratch}]}}}))?)?;
    }
    let LocalPackage::Installed { backings } = local_package(&catalog, Some(&manifest), 2, rows)
        else { anyhow::bail!("complete native package owners were not admitted") };
    let output = rows * catalog.routed_experts().hidden as u64 * 2;
    assert_eq!(backings.len(), 5);
    assert!(backings[0].is_empty());
    assert_eq!(backings[1].len(), 2, "one homogeneous group per physical rank");
    assert_eq!(backings[4].len(), 4, "mixed formats retain two independently retired backends per rank");
    let host = crate::families::inkling2::weight_layout::push_host_envelope(&catalog, &backings[4], 2)?;
    assert_eq!(host.dense_bytes, [0, 0], "raw dense delivery and device transpose need no staged host transform");
    assert_eq!(host.retained_bytes, [0, 0]);
    assert_eq!(host.slots, cuteafd_ffi::push_ring::DEFAULT_SLOTS as u64);
    assert_eq!(host.slot_bytes, cuteafd_ffi::push_ring::DEFAULT_SLOT_BYTES as u64);
    assert!(host.expert_bytes.iter().all(|&bytes| bytes > 0), "native dynamic scalar validation is explicit");
    let empty_host = crate::families::inkling2::weight_layout::push_host_envelope(&catalog, &backings[0], 2)?;
    assert_eq!(empty_host.expert_bytes, [0, 0]);
    for gpu in 0..2 {
        let workspace = backings[4].iter().filter(|backing| backing.gpu == gpu)
            .flat_map(|backing| &backing.workspace).map(|(_, bytes)| bytes).sum::<u64>();
        assert_eq!(workspace, 12345 + 23456 + 2*output);
        for backing in backings[4].iter().filter(|backing| backing.gpu == gpu) {
            let tensors = catalog.fp8().unwrap().for_coordinator().for_layer_metadata(backing.layers[0])?;
            let regions = crate::serving_capacity::fp8_expert_regions(&tensors, &backing.layers, 2, gpu,
                crate::formats::fp8_experts::Slicing::Blocks(128))?;
            let id = crate::serving_capacity::expert_weight_arena_id("fp8", gpu as i32, &regions)?;
            let runtime = crate::serving_capacity::WeightArenaLayout::new(id, regions)?;
            assert_eq!(backing.arena, runtime, "planner backing must be the shared runtime arena");
            assert_eq!(backing.workspace[1], ("output".into(), output));
        }
    }
    // Each native group is independently retained; no TP2 output from the
    // shared request list is charged again beside each backend's output.
    assert!(matches!(local_package(&catalog, Some(&manifest), 2, 81), LocalPackage::Missing(_)));
    Ok(())
}

#[test]
fn spark_charges_match_shared_group_arenas_and_typed_owner_census() -> anyhow::Result<()> {
    use crate::families::inkling2::{admission::spark_charges, experts::layer_experts};
    use crate::placement::{SparkAssignment, SparkResidentLayout, SparkWorkingSet};
    use crate::placement::families::inkling2::{self as shared, SparkCharge};
    use crate::placement::planner::PrefillShape;
    for kind in [Fixture::Mxfp8, Fixture::Mixed] {
        let (dir, config, _) = fixture(kind);
        let checkpoint = Checkpoint::open(dir.path())?;
        let cfg = Inkling2Config::from_hf(&config)?;
        let catalog = crate::read_expert_catalog(dir.path())?;
        let layers = (0..cfg.layers).collect::<Vec<_>>();
        let experts = layer_experts(&checkpoint, &cfg)?;
        for world in [3, 6] {
            let shape = PrefillShape { rows: 256, waves: 3 };
            let ingress = if kind == Fixture::Mxfp8 { cuteafd_core::SparkIngress::Fp8K32 }
                else if kind == Fixture::Mixed { cuteafd_core::SparkIngress::Native }
                else { cuteafd_core::SparkIngress::Bf16 };
            let charges = spark_charges(&catalog, &experts, &layers, world, shape, ingress)?;
            assert_eq!(charges.len(), layers.len()+1);
            assert!(charges.last().unwrap().iter().all(|charge| *charge == SparkCharge { experts:0,owners:0,runtime:0 }));
            let executions = shared::spark_executions(shape, kind == Fixture::Exl3)?;
            for rank in 0..world {
                let working = SparkWorkingSet { rtx_layers:vec![], spark_layers:layers.clone(),
                    spark_ranks:world,prefill_lanes:shared::LANES };
                let expected = if let Some(tensors) = catalog.fp8() {
                    let regions = tensors.homogeneous_layers(&layers)?.into_iter().map(|(view,layers)| {
                        let format = if view.dynamic_nvfp4() {"nvfp4a4"} else {"mxfp8"};
                        let recipe = crate::plan::experts::spark_recipe(&format!("inkling2:{format}"),cfg.intermediate,world)?;
                        crate::serving_capacity::fp8_expert_regions(&view,&layers,world,rank,recipe.slicing)
                    }).collect::<anyhow::Result<Vec<_>>>()?.into_iter().flatten();
                    crate::serving_capacity::WeightArenaLayout::new("worker-fixture",regions)?.bytes() as u64
                } else {
                    SparkAssignment::resident_from_catalog(&catalog,&working,rank,SparkResidentLayout::Exl3{world,rank})?.resident_bytes
                };
                assert_eq!(charges[0][rank].experts,expected);
                let owners = executions.workspace_bytes(|extent| crate::plan::layout::spark_slots::catalog_owner_bytes(
                    &catalog,"inkling2",&working,rank,extent))?.unwrap().total().unwrap();
                let rings = crate::plan::layout::spark_ring_geometry(shared::LANES as usize);
                let rings = cuteafd_core::SparkRings { max_frame_bytes:rings.max_frame_bytes,depth:rings.depth,
                    slot_bytes:rings.slot_bytes,alignment:rings.alignment };
                let endpoint = cuteafd_core::SparkEndpoint { ring_depth:None,capacity:shape.rows as u32,
                    ingress,partial_reply_bytes:None };
                let io = working.rank_endpoints_io(rank,cuteafd_core::ExpertGeometry::INKLING2,&[endpoint],rings)?;
                assert_eq!(charges[0][rank].owners, owners+io.intake+io.host_exchange+io.row_indices
                    +io.mapped_rings+io.endpoint_staging);
            }
        }
    }
    Ok(())
}

#[test]
fn installed_native_measured_five_sparks_fit_with_exact_ledgers() -> anyhow::Result<()> {
    use std::path::PathBuf;
    use crate::plan::{ExpertPlacement, MediaMode, layout::LayoutOptions};
    let home = PathBuf::from(std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME")
        .expect("CUTEAFD_DEPLOYMENT_TEST_HF_HOME is required"));
    let snapshot = crate::model_cache_dir(&home, "thinkingmachines/Inkling-2-Small-MXFP8")
        .join("snapshots/3febc62259ee92ea36e2fb6e8303371c4eb3922c");
    let manifest = PathBuf::from(std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS")
        .expect("CUTEAFD_DEPLOYMENT_TEST_PROGRAMS is required"));
    anyhow::ensure!(snapshot.is_dir() && manifest.is_file(), "installed Inkling inputs missing");
    let input = testing::BudgetedPlanInput::open(&snapshot)?;
    let options = PlanOptions { placement: ExpertPlacement::from_spark_ranks(6),
        vision: MediaMode::Off, audio: MediaMode::Off,
        layout: Some(LayoutOptions { workspace_manifest: Some(manifest), inkling_kv: crate::families::inkling2::InklingKvCache::Bf16, ..Default::default() }),
        ..Default::default() };
    let report = input.plan(&options)?;
    assert_eq!(report.runtime, Some(RuntimeStatus::Planned));
    assert!(report.fits && report.placement_supported && report.native_serving_admitted, "{:?}", report.hints);
    assert!(report.wip_launchable() && !report.executable());
    assert_eq!(report.memory_layout.as_ref().unwrap().pool_tokens, 2_097_152);
    let mut fp8_options = options.clone();
    fp8_options.layout.as_mut().unwrap().inkling_kv = Default::default();
    assert_eq!(fp8_options.layout.as_ref().unwrap().inkling_kv,
        crate::families::inkling2::InklingKvCache::Fp8);
    let fp8 = input.plan(&fp8_options)?;
    assert!(fp8.fits && fp8.placement_supported && fp8.native_serving_admitted, "{:?}", fp8.hints);
    assert_eq!(fp8.memory_layout.as_ref().unwrap().pool_tokens, 2_097_152);
    println!("Inkling native 1RTX+6S pool2097152: BF16->FP8 onboard {:?}->{:?}, global KV 64GiB->33GiB",
        report.onboard_layers, fp8.onboard_layers);
    assert!(fp8.onboard_layers > report.onboard_layers, "freed global KV must admit more native expert layers");
    let five = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(5), ..fp8_options.clone() })?;
    assert!(five.fits && five.placement_supported && five.native_serving_admitted, "{:?}", five.hints);
    for device in five.memory_layout.as_ref().unwrap().devices.iter()
        .filter(|device| device.kind == cuteafd_core::memory_layout::DeviceKind::Spark) {
        assert_eq!(device.capacity_bytes, crate::serving_capacity::spark_memory::serving_bytes()[device.index as usize]);
        assert!(device.used_bytes() <= device.capacity_bytes);
        println!("Inkling native default FP8 five-Spark rank {}: used={} capacity={} headroom={}",
            device.index, device.used_bytes(), device.capacity_bytes, device.capacity_bytes - device.used_bytes());
    }
    let refused = input.plan(&PlanOptions { placement: ExpertPlacement::from_spark_ranks(4), ..fp8_options })?;
    println!("Inkling native default FP8 four-Spark fit={} supported={}: {:?}",
        refused.fits, refused.placement_supported, refused.hints);
    assert!(!refused.fits && !refused.placement_supported);
    assert!(refused.hints.iter().any(|hint| hint.what.contains("joint admission refused")), "{:?}", refused.hints);
    Ok(())
}

#[test]
fn serving_header_inventory_preserves_remote_native_expert_scales() -> anyhow::Result<()> {
    use crate::families::inkling2::experts::layer_experts;
    let (directory, config, _) = fixture(Fixture::Mxfp8);
    let view = crate::plan::checkpoint::CheckpointView::select(directory.path(),
        crate::plan::checkpoint::CoreView::Original)?;
    let cfg = Inkling2Config::from_hf(&config)?;
    let coordinator = Checkpoint::coordinator_view(&view, false, false)?;
    assert!(layer_experts(&coordinator, &cfg).is_err());
    let inventory = Checkpoint::open_view(&view)?;
    let experts = layer_experts(&inventory, &cfg)?;
    assert_eq!(experts.len(), cfg.layers);
    assert!(experts.iter().all(|layer| layer.source > 0 && layer.whole > 0));
    Ok(())
}

#[test]
fn structural_open_names_missing_tensors_and_accepts_reports_drift_per_tensor() {
    let (dir, small, tensors) = fixture(Fixture::Mxfp8);
    // A missing conv tensor is a typed open error naming it.
    let pruned = tensors.iter().filter(|t| !t.0.ends_with("layers.1.attn_sconv.weight")).cloned().collect::<Vec<_>>();
    let dir2 = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir2.path(), &small, &pruned, None);
    let error = match Model::open(&Checkpoint::open(dir2.path()).unwrap()) {
        Ok(_) => panic!("open accepted a checkpoint missing layers.1.attn_sconv.weight"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("layers.1.attn_sconv.weight"), "{error}");
    // A drifted dtype is an `accepts` rejection, not an open failure.
    let drifted = tensors.iter().map(|t| if t.0.ends_with("layers.2.mlp.gate.weight") { (t.0.clone(), "F32", t.2.clone()) } else { t.clone() })
        .collect::<Vec<_>>();
    let dir3 = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir3.path(), &small, &drifted, None);
    let report = plan::plan(dir3.path(), &PlanOptions::default()).unwrap();
    assert!(report.config_error.is_none(), "{:?}", report.config_error);
    let router = report.components.iter().find(|c| c.component == Component::Router).unwrap();
    assert_eq!(router.rejected, 1, "{router:?}");
    assert!(router.rejections[0].tensor.ends_with("layers.2.mlp.gate"), "{:?}", router.rejections);
    // An unknown tensor is a named census rejection.
    let mut extra = tensors.clone();
    extra.push((format!("{TEXT}layers.0.attn.extra.weight"), "BF16", vec![8]));
    let dir4 = tempfile::tempdir().unwrap();
    testing::write_snapshot(dir4.path(), &small, &extra, None);
    let report = plan::plan(dir4.path(), &PlanOptions::default()).unwrap();
    let attention = report.components.iter().find(|c| c.component == Component::Attention).unwrap();
    assert!(attention.rejections.iter().any(|r| r.tensor.contains("attn.extra") && r.reason.contains("outside the Inkling 2 tensor contract")),
        "{:?}", attention.rejections);
    drop(dir);
}

#[test]
fn compact_core_headers_preserve_native_keeps_and_plan_exact_trellis_arenas() -> anyhow::Result<()> {
    use crate::families::inkling2::weight_layout::CoordinatorWeights;
    use crate::placement::{families::inkling2::layer_modes, AttentionPlacement};
    let (_native, config, tensors) = fixture(Fixture::Exl3);
    for bits in [6, 8] {
        let dir = tempfile::tempdir()?;
        let mut compact = Vec::new();
        for (name, dtype, shape) in &tensors {
            let selected = name == "model.llm.unembed.weight"
                || (name.contains(".attn.") && name.ends_with(".weight") && shape.len() == 2
                    && shape.iter().all(|width| width % 128 == 0));
            if selected {
                let outputs = if name == "model.llm.unembed.weight" { shape[0].next_multiple_of(128) } else { shape[0] };
                compact.extend(testing::exl3(name.strip_suffix(".weight").unwrap(), outputs, shape[1], bits));
            } else { compact.push((name.clone(), *dtype, shape.clone())); }
        }
        testing::write_snapshot(dir.path(), &config, &compact, None);
        let checkpoint = Checkpoint::open(dir.path())?;
        let cfg = Inkling2Config::from_hf(&config)?;
        let model = Model::open(&checkpoint)?;
        assert_eq!(model.core.len(), 4 * 6 + 1);
        let report = plan::plan(dir.path(), &PlanOptions::default())?;
        assert!(report.config_error.is_none(), "{:?}", report.config_error);
        assert!(report.components.iter().all(|component| component.rejected == 0), "{:?}", report.components);
        for (ranks, attention) in [(1, AttentionPlacement::Heads), (2, AttentionPlacement::Heads),
            (2, AttentionPlacement::Layers)] {
            let modes = layer_modes(&cfg, ranks, attention);
            let layouts = CoordinatorWeights::plan(&checkpoint, &cfg, ranks, &modes)?;
            for layout in layouts {
                assert!(!layout.core.is_empty());
                for (stem, core) in &layout.core {
                    assert_eq!(core.bits, bits);
                    assert!(!checkpoint.contains_tensor(&format!("{stem}.weight")));
                    let prefix = if core.ranks == 2 { format!("{stem}#rank{}", core.rank) } else { stem.clone() };
                    for (name, bytes) in core.arena_tensors(&prefix)? {
                        assert_eq!(layout.arena.view(&name).unwrap().bytes(), bytes);
                    }
                }
                layout.mount_load(&checkpoint)?;
                assert!(layout.arena.views().keys().all(|name| !name.ends_with(".mcg")));
            }
        }
    }
    Ok(())
}

#[test]
fn shared_recipe_charges_every_owner_once_on_one_and_two_rtx() {
    use crate::families::inkling2::{experts::layer_experts, weight_layout::CoordinatorWeights};
    use crate::placement::families::inkling2::{self as shared, InklingInputs, LocalPackage, SparkCharge};
    use crate::placement::planner::{Context, Location, PrefillShape, ComponentKind, Recipe, serving};
    let (dir, small, _) = fixture(Fixture::Mxfp8);
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = Inkling2Config::from_hf(&small).unwrap();
    let routed = (0..cfg.layers).collect::<Vec<_>>();
    use crate::placement::{AttentionPlacement, LayerMode};
    for (gpus, placement) in [(1usize, AttentionPlacement::Heads), (2, AttentionPlacement::Heads),
        (2, AttentionPlacement::Context), (2, AttentionPlacement::Layers)] {
        let modes = shared::layer_modes(&cfg, gpus, placement);
        let owners = modes.iter().map(|mode| match mode { LayerMode::Whole { gpu, .. } => Some(*gpu), _ => None }).collect::<Vec<_>>();
        let targets = CoordinatorWeights::plan(&checkpoint, &cfg, gpus, &modes).unwrap();
        let experts = layer_experts(&checkpoint, &cfg).unwrap();
        let ranks = 2;
        let prefill = PrefillShape { rows: 1024, waves: 3 };
        let charge = |count: usize| (0..ranks).map(|_| if count == routed.len() {
            SparkCharge { experts: 0, owners: 0, runtime: 0 } } else {
            SparkCharge { experts: ((routed.len() - count) as u64) << 30, owners: 1 << 30, runtime: 1 << 30 } }).collect();
        let budgets = (0..gpus).map(|gpu| shared::planned_budget(gpu as u8, 96 << 30, 96 << 30, 1 << 30))
            .chain((0..ranks).map(|rank| shared::spark_budget(rank as u8, 100 << 30)))
            .chain([crate::placement::families::kimi::host_budget(512 << 30, 480 << 30)]).collect();
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let local = LocalPackage::Installed { backings: (0..=routed.len()).map(|count| {
            if count == 0 { return vec![]; }
            (0..gpus).map(|gpu| {
                let layers = routed[..count].to_vec();
                let regions = crate::serving_capacity::fp8_expert_regions(&catalog.fp8().unwrap().for_coordinator(),
                    &layers, gpus, gpu, crate::formats::fp8_experts::Slicing::Blocks(128)).unwrap();
                let id = crate::serving_capacity::expert_weight_arena_id("fp8", gpu as i32, &regions).unwrap();
                shared::LocalBacking { gpu, layers, arena: crate::serving_capacity::WeightArenaLayout::new(id, regions).unwrap(),
                    workspace: vec![("scratch".into(), (64 << 20) - 1024*4608*2), ("output".into(), 1024*4608*2)] }
            }).collect()
        }).collect() };
        let inputs = InklingInputs { cfg: cfg.clone(), kv_format: crate::families::inkling2::InklingKvCache::Bf16, gpus, spark_ranks: ranks, context: Context::automatic(1 << 20),
            pool: crate::placement::prepare_pool(crate::placement::PoolFacts::new(
            &vec![96 << 30; gpus], Context::automatic(1 << 20), None, shared::PAGE_ROWS)).unwrap(), prefill, sequences: 16, prefix_entries: 8, prefix_mark_budget: 2 << 30, targets: targets.clone(), placement, modes: modes.clone(),
            routed: routed.clone(), experts,
            local,
            spark: (0..=routed.len()).map(charge).collect(),
            links: shared::spark_link((gpus - 1) as u8, ranks, prefill, Default::default(), cuteafd_core::SparkIngress::Fp8K32).unwrap(),
            spark_reply_formats: Default::default(), drafter: None, vision: None, budgets,
            code: vec![64 << 20; gpus], graph_reserve: vec![500 << 20; gpus], head: serving::BlasHead::Gemm,
            host_prefix: serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }, device_sampling: true,
            expert_usage: None, resident_weight_limit: None,
            mount: targets.iter().map(|t| t.mount_load(&checkpoint).unwrap()).collect(),
            push_host: vec![None; routed.len()+1] };
        let request = shared::request(inputs.context, None, crate::placement::Onboard::Auto, prefill,
            crate::placement::planner::Operation::Serving, None, Default::default());
        let contract = inputs.prepare(&request).unwrap();
        assert_eq!(contract.components.iter().filter(|component| matches!(&component.kind,
            ComponentKind::Allocation { template, .. } if template.group == "residual hop control")).count(),
            if gpus == 2 && placement == AttentionPlacement::Layers { 2 } else { 0 });

        // Every allocation id is unique within one activation class.
        let mut seen = std::collections::BTreeSet::new();
        for component in &contract.components {
            if let ComponentKind::Allocation { template, .. } = &component.kind {
                assert!(seen.insert((template.id.clone(), format!("{:?}", component.activation))), "{} twice", template.id);
            }
        }
        // Each prefix uses shared runtime weight arenas plus independent
        // scratch/output requests. No aggregate expert CUDA allocation exists.
        assert!(!contract.components.iter().any(|component| matches!(&component.kind,
            ComponentKind::Allocation { template, .. } if template.id.starts_with("inkling2.experts."))));
        let LocalPackage::Installed { backings } = &inputs.local else { unreachable!() };
        for (count, groups) in backings.iter().enumerate().skip(1).take(routed.len()-1) {
            for (group, backing) in groups.iter().enumerate() {
                let arena = contract.components.iter().find(|component| matches!(&component.kind,
                    ComponentKind::Allocation { template, .. } if template.id == backing.arena.id())
                    && component.activation == crate::placement::planner::Activation::choice(cfg.layers + 2, count)).unwrap();
                assert_eq!(arena.bytes.at(1).unwrap(), backing.arena.bytes() as u64);
                for (name, bytes) in &backing.workspace {
                    let id = format!("inkling2.expert_workspace.{}.{}.{}.{}", backing.gpu, count, group, name);
                    let component = contract.components.iter().find(|component| matches!(&component.kind,
                        ComponentKind::Allocation { template, .. } if template.id == id)).unwrap();
                    assert_eq!(component.bytes.at(1).unwrap(), *bytes, "{id}");
                }
            }
        }
        let all_local = shared::request(inputs.context, None, crate::placement::Onboard::Layers(routed.len()), prefill,
            crate::placement::planner::Operation::Serving, None, Default::default());
        assert!(inputs.admit(all_local).is_err(), "a linked-world recipe must refuse the all-local prefix");
        assert!(contract.requires.iter().any(|required|
            required.when == crate::placement::planner::Choice { variable: cfg.layers + 2, option: routed.len() }
                && required.any_of.is_empty()), "the full local prefix is explicitly impossible");
        // The planner inventories the exact allocator recipe, per owner, not
        // a BF16 approximation of the native FP32 forward planes.
        for gpu in 0..gpus {
            for (prefix, rows) in std::iter::once((format!("inkling2.decode.{gpu}"), 64usize))
                .chain((0..prefill.waves).map(|slot| (format!("inkling2.unit.{gpu}.{slot}"), prefill.rows as usize))) {
                let expected = crate::families::inkling2::execution::UnitLayout::serving_format(&cfg, rows, 1 << 20, inputs.kv_format, inputs.sequences as usize).unwrap();
                let actual = contract.components.iter().filter_map(|component| match &component.kind {
                    ComponentKind::Allocation { template, .. } if template.location == Location::coordinator(gpu as u8) => template.id.strip_prefix(&format!("{prefix}."))
                        .filter(|name| *name != "uploads")
                        .map(|name| (name.to_owned(), component.bytes.at(1).unwrap() as usize)),
                    _ => None,
                }).collect::<std::collections::BTreeMap<_, _>>();
                assert_eq!(actual, expected.buffers, "{prefix}");
                let upload = contract.components.iter().find(|component| matches!(&component.kind,
                    ComponentKind::Allocation { template, .. } if template.id == format!("{prefix}.uploads"))).unwrap();
                assert_eq!(upload.bytes.at(1).unwrap(), expected.upload_shape(&cfg, 16.min(rows)).unwrap().capacity_bytes().unwrap() as u64);
            }
        }
        if gpus == 1 {
            let cache = crate::families::inkling2::execution::CacheLayout::new(&cfg, 16, 3).unwrap();
            let owner = |id: &str| contract.components.iter().find(|component| matches!(&component.kind,
                ComponentKind::Allocation { template, .. } if template.id == id)).unwrap();
            assert_eq!(owner("inkling2.cache.0.active").bytes.at(1).unwrap(), cache.active_bytes().unwrap() as u64);
            let slots = cuteafd_core::prefix::mark_slots_with_host(16, 8, cache.sequence_bytes as u64, 2 << 30, false);
            assert_eq!(owner("inkling2.cache.0.marks").bytes.at(1).unwrap(), cache.retained_bytes(slots as usize).unwrap() as u64);
            for &layer in &cache.global_layers {
                assert_eq!(owner(&format!("inkling2.kv.{layer}.0.records")).bytes.at(3).unwrap(), cache.layer_page_bytes().unwrap() as u64);
            }
        }
        // One target arena backing per rank.
        for gpu in 0..gpus {
            let arenas = contract.components.iter().filter(|c| matches!(&c.kind, ComponentKind::Allocation { template, .. }
                if template.group == "weight arena" && template.id == targets[gpu].arena.id()
                    && template.location == Location::coordinator(gpu as u8))).count();
            assert_eq!(arenas, 1, "rtx{gpu}");
            let execution = targets[gpu].execution_arena().unwrap();
            for &layer in &targets[gpu].layers {
                assert_eq!(execution.view(&format!("{TEXT}layers.{layer}.attn.rel_logits_proj.padded")).unwrap().bytes(),
                    1024*64*2);
            }
            let backing = contract.components.iter().find(|component| matches!(&component.kind,
                ComponentKind::Allocation { template, .. } if template.group == "weight arena"
                    && template.id == targets[gpu].arena.id()
                    && template.location == Location::coordinator(gpu as u8))).unwrap();
            assert_eq!(backing.bytes.at(1).unwrap(), execution.bytes() as u64);
        }
        // KV pages exist for the global layer only: on its owner under `layers`, on both ranks under a split.
        for (layer, owner) in owners.iter().enumerate() {
            let homes = contract.components.iter().filter_map(|c| match &c.kind {
                ComponentKind::Allocation { template, .. } if template.id.starts_with(&format!("inkling2.kv.{layer}.")) =>
                    Some(template.location), _ => None }).collect::<std::collections::BTreeSet<_>>();
            if cfg.local[layer] { assert!(homes.is_empty(), "layer {layer} is sliding-window"); }
            else if let Some(owner) = owner { assert_eq!(homes, [Location::coordinator(*owner)].into(), "layer {layer}"); }
            else { assert_eq!(homes.len(), 2, "layer {layer} split over the pair: {homes:?}"); }
        }
        if placement == AttentionPlacement::Context && gpus == 2 {
            assert_eq!(contract.page_rows, 2 * shared::PAGE_ROWS, "context parity doubles the page");
            assert!(contract.components.iter().any(|c| matches!(&c.kind, ComponentKind::Allocation { template, .. }
                if template.id.starts_with("inkling2.context.0."))), "context exchange owners");
        }
        // The per-sequence ring/lag marks exist on every rank that owns a layer.
        for gpu in 0..gpus {
            assert!(contract.components.iter().any(|c| matches!(&c.kind, ComponentKind::Allocation { template, .. }
                if template.id == format!("inkling2.cache.{gpu}.active"))), "rtx{gpu} marks");
        }
        let small = crate::placement::prepare_pool(crate::placement::PoolFacts::new(
            &vec![32 << 30; gpus], inputs.context, None, inputs.page_rows())).unwrap();
        assert_eq!(small.target(), 1 << 20);
        let admitted = inputs.admit(request.clone()).unwrap();
        assert_eq!(admitted.solved.pool_policy, Some(inputs.pool));
        let mut sampled = inputs.clone();
        for budget in sampled.budgets.iter_mut().take(gpus) {
            let crate::placement::planner::Baseline::Forecast { occupied, .. } = budget.baseline else { panic!("forecast baseline") };
            budget.baseline = crate::placement::planner::Baseline::Sampled { free: budget.physical_bytes - occupied,
                phase: crate::placement::planner::Phase::Load, source: "same preweight sample".into() };
        }
        assert_eq!(admitted.solved.pool_policy, sampled.admit(request.clone()).unwrap().solved.pool_policy);
        use cuteafd_core::memory_policy::MemoryPolicy;
        let grow = MemoryPolicy { kv_max: Some(100_000_000), ..Default::default() };
        let grown = grow.scoped(|| {
            let mut growing = inputs.clone();
            growing.pool = crate::placement::prepare_pool(crate::placement::PoolFacts::new(
                &vec![96 << 30; gpus], inputs.context, None, inputs.page_rows())).unwrap();
            growing.admit(request.clone()).unwrap()
        });
        assert_eq!(grown.onboard, admitted.onboard, "growth must preserve Inkling onboarding");
        assert!(grown.solved.pool_tokens > admitted.solved.pool_tokens,
            "{placement:?} on {gpus} GPUs: {} -> {}", admitted.solved.pool_tokens, grown.solved.pool_tokens);
        let explicit = grow.scoped(|| {
            let mut pinned = inputs.clone();
            pinned.pool = crate::placement::prepare_pool(crate::placement::PoolFacts::new(
                &vec![96 << 30; gpus], inputs.context, Some(admitted.solved.pool_tokens), inputs.page_rows())).unwrap();
            let mut exact = request.clone();
            exact.pool = crate::placement::planner::Pin::Exact(admitted.solved.pool_tokens);
            pinned.admit(exact).unwrap()
        });
        assert_eq!(explicit.solved.pool_tokens, admitted.solved.pool_tokens);
        let mut fp8_inputs = inputs.clone();
        fp8_inputs.kv_format = crate::families::inkling2::InklingKvCache::Fp8;
        let fp8_contract = fp8_inputs.prepare(&request).unwrap();
        for component in &contract.components {
            let ComponentKind::Allocation { template, .. } = &component.kind else { continue };
            if !template.id.starts_with("inkling2.kv.") { continue; }
            let fp8 = fp8_contract.components.iter().find(|candidate| matches!(&candidate.kind,
                ComponentKind::Allocation { template: other, .. } if other.id == template.id)).unwrap();
            let bf16_bytes = component.bytes.at(2048).unwrap();
            let fp8_bytes = fp8.bytes.at(2048).unwrap();
            // Admission includes exactly one measured allocation tail, so
            // compare each record-format payload through the allocator recipe.
            let mode = if placement == AttentionPlacement::Context && gpus == 2 { 2 } else { 1 };
            let heads = if gpus == 2 && placement != AttentionPlacement::Layers { cfg.kv_heads/2 } else { cfg.kv_heads };
            let tokens = 2048*64*mode;
            assert_eq!(bf16_bytes, crate::placement::planner::sm120_request_backing(
                (tokens*4*heads*cfg.head_dim) as u64).unwrap());
            assert_eq!(fp8_bytes, crate::placement::planner::sm120_request_backing(
                (tokens*(2*heads*cfg.head_dim+8*heads)) as u64).unwrap());
        }
        let fp8_admitted = fp8_inputs.admit(request).unwrap();
        assert!(fp8_admitted.solved.pool_tokens >= admitted.solved.pool_tokens);
        assert!(fp8_admitted.onboard >= admitted.onboard);
        assert!(admitted.solved.pool_tokens >= 1 << 20, "{placement:?} on {gpus} GPUs: {}", admitted.solved.pool_tokens);
        assert!(admitted.placement.layers.iter().all(|l| shared::INKLING2.runs(l.mode)));
        assert_eq!(admitted.placement.attention_placement, if gpus == 2 { placement } else { AttentionPlacement::Heads });
        if gpus == 2 {
            for layer in 0..cfg.layers {
                let stem = format!("{TEXT}layers.{layer}.mlp_norm.weight");
                let homes = targets.iter().filter(|t| t.arena.view(&stem).is_some()).map(|t| t.rank).collect::<Vec<_>>();
                match owners[layer] {
                    Some(owner) => assert_eq!(homes, [usize::from(owner)], "layer {layer}"),
                    None => assert_eq!(homes, [0, 1], "split layer {layer} replicates its norms"),
                }
                if owners[layer].is_none() {
                    // Half the heads' projections per rank, named per rank.
                    let q = format!("{TEXT}layers.{layer}.attn.wq_du.weight");
                    assert!(targets[0].arena.view(&format!("{q}#rank0")).is_some() && targets[1].arena.view(&format!("{q}#rank1")).is_some());
                    assert_eq!(targets[0].arena.view(&format!("{q}#rank0")).unwrap().bytes(), 6144 * 4608 * 2 / 2);
                }
            }
            assert!(admitted.placement.tp2.is_none_or(|t| t.peak_bytes[0] >= t.peak_bytes[1]));
            assert_eq!(admitted.placement.peer_row_bytes > 0, placement == AttentionPlacement::Context);
        }
    }
}

#[test]
fn eager_expert_headers_classify_inventory_and_roles_without_payload_ownership() {
    use crate::plan::checkpoint::{CheckpointView, CoreView};
    for kind in [Fixture::Mxfp8, Fixture::Mixed, Fixture::Exl3] {
        let (dir, _, _) = fixture(kind);
        let view = CheckpointView::select(dir.path(), CoreView::Original).unwrap();
        let inventory = Checkpoint::inventory_view(&view).unwrap();
        assert!(inventory.tensors.is_empty(), "classification evidence must not own payload");
        assert_eq!(inventory.shard_bytes, 0);
        for layer in 0..4 {
            let expected = match kind { Fixture::Mxfp8 => ExpertStorage::Mxfp8,
                Fixture::Mixed if layer < 3 => ExpertStorage::Nvfp4,
                Fixture::Mixed => ExpertStorage::Mxfp8, Fixture::Exl3 => ExpertStorage::Exl3 { bits: 4 } };
            assert_eq!(expert_storage(&inventory, layer).unwrap(), expected);
        }
        for local in [false, true] {
            let role = Checkpoint::coordinator_view(&view, local, false).unwrap();
            assert!(role.tensors.iter().all(|tensor| local || !tensor.meta.name.contains(".mlp.experts.")));
            assert_eq!(expert_storage(&role, 0).unwrap(), expert_storage(&inventory, 0).unwrap());
        }
    }
    for name in ["model.llm.layers.0.mlp.experts.w13_weight", "model.llm.layers.1.mlp.experts.w2_weight",
        "model.llm.layers.2.mlp.experts.0.gate_proj.trellis", "model.llm.layers.2.mlp.experts.0.up_proj.trellis",
        "model.llm.layers.2.mlp.experts.0.down_proj.trellis"] {
        assert!(INKLING2.eager_expert_header(name));
    }
    for name in ["mtp.model.llm.layers.0.mlp.experts.w13_weight", "model.llm.layers.bad.mlp.experts.w13_weight",
        "model.llm.layers.0.mlp.experts.1.gate_proj.trellis", "model.llm.layers.0.mlp.experts.w13_weight_scale"] {
        assert!(!INKLING2.eager_expert_header(name));
    }
}
