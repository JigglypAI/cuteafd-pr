use super::*;
use crate::shared::token_io::{EmbedPlacement, EmbedSource};
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::placement::{
    families::glm5::buffers::TargetBuffers,
    planner::{self, *},
};
use cuteafd_loader::{
    families::glm5::{
        core::{CoreRankLayout, CoreStageLayout},
        native::ScratchLayout,
    },
    formats::trellis_core::ProjectionScratchLayout,
    placement::families::glm5::hooks::HookBuffers,
    plan::exlr8::dense::scratch::LinearScratchLayout,
};

fn native_layout(cfg: &GlmDsaConfig, rows: usize) -> Result<ScratchLayout> {
    ScratchLayout::new(
        cfg,
        cfg.heads / 2,
        cfg.dense_intermediate / 2,
        rows,
        LinearScratchLayout {
            rotated: rows * 128 * 2,
            table: 256,
            block_scales: 256,
            partial: 1024,
            records: (rows > 64).then_some(640),
            acc: rows * 256 * 4,
        },
    )
}

fn core_layout(cfg: &GlmDsaConfig, rows: usize) -> Result<CoreRankLayout> {
    Ok(CoreRankLayout {
        body: vec![],
        stages: Some([
            CoreStageLayout::new(cfg, DECODE_ROWS, 2, false)?,
            CoreStageLayout::new(cfg, rows, 2, false)?,
        ]),
        decode: Some(ProjectionScratchLayout::new(128, 256, DECODE_ROWS)?),
        prefill: Some(ProjectionScratchLayout::new(128, 256, rows)?),
        target_head: None,
        draft_head: None,
        retained_rows: 0,
        receive_suffix: 0,
    })
}

fn plan(
    cfg: &GlmDsaConfig,
    partitioned: bool,
    owners: Option<&[u8]>,
    pages: u64,
    shape: PrefillShape,
    full: bool,
    native: bool,
    core: bool,
) -> Result<ResolvedPlan> {
    let mut contract = Contract::fixed(
        128,
        128,
        2,
        Facts {
            allocations: vec![],
            deferred: vec![],
            sequential: vec![],
            allowances: vec![],
        },
    );
    contract.variables = vec![Variable::Prefill(vec![shape]), Variable::Lanes(vec![1])];
    for rank in 0..2u8 {
        contract.components.extend(
            TargetBuffers {
                device: rank,
                lead: rank == 0,
                page_tokens: 128,
                maximum_pages: 128,
                context: 4096,
                partitioned,
                owned_layers: (0..cfg.layers)
                    .map(|layer| owners.is_none_or(|owners| owners[layer] == rank))
                    .collect(),
                split_heads: owners.is_none(),
                router: false,
                native_head: false,
                decode: Some(Default::default()),
                prefill: Some((shape, Default::default())),
                full_prefill_logits: full,
                payload_row_bytes: 0,
                core_rows: false,
            }
            .components(cfg)?,
        );
        contract.components.extend(
            HookBuffers {
                device: rank,
                maximum_pages: 128,
                native: native
                    .then(|| {
                        Ok::<_, anyhow::Error>([
                            native_layout(cfg, DECODE_ROWS)?,
                            native_layout(cfg, shape.rows as usize)?,
                        ])
                    })
                    .transpose()?,
                core: core
                    .then(|| core_layout(cfg, shape.rows as usize))
                    .transpose()?,
                shared_workspace: None,
            }
            .components()?,
        );
    }
    let facts = contract.facts(&[0, 0], pages / 2)?;
    let budgets: Vec<_> = [
        Location::coordinator(0),
        Location::coordinator(1),
        Location::PINNED,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, location)| Budget {
        id: format!("fixture.{i}"),
        locations: vec![location],
        physical_bytes: 1 << 40,
        logical_bytes: 1 << 40,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "CPU fake DSO only".into(),
            assumed: false,
        },
        headroom: 0,
    })
    .collect();
    let phases = planner::evaluate(&facts, &budgets)?;
    Ok(ResolvedPlan {
        request: Request {
            context: 4096,
            pool: Pin::Exact(pages * 64),
            onboard: Pin::Auto,
            prefill: Pin::Exact(shape),
            operation: Operation::Serving,
        },
        decisions: contract.decisions(&[0, 0])?,
        pool_tokens: pages * 64,
        budgets,
        facts,
        phases,
    })
}

#[test]
#[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
fn actual_target_buffers_and_aliases_match_independent_contract() -> Result<()> {
    let fixture = Fixture::build()?;
    let cfg = GlmDsaConfig::from_hf(&cuteafd_loader::plan::testing::glm5_config())?;
    let directory = tempfile::tempdir_in(fixture.directory())?;
    let path = directory.path().join("embedding.bf16");
    std::fs::write(&path, vec![0; cfg.vocab_size * cfg.hidden * 2])?;
    for (partitioned, owned, waves, full, pages, native, core) in [
        (false, false, 1, false, 2, false, false),
        (true, false, 3, true, 64, false, false),
        (false, true, 3, false, 66, false, false),
        (false, false, 3, false, 2, true, true),
    ] {
        let library = fixture.load()?;
        fixture.quiet_events(&library, true)?;
        fixture.immediate_copies(&library)?;
        fixture.glm_target_programs(&library)?;
        for device in 0..2 {
            fixture.configure_memory(&library, device, 8 << 30)?;
        }
        let programs = library.programs()?;
        // Only metadata and real allocation constructors run; no numerical program launches.
        assert!(programs.spec("glm_index_producer_m64").is_ok());
        library.cuda_set_device(0)?;
        let owners = owned.then_some([0, 1]);
        let owners = owners.as_ref().map(|a| a.as_slice());
        let shape = PrefillShape { rows: 128, waves };
        let weights = GlmWeights {
            layers: GlmLayer::allocation_fixture(&library, &cfg, 0, owners)?,
            norm: DeviceAllocation::new(&library, 256)?,
            head: super::super::super::weights::GlmHead::Bf16(DeviceAllocation::new(
                &library, 256,
            )?),
        };
        let (embedding, ()) = TokenEmbedding::load(
            &library,
            EmbedSource {
                path: path.clone(),
                offset: 0,
                vocab: cfg.vocab_size,
                hidden: cfg.hidden,
                snapshot: None,
                tensor_name: "fixture.embedding".into(),
            },
            EmbedPlacement::Gpu,
            || Ok(()),
        )?;
        let stream = std::ptr::null_mut();
        let mut engine = GlmEngine::new(
            &library,
            &programs,
            cfg.clone(),
            weights,
            stream,
            4096,
            128,
            pages as usize,
            embedding,
            crate::shared::prefill_depth::PrefillDepth::new(waves as usize, 128)?,
            partitioned.then_some(GlmContext::default()),
        )?;
        engine.full_prefill_logits = full;
        let layers = crate::shared::peer_split::on_device(&library, 1, || {
            GlmLayer::allocation_fixture(&library, &cfg, 1, owners)
        })?;
        engine.attach_peer(1, stream, layers)?;
        if native {
            // Only an ownership marker is loaded; no numerical native programs run.
            engine.weights.layers[0]
                .native
                .insert("fixture".into(), DeviceAllocation::new(&library, 256)?);
            engine.native_scratch = (0..2)
                .map(|_| {
                    Ok([
                        native_layout(&cfg, DECODE_ROWS)?,
                        native_layout(&cfg, shape.rows as usize)?,
                    ])
                })
                .collect::<Result<_>>()?;
        }
        let expected = plan(&cfg, partitioned, owners, pages, shape, full, native, core)?;
        assert!(
            planner::reconcile(&expected, Phase::Warmup, &engine.target_buffer_receipts()?)
                .is_err()
        );
        *engine.decode_workspace.borrow_mut() = Some(engine.workspace(DECODE_ROWS, true)?);
        let _ = engine.peer_workspaces(true, 1)?;
        engine.ensure_prefill_slots()?;
        if core {
            // Allocate through the production CoreScratch constructor; program binding is not
            // emulated. The retained workspaces expose the actual projection owners and views.
            for rank in 0..2 {
                let device = crate::shared::memory::device::Device {
                    library: &library,
                    id: rank as i32,
                };
                let layout = core_layout(&cfg, shape.rows as usize)?;
                let allocate = |index: usize| {
                    let projection = if index == 0 {
                        layout.decode.as_ref()
                    } else {
                        layout.prefill.as_ref()
                    }
                    .unwrap();
                    super::super::super::core_runtime::CoreScratch::new(
                        device,
                        &crate::shared::dense_trellis::ProjectionLayout {
                            extents: projection.extents.clone(),
                            rows: projection.rows,
                        },
                        &layout.stages.as_ref().unwrap()[index],
                    )
                    .map(Rc::new)
                };
                let decode = allocate(0)?;
                let prefill = allocate(1)?;
                let (decode_slot, prefill_slots) = if rank == 0 {
                    (&engine.decode_workspace, &engine.prefill_slots)
                } else {
                    let peer = engine.peer()?;
                    (&peer.decode_workspace, &peer.prefill_slots)
                };
                decode_slot.borrow_mut().as_mut().unwrap().core = Some(decode);
                for slot in prefill_slots.borrow_mut().iter_mut() {
                    slot.core = Some(prefill.clone());
                }
            }
        }
        // Emulate the actual warmup metadata writers without launching a numerical kernel.
        for rank in 0..2 {
            let decode = if rank == 0 {
                &engine.decode_workspace
            } else {
                &engine.peer()?.decode_workspace
            };
            let decode = decode.borrow();
            let w = decode.as_ref().context("CPU decode workspace")?;
            engine.on(rank, || {
                let tables = StepTables {
                    segments: Vec::new(),
                    decode: true,
                    positions: vec![0],
                    slots: vec![0],
                    page_table: vec![0],
                    table_width: 1,
                    table_stride: 1,
                    cache_lengths: vec![1],
                    lengths: vec![1],
                    exchange_rows: 1,
                };
                engine.put_tables(w, &tables)?;
                if rank == 0 {
                    engine.put(&w.ids, &[0u32])?;
                }
                Ok(())
            })?;
        }
        let receipts = || -> Result<Vec<Receipt>> {
            let mut actual = engine.target_buffer_receipts()?;
            actual.extend(engine.target_hook_receipts()?);
            Ok(actual)
        };
        let actual = receipts()?;
        planner::reconcile(&expected, Phase::Warmup, &actual)?;
        if core {
            let slots = engine.prefill_slots.borrow();
            let original = slots[0].core.as_ref().unwrap().clone();
            drop(slots);
            let device = crate::shared::memory::device::Device {
                library: &library,
                id: 0,
            };
            let layout = core_layout(&cfg, shape.rows as usize)?;
            let projection = layout.prefill.as_ref().unwrap();
            let replacement = Rc::new(super::super::super::core_runtime::CoreScratch::new(
                device,
                &crate::shared::dense_trellis::ProjectionLayout {
                    extents: projection.extents.clone(),
                    rows: projection.rows,
                },
                &layout.stages.as_ref().unwrap()[1],
            )?);
            engine.prefill_slots.borrow_mut()[1].core = Some(replacement);
            assert!(engine.target_hook_receipts().is_err());
            engine.prefill_slots.borrow_mut()[1].core = Some(original);
            planner::reconcile(&expected, Phase::Warmup, &receipts()?)?;
        }
        let mut wrong = actual.clone();
        let index = wrong
            .iter()
            .position(|r| r.allocation.id.ends_with("prefill.rows.h"))
            .unwrap();
        wrong[index].allocation.bytes -= 1;
        assert!(planner::reconcile(&expected, Phase::Warmup, &wrong).is_err());
        wrong = actual.clone();
        wrong.remove(index);
        assert!(planner::reconcile(&expected, Phase::Warmup, &wrong).is_err());
        wrong = actual.clone();
        let mut extra = actual[index].clone();
        extra.allocation.id.push_str(".unexpected");
        wrong.push(extra);
        assert!(planner::reconcile(&expected, Phase::Warmup, &wrong).is_err());
        wrong = actual.clone();
        let view = wrong
            .iter_mut()
            .find(|r| r.allocation.id.ends_with("prefill.slot.0.h"))
            .unwrap();
        view.address += 1;
        assert!(planner::reconcile(&expected, Phase::Warmup, &wrong).is_err());
        if waves > 1 {
            let slots = engine.prefill_slots.borrow();
            let original = slots[0].h.clone();
            drop(slots);
            let saved = engine.prefill_slots.borrow()[1].h.clone();
            engine.prefill_slots.borrow_mut()[1].h = original;
            let aliased = receipts()?;
            assert!(planner::reconcile(&expected, Phase::Warmup, &aliased).is_err());
            engine.prefill_slots.borrow_mut()[1].h = saved;
        }
    }
    Ok(())
}
