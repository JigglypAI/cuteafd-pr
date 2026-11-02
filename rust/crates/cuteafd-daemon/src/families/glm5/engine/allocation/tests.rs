use super::*;
use crate::shared::token_io::{EmbedPlacement, EmbedSource};
use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
use cuteafd_loader::placement::{
    families::glm5::buffers::TargetBuffers,
    planner::{self, *},
};

fn plan(
    cfg: &GlmDsaConfig,
    partitioned: bool,
    owners: Option<&[u8]>,
    pages: u64,
    shape: PrefillShape,
    full: bool,
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
    for (partitioned, owned, waves, full, pages) in [
        (false, false, 1, false, 2),
        (true, false, 3, true, 64),
        (false, true, 3, false, 66),
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
        let expected = plan(&cfg, partitioned, owners, pages, shape, full)?;
        assert!(
            planner::reconcile(&expected, Phase::Warmup, &engine.target_buffer_receipts()?)
                .is_err()
        );
        *engine.decode_workspace.borrow_mut() = Some(engine.workspace(DECODE_ROWS, true)?);
        let _ = engine.peer_workspaces(true, 1)?;
        engine.ensure_prefill_slots()?;
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
        let actual = engine.target_buffer_receipts()?;
        planner::reconcile(&expected, Phase::Warmup, &actual)?;
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
            let aliased = engine.target_buffer_receipts()?;
            assert!(planner::reconcile(&expected, Phase::Warmup, &aliased).is_err());
            engine.prefill_slots.borrow_mut()[1].h = saved;
        }
    }
    Ok(())
}
