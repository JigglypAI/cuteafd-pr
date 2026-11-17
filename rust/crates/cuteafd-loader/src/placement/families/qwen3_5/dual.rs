//! Contiguous native whole-layer owners, admitted by the shared phase solver.
use super::{admission::DenseFacts, *};
use crate::placement::planner::*;
use crate::placement::planner::Lifetime;
use crate::serving_capacity::WeightArenaLayout;

/// The executor uses the same shared hop geometry as Qwen4's layer owners.
pub fn hop_spec(cfg: &Qwen35Config) -> HopSpec {
    HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2,
        rows: PREFILL_ROWS as u64, lanes: 1, entry_gpu: 0, head_gpu: 0,
        terminal: TerminalHome::AttentionOwner }
}

pub struct DualAdmission {
    pub plan: ResolvedPlan,
    pub owners: Vec<u8>,
    pub arenas: [WeightArenaLayout; 2],
}

/// Try every legal contiguous cut using exact per-owner allocations, not a
/// proportional estimate of a single-device ledger. Equal pools retain the
/// cut with the smallest residual-headroom gap, including KV and workspaces.
pub fn admit_dual(request: Request, budgets: Vec<Budget>, facts: DenseFacts<'_>)
    -> Result<DualAdmission, Refusal> {
    let cards = budgets.iter().filter(|budget| budget.locations.iter()
        .any(|location| matches!(location.device, Device::Coordinator(_))))
        .map(|budget| budget.logical_bytes).collect::<Vec<_>>();
    let requested = match request.pool { Pin::Auto => None, Pin::Exact(tokens) => Some(tokens) };
    let mut pool_facts = crate::placement::PoolFacts::new(&cards, request.context, requested, PAGE_ROWS);
    pool_facts.encoding_limit = Some(if facts.drafter.is_some() { i32::MAX as u64 } else { u32::MAX as u64 });
    pool_facts.target = facts.target_pool;
    let pool = crate::placement::prepare_pool(pool_facts).map_err(|error| Refusal::InvalidFact(error.to_string()))?;
    let groups = vec![(true, None); facts.cfg.attention.len()];
    let mut best: Option<((u64, std::cmp::Reverse<u64>), DualAdmission)> = None;
    let mut failure = None;
    for cut in crate::placement::attention::owner_boundaries(&groups, true) {
        let owners = (0..groups.len()).map(|i| u8::from(i >= cut)).collect::<Vec<_>>();
        let arenas = crate::families::qwen3_5::weight_layout::placed_arenas(facts.target_arena, &owners)
            .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        let candidate = prepare(&facts, &owners, &arenas).and_then(|contract| {
            super::admission::admit_contract(request.clone(), budgets.clone(), contract, &pool)
        });
        match candidate {
            Ok(plan) => {
                let free = plan.phases.iter().filter(|row| row.phase == Phase::Growth && row.budget.starts_with("rtx"))
                    .map(|row| row.available.saturating_sub(row.owned + row.opaque)).collect::<Vec<_>>();
                let gap = free.iter().max().copied().unwrap_or(0).saturating_sub(free.iter().min().copied().unwrap_or(0));
                let score = (plan.pool_tokens, std::cmp::Reverse(gap));
                if best.as_ref().is_none_or(|(old, _)| score > *old) {
                    best = Some((score, DualAdmission { plan, owners, arenas }));
                }
            }
            Err(error) => { if failure.is_none() { failure = Some(error); } }
        }
    }
    best.map(|(_, admission)| admission).ok_or_else(|| failure.unwrap_or_else(||
        Refusal::InvalidFact("dense two-RTX execution needs two nonempty layer ranges".into())))
}

fn namespace(component: &mut Component, rank: u8) {
    let name = |id: &str| if id.starts_with("qwen3_5/") { id.to_owned() } else { format!("rtx{rank}.{id}") };
    let location = |at: &mut Location| { if *at == Location::coordinator(0) { *at = Location::coordinator(rank); } };
    match &mut component.kind {
        ComponentKind::Allocation { template, .. } => {
            template.id = name(&template.id);
            location(&mut template.location);
            if let Some(view) = &mut template.view { view.backing = name(&view.backing); }
        }
        ComponentKind::Allowance(owner) => {
            location(&mut owner.location);
            if crate::placement::graph_reserve::is_source(&owner.source) {
                owner.source = crate::placement::graph_reserve::source(&format!("qwen3_5.graphs.{rank}"));
            }
        }
        ComponentKind::AllocationGranularity(owner) => {
            owner.id = name(&owner.id);
            location(&mut owner.location);
            for backing in &mut owner.backings { backing.backing = name(&backing.backing); }
        }
        ComponentKind::ResidentWeight(owner) => { owner.id = name(&owner.id); location(&mut owner.location); }
        ComponentKind::DriverPageTables { location: at, backing, .. } => { location(at); *backing = name(backing); }
        _ => {}
    }
}

fn prepare(facts: &DenseFacts<'_>, owners: &[u8], arenas: &[WeightArenaLayout; 2]) -> Result<Contract, Refusal> {
    let mut contracts = Vec::new();
    for rank in 0..2u8 {
        let indices = owners.iter().enumerate().filter_map(|(i, &owner)| (owner == rank).then_some(i)).collect::<Vec<_>>();
        let mut cfg = facts.cfg.clone();
        cfg.attention = indices.iter().map(|&i| cfg.attention[i]).collect();
        let weights = NativeWeights { layers: indices.iter().map(|&i| facts.weights.layers[i]).collect(),
            norm: if rank == 0 { facts.weights.norm } else { 0 },
            head: if rank == 0 { facts.weights.head } else { 0 },
            embedding: if rank == 0 { facts.weights.embedding } else { 0 } };
        let mut buffers = NativeBuffers::new(&cfg, facts.buffers.planes[6], facts.slots, facts.buffers.marks)
            .map_err(|error| Refusal::InvalidFact(error.to_string()))?;
        if rank == 0 { buffers.verify[2] = facts.buffers.verify[2]; }
        else { buffers.ids = 0; buffers.logits = 0; buffers.head = 0; buffers.verify[1] = 0; buffers.verify[2] = 0; }
        let local = DenseFacts { cfg: &cfg, weights: &weights, target_arena: &arenas[rank as usize], buffers: &buffers,
            graph_variant: format!("{}-layers-rtx2-rank{rank}", facts.graph_variant), graphs_enabled: facts.graphs_enabled,
            drafter: if rank == 0 { facts.drafter.clone() } else { None },
            blas_head: if rank == 0 { facts.blas_head } else { serving::BlasHead::Native },
            serving: if rank == 0 { facts.serving } else { None },
            host_prefix: if rank == 0 { facts.host_prefix } else { serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 } },
            ..facts.clone() };
        let mut contract = local.prepare(&Request { context: (cfg.max_context as u64).into(),
            pool: Pin::Auto, onboard: Pin::Auto, prefill: Pin::Auto, operation: Operation::Serving,
            resident_weight_limit: None, spark_reply_formats: Default::default() })?;
        // Full-context drafter page indices are signed. Both ranks must expose
        // the same domain when merging; target-only retains its unsigned extent.
        let maximum = if facts.drafter.is_some() { i32::MAX as u64 / PAGE_ROWS } else { u32::MAX as u64 / PAGE_ROWS };
        if contract.maximum_pages != maximum {
            for component in &mut contract.components {
                let truncate = |curve: &mut ByteCurve| {
                    curve.0.retain(|piece| piece.first <= maximum);
                    for piece in &mut curve.0 { piece.last = piece.last.min(maximum); }
                };
                truncate(&mut component.bytes);
                if let ComponentKind::Allocation { shape, offset, .. } = &mut component.kind {
                    for curve in shape { truncate(curve); }
                    if let Some(curve) = offset { truncate(curve); }
                }
            }
            contract.maximum_pages = maximum;
        }
        contract.components.retain(|component| match &component.kind {
            ComponentKind::DriverPageTables { .. } => false,
            ComponentKind::Allocation { template, .. } if rank == 1 && template.id == "weights.upload" => false,
            ComponentKind::Allowance(owner) if rank == 1 && owner.location == Location::HOST => false,
            ComponentKind::Allowance(owner) if owner.source.starts_with("dense Qwen driver stream slabs (") => false,
            _ => true,
        });
        // Each rank has its execution stream and PrefixCopies stream. The
        // registered host tier adds store/restore streams and their two probes
        // on every registered device, although its quota is charged only once.
        let peak = 2 + if facts.host_prefix.quota_bytes > 0 { 4 } else { 0 };
        let slabs = cuda_stream_slab_bytes(peak)?;
        contract.components.push(Component { activation: Activation::always(),
            bytes: ByteCurve::fixed(slabs, maximum),
            kind: ComponentKind::Allowance(Allowance { location: Location::coordinator(0),
                bytes: slabs, lifetime: Lifetime::RESIDENT, assumed: false,
                source: format!("dense Qwen driver stream slabs ({peak} peak live streams, \
                    SM120 driver/runtime13020 LAZY, #1511 2026-11-08)") }) });
        for component in &mut contract.components { namespace(component, rank); }
        contracts.push(contract);
    }
    let mut result = contracts.remove(0);
    // Each local recipe has only fixed singleton domains; the merged candidate
    // replaces them with global layer IDs before the ordinary shared solve.
    result.components.extend(contracts.remove(0).components);
    for component in &mut result.components { component.activation = Activation::always(); }
    result.gpus = 2;
    result.variables = owners.iter().enumerate().map(|(layer, &gpu)| Variable::Attention {
        layer, values: vec![LayerMode::Whole { gpu, ffn: FfnMode::Owner }] }).collect();
    result.variables.extend([Variable::Prefill(vec![PrefillShape { rows: PREFILL_ROWS as u64, waves: 1 }]),
        Variable::Lanes(vec![1]), Variable::Embedding(vec![EmbeddingHome::Device(0)]), Variable::HeadHome(vec![0])]);
    if facts.drafter.is_some() { result.variables.push(Variable::DraftHome(vec![0])); }
    let modes = owners.iter().map(|&gpu| LayerMode::Whole { gpu, ffn: FfnMode::Owner }).collect::<Vec<_>>();
    let spec = hop_spec(facts.cfg);
    let hops = plan_hops(&modes, &spec).map_err(|error| Refusal::InvalidFact(error.to_string()))?;
    let layout = HopLinkLayout::new(&hops, &spec).ok_or(Refusal::Overflow("dense residual hops"))?;
    let mut transport = layout.receive_components("dense.hops", Lifetime::TARGET, result.maximum_pages);
    transport.extend(layout.control_components("dense.hops", Lifetime::TARGET, result.maximum_pages));
    fixed_request_tails(&mut transport, result.maximum_pages, &[])?;
    result.components.extend(transport);
    result.add_driver_page_tables();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_rank_stream_slab_owners_match_execution_prefix_and_host_copy_streams() {
        let (dir, _, buffers, _) = crate::families::qwen3_5::tests::native_planner_fixture();
        let checkpoint = crate::plan::checkpoint::Checkpoint::open(dir.path()).unwrap();
        let cfg = Qwen35Config::from_hf(&checkpoint.config).unwrap();
        let weights = NativeWeights::native(&checkpoint, &cfg, cuteafd_core::DType::F32).unwrap();
        let target = crate::families::qwen3_5::weight_layout::target_arena(&checkpoint, &cfg,
            cuteafd_core::DType::F32, crate::families::qwen3_5::weight_layout::DenseFormat::Bf16).unwrap();
        let owners = (0..cfg.attention.len()).map(|i| u8::from(i >= 2)).collect::<Vec<_>>();
        let arenas = crate::families::qwen3_5::weight_layout::placed_arenas(&target, &owners).unwrap();
        for host in [false, true] {
            let facts = DenseFacts { target_pool: None, cfg: &cfg, weights: &weights, target_arena: &target, buffers: &buffers,
                slots: 4, record_bytes: 4096, graph_variant: "fixture".into(), graphs_enabled: false,
                drafter: None, blas_head: serving::BlasHead::Native, serving: None,
                host_prefix: serving::HostPrefixOwners { quota_bytes: if host { 1 << 30 } else { 0 },
                    chunk_bytes: if host { 256 << 20 } else { 0 } }, verify_check: false };
            let contract = prepare(&facts, &owners, &arenas).unwrap();
            let execution = 1;
            let prefix_copies = 1;
            let host_store_restore = if host { 2 } else { 0 };
            let host_completion_probes = if host { 2 } else { 0 };
            let created = execution + prefix_copies + host_store_restore + host_completion_probes;
            assert_eq!(created, if host { 6 } else { 2 });
            for rank in [0, 1] {
                let slabs = contract.components.iter().filter_map(|component| match &component.kind {
                    ComponentKind::Allowance(owner) if owner.location == Location::coordinator(rank)
                        && owner.source.starts_with("dense Qwen driver stream slabs (") => Some((component, owner)),
                    _ => None,
                }).collect::<Vec<_>>();
                assert_eq!(slabs.len(), 1, "rank {rank}: one named owner, no legacy double charge");
                assert_eq!(slabs[0].1.bytes, cuda_stream_slab_bytes(created).unwrap());
                assert_eq!(slabs[0].1.bytes, if host { 4 << 20 } else { 2 << 20 });
                assert!(slabs[0].1.source.contains(&format!("{created} peak live streams")));
                assert_eq!(slabs[0].0.bytes.at(1).unwrap(), slabs[0].1.bytes);
            }
            // The peer receives host streams, not a second pinned host quota.
            let pinned = contract.components.iter().filter_map(|component| match &component.kind {
                ComponentKind::Allocation { template, .. } if template.group == "retained host prefix chunk" => Some(template),
                _ => None,
            }).collect::<Vec<_>>();
            assert_eq!(pinned.len(), if host { 4 } else { 0 });
            assert!(pinned.iter().all(|allocation| !allocation.id.starts_with("rtx1.")));
        }
    }

    #[test]
    fn speculation_alone_selects_common_signed_domain_and_exact_owner_balance() {
        let (dir, _, buffers, _) = crate::families::qwen3_5::tests::native_planner_fixture();
        let checkpoint = crate::plan::checkpoint::Checkpoint::open(dir.path()).unwrap();
        let cfg = Qwen35Config::from_hf(&checkpoint.config).unwrap();
        let weights = NativeWeights::native(&checkpoint, &cfg, cuteafd_core::DType::F32).unwrap();
        let target = crate::families::qwen3_5::weight_layout::target_arena(&checkpoint, &cfg,
            cuteafd_core::DType::F32, crate::families::qwen3_5::weight_layout::DenseFormat::Bf16).unwrap();
        let mut budgets = forecast_budgets(&crate::placement::ARCH_CONTEXTS[1], 96 << 30, 0,
            crate::plan::testing::SimulatedHost::raptor().host_ram).unwrap();
        let mut peer = budgets[0].clone();
        peer.id = "rtx1".into(); peer.locations = vec![Location::coordinator(1)]; budgets.insert(1, peer);
        let request = Request { context: (262144).into(), pool: Pin::Exact(262144),
            onboard: Pin::Auto, prefill: Pin::Auto, operation: Operation::Serving,
            resident_weight_limit: None, spark_reply_formats: Default::default() };
        for draft in [false, true] {
            let facts = DenseFacts { target_pool: None, cfg: &cfg, weights: &weights, target_arena: &target, buffers: &buffers,
                slots: 4, record_bytes: 4096, graph_variant: "fixture".into(), graphs_enabled: false,
                drafter: draft.then(|| DrafterBytes { owned: 4096, weights: ResidentWeights(1024),
                    weight_arena: Some(("qwen3_5/mtp/gpu0", 1280)), pools: vec![4096],
                    workspace_requests: vec![2048, 1024], ..Default::default() }),
                blas_head: serving::BlasHead::Native, serving: None,
                host_prefix: serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }, verify_check: false };
            let admission = admit_dual(request.clone(), budgets.clone(), facts.clone()).unwrap();
            let contract = prepare(&facts, &admission.owners, &admission.arenas).unwrap();
            let maximum = if draft { i32::MAX as u64 } else { u32::MAX as u64 } / PAGE_ROWS;
            assert_eq!(contract.maximum_pages, maximum);
            assert_eq!(maximum * PAGE_ROWS, if draft { 2147483584 } else { 4294967232 });
            assert!(contract.components.iter().all(|component| component.bytes.0.last().unwrap().last == maximum));
            assert_eq!(admission.plan.served_context(), 262144);
            let slack = |plan: &ResolvedPlan| plan.phases.iter().filter(|row|
                row.phase == Phase::Growth && row.budget.starts_with("rtx"))
                .map(|row| row.available - row.owned - row.opaque).collect::<Vec<_>>();
            let selected = slack(&admission.plan);
            let gap = selected[0].abs_diff(selected[1]);
            for cut in 1..cfg.attention.len() {
                let owners = (0..cfg.attention.len()).map(|i| u8::from(i >= cut)).collect::<Vec<_>>();
                let arenas = crate::families::qwen3_5::weight_layout::placed_arenas(&target, &owners).unwrap();
                let candidate = super::super::admission::admit_contract(request.clone(), budgets.clone(),
                    prepare(&facts, &owners, &arenas).unwrap(), &admission.plan.pool_policy.unwrap()).unwrap();
                let free = slack(&candidate);
                assert!(gap <= free[0].abs_diff(free[1]), "cut {cut} has a smaller admitted headroom gap");
            }
        }
    }
}
