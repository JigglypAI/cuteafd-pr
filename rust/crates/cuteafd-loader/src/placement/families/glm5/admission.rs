//! Finite GLM component composition. This is not yet wired to live admission:
//! expert/serving owners and startup history must join before Ready is allowed.
//! Loading provenance never changes an owner's charge or its activation.
use super::{PAGE_ROWS, weights::WeightInventory};
use super::{buffers::TargetBuffers, exchanges::ExchangeBuffers, hooks::HookBuffers};
use crate::families::glm5::GlmDsaConfig;
use crate::placement::inventory::{DeclaredGraphSet, FigureAttribution, FigureIdentity, MemoryFigure};
use crate::placement::{AttentionPlacement, LayerMode, planner::*};
use crate::serving_capacity::glm_graphs::{GlmGraphKey, GlmGraphLayer, glm_declared_graphs, glm_graph_shapes};
use cuteafd_core::spark_reply::SparkReplyFormats;
use std::collections::{BTreeMap, BTreeSet};

/// A condition names a domain value, never a solver variable index. Conditions
/// in one case are conjunctive; membership in a count interval is disjunctive.
#[derive(Debug, Clone)]
pub enum When {
    Prefill(PrefillShape),
    Lanes(u64),
    Embedding(EmbeddingHome),
    Onboard { first: usize, last: usize },
    LocalLayer(usize),
    RemoteLayer(usize),
    SparkRanks(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct Case<T> {
    pub when: Vec<When>,
    pub value: T,
}

/// One canonical target-only prefix. TP2 names both destinations but counts a
/// routed layer once. A fallback rank domain is absent only for all-local work.
#[derive(Debug, Clone)]
pub struct ExpertProfile {
    pub layers: Vec<usize>,
    pub destination: Vec<u8>,
    pub counts: Vec<usize>,
    pub rank_sets: Vec<Vec<u8>>,
}

/// Physical subrecipes prepared from headers and selected program tables, not
/// pre-subtracted free-memory totals. This builder does not select a pool or
/// solve once per count. Additional owners use the same named finite domains.
#[derive(Debug, Clone)]
pub struct GlmRecipe<'a> {
    pub cfg: &'a GlmDsaConfig,
    pub context: u64,
    pub page_tokens: u64,
    pub maximum_pages: u64,
    /// The executor's existing attention ownership, unchanged by this port.
    pub attention: Vec<LayerMode>,
    pub gpus: usize,
    /// Retained links, even when one assignment onboards every target layer.
    pub spark_links: bool,
    pub prefill: Vec<PrefillShape>,
    pub lanes: Vec<u64>,
    pub embedding: Vec<EmbeddingHome>,
    pub experts: Option<ExpertProfile>,
    pub targets: Vec<Case<TargetBuffers>>,
    pub hooks: Vec<Case<HookBuffers>>,
    pub exchanges: Vec<Case<ExchangeBuffers>>,
    pub weights: Vec<Case<WeightInventory>>,
    pub graphs: Vec<Case<StartupGraphs>>,
    /// Process-wide ordinary RSS; subsumes heap subsets, never pinned storage.
    pub host: Option<HostFigures>,
    /// Separate physical decode/prefill links; each rank retains one return reserve.
    pub returns: Vec<Case<ReplyReturns>>,
    /// Expert, drafter, serving, code and temporary physical components.
    /// Their constructors remain responsible for independent actual receipts.
    pub owners: Vec<Case<Vec<Component>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyPhase {
    Decode,
    Prefill(PrefillShape),
}

impl GlmRecipe<'_> {
    /// Mechanical executor output. Backend bytes come from the selected local
    /// physical inventory, not the report's route/payload/check/view totals.
    pub fn execution(&self, solved: &ResolvedPlan, local: &super::local_physical::LocalPhysicalInventory)
        -> Result<(crate::placement::Placement, crate::placement::SparkWorkingSet), Refusal> {
        use crate::placement::{AttentionClass, ExpertHome, ExpertRange, FfnMode, HopSpec,
            HopStorage, LayerAssignment, Placement, PoolLimits, SparkWorkingSet, TerminalHome,
            Tp2Range, plan_hops, residual_homes};
        if !(1..=2).contains(&self.gpus) || solved.decisions.attention != self.attention
            || self.attention.is_empty() || self.attention.len() > self.cfg.layers
            || solved.decisions.onboard_layers.len() != self.gpus
            || !self.prefill.contains(&solved.decisions.prefill)
            || !self.lanes.contains(&solved.decisions.lanes)
            || solved.decisions.embedding.is_none_or(|home| !self.embedding.contains(&home))
        {
            return Err(invalid("resolved execution differs from prepared geometry"));
        }
        let backend_bytes = [
            local.backend_bytes(0).map_err(|error| invalid(&error.to_string()))?,
            local.backend_bytes(1).map_err(|error| invalid(&error.to_string()))?,
        ];
        let modes = &solved.decisions.attention;
        let attention = if modes.iter().all(|mode| *mode == LayerMode::ContextSplit) {
            AttentionPlacement::Context
        } else if modes.iter().all(|mode| *mode == LayerMode::HeadSplit)
            || self.gpus == 1 && modes.iter().all(|mode|
                *mode == LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner })
        {
            AttentionPlacement::Heads
        } else if modes.iter().all(|mode| matches!(mode, LayerMode::Whole { .. })) {
            AttentionPlacement::Layers
        } else {
            return Err(invalid("executor cannot report mixed GLM attention modes"));
        };
        let mut local = Vec::new();
        let mut remote = Vec::new();
        let mut local_by_rank = vec![Vec::new(); self.gpus];
        let mut ranks = None;
        let mut assignments = Vec::with_capacity(modes.len());
        for (layer, &mode) in modes.iter().enumerate() {
            let experts = match solved.decisions.expert_homes.get(&layer) {
                None if self.experts.is_none() || layer < self.cfg.first_moe_layer => ExpertHome::Dense,
                None => return Err(invalid("resolved routed layer has no expert home")),
                Some(_) if self.experts.is_none() || layer < self.cfg.first_moe_layer =>
                    return Err(invalid("resolved expert home names a dense layer")),
                Some(ExpertDestination::Coordinator { gpus }) => {
                    if self.experts.as_ref().is_none_or(|profile| *gpus != profile.destination) {
                        return Err(invalid("resolved local home differs from prepared destination"));
                    }
                    let home = match gpus.as_slice() {
                        [0, 1] if self.gpus == 2 => ExpertHome::RtxTp2,
                        [gpu] if usize::from(*gpu) < self.gpus => ExpertHome::RtxWhole { gpu: *gpu },
                        _ => return Err(invalid("executor cannot bind resolved local expert home")),
                    };
                    for &gpu in gpus { local_by_rank[usize::from(gpu)].push(layer); }
                    local.push(layer);
                    home
                }
                Some(ExpertDestination::Sparks { ranks: selected }) => {
                    if selected.is_empty() || selected.len() > 8
                        || selected.iter().enumerate().any(|(rank, &value)| usize::from(value) != rank)
                    {
                        return Err(invalid("executor requires a Spark rank prefix starting at logical rank 0; non-prefix rank sets cannot be renumbered"));
                    }
                    if self.experts.as_ref().is_none_or(|profile| !profile.rank_sets.contains(selected))
                        || ranks.as_ref().is_some_and(|ranks| ranks != selected)
                    {
                        return Err(invalid("resolved Spark layers do not share a prepared rank set"));
                    }
                    ranks = Some(selected.clone());
                    remote.push(layer);
                    ExpertHome::Spark
                }
            };
            assignments.push(LayerAssignment { mode, experts });
        }
        if solved.decisions.expert_homes.keys().any(|&layer| layer >= modes.len())
            || local_by_rank != solved.decisions.onboard_layers
            || local.iter().copied().ne(self.cfg.first_moe_layer..self.cfg.first_moe_layer + local.len())
            || self.experts.as_ref().is_some_and(|profile| !profile.counts.contains(&local.len()))
            || backend_bytes.iter().enumerate().any(|(rank, &bytes)|
                (rank >= self.gpus || local_by_rank[rank].is_empty()) && bytes != 0)
        {
            return Err(invalid("resolved onboarding is not the declared ordered prefix"));
        }
        let whole = local.iter().any(|&layer| matches!(assignments[layer].experts, ExpertHome::RtxWhole { .. }));
        let tp2 = if !local.is_empty() && !whole {
            Some(Tp2Range { first: local[0], layers: local.len(), peak_bytes: backend_bytes })
        } else { None };
        let ranges = local_by_rank.iter().enumerate().map(|(rank, layers)| {
            if tp2.is_some() || layers.is_empty() { ExpertRange { first: 0, layers: 0, peak_bytes: 0 } }
            else { ExpertRange { first: layers[0], layers: layers.len(), peak_bytes: backend_bytes[rank] } }
        }).collect();
        let hop_spec = HopSpec { storage: HopStorage::ReceiveSlots,
            row_bytes: (self.cfg.hidden as u64).checked_mul(2).ok_or(Refusal::Overflow("GLM hop rows"))?,
            rows: solved.decisions.prefill.rows.max(super::DECODE_ROWS),
            lanes: solved.decisions.prefill.waves, entry_gpu: 0, head_gpu: 0,
            terminal: if self.gpus == 2 { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } };
        let residual = residual_homes(modes, &hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let hops = plan_hops(modes, &hop_spec).map_err(|error| invalid(&error.to_string()))?;
        let context = super::context_buffers(self.cfg, self.context, solved.decisions.prefill.rows,
            solved.decisions.prefill.waves, super::GlmContext::default());
        let mut peer_row_bytes = 0u64;
        for (layer, mode) in modes.iter().enumerate() {
            if *mode != LayerMode::ContextSplit { continue; }
            peer_row_bytes = peer_row_bytes.checked_add(context.query_row_bytes)
                .and_then(|bytes| bytes.checked_add(context.partial_row_bytes))
                .and_then(|bytes| bytes.checked_add(if self.cfg.indexers[layer] == crate::families::glm5::GlmIndexer::Full {
                    context.candidate_row_bytes
                } else { 0 })).ok_or(Refusal::Overflow("GLM context peer rows"))?;
        }
        let placement = Placement { pool_tokens: solved.pool_tokens, pool_limits: PoolLimits(vec![]),
            attention_placement: attention, attention_by_kind: vec![(AttentionClass::Dsa, attention)],
            peer_row_bytes, onboard_layers: local.len(), max_pool_tokens_no_onboard: None,
            layers: assignments, movables: vec![], expert_ranges: ranges, residual, hops, tp2,
            items: (0..self.gpus).map(|rank| solved.items(Location::coordinator(rank as u8)))
                .collect::<Result<_, _>>()? };
        super::super::GLM5.check(&placement).map_err(|error| invalid(&error.to_string()))?;
        let working = SparkWorkingSet { rtx_layers: local, spark_layers: remote,
            spark_ranks: ranks.map_or(0, |ranks| ranks.len()), prefill_lanes: solved.decisions.prefill.waves };
        Ok((placement, working))
    }
}

/// Device return reserves only. Transport staging and registrations have their
/// own owners; compressed replies reuse these BF16 fallback backings.
#[derive(Debug, Clone)]
pub struct ReplyReturns {
    pub id: String,
    pub device: u8,
    pub ranks: Vec<u8>,
    pub phase: ReplyPhase,
}

impl ReplyReturns {
    fn components(
        &self,
        cfg: &GlmDsaConfig,
        maximum_pages: u64,
        formats: SparkReplyFormats,
    ) -> Result<Vec<Component>, Refusal> {
        if self.id.is_empty()
            || self.ranks.is_empty()
            || self.ranks.iter().any(|&rank| rank >= 8)
            || self.ranks.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid("invalid Spark return identity or rank set"));
        }
        let (rows, waves, format) = match self.phase {
            ReplyPhase::Decode => (super::DECODE_ROWS, 1, formats.decode()),
            ReplyPhase::Prefill(shape) => (
                shape.rows,
                shape.waves,
                formats.prefill(cuteafd_core::ExpertGeometry::GLM5.partial_replies()),
            ),
        };
        let bytes = crate::serving_capacity::prefill_return_bytes_with_format(
            waves,
            rows,
            1,
            cfg.hidden as u64,
            format,
        )
        .map_err(|error| Refusal::InvalidFact(format!("GLM Spark return: {error}")))?;
        Ok(self
            .ranks
            .iter()
            .map(|rank| {
                let owner = Allocation::buffer(
                    format!("{}.rank.{rank}", self.id),
                    Location::coordinator(self.device),
                    bytes,
                    Lifetime::TARGET,
                    cuteafd_core::memory_layout::Category::Transport,
                    "GLM Spark returns",
                );
                Component::allocation(
                    owner,
                    ByteCurve::fixed(bytes, maximum_pages),
                    false,
                    maximum_pages,
                )
            })
            .collect())
    }
}

/// The current DFlash prefetch reads one entire file while target weights load.
/// Its ordinary Vec payload is an assumption, not an allocator/RSS receipt.
#[derive(Debug, Clone)]
pub struct DflashPrefetch {
    file_bytes: u64,
}

impl DflashPrefetch {
    pub fn from_snapshot(snapshot: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::{Context, ensure};
        let path = snapshot.join("model.safetensors");
        let metadata = std::fs::metadata(&path)
            .with_context(|| format!("DFlash prefetch file {}", path.display()))?;
        ensure!(
            metadata.is_file() && metadata.len() > 0,
            "DFlash prefetch needs a nonempty regular file: {}",
            path.display()
        );
        Ok(Self {
            file_bytes: metadata.len(),
        })
    }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 {
            return Err(invalid("zero DFlash prefetch pool extent"));
        }
        Ok(vec![Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(self.file_bytes, maximum_pages),
            kind: ComponentKind::Allowance(Allowance {
                location: Location::HOST,
                bytes: self.file_bytes,
                lifetime: Lifetime { first: Phase::Load, last: Phase::DraftLoad },
                source: "DFlash whole-file prefetch Vec payload: model.safetensors metadata.len(); overlaps target read/pack heap in Load and is released when drafter load returns; ordinary Rust heap, unmeasured assumption; excludes allocator/tensor-map metadata overhead".into(),
                assumed: true,
            }),
        }])
    }
}

const CENSUS_PROGRAMS: &str = "15a9aea77733f9be49e7871642e92fd65a64a46d4c7eb1711026307c90c391cf";
const CENSUS_NATIVE: &str = "f5b8d1e1371ae31c97164f08a6da5de293241ef0a4c1c89beeeba7c52a479513";
const CENSUS_TOOLCHAIN: &str = "rustc 1.98.1 (48a229cea 2026-09-01); binary: rustc; commit-hash: 48a229ceaefd4985c50990b14116b6d856af0985; commit-date: 2026-09-01; host: x86_64-unknown-linux-gnu; release: 1.98.1; LLVM version: 22.1.8";
const MIN_LAYOUT: &str = "glm53-exl3-k4/47af23347db743b4666d952e2eb48f2b01c3fede; dflash/425aa615ce320caac34400208b30808c8f14f76c/fp8; heads/1; sparks/4; onboard/0; context/1048576; compiled/1048576; pool/1257856; C16; prefill/2752x3; lanes/1; embedding/device; selector/device; replies/bf16,bf16; l2/auto; host-prefix/34359738368; instrumentation/census";
const MAX_LAYOUT: &str = "glm53-exl3-k4/47af23347db743b4666d952e2eb48f2b01c3fede; dflash/425aa615ce320caac34400208b30808c8f14f76c/fp8; context/2; sparks/6; onboard/3..9/tp2; context/1048576; compiled/1048576; pool/2097152; C16; prefill/2752x3; lanes/1; embedding/device; selector/device; replies/bf16,bf16; l2/auto; host-prefix/34359738368; instrumentation/census";
const MIN_GRAPHS: &str = "2bae8417cb9862ba77083453922d96de79b52db8de16a9ddebbd821d476afaf6";
const MAX_GRAPHS: [&str; 2] = [
    "b3acfa6e5d560b9f0440af62250a9f3b32424f9186deeb769f1a1ed0d5311543",
    "7f4662ddadd8137dc854f960c69c41469d82662e9e51073d7747ff4b31a61ccf",
];

const fn measured(bytes: u64, graphs: &'static str, layout: &'static str,
    commit: &'static str, evidence: &'static str) -> MemoryFigure {
    MemoryFigure { bytes, attribution: FigureAttribution::Measured, evidence,
        producing_commit: commit, identity: FigureIdentity { declared_graphs: graphs,
            programs_sha256: CENSUS_PROGRAMS, native_sha256: CENSUS_NATIVE,
            toolchain: CENSUS_TOOLCHAIN, gpu_class: "RTX PRO 6000 Blackwell/sm_120/188",
            driver: "13.2", layout } }
}

/// Measured parameters, #1242 min census (2026-11-04), retained under
/// builds/shared-planner-glm-census-min-4a56925e7/glm53-exl3-min-census-r1.
/// Context is normalized at preweight (context + streams + selected modules).
/// Graph bytes are the signed immediate instantiate sum, not whole warm growth.
/// These are not independently qualified bounds, receipts or post-Ready Growth.
pub const GLM_MIN_CONTEXT: MemoryFigure = measured(598_999_040, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519", "#1242 glm53-exl3-min-census-r1/preweight");
pub const GLM_MIN_GRAPH: MemoryFigure = measured(2_560_622_592, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519", "#1242 glm53-exl3-min-census-r1/instantiate");

/// Measured parameters, #1242 max census (2026-11-04), retained under
/// builds/shared-planner-glm-census-max-5d180b5c8/glm53-exl3-max-census-r1.
/// Preserve logical-rank asymmetry; rank 0 alone captures the terminal head.
pub const GLM_MAX_CONTEXT: [MemoryFigure; 2] = [
    measured(594_804_736, MAX_GRAPHS[0], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/preweight/rank0"),
    measured(603_193_344, MAX_GRAPHS[1], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/preweight/rank1"),
];
pub const GLM_MAX_GRAPH: [MemoryFigure; 2] = [
    measured(3_042_967_552, MAX_GRAPHS[0], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/instantiate/rank0"),
    measured(2_971_664_384, MAX_GRAPHS[1], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/instantiate/rank1"),
];

/// Measured #1242 peak over recorded startup phases after subtracting the
/// normalized preweight context, exact native backings and signed instantiate
/// bytes. Ready was the largest recorded remainder on all three ranks. These
/// are observed peaks, not independently qualified bounds or attributed owners.
pub const GLM_MIN_RUNTIME_UNATTRIBUTED: MemoryFigure = measured(621_275_328, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519", "#1242 glm53-exl3-min-census-r1/runtime-unattributed-peak");
pub const GLM_MAX_RUNTIME_UNATTRIBUTED: [MemoryFigure; 2] = [
    measured(912_069_460, MAX_GRAPHS[0], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/runtime-unattributed-peak/rank0"),
    measured(581_053_972, MAX_GRAPHS[1], MAX_LAYOUT, "5d180b5c8d76636d5937336638dfd71c15a8d226",
        "#1242 glm53-exl3-max-census-r1/runtime-unattributed-peak/rank1"),
];

/// Opaque non-graph runtime parameters, charged independently of physical
/// weights, payloads, pinned storage and graph instantiate bytes.
#[derive(Debug, Clone)]
pub struct RuntimeFigures {
    context: Vec<MemoryFigure>,
    unattributed: Vec<MemoryFigure>,
}

impl RuntimeFigures {
    /// `layout` is the complete selected layout identity, not a graph count or
    /// placement name. Uncovered layouts keep the measured bytes unchanged and
    /// explicitly record the one-to-one derivation; staleness never adds bytes.
    pub fn new(gpus: usize, layout: &str) -> Result<Self, Refusal> {
        if !(1..=2).contains(&gpus) || layout.trim().is_empty() {
            return Err(invalid("invalid runtime figure layout or rank count"));
        }
        let context = if gpus == 1 { vec![GLM_MIN_CONTEXT] } else { GLM_MAX_CONTEXT.to_vec() };
        let unattributed = if gpus == 1 { vec![GLM_MIN_RUNTIME_UNATTRIBUTED] }
            else { GLM_MAX_RUNTIME_UNATTRIBUTED.to_vec() };
        let derive = |figures: Vec<MemoryFigure>, derivation| -> Result<_, Refusal> {
            figures.into_iter().map(|figure| {
                if layout == figure.identity.layout { Ok(figure) }
                else { figure.scaled(1, 1, derivation).map_err(invalid) }
            }).collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            context: derive(context,
                "unchanged measured min/max preweight context; different layout, no finer context measurement")?,
            unattributed: derive(unattributed,
                "unchanged measured min/max recorded runtime-unattributed peak; different layout, no finer runtime measurement")?,
        })
    }

    pub fn context(&self) -> &[MemoryFigure] { &self.context }
    pub fn unattributed(&self) -> &[MemoryFigure] { &self.unattributed }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 { return Err(invalid("zero runtime figure pool extent")); }
        Ok([("context and selected modules", &self.context),
            ("runtime unattributed", &self.unattributed)].into_iter().flat_map(|(name, figures)| {
                figures.iter().enumerate().map(move |(rank, figure)| Component {
                    activation: Activation::always(),
                    bytes: ByteCurve::fixed(figure.bytes, maximum_pages),
                    kind: ComponentKind::Allowance(Allowance {
                        location: Location::coordinator(rank as u8), bytes: figure.bytes,
                        lifetime: Lifetime::RESIDENT,
                        source: format!("GLM {name}: {:?}; {}; producer {}",
                            figure.attribution, figure.evidence, figure.producing_commit),
                        assumed: matches!(figure.attribution, FigureAttribution::Estimated { .. }),
                    }),
                })
            }).collect())
    }
}

/// Process-wide ordinary RSS from #1242, not one charge per coordinator GPU.
/// Warm-up samples are included in the load peak. Max uses its larger Ready
/// sample conservatively before Ready as well; these are sampled envelopes,
/// not independently qualified bounds on unobserved intra-phase peaks.
pub const GLM_MIN_HOST_LOAD: MemoryFigure = measured(5_759_569_920, MIN_GRAPHS, MIN_LAYOUT,
    "4a56925e7986a57e3194c0cca6df256838386519",
    "#1242 glm53-exl3-min-census-r1/glm/weights-before-draft-pool/ordinary_rss_bytes");
pub const GLM_MIN_HOST_STEADY: MemoryFigure = MemoryFigure {
    attribution: FigureAttribution::Estimated {
        derivation: "Ready smaps_rss_bytes 40403427328 - file_mapping_resident_bytes 36902883328 = 3500544000 B; ordinary_rss_bytes missing and complete=false, not a raw Ready ordinary-RSS receipt",
    },
    ..measured(3_500_544_000, MIN_GRAPHS, MIN_LAYOUT,
        "4a56925e7986a57e3194c0cca6df256838386519",
        "#1242 glm53-exl3-min-census-r1/glm/ready/incomplete-host-reconstruction")
};
const MAX_HOST_GRAPHS: &str = "b3acfa6e5d560b9f0440af62250a9f3b32424f9186deeb769f1a1ed0d5311543,7f4662ddadd8137dc854f960c69c41469d82662e9e51073d7747ff4b31a61ccf";
pub const GLM_MAX_HOST_LOAD: MemoryFigure = measured(8_439_906_304, MAX_HOST_GRAPHS, MAX_LAYOUT,
    "5d180b5c8d76636d5937336638dfd71c15a8d226",
    "#1242 glm53-exl3-max-census-r1/glm/ready/ordinary_rss_bytes; maximum including startup warm-up");
pub const GLM_MAX_HOST_STEADY: MemoryFigure = measured(8_439_906_304, MAX_HOST_GRAPHS, MAX_LAYOUT,
    "5d180b5c8d76636d5937336638dfd71c15a8d226",
    "#1242 glm53-exl3-max-census-r1/glm/ready/ordinary_rss_bytes");

#[derive(Debug, Clone)]
pub struct HostFigures {
    gpus: usize,
    load: MemoryFigure,
    steady: MemoryFigure,
}

impl HostFigures {
    /// Other one-/two-coordinator layouts use the matching reference unchanged.
    /// Spark count does not scale process RSS; uncovered layouts stay estimated.
    pub fn for_layout(gpus: usize, layout: &str) -> Result<Self, Refusal> {
        if !(1..=2).contains(&gpus) || layout.trim().is_empty() {
            return Err(invalid("invalid host figure layout or rank count"));
        }
        let (load, steady, derivation) = if gpus == 1 {
            (GLM_MIN_HOST_LOAD, GLM_MIN_HOST_STEADY,
                "unchanged one-coordinator host envelope from source layout glm53-exl3-min-census-r1 (heads/1, sparks/4, onboard/0), #1242 retained coordinator log; no Spark-count scaling; min Ready remains an incomplete smaps-minus-file estimate")
        } else {
            (GLM_MAX_HOST_LOAD, GLM_MAX_HOST_STEADY,
                "unchanged two-coordinator host envelope from source layout glm53-exl3-max-census-r1 (context/2, sparks/6, onboard/3..9/tp2), #1242 retained coordinator log; no Spark-count scaling")
        };
        let derive = |figure: MemoryFigure| {
            if layout == figure.identity.layout { Ok(figure) }
            else { figure.scaled(1, 1, derivation).map_err(invalid) }
        };
        Ok(Self { gpus, load: derive(load)?, steady: derive(steady)? })
    }

    pub fn load(&self) -> MemoryFigure { self.load }
    pub fn steady(&self) -> MemoryFigure { self.steady }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 { return Err(invalid("zero host figure pool extent")); }
        Ok([("load", self.load, Lifetime { first: Phase::Load, last: Phase::Warmup }),
            ("steady", self.steady, Lifetime { first: Phase::Ready, last: Phase::Growth })]
            .into_iter().map(|(name, figure, lifetime)| Component {
                activation: Activation::always(),
                bytes: ByteCurve::fixed(figure.bytes, maximum_pages),
                kind: ComponentKind::Allowance(Allowance {
                    location: Location::HOST, bytes: figure.bytes, lifetime,
                    source: format!("GLM process ordinary RSS {name}: {:?}; {}; source layout {}; producer {}",
                        figure.attribution, figure.evidence, figure.identity.layout, figure.producing_commit),
                    assumed: matches!(figure.attribution, FigureAttribution::Estimated { .. }),
                }),
            }).collect())
    }
}

/// Whole ordinary RSS already includes these physical heap subsets. Resource
/// facts remain independent, and pinned memory never belongs to this envelope.
fn ordinary_heap_subset(component: &Component) -> bool {
    match &component.kind {
        ComponentKind::Allocation { template, .. } => template.location == Location::HOST,
        ComponentKind::Sequential(group) => !group.members.is_empty()
            && group.members.iter().all(|owner| owner.location == Location::HOST),
        ComponentKind::Allowance(allowance) => allowance.location == Location::HOST,
        ComponentKind::Reservation(_) | ComponentKind::ResidentWeight(_) => false,
    }
}

/// Independently declared full-context keys plus maintained whole-set figures.
/// No per-executable calibration, margin or invented post-Ready reserve.
#[derive(Debug, Clone)]
pub struct StartupGraphs {
    context: u64,
    attention: AttentionPlacement,
    layers: Vec<GlmGraphLayer>,
    declared: Vec<DeclaredGraphSet<GlmGraphKey>>,
    figures: Vec<MemoryFigure>,
}

impl StartupGraphs {
    pub fn new(context: u64, attention: AttentionPlacement, gpus: usize,
        layers: &[GlmGraphLayer], return_ranks: usize) -> Result<Self, Refusal> {
        if context == 0 { return Err(invalid("zero graph context")); }
        let extent = usize::try_from(context).map_err(|_| Refusal::Overflow("GLM graph context"))?;
        let declared = glm_declared_graphs(&glm_graph_shapes(extent, extent.div_ceil(PAGE_ROWS as usize)),
            attention, gpus, layers, return_ranks).map_err(Refusal::InvalidFact)?;
        let mut figures = Vec::with_capacity(gpus);
        for (rank, keys) in declared.iter().enumerate() {
            let source = if gpus == 1 { GLM_MIN_GRAPH } else { GLM_MAX_GRAPH[rank] };
            let identity = keys.fingerprint(rank as u8).map_err(Refusal::InvalidFact)?;
            figures.push(if identity == source.identity.declared_graphs { source } else {
                // Estimated, not measured: a new declaration changes the graph
                // set. Scale the named whole min/max set by exact key counts.
                source.scaled(keys.len() as u64, if rank == 0 { 16_827 } else { 16_614 },
                    "whole measured min/max instantiate bytes * declared keys / measured declared keys; different graph layout, not per-executable calibration")
                    .map_err(|error| invalid(error))?
            });
        }
        Ok(Self { context, attention, layers: layers.to_vec(), declared, figures })
    }

    pub fn declared(&self) -> &[DeclaredGraphSet<GlmGraphKey>] { &self.declared }
    pub fn figures(&self) -> &[MemoryFigure] { &self.figures }

    pub fn components(&self, maximum_pages: u64) -> Result<Vec<Component>, Refusal> {
        if maximum_pages == 0 { return Err(invalid("zero graph pool extent")); }
        Ok(self.figures.iter().enumerate().map(|(rank, figure)| Component {
            activation: Activation::always(),
            bytes: ByteCurve::fixed(figure.bytes, maximum_pages),
            kind: ComponentKind::Allowance(Allowance { location: Location::coordinator(rank as u8),
                bytes: figure.bytes, lifetime: Lifetime { first: Phase::Warmup, last: Phase::Growth },
                source: format!("GLM startup graph instantiate: {:?}; {}; exact {} declared keys at context {}; producer {}",
                    figure.attribution, figure.evidence, self.declared[rank].len(), self.context, figure.producing_commit),
                assumed: matches!(figure.attribution, FigureAttribution::Estimated { .. }),
            }),
        }).collect())
    }
}

struct Domains<'a> {
    prefill: (usize, &'a [PrefillShape]),
    lanes: (usize, &'a [u64]),
    embedding: (usize, &'a [EmbeddingHome]),
    onboard: Option<(usize, &'a ExpertProfile)>,
    remote: Option<usize>,
}

fn invalid(detail: &str) -> Refusal {
    Refusal::InvalidFact(format!("GLM component contract: {detail}"))
}

impl Domains<'_> {
    fn activation(&self, conditions: &[When]) -> Result<Activation, Refusal> {
        let mut domains = BTreeMap::<usize, BTreeSet<usize>>::new();
        for condition in conditions {
            let (variable, options): (_, Vec<_>) = match condition {
                When::Prefill(shape) => (
                    self.prefill.0,
                    self.prefill
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| (value == shape).then_some(i))
                        .collect(),
                ),
                When::Lanes(lanes) => (
                    self.lanes.0,
                    self.lanes
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| (value == lanes).then_some(i))
                        .collect(),
                ),
                When::Embedding(home) => (
                    self.embedding.0,
                    self.embedding
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| (value == home).then_some(i))
                        .collect(),
                ),
                When::Onboard { first, last } => {
                    let (variable, profile) = self
                        .onboard
                        .ok_or_else(|| invalid("count without routed experts"))?;
                    if first > last || *last > profile.layers.len() {
                        return Err(invalid("invalid onboarding interval"));
                    }
                    (
                        variable,
                        profile
                            .counts
                            .iter()
                            .enumerate()
                            .filter_map(|(i, count)| (first <= count && count <= last).then_some(i))
                            .collect(),
                    )
                }
                When::LocalLayer(layer) | When::RemoteLayer(layer) => {
                    let (variable, profile) = self
                        .onboard
                        .ok_or_else(|| invalid("layer without routed experts"))?;
                    let position = profile
                        .layers
                        .iter()
                        .position(|value| value == layer)
                        .ok_or_else(|| invalid("condition names a non-target routed layer"))?;
                    let local = matches!(condition, When::LocalLayer(_));
                    (
                        variable,
                        profile
                            .counts
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &count)| ((position < count) == local).then_some(i))
                            .collect(),
                    )
                }
                When::SparkRanks(ranks) => {
                    let (_, profile) = self
                        .onboard
                        .ok_or_else(|| invalid("ranks without routed experts"))?;
                    (
                        self.remote
                            .ok_or_else(|| invalid("ranks without remote homes"))?,
                        profile
                            .rank_sets
                            .iter()
                            .enumerate()
                            .filter_map(|(i, value)| (value == ranks).then_some(i))
                            .collect(),
                    )
                }
            };
            if options.is_empty() {
                return Err(invalid("condition has no domain value"));
            }
            let options: BTreeSet<_> = options.into_iter().collect();
            if let Some(existing) = domains.get_mut(&variable) {
                existing.retain(|option| options.contains(option));
                if existing.is_empty() {
                    return Err(invalid("contradictory component conditions"));
                }
            } else {
                domains.insert(variable, options);
            }
        }
        let mut clauses = vec![vec![]];
        for (variable, options) in domains {
            clauses = clauses
                .into_iter()
                .flat_map(|clause| {
                    options.iter().map(move |&option| {
                        let mut clause = clause.clone();
                        clause.push(Choice { variable, option });
                        clause
                    })
                })
                .collect();
        }
        Ok(Activation { clauses })
    }

    fn append(
        &self,
        contract: &mut Contract,
        conditions: &[When],
        components: Vec<Component>,
    ) -> Result<(), Refusal> {
        let activation = self.activation(conditions)?;
        for mut component in components {
            // Subrecipes cannot smuggle private variable indexes into the family contract.
            if component.activation != Activation::always() {
                return Err(invalid("subrecipe supplied a private activation"));
            }
            component.activation = activation.clone();
            contract.components.push(component);
        }
        Ok(())
    }
}

impl Recipe for GlmRecipe<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if !(1..=2).contains(&self.gpus)
            || self.attention.is_empty()
            || self.attention.len() > self.cfg.layers
            || self.context == 0
            || request.context == 0
            || request.context != self.context
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
            || self.page_tokens == 0
            || self.page_tokens % PAGE_ROWS != 0
            || self.prefill.is_empty()
            || self.lanes.is_empty()
            || self.embedding.is_empty()
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        self.page_tokens
            .checked_mul(self.maximum_pages)
            .ok_or(Refusal::Overflow("GLM pool extent"))?;
        for (i, shape) in self.prefill.iter().enumerate() {
            if !(64..=4096).contains(&shape.rows)
                || shape.rows % PAGE_ROWS != 0
                || !(1..=4).contains(&shape.waves)
                || self.prefill[..i].contains(shape)
            {
                return Err(invalid("invalid or duplicate prefill shape"));
            }
        }
        if self
            .lanes
            .iter()
            .enumerate()
            .any(|(i, lane)| *lane == 0 || self.lanes[..i].contains(lane))
            || self.embedding.iter().enumerate().any(|(i, home)| {
                self.embedding[..i].contains(home)
                    || matches!(home, EmbeddingHome::Device(gpu) if usize::from(*gpu) >= self.gpus)
            })
        {
            return Err(invalid("invalid or duplicate lane/embedding value"));
        }
        let partitioned = self
            .attention
            .iter()
            .any(|mode| *mode == LayerMode::ContextSplit);
        if self.attention.iter().any(|mode| match mode {
            LayerMode::HeadSplit | LayerMode::ContextSplit => self.gpus != 2,
            LayerMode::Whole { gpu, .. } => usize::from(*gpu) >= self.gpus,
        }) || partitioned && self.page_tokens % (2 * PAGE_ROWS) != 0
        {
            return Err(invalid("attention ownership or context page quantum"));
        }
        let mut contract = Contract {
            resident_weight_limit: request.resident_weight_limit,
            phase_order: crate::placement::planner::PhaseOrder::DRAFT_THEN_TARGET,
            spark_reply_formats: self.spark_links.then(|| SparkReplyFormats {
                prefill: Some(
                    request
                        .spark_reply_formats
                        .prefill(cuteafd_core::ExpertGeometry::GLM5.partial_replies()),
                ),
                decode: Some(request.spark_reply_formats.decode()),
            }),
            page_rows: self.page_tokens,
            maximum_pages: self.maximum_pages,
            gpus: self.gpus,
            variables: self
                .attention
                .iter()
                .enumerate()
                .map(|(layer, &mode)| Variable::Attention {
                    layer,
                    values: vec![mode],
                })
                .collect(),
            requires: vec![],
            components: vec![],
        };
        let prefill = contract.variables.len();
        contract
            .variables
            .push(Variable::Prefill(self.prefill.clone()));
        let lanes = contract.variables.len();
        contract.variables.push(Variable::Lanes(self.lanes.clone()));
        let embedding = contract.variables.len();
        contract
            .variables
            .push(Variable::Embedding(self.embedding.clone()));
        let onboard = self
            .experts
            .as_ref()
            .map(|profile| -> Result<_, Refusal> {
                if profile.layers.is_empty()
                    || profile.counts.is_empty()
                    || profile.layers
                        != (self.cfg.first_moe_layer..self.attention.len()).collect::<Vec<_>>()
                    || profile.layers.iter().collect::<BTreeSet<_>>().len() != profile.layers.len()
                    || profile.destination.is_empty()
                    || profile
                        .destination
                        .iter()
                        .any(|&gpu| usize::from(gpu) >= self.gpus)
                    || profile
                        .destination
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || profile
                        .counts
                        .iter()
                        .any(|&count| count > profile.layers.len())
                    || profile.counts.windows(2).any(|pair| pair[0] >= pair[1])
                    || profile.rank_sets.iter().enumerate().any(|(i, ranks)| {
                        ranks.is_empty()
                            || ranks.windows(2).any(|pair| pair[0] >= pair[1])
                            || profile.rank_sets[..i].contains(ranks)
                    })
                    || profile.rank_sets.is_empty() && profile.counts != [profile.layers.len()]
                {
                    return Err(invalid("invalid canonical expert profile"));
                }
                let variable = contract.variables.len();
                contract.variables.push(Variable::Onboard {
                    layers: profile.layers.clone(),
                    destination: profile.destination.clone(),
                    counts: profile.counts.clone(),
                });
                Ok((variable, profile))
            })
            .transpose()?;
        let remote = onboard.and_then(|(_, profile)| {
            (!profile.rank_sets.is_empty()).then(|| {
                let variable = contract.variables.len();
                contract.variables.push(Variable::ExpertHome {
                    layers: profile.layers.clone(),
                    rank_sets: profile.rank_sets.clone(),
                });
                variable
            })
        });
        let domains = Domains {
            prefill: (prefill, &self.prefill),
            lanes: (lanes, &self.lanes),
            embedding: (embedding, &self.embedding),
            onboard,
            remote,
        };
        for &shape in &self.prefill {
            let owner = super::buffers::sync_h2d_successor(shape)?;
            let bytes = ByteCurve::fixed(owner.bytes, self.maximum_pages);
            domains.append(
                &mut contract,
                &[When::Prefill(shape)],
                vec![Component::allocation(
                    owner,
                    bytes,
                    false,
                    self.maximum_pages,
                )],
            )?;
        }
        for case in &self.targets {
            let target = &case.value;
            let expected: Vec<_> = self
                .attention
                .iter()
                .map(|mode| match mode {
                    LayerMode::Whole { gpu, .. } => *gpu == target.device,
                    _ => true,
                })
                .collect();
            let whole = self
                .attention
                .iter()
                .any(|mode| matches!(mode, LayerMode::Whole { .. }));
            if usize::from(target.device) >= self.gpus
                || target.lead != (target.device == 0)
                || target.page_tokens != self.page_tokens
                || target.maximum_pages != self.maximum_pages
                || target.context != self.context
                || target.partitioned != partitioned
                || target.owned_layers != expected
                || target.split_heads != (self.gpus == 2 && !whole)
                || target.full_prefill_logits != (request.operation == Operation::Scoring)
                || target.prefill.is_some_and(|(shape, _)| {
                    !case
                        .when
                        .iter()
                        .any(|when| matches!(when, When::Prefill(value) if *value == shape))
                })
            {
                return Err(invalid(
                    "target geometry differs from declared execution domains",
                ));
            }
            domains.append(&mut contract, &case.when, target.components(self.cfg)?)?;
        }
        for case in &self.hooks {
            if usize::from(case.value.device) >= self.gpus
                || case.value.maximum_pages != self.maximum_pages
            {
                return Err(invalid("hook geometry differs from contract"));
            }
            domains.append(&mut contract, &case.when, case.value.components()?)?;
        }
        for case in &self.exchanges {
            if self.gpus != 2 || case.value.maximum_pages != self.maximum_pages
                || case.value.context.is_some() != partitioned
                || case.value.context.is_some_and(|context| context.extent != self.context
                    || !case.when.iter().any(|when| matches!(when, When::Prefill(value) if *value == context.prefill))) {
                return Err(invalid("exchange geometry differs from contract"));
            }
            domains.append(&mut contract, &case.when, case.value.components(self.cfg)?)?;
        }
        for case in &self.returns {
            let formats = contract
                .spark_reply_formats
                .ok_or_else(|| invalid("Spark returns without a physical link"))?;
            if usize::from(case.value.device) >= self.gpus
                || matches!(case.value.phase, ReplyPhase::Prefill(shape)
                    if !case.when.iter().any(|when| matches!(when, When::Prefill(value) if *value == shape)))
            {
                return Err(invalid("Spark return geometry differs from contract"));
            }
            domains.append(
                &mut contract,
                &case.when,
                case.value
                    .components(self.cfg, self.maximum_pages, formats)?,
            )?;
        }
        for case in &self.weights {
            if case.value.ranks != self.gpus {
                return Err(invalid("weight rank count differs from contract"));
            }
            let mut components = case.value.components(self.maximum_pages)?;
            components.extend(case.value.loading_components(self.maximum_pages)?);
            domains.append(&mut contract, &case.when, components)?;
        }
        for case in &self.graphs {
            let graph = &case.value;
            if graph.declared.len() != self.gpus
                || graph.context != self.context
                || graph.layers.len() != self.attention.len()
                || self.attention.iter().zip(&graph.layers).enumerate().any(|(index, (mode, layer))| {
                    layer.dense != (index < self.cfg.first_moe_layer) || match (graph.attention, mode) {
                        (AttentionPlacement::Heads, LayerMode::HeadSplit) => layer.attention_owner.is_some(),
                        (AttentionPlacement::Heads, LayerMode::Whole { gpu: 0, .. }) if self.gpus == 1 => layer.attention_owner.is_some(),
                        (AttentionPlacement::Context, LayerMode::ContextSplit) => layer.attention_owner.is_some(),
                        (AttentionPlacement::Layers, LayerMode::Whole { gpu, .. }) => layer.attention_owner != Some(*gpu),
                        _ => true,
                    }
                })
            {
                return Err(invalid("startup graph geometry differs from contract"));
            }
            let activation = domains.activation(&case.when)?;
            for clause in &activation.clauses {
                let counts: Vec<_> = if let Some((variable, profile)) = onboard {
                    profile
                        .counts
                        .iter()
                        .enumerate()
                        .filter(|(option, _)| {
                            clause.iter().all(|choice| {
                                choice.variable != variable || choice.option == *option
                            })
                        })
                        .map(|(_, &count)| count)
                        .collect()
                } else {
                    vec![0]
                };
                for count in counts {
                    let mut tp2 = vec![false; self.attention.len()];
                    if let Some((_, profile)) = onboard {
                        if profile.destination == [0, 1] {
                            for &layer in &profile.layers[..count] {
                                tp2[layer] = true;
                            }
                        }
                    }
                    if !graph.layers.iter().map(|layer| layer.tp2).eq(tp2) {
                        return Err(invalid("startup graph TP2 declaration differs from activation"));
                    }
                }
            }
            domains.append(
                &mut contract,
                &case.when,
                graph.components(self.maximum_pages)?,
            )?;
        }
        for case in &self.owners {
            domains.append(&mut contract, &case.when, case.value.clone())?;
        }
        if let Some(host) = &self.host {
            if host.gpus != self.gpus {
                return Err(invalid("host figure rank count differs from contract"));
            }
            contract.components.retain(|component| !ordinary_heap_subset(component));
            contract.components.extend(host.components(self.maximum_pages)?);
        }
        Ok(contract)
    }
}

#[cfg(test)]
mod tests;
