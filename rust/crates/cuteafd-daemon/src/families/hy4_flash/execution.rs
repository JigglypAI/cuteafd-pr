//! Full 42-layer Hy4-flash coordinator forward pass (1x RTX TP1 and 2x RTX TP2).
use super::cache::Hy4CacheRank;
use super::native::Hy4NativeScratch;
use super::programs::Hy4Programs;
use super::weights::{Hy4EngramHost, Hy4RankWeights};
use crate::shared::experts::rtx::exl3::Exl3Tp2;
use crate::shared::experts::rtx::{ExpertInput, Routes, RtxExpertLayer};
use crate::shared::launch_grid::Fp8QuantizeGrid;
use crate::shared::memory::device::{Allocation, Device, Event, Stream};
use crate::shared::memory::Tag;
use crate::shared::target_head::{self, TargetHead};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::programs::{Program, Programs, Scalar};
use cuteafd_ffi::{Bf16Add, CuteafdDeviceBuffer, F32Add};
use cuteafd_loader::families::hy4_flash::engram::Hy4EngramHistory;
use cuteafd_loader::families::hy4_flash::execution::PAGE_ROWS;
use cuteafd_loader::families::hy4_flash::{Hy4FlashConfig, Hy4KvCache};
use cuteafd_loader::formats::trellis_core::ProjectionScratchLayout;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use std::collections::BTreeMap;

const H: usize = 3072;
const HC: usize = 4;
const Q_LORA: usize = 1024;
const KV_A_DIM: usize = 576;
const KV_A_PADDED: usize = 640;
const LATENT_H: usize = 1536;
const EXPERTS: usize = 512;
const TOPK: usize = 16;
const IDX_HEADS: usize = 16;
const IDX_DIM: usize = 128;
const CANDIDATE_TOPK: usize = 2048;

/// Per-rank sequential scratch workspace for one Hy4-flash prefill/decode chunk.
struct RankScratch<'a> {
    ids: Allocation<'a>,
    positions_i64: Allocation<'a>,
    positions_i32: Allocation<'a>,
    dsa_slots_i64: Allocation<'a>,
    swa_slots_i64: Allocation<'a>,
    r_a: Allocation<'a>,
    r_b: Allocation<'a>,
    post: Allocation<'a>,
    post_ffn: Allocation<'a>,
    y: Allocation<'a>,
    delta: Allocation<'a>,
    delta_f32: Allocation<'a>,
    delta_peer_f32: Allocation<'a>,
    delta_sum_f32: Allocation<'a>,
    delta_norm: Allocation<'a>,
    // Engram (one staging slot per Engram layer so all 4 upload once before the layer loop)
    engram_raw: Vec<Allocation<'a>>,
    engram_masked: Vec<Allocation<'a>>,
    engram_normed: Allocation<'a>,
    engram_val: Allocation<'a>,
    engram_keys: Allocation<'a>,
    // MLA
    q_a: Allocation<'a>,
    kv_a: Allocation<'a>,
    kv_a_padded: Allocation<'a>,
    q_resid: Allocation<'a>,
    q_raw: Allocation<'a>,
    gate: Allocation<'a>,
    query: Allocation<'a>,
    attn_out: Allocation<'a>,
    lse_out: Allocation<'a>,
    uv_out: Allocation<'a>,
    attn_scratch: Allocation<'a>,
    dsa_page_table: Allocation<'a>,
    swa_page_table: Allocation<'a>,
    swa_seqlens: Allocation<'a>,
    dsa_cu_seqlens: Allocation<'a>,
    swa_cu_seqlens: Allocation<'a>,
    // Indexer
    idx_q_raw: Allocation<'a>,
    idx_q_rot: Allocation<'a>,
    idx_w_raw: Allocation<'a>,
    idx_w_out: Allocation<'a>,
    idx_tk_raw: Allocation<'a>,
    idx_tk: Allocation<'a>,
    idx_group_tok: Allocation<'a>,
    idx_group_pos: Allocation<'a>,
    idx_group_slots: Allocation<'a>,
    idx_scores: Allocation<'a>,
    idx_limits: Allocation<'a>,
    idx_cand_slots: Allocation<'a>,
    idx_cand_lens: Allocation<'a>,
    // FFN / LatentMoE
    ffn_gate: Allocation<'a>,
    ffn_up: Allocation<'a>,
    ffn_act: Allocation<'a>,
    router_logits: Allocation<'a>,
    router_select: Allocation<'a>,
    route_ids: Allocation<'a>,
    route_weights: Allocation<'a>,
    latent_in: Allocation<'a>,
    wire_in: Allocation<'a>,
    routed_latent: Allocation<'a>,
    routed_out: Allocation<'a>,
    shared_out: Allocation<'a>,
    shared_f32: Allocation<'a>,
    // Head logits tile (<=64 rows)
    logits_tile: Allocation<'a>,
    // Compact-core serialized scratch (`k8-compact`, `k6-compact`, `fp8-compact`)
    native: Option<Hy4NativeScratch<'a>>,
}

impl<'a> RankScratch<'a> {
    fn new(
        device: Device<'a>,
        tp: usize,
        max_rows: usize,
        max_pages: usize,
        engram_count: usize,
        local_vocab: usize,
        attn_scratch_bytes: usize,
        native_layout: Option<&ProjectionScratchLayout>,
    ) -> Result<Self> {
        let alloc = |bytes: usize, label: &'static str| {
            Allocation::new(device, bytes.max(256), Tag::workspace(label))
        };
        let max_heads = 64 / tp;
        let max_inter = 13824 / tp;
        let max_groups = max_pages * (PAGE_ROWS / 4);
        let wire_row = LATENT_H + LATENT_H / 32;
        let native = native_layout
            .map(|l| Hy4NativeScratch::allocate(device, l))
            .transpose()?;
        let mut engram_raw = Vec::with_capacity(engram_count);
        let mut engram_masked = Vec::with_capacity(engram_count);
        for _ in 0..engram_count {
            engram_raw.push(alloc(max_rows * 16 * 96 * 2, "engram-raw")?);
            engram_masked.push(alloc(max_rows, "engram-masked")?);
        }
        Ok(Self {
            ids: alloc(max_rows * 4, "ids")?,
            positions_i64: alloc(max_rows * 8, "pos-i64")?,
            positions_i32: alloc(max_rows * 4, "pos-i32")?,
            dsa_slots_i64: alloc(max_rows * 8, "dsa-slots-i64")?,
            swa_slots_i64: alloc(max_rows * 8, "swa-slots-i64")?,
            r_a: alloc(max_rows * HC * H * 2, "r-a")?,
            r_b: alloc(max_rows * HC * H * 2, "r-b")?,
            post: alloc(max_rows * HC * 4, "post")?,
            post_ffn: alloc(max_rows * HC * 4, "post-ffn")?,
            y: alloc(max_rows * H * 2, "y")?,
            delta: alloc(max_rows * H * 2, "delta")?,
            delta_f32: alloc(max_rows * H * 4, "delta-f32")?,
            delta_peer_f32: alloc(max_rows * H * 4, "delta-peer-f32")?,
            delta_sum_f32: alloc(max_rows * H * 4, "delta-sum-f32")?,
            delta_norm: alloc(max_rows * H * 2, "delta-norm")?,
            engram_raw,
            engram_masked,
            engram_normed: alloc(max_rows * 1536 * 2, "engram-normed")?,
            engram_val: alloc(max_rows * H * 2, "engram-val")?,
            engram_keys: alloc(max_rows * HC * H * 2, "engram-keys")?,
            q_a: alloc(max_rows * Q_LORA * 2, "q-a")?,
            kv_a: alloc(max_rows * KV_A_DIM * 2, "kv-a")?,
            kv_a_padded: alloc(max_rows * KV_A_PADDED * 2, "kv-a-padded")?,
            q_resid: alloc(max_rows * Q_LORA * 2, "q-resid")?,
            q_raw: alloc(max_rows * max_heads * 256 * 2, "q-raw")?,
            gate: alloc(max_rows * max_heads * 256 * 2, "gate")?,
            query: alloc(max_rows * max_heads * 576 * 2, "query")?,
            attn_out: alloc(max_rows * max_heads * 512 * 2, "attn-out")?,
            lse_out: alloc(max_rows * max_heads * 4, "lse-out")?,
            uv_out: alloc(max_rows * max_heads * 256 * 2, "uv-out")?,
            attn_scratch: alloc(attn_scratch_bytes, "attn-scratch")?,
            dsa_page_table: alloc(max_rows * max_pages.max(16) * 4, "dsa-page-table")?,
            swa_page_table: alloc(max_pages.max(16) * 4, "swa-page-table")?,
            swa_seqlens: alloc(4, "swa-seqlens")?,
            dsa_cu_seqlens: alloc((max_rows + 1) * 4, "dsa-cu-seqlens")?,
            swa_cu_seqlens: alloc(2 * 4, "swa-cu-seqlens")?,
            idx_q_raw: alloc(max_rows * IDX_HEADS * IDX_DIM * 2, "idx-q-raw")?,
            idx_q_rot: alloc(max_rows * IDX_HEADS * IDX_DIM * 2, "idx-q-rot")?,
            idx_w_raw: alloc(max_rows * IDX_HEADS * 2, "idx-w-raw")?,
            idx_w_out: alloc(max_rows * IDX_HEADS * 4, "idx-w-out")?,
            idx_tk_raw: alloc(max_rows * IDX_DIM * 2, "idx-tk-raw")?,
            idx_tk: alloc(max_rows * IDX_DIM * 2, "idx-tk")?,
            idx_group_tok: alloc((max_rows + 4) * IDX_DIM * 2, "idx-group-tok")?,
            idx_group_pos: alloc((max_rows / 4 + 2) * 8, "idx-group-pos")?,
            idx_group_slots: alloc((max_rows / 4 + 2) * 8, "idx-group-slots")?,
            idx_scores: alloc(max_rows * max_groups.max(1) * 4, "idx-scores")?,
            idx_limits: alloc(max_rows * 4, "idx-limits")?,
            idx_cand_slots: alloc(max_rows * CANDIDATE_TOPK * 4, "idx-cand-slots")?,
            idx_cand_lens: alloc(max_rows * 4, "idx-cand-lens")?,
            ffn_gate: alloc(max_rows * max_inter * 2, "ffn-gate")?,
            ffn_up: alloc(max_rows * max_inter * 2, "ffn-up")?,
            ffn_act: alloc(max_rows * max_inter * 2, "ffn-act")?,
            router_logits: alloc(max_rows * EXPERTS * 4, "router-logits")?,
            router_select: alloc(max_rows * EXPERTS * 4, "router-select")?,
            route_ids: alloc(max_rows * TOPK * 4, "route-ids")?,
            route_weights: alloc(max_rows * TOPK * 4, "route-weights")?,
            latent_in: alloc(max_rows * LATENT_H * 2, "latent-in")?,
            wire_in: alloc(max_rows * wire_row, "wire-in")?,
            routed_latent: alloc(max_rows * LATENT_H * 2, "routed-latent")?,
            routed_out: alloc(max_rows * H * 2, "routed-out")?,
            shared_out: alloc(max_rows * H * 2, "shared-out")?,
            shared_f32: alloc(max_rows * H * 4, "shared-f32")?,
            logits_tile: alloc(64 * local_vocab * 4, "logits-tile")?,
            native,
        })
    }
}

fn launch_proj_out(
    r: &RankRuntime<'_>,
    stem: &str,
    bf16_prog: &Program<'_>,
    rows: usize,
    input: CuteafdDeviceBuffer,
    output: CuteafdDeviceBuffer,
) -> Result<()> {
    if let Some(linear) = r.weights.native.get(stem) {
        let ns = r.scratch.native.as_ref().context("Hy4 native core missing scratch")?;
        return linear.launch(rows, input, output, &ns.scratch, r.stream.raw);
    }
    // SAFETY: caller ensures device active and all buffers live through stream drain.
    unsafe {
        bf16_prog.launch(
            &[input.ptr, r.weights.weight(stem)?.ptr, output.ptr],
            &[Scalar::I32(rows as i32)],
            r.stream.raw,
        )
    }
}

fn launch_proj_kv_a(
    r: &RankRuntime<'_>,
    stem: &str,
    rows: usize,
    input: CuteafdDeviceBuffer,
    output_576: CuteafdDeviceBuffer,
) -> Result<()> {
    if let Some(linear) = r.weights.native.get(stem) {
        let ns = r.scratch.native.as_ref().context("Hy4 native core missing scratch")?;
        linear.launch(rows, input, r.scratch.kv_a_padded.buffer, &ns.scratch, r.stream.raw)?;
        // SAFETY: kv_a_padded [rows, 640] BF16 and output_576 [rows, 576] BF16 are disjoint admitted buffers.
        return unsafe {
            r.device.library.copy_d2d_2d_async(
                output_576,
                KV_A_DIM * 2,
                r.scratch.kv_a_padded.buffer,
                KV_A_PADDED * 2,
                KV_A_DIM * 2,
                rows,
                r.stream.raw,
            )
        };
    }
    // SAFETY: caller ensures device active and all buffers live through stream drain.
    unsafe {
        r.programs.proj_kv_a.launch(
            &[input.ptr, r.weights.weight(stem)?.ptr, output_576.ptr],
            &[Scalar::I32(rows as i32)],
            r.stream.raw,
        )
    }
}

fn launch_proj_split(
    r: &RankRuntime<'_>,
    gpus: usize,
    stem: &str,
    bf16_prog: &Program<'_>,
    rows: usize,
    input: CuteafdDeviceBuffer,
    out_tp1_bf16: CuteafdDeviceBuffer,
    out_tp2_f32: CuteafdDeviceBuffer,
) -> Result<()> {
    if let Some(linear) = r.weights.native.get(stem) {
        let ns = r.scratch.native.as_ref().context("Hy4 native core missing scratch")?;
        return if gpus == 1 {
            linear.launch(rows, input, out_tp1_bf16, &ns.scratch, r.stream.raw)
        } else {
            linear.partial(rows, input, out_tp2_f32, &ns.scratch, r.stream.raw)
        };
    }
    let out = if gpus == 1 { out_tp1_bf16.ptr } else { out_tp2_f32.ptr };
    // SAFETY: caller ensures device active and all buffers live through stream drain.
    unsafe {
        bf16_prog.launch(
            &[input.ptr, r.weights.weight(stem)?.ptr, out],
            &[Scalar::I32(rows as i32)],
            r.stream.raw,
        )
    }
}

struct RankRuntime<'a> {
    device: Device<'a>,
    stream: Stream<'a>,
    peer_ready: Event<'a>,
    programs: Hy4Programs<'a>,
    weights: Hy4RankWeights<'a>,
    cache: Hy4CacheRank<'a>,
    scratch: RankScratch<'a>,
    bf16_add: Bf16Add<'a>,
    f32_add: F32Add<'a>,
    quantize_grid: Fp8QuantizeGrid,
    onboarded_experts: Option<Exl3Tp2<'a>>,
}

/// Coordinator-only Hy4-flash execution engine (1x RTX TP1 or 2x RTX TP2).
pub(crate) struct Hy4Engine<'a> {
    pub(crate) cfg: Hy4FlashConfig,
    pub(crate) max_rows: usize,
    pub(crate) max_pages: usize,
    pub(crate) engram: Hy4EngramHost,
    pub(crate) skip_routed_experts: bool,
    pub(crate) dump_dir: Option<std::path::PathBuf>,
    ranks: Vec<RankRuntime<'a>>,
    dsa_index_of: BTreeMap<usize, usize>,
    swa_index_of: BTreeMap<usize, usize>,
    engram_index_of: BTreeMap<usize, usize>,
    engram_histories: BTreeMap<usize, Hy4EngramHistory>,
    committed_pos: usize,
}

impl<'a> Hy4Engine<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        devices: &[Device<'a>],
        programs: &'a Programs<'a>,
        checkpoint: &Checkpoint,
        cfg: Hy4FlashConfig,
        kv_format: Hy4KvCache,
        max_rows: usize,
        max_pages: usize,
        force_engram_bf16: bool,
        skip_routed_experts: bool,
        mut onboarded_experts: Vec<Option<Exl3Tp2<'a>>>,
    ) -> Result<Self> {
        let gpus = devices.len();
        ensure!(matches!(gpus, 1 | 2), "Hy4Engine supports 1 or 2 GPUs, got {gpus}");
        ensure!(onboarded_experts.len() == gpus, "onboarded_experts rank count mismatch");
        if gpus == 2 {
            let probed = crate::shared::peer_split::probed_device(
                devices[0].library,
                devices[0].id,
                Some(devices[1].id),
            )?;
            ensure!(
                probed == Some(devices[1].id),
                "Hy4Engine TP2 requires bidirectional GPU-to-GPU peer access between devices {} and {}",
                devices[0].id,
                devices[1].id
            );
        }

        let weights_all = Hy4RankWeights::load_all(devices, programs, checkpoint, &cfg, max_rows, max_rows.min(64))?;
        // SAFETY: checkpoint shard files remain immutable for the duration of the engine.
        let engram = unsafe { Hy4EngramHost::open(checkpoint, &cfg, force_engram_bf16)? };
        let engram_histories = engram.start_histories();

        let mut ranks = Vec::with_capacity(gpus);
        for (rank, weights) in weights_all.into_iter().enumerate() {
            let device = devices[rank];
            let stream = Stream::new(device)?;
            let peer_ready = Event::new(device)?;
            let progs = device.run(|| Hy4Programs::load(programs, gpus, kv_format, max_rows))?;
            let attn_scratch = progs
                .dsa_scratch_bytes
                .max(progs.swa_decode_scratch_bytes)
                .max(progs.swa_extend_scratch_bytes);
            let cache = Hy4CacheRank::new(device, &cfg, kv_format, max_pages, 1, max_rows, 0, &stream)?;
            let native_layout = weights.native.scratch_layout(max_rows)?;
            let scratch = RankScratch::new(
                device,
                gpus,
                max_rows,
                max_pages,
                cfg.engram_layers.len(),
                weights.local_vocab,
                attn_scratch,
                native_layout.as_ref(),
            )?;
            let bf16_add = device.run(|| device.library.bf16_add())?;
            let f32_add = device.run(|| device.library.f32_add())?;
            let quantize_grid = Fp8QuantizeGrid::new(device.run(|| device.library.sm_count())?, None)?;
            ranks.push(RankRuntime {
                device,
                stream,
                peer_ready,
                programs: progs,
                weights,
                cache,
                scratch,
                bf16_add,
                f32_add,
                quantize_grid,
                onboarded_experts: onboarded_experts[rank].take(),
            });
        }

        let dsa_index_of: BTreeMap<usize, usize> = cfg
            .sparse_layers()
            .into_iter()
            .enumerate()
            .map(|(i, l)| (l, i))
            .collect();
        let swa_index_of: BTreeMap<usize, usize> = cfg
            .swa_layers()
            .into_iter()
            .enumerate()
            .map(|(i, l)| (l, i))
            .collect();
        let engram_index_of: BTreeMap<usize, usize> = cfg
            .engram_layers
            .iter()
            .copied()
            .enumerate()
            .map(|(i, l)| (l, i))
            .collect();

        Ok(Self {
            cfg,
            max_rows,
            max_pages,
            engram,
            skip_routed_experts,
            dump_dir: None,
            ranks,
            dsa_index_of,
            swa_index_of,
            engram_index_of,
            engram_histories,
            committed_pos: 0,
        })
    }

    fn dump_bf16(&self, name: &str, buf: CuteafdDeviceBuffer, elems: usize) -> Result<()> {
        let Some(dir) = &self.dump_dir else { return Ok(()); };
        let r0 = &self.ranks[0];
        r0.stream.drain()?;
        std::fs::create_dir_all(dir)?;
        let mut raw = vec![0u8; elems * 2];
        r0.device.run(|| {
            r0.device.library.copy_d2h(&mut raw, CuteafdDeviceBuffer { bytes: elems * 2, ..buf })
        })?;
        let f32_bytes: Vec<u8> = raw
            .chunks_exact(2)
            .flat_map(|b| (u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes())
            .collect();
        std::fs::write(dir.join(format!("{name}.f32")), f32_bytes)?;
        Ok(())
    }

    fn dump_f32(&self, name: &str, buf: CuteafdDeviceBuffer, elems: usize) -> Result<()> {
        let Some(dir) = &self.dump_dir else { return Ok(()); };
        let r0 = &self.ranks[0];
        r0.stream.drain()?;
        std::fs::create_dir_all(dir)?;
        let mut raw = vec![0u8; elems * 4];
        r0.device.run(|| {
            r0.device.library.copy_d2h(&mut raw, CuteafdDeviceBuffer { bytes: elems * 4, ..buf })
        })?;
        std::fs::write(dir.join(format!("{name}.f32")), raw)?;
        Ok(())
    }

    pub(crate) fn reset(&mut self) -> Result<()> {
        for r in &mut self.ranks {
            r.cache.zero_all(&r.stream)?;
            for seq in &mut r.cache.pending_counts {
                seq.fill(0);
            }
        }
        self.engram_histories = self.engram.start_histories();
        self.committed_pos = 0;
        Ok(())
    }

    /// Device-driven GPU-to-GPU P2P sum of `scratch.delta_f32 [rows, width]` across TP2 ranks,
    /// followed by native `egress` (`svh` + BF16 narrow for EXL3, or BF16 narrow for FP8/BF16)
    /// into `scratch.delta [rows, width]` on both ranks with zero host hop.
    fn reduce_tp2_f32_to_bf16(&mut self, rows: usize, width: usize, egress_stem: Option<&str>) -> Result<()> {
        if self.ranks.len() == 1 {
            return Ok(());
        }
        let elems = rows * width;
        let bytes = elems * 4;
        for r in &mut self.ranks {
            r.peer_ready.record(&r.stream)?;
        }
        let ev0 = self.ranks[0].peer_ready.raw;
        let ev1 = self.ranks[1].peer_ready.raw;
        let src0 = CuteafdDeviceBuffer { bytes, ..self.ranks[0].scratch.delta_f32.buffer };
        let src1 = CuteafdDeviceBuffer { bytes, ..self.ranks[1].scratch.delta_f32.buffer };
        // Wait for peer partial to finish, then copy GPU-to-GPU directly into delta_peer_f32.
        self.ranks[0].device.run(|| unsafe {
            self.ranks[0].device.library.cuda_stream_wait_event(self.ranks[0].stream.raw, ev1)?;
            self.ranks[0].device.library.copy_peer_async(
                CuteafdDeviceBuffer { bytes, ..self.ranks[0].scratch.delta_peer_f32.buffer },
                src1,
                bytes,
                self.ranks[0].stream.raw,
            )
        })?;
        self.ranks[1].device.run(|| unsafe {
            self.ranks[1].device.library.cuda_stream_wait_event(self.ranks[1].stream.raw, ev0)?;
            self.ranks[1].device.library.copy_peer_async(
                CuteafdDeviceBuffer { bytes, ..self.ranks[1].scratch.delta_peer_f32.buffer },
                src0,
                bytes,
                self.ranks[1].stream.raw,
            )
        })?;
        // Ensure both peer copies finished reading the source buffers before summing in-place.
        for r in &mut self.ranks {
            r.peer_ready.record(&r.stream)?;
        }
        let cp0 = self.ranks[0].peer_ready.raw;
        let cp1 = self.ranks[1].peer_ready.raw;
        let waits = [cp1, cp0];
        for (idx, r) in self.ranks.iter().enumerate() {
            let wait_ev = waits[idx];
            r.device.run(|| {
                // SAFETY: peer copy into delta_peer_f32 completed before wait_ev; all three buffers are disjoint.
                unsafe {
                    r.device.library.cuda_stream_wait_event(r.stream.raw, wait_ev)?;
                    r.f32_add.launch(
                        r.scratch.delta_f32.buffer,
                        r.scratch.delta_peer_f32.buffer,
                        r.scratch.delta_sum_f32.buffer,
                        elems,
                        r.stream.raw,
                    )?;
                }
                if let Some(linear) = egress_stem.and_then(|s| r.weights.native.get(s)) {
                    linear.egress(rows, r.scratch.delta_sum_f32.buffer, r.scratch.delta.buffer, r.stream.raw)
                } else {
                    // SAFETY: delta_sum_f32 sum finished in stream order before narrowing into disjoint delta.
                    unsafe {
                        r.device.library.cuda_f32_to_bf16_async(
                            r.scratch.delta_sum_f32.buffer,
                            r.scratch.delta.buffer,
                            elems,
                            r.stream.raw,
                        )
                    }
                }
            })?;
        }
        Ok(())
    }

    /// Run a contiguous chunk of `tokens` starting at `self.committed_pos` through layers `0..=last_layer`.
    /// Returns FP32 logits `[tokens.len(), vocab_size]` if `last_layer == self.cfg.layers - 1`, else empty.
    pub(crate) fn forward_chunk(&mut self, tokens: &[u32], last_layer: usize) -> Result<Vec<f32>> {
        let rows = tokens.len();
        ensure!(rows > 0 && rows <= self.max_rows, "Hy4 chunk rows {rows} outside 1..={}", self.max_rows);
        ensure!(last_layer < self.cfg.layers, "last_layer {last_layer} >= {}", self.cfg.layers);
        let pos_start = self.committed_pos;
        let pos_end = pos_start + rows;
        let total_pages = pos_end.div_ceil(PAGE_ROWS);
        ensure!(total_pages <= self.max_pages, "sequence length {pos_end} exceeds cache pages {}", self.max_pages);

        // 1. Upload positions and DSA/SWA KV slots; embed tokens into 4-stream iHC residual r_a.
        let swa_ring_pages = self.ranks[0].cache.swa_ring_pages;
        let swa_ring_rows = self.ranks[0].cache.swa_ring_rows;
        let pos_i64: Vec<u8> = (pos_start..pos_end).flat_map(|p| (p as i64).to_le_bytes()).collect();
        let pos_i32: Vec<u8> = (pos_start..pos_end).flat_map(|p| (p as i32).to_le_bytes()).collect();
        let swa_slots_i64: Vec<u8> = (pos_start..pos_end)
            .flat_map(|p| ((p % swa_ring_rows) as i64).to_le_bytes())
            .collect();
        let cu_singleton: Vec<u8> = (0..=rows).flat_map(|i| (i as i32).to_le_bytes()).collect();
        let identity_pt_row: Vec<i32> = (0..total_pages as i32).collect();
        let mut dsa_pt_bytes = Vec::with_capacity(rows * total_pages * 4);
        for _ in 0..rows {
            for &p in &identity_pt_row {
                dsa_pt_bytes.extend_from_slice(&p.to_le_bytes());
            }
        }
        let swa_pt_bytes: Vec<u8> = (0..total_pages)
            .flat_map(|p| ((p % swa_ring_pages) as i32).to_le_bytes())
            .collect();
        let swa_seqlens_bytes: Vec<u8> = (pos_end as i32).to_le_bytes().to_vec();
        let swa_cu_bytes: Vec<u8> = [0i32, rows as i32].into_iter().flat_map(i32::to_le_bytes).collect();
        let limits_bytes: Vec<u8> = (pos_start..pos_end).flat_map(|p| ((p / 4) as i32).to_le_bytes()).collect();
        // Every DSA layer processes the same token stream in lockstep, so pending/complete_groups match.
        let pending_common = self.ranks[0].cache.pending_counts[0].first().copied().unwrap_or(0);
        let complete_groups_common = (pending_common + rows) / 4;
        let (g_pos_bytes, g_slots_bytes) = if complete_groups_common > 0 {
            let first_gid = (pos_start - pending_common) / 4;
            let gp: Vec<u8> = (0..complete_groups_common).flat_map(|g| (((first_gid + g) * 4) as i64).to_le_bytes()).collect();
            let gs: Vec<u8> = (0..complete_groups_common).flat_map(|g| ((first_gid + g) as i64).to_le_bytes()).collect();
            (gp, gs)
        } else {
            (Vec::new(), Vec::new())
        };

        // Precompute Engram lookups on host for every Engram layer we will visit.
        let mut engram_chunks: BTreeMap<usize, (Vec<u8>, Vec<u8>)> = BTreeMap::new();
        for &layer in &self.cfg.engram_layers {
            if layer <= last_layer {
                let hist = self.engram_histories.get_mut(&layer).context("missing engram history")?;
                let (bf16, masked) = self.engram.lookup_chunk(layer, hist, tokens)?;
                let raw_bytes: Vec<u8> = bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
                engram_chunks.insert(layer, (raw_bytes, masked));
            }
        }

        // Embed on rank 0 into r_a [rows, 4, H] BF16, replicate GPU-to-GPU to rank 1 if TP2.
        {
            let r0 = &mut self.ranks[0];
            let emb = r0.weights.embedding.as_ref().context("rank 0 missing TokenEmbedding")?;
            r0.device.run(|| {
                emb.embed(tokens, r0.scratch.ids.buffer, HC, r0.scratch.r_a.buffer, r0.stream.raw)
            })?;
            r0.peer_ready.record(&r0.stream)?;
        }
        if self.ranks.len() == 2 {
            let bytes = rows * HC * H * 2;
            let ev0 = self.ranks[0].peer_ready.raw;
            let src0 = CuteafdDeviceBuffer { bytes, ..self.ranks[0].scratch.r_a.buffer };
            self.ranks[1].device.run(|| unsafe {
                self.ranks[1].device.library.cuda_stream_wait_event(self.ranks[1].stream.raw, ev0)?;
                self.ranks[1].device.library.copy_peer_async(
                    CuteafdDeviceBuffer { bytes, ..self.ranks[1].scratch.r_a.buffer },
                    src0,
                    bytes,
                    self.ranks[1].stream.raw,
                )
            })?;
        }
        // Upload every chunk metadata & Engram staging buffer ONCE on each rank's non-blocking
        // stream with copy_h2d_ordered (whole admitted destinations, zero stream-0 races).
        for r in &self.ranks {
            r.device.run(|| unsafe {
                let s = r.stream.raw;
                r.device.library.copy_h2d_ordered(r.scratch.positions_i64.buffer, &pos_i64, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.positions_i32.buffer, &pos_i32, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.dsa_slots_i64.buffer, &pos_i64, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.swa_slots_i64.buffer, &swa_slots_i64, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.dsa_page_table.buffer, &dsa_pt_bytes, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.swa_page_table.buffer, &swa_pt_bytes, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.dsa_cu_seqlens.buffer, &cu_singleton, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.swa_cu_seqlens.buffer, &swa_cu_bytes, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.swa_seqlens.buffer, &swa_seqlens_bytes, s)?;
                r.device.library.copy_h2d_ordered(r.scratch.idx_limits.buffer, &limits_bytes, s)?;
                if !g_pos_bytes.is_empty() {
                    r.device.library.copy_h2d_ordered(r.scratch.idx_group_pos.buffer, &g_pos_bytes, s)?;
                    r.device.library.copy_h2d_ordered(r.scratch.idx_group_slots.buffer, &g_slots_bytes, s)?;
                }
                for (&layer, (raw_bytes, masked)) in &engram_chunks {
                    let slot = self.engram_index_of[&layer];
                    r.device.library.copy_h2d_ordered(r.scratch.engram_raw[slot].buffer, raw_bytes, s)?;
                    r.device.library.copy_h2d_ordered(r.scratch.engram_masked[slot].buffer, masked, s)?;
                }
                Ok(())
            })?;
        }

        let dump_rows = if self.dump_dir.is_some() && pos_start == 0 {
            rows.min(16)
        } else {
            0
        };
        if dump_rows > 0 {
            let dir = self.dump_dir.as_ref().unwrap();
            std::fs::create_dir_all(dir)?;
            let tok_bytes: Vec<u8> = tokens[..dump_rows].iter().flat_map(|t| t.to_le_bytes()).collect();
            std::fs::write(dir.join("tokens.u32"), tok_bytes)?;
            self.dump_bf16("embed", self.ranks[0].scratch.r_a.buffer, dump_rows * HC * H)?;
        }

        let mut use_a = true;
        let gpus = self.ranks.len();

        for layer in 0..=last_layer {
            let p = format!("model.layers.{layer}");
            let is_sparse = self.cfg.sparse_layer[layer];
            let attn = self.cfg.attention(layer);
            let local_heads = attn.heads / gpus;

            // (A) Engram layer (in-place update on active residual stream r_cur)
            if let Some(&slot) = self.engram_index_of.get(&layer) {
                for r in &self.ranks {
                    let r_cur = if use_a { r.scratch.r_a.buffer } else { r.scratch.r_b.buffer };
                    r.device.run(|| {
                        let ep = format!("{p}.engram_layer");
                        let s = [Scalar::I32(rows as i32)];
                        // SAFETY: admitted arena weight views & rank scratch live through stream drain.
                        unsafe {
                            r.programs.engram_head_norm.launch(
                                &[
                                    r.scratch.engram_raw[slot].buffer.ptr,
                                    r.weights.buf(&format!("{ep}.lookup_per_head_norms.packed"))?.ptr,
                                    r.scratch.engram_normed.buffer.ptr,
                                ],
                                &s, r.stream.raw,
                            )?;
                            r.programs.proj_engram_value.launch(
                                &[r.scratch.engram_normed.buffer.ptr, r.weights.weight(&format!("{ep}.value_proj"))?.ptr, r.scratch.engram_val.buffer.ptr],
                                &s, r.stream.raw,
                            )?;
                            r.programs.proj_engram_key.launch(
                                &[r.scratch.engram_normed.buffer.ptr, r.weights.buf(&format!("{ep}.key_projs.packed.weight"))?.ptr, r.scratch.engram_keys.buffer.ptr],
                                &s, r.stream.raw,
                            )?;
                            r.programs.engram_gate_add.launch(
                                &[
                                    r_cur.ptr,
                                    r.scratch.engram_keys.buffer.ptr,
                                    r.weights.buf(&format!("{ep}.key_projs.packed.bias"))?.ptr,
                                    r.scratch.engram_val.buffer.ptr,
                                    r.weights.buf(&format!("{ep}.value_proj.bias"))?.ptr,
                                    r.weights.buf(&format!("{ep}.norm_key.packed"))?.ptr,
                                    r.weights.buf(&format!("{ep}.norm_query.packed"))?.ptr,
                                    r.weights.buf(&format!("{ep}.unified_pre_gamma"))?.ptr,
                                    r.weights.buf(&format!("{ep}.unified_per_hc_scales.packed"))?.ptr,
                                    r.scratch.engram_masked[slot].buffer.ptr,
                                ],
                                &s, r.stream.raw,
                            )?;
                        }
                        Ok(())
                    })?;
                }
                if dump_rows > 0 && layer <= 3 {
                    let r_cur = if use_a { self.ranks[0].scratch.r_a.buffer } else { self.ranks[0].scratch.r_b.buffer };
                    self.dump_bf16(&format!("l{layer:02}_after_engram"), r_cur, dump_rows * HC * H)?;
                }
            }

            // (B) Attention iHC pre + absorbed MLA + compressed indexer (if DSA) + sink/gate W_UV + o_proj
            for r in &mut self.ranks {
                let r_cur = if use_a { r.scratch.r_a.buffer } else { r.scratch.r_b.buffer };
                let s_rows = [Scalar::I32(rows as i32)];
                let inv_freq = if is_sparse { r.weights.inv_freq_dsa.buffer.ptr } else { r.weights.inv_freq_swa.buffer.ptr };
                let kv_slots_ptr = if is_sparse {
                    r.scratch.dsa_slots_i64.buffer.ptr
                } else {
                    r.scratch.swa_slots_i64.buffer.ptr
                };
                let kv_cache_buf = if is_sparse {
                    r.cache.dsa_kv_layer(self.dsa_index_of[&layer])?
                } else {
                    r.cache.swa_ring(0, self.swa_index_of[&layer])?
                };
                let (w_uk, w_uv) = r.weights.kv_b_absorbed(layer, local_heads)?;
                let dev = r.device;

                dev.run(|| {
                    // SAFETY: all arena weight views, cache planes, and scratch stay live through stream drain.
                    unsafe {
                        r.programs.ihc_pre.launch(
                            &[
                                r_cur.ptr,
                                r.weights.buf(&format!("{p}.hc_attn_layer.hc_pre.hc_fn"))?.ptr,
                                r.weights.buf(&format!("{p}.hc_attn_layer.hc_pre.hc_scale"))?.ptr,
                                r.weights.buf(&format!("{p}.hc_attn_layer.hc_pre.hc_base"))?.ptr,
                                r.weights.buf(&format!("{p}.input_layernorm.weight"))?.ptr,
                                r.scratch.post.buffer.ptr,
                                r.scratch.y.buffer.ptr,
                            ],
                            &s_rows, r.stream.raw,
                        )?;
                        launch_proj_out(
                            r,
                            &format!("{p}.self_attn.q_a_proj"),
                            &r.programs.proj_q_a,
                            rows,
                            r.scratch.y.buffer,
                            r.scratch.q_a.buffer,
                        )?;
                        launch_proj_kv_a(
                            r,
                            &format!("{p}.self_attn.kv_a_proj_with_mqa"),
                            rows,
                            r.scratch.y.buffer,
                            r.scratch.kv_a.buffer,
                        )?;
                        r.programs.mla_rank_norm_pack.launch(
                            &[
                                r.scratch.q_a.buffer.ptr,
                                r.scratch.kv_a.buffer.ptr,
                                r.weights.buf(&format!("{p}.self_attn.q_a_layernorm.weight"))?.ptr,
                                r.weights.buf(&format!("{p}.self_attn.kv_a_layernorm.weight"))?.ptr,
                                kv_slots_ptr,
                                r.scratch.positions_i64.buffer.ptr,
                                inv_freq,
                                r.scratch.q_resid.buffer.ptr,
                                kv_cache_buf.ptr,
                            ],
                            &s_rows, r.stream.raw,
                        )?;
                        let (proj_qb, proj_gate, q_abs, sink_uv, proj_o) = if is_sparse {
                            (&r.programs.proj_q_b_dsa, &r.programs.proj_linear_gate_dsa, &r.programs.mla_q_absorb_dsa, &r.programs.mla_sink_gate_uv_dsa, &r.programs.proj_o_dsa)
                        } else {
                            (&r.programs.proj_q_b_swa, &r.programs.proj_linear_gate_swa, &r.programs.mla_q_absorb_swa, &r.programs.mla_sink_gate_uv_swa, &r.programs.proj_o_swa)
                        };
                        launch_proj_out(
                            r,
                            &format!("{p}.self_attn.q_b_proj"),
                            proj_qb,
                            rows,
                            r.scratch.q_resid.buffer,
                            r.scratch.q_raw.buffer,
                        )?;
                        launch_proj_out(
                            r,
                            &format!("{p}.self_attn.linear_gate"),
                            proj_gate,
                            rows,
                            r.scratch.y.buffer,
                            r.scratch.gate.buffer,
                        )?;
                        q_abs.launch(
                            &[r.scratch.q_raw.buffer.ptr, w_uk, r.scratch.positions_i64.buffer.ptr, inv_freq, r.scratch.query.buffer.ptr],
                            &s_rows, r.stream.raw,
                        )?;

                        if is_sparse {
                            let dsa_idx = self.dsa_index_of[&layer];
                            let idx_cache = r.cache.dsa_index_layer(dsa_idx)?;
                            let ip = format!("{p}.self_attn.compressed_indexer");
                            // Token keys + RMSNorm -> complete 4-token groups into group_cache.
                            r.programs.proj_indexer_wk.launch(
                                &[r.scratch.y.buffer.ptr, r.weights.weight(&format!("{ip}.token_key_projection"))?.ptr, r.scratch.idx_tk_raw.buffer.ptr],
                                &s_rows, r.stream.raw,
                            )?;
                            let tk_w = r.weights.buf(&format!("{ip}.token_key_norm.weight"))?.ptr;
                            r.programs.rms_norm_128.launch(
                                &[r.scratch.idx_tk_raw.buffer.ptr, tk_w, tk_w, tk_w, tk_w, tk_w, r.scratch.idx_tk.buffer.ptr],
                                &s_rows, r.stream.raw,
                            )?;

                            let pending = r.cache.pending_counts[0][dsa_idx];
                            ensure!(
                                pending == pending_common,
                                "Hy4 DSA layer {layer} pending count {pending} diverged from {pending_common}"
                            );
                            let total_avail = pending + rows;
                            let complete_groups = total_avail / 4;
                            let new_pending = total_avail % 4;
                            let pending_buf = r.cache.pending_key_slot(0, dsa_idx)?;
                            if complete_groups > 0 {
                                if pending > 0 {
                                    r.device.library.copy_d2d_async(
                                        r.scratch.idx_group_tok.buffer,
                                        pending_buf,
                                        pending * IDX_DIM * 2,
                                        r.stream.raw,
                                    )?;
                                }
                                let copy_from_cur = complete_groups * 4 - pending;
                                let dst_cur = CuteafdDeviceBuffer {
                                    ptr: r.scratch.idx_group_tok.buffer.ptr.cast::<u8>().wrapping_add(pending * IDX_DIM * 2).cast(),
                                    bytes: copy_from_cur * IDX_DIM * 2,
                                    ..r.scratch.idx_group_tok.buffer
                                };
                                r.device.library.copy_d2d_async(dst_cur, r.scratch.idx_tk.buffer, copy_from_cur * IDX_DIM * 2, r.stream.raw)?;
                                r.programs.indexer_pool_group.launch(
                                    &[
                                        r.scratch.idx_group_tok.buffer.ptr,
                                        r.weights.weight(&format!("{ip}.key_value_projection"))?.ptr,
                                        r.weights.weight(&format!("{ip}.key_gate_projection"))?.ptr,
                                        r.weights.buf(&format!("{ip}.intra_group_ape"))?.ptr,
                                        r.weights.buf(&format!("{ip}.key_norm.weight"))?.ptr,
                                        r.scratch.idx_group_pos.buffer.ptr,
                                        r.scratch.idx_group_slots.buffer.ptr,
                                        inv_freq,
                                        idx_cache.ptr,
                                    ],
                                    &[Scalar::I32(complete_groups as i32)],
                                    r.stream.raw,
                                )?;
                            }
                            if new_pending > 0 {
                                if complete_groups == 0 {
                                    let dst = CuteafdDeviceBuffer {
                                        ptr: pending_buf.ptr.cast::<u8>().wrapping_add(pending * IDX_DIM * 2).cast(),
                                        bytes: rows * IDX_DIM * 2,
                                        ..pending_buf
                                    };
                                    r.device.library.copy_d2d_async(dst, r.scratch.idx_tk.buffer, rows * IDX_DIM * 2, r.stream.raw)?;
                                } else {
                                    let tail_off = (complete_groups * 4 - pending) * IDX_DIM * 2;
                                    let src = CuteafdDeviceBuffer {
                                        ptr: r.scratch.idx_tk.buffer.ptr.cast::<u8>().wrapping_add(tail_off).cast(),
                                        bytes: new_pending * IDX_DIM * 2,
                                        ..r.scratch.idx_tk.buffer
                                    };
                                    r.device.library.copy_d2d_async(pending_buf, src, new_pending * IDX_DIM * 2, r.stream.raw)?;
                                }
                            }
                            r.cache.pending_counts[0][dsa_idx] = new_pending;

                            let total_groups = pos_end / 4;
                            if pos_end > CANDIDATE_TOPK && total_groups > 0 {
                                r.programs.proj_indexer_wq_b.launch(
                                    &[r.scratch.q_resid.buffer.ptr, r.weights.weight(&format!("{ip}.query_projection"))?.ptr, r.scratch.idx_q_raw.buffer.ptr],
                                    &s_rows, r.stream.raw,
                                )?;
                                r.programs.proj_indexer_weights.launch(
                                    &[r.scratch.y.buffer.ptr, r.weights.weight(&format!("{ip}.head_weight_projection"))?.ptr, r.scratch.idx_w_raw.buffer.ptr],
                                    &s_rows, r.stream.raw,
                                )?;
                                r.programs.indexer_query_rope.launch(
                                    &[
                                        r.scratch.idx_q_raw.buffer.ptr,
                                        r.scratch.idx_w_raw.buffer.ptr,
                                        r.scratch.positions_i64.buffer.ptr,
                                        inv_freq,
                                        r.scratch.idx_q_rot.buffer.ptr,
                                        r.scratch.idx_w_out.buffer.ptr,
                                    ],
                                    &s_rows, r.stream.raw,
                                )?;
                                r.programs.indexer_score_groups.launch(
                                    &[
                                        r.scratch.idx_q_rot.buffer.ptr,
                                        r.scratch.idx_w_out.buffer.ptr,
                                        idx_cache.ptr,
                                        r.scratch.dsa_page_table.buffer.ptr,
                                        r.scratch.idx_limits.buffer.ptr,
                                        r.scratch.idx_scores.buffer.ptr,
                                    ],
                                    &[Scalar::I32(rows as i32), Scalar::I32(total_groups as i32), Scalar::I32(total_pages as i32)],
                                    r.stream.raw,
                                )?;
                            }
                            r.programs.indexer_topk_expand.launch(
                                &[
                                    r.scratch.idx_scores.buffer.ptr,
                                    r.scratch.positions_i32.buffer.ptr,
                                    r.scratch.dsa_page_table.buffer.ptr,
                                    r.scratch.idx_cand_slots.buffer.ptr,
                                    r.scratch.idx_cand_lens.buffer.ptr,
                                ],
                                &[Scalar::I32(rows as i32), Scalar::I32(total_groups.max(1) as i32), Scalar::I32(total_pages as i32)],
                                r.stream.raw,
                            )?;
                            r.programs.mla_dsa_decode.launch(
                                &[
                                    r.scratch.query.buffer.ptr,
                                    kv_cache_buf.ptr,
                                    r.scratch.idx_cand_slots.buffer.ptr,
                                    r.scratch.idx_cand_lens.buffer.ptr,
                                    r.scratch.dsa_cu_seqlens.buffer.ptr,
                                    r.scratch.attn_out.buffer.ptr,
                                    r.scratch.lse_out.buffer.ptr,
                                    r.scratch.attn_scratch.buffer.ptr,
                                ],
                                &[Scalar::I32(rows as i32), Scalar::I32(rows as i32), Scalar::I64(CANDIDATE_TOPK as i64), Scalar::I32(1)],
                                r.stream.raw,
                            )?;
                        } else {
                            // SWA-512 circular ring attention
                            let swa_prog = if rows == 1 { &r.programs.mla_swa_decode } else { &r.programs.mla_swa_extend };
                            swa_prog.launch(
                                &[
                                    r.scratch.query.buffer.ptr,
                                    kv_cache_buf.ptr,
                                    r.scratch.swa_page_table.buffer.ptr,
                                    r.scratch.swa_seqlens.buffer.ptr,
                                    r.scratch.swa_cu_seqlens.buffer.ptr,
                                    r.scratch.attn_out.buffer.ptr,
                                    r.scratch.lse_out.buffer.ptr,
                                    r.scratch.attn_scratch.buffer.ptr,
                                ],
                                &[Scalar::I32(rows as i32), Scalar::I32(1), Scalar::I64(total_pages as i64), Scalar::I32(1)],
                                r.stream.raw,
                            )?;
                        }

                        sink_uv.launch(
                            &[
                                r.scratch.attn_out.buffer.ptr,
                                w_uv,
                                r.scratch.gate.buffer.ptr,
                                r.scratch.lse_out.buffer.ptr,
                                r.weights.buf(&format!("{p}.self_attn.learnable_sink_param"))?.ptr,
                                r.scratch.uv_out.buffer.ptr,
                            ],
                            &s_rows, r.stream.raw,
                        )?;
                        launch_proj_split(
                            r,
                            gpus,
                            &format!("{p}.self_attn.o_proj"),
                            proj_o,
                            rows,
                            r.scratch.uv_out.buffer,
                            r.scratch.delta.buffer,
                            r.scratch.delta_f32.buffer,
                        )?;
                    }
                    Ok(())
                })?;
            }
            self.reduce_tp2_f32_to_bf16(rows, H, Some(&format!("{p}.self_attn.o_proj")))?;
            if dump_rows > 0 && layer <= 3 {
                let s0 = &self.ranks[0].scratch;
                self.dump_bf16(&format!("l{layer:02}_attn_y"), s0.y.buffer, dump_rows * H)?;
                self.dump_bf16(&format!("l{layer:02}_attn_q_resid"), s0.q_resid.buffer, dump_rows * Q_LORA)?;
                self.dump_bf16(&format!("l{layer:02}_attn_query"), s0.query.buffer, dump_rows * local_heads * 576)?;
                self.dump_bf16(&format!("l{layer:02}_attn_out"), s0.attn_out.buffer, dump_rows * local_heads * 512)?;
                self.dump_f32(&format!("l{layer:02}_attn_lse"), s0.lse_out.buffer, dump_rows * local_heads)?;
                self.dump_bf16(&format!("l{layer:02}_attn_uv"), s0.uv_out.buffer, dump_rows * local_heads * 256)?;
                self.dump_bf16(&format!("l{layer:02}_attn_delta"), s0.delta.buffer, dump_rows * H)?;
            }

            // (C) post_proj_norm + fused attention iHC post + FFN iHC pre + FFN
            for r in &mut self.ranks {
                let (r_in, r_out) = if use_a {
                    (r.scratch.r_a.buffer, r.scratch.r_b.buffer)
                } else {
                    (r.scratch.r_b.buffer, r.scratch.r_a.buffer)
                };
                let s_rows = [Scalar::I32(rows as i32)];
                let dev = r.device;
                dev.run(|| {
                    // SAFETY: all buffers live through stream drain.
                    unsafe {
                        let ppn = r.weights.buf(&format!("{p}.self_attn.post_proj_norm.weight"))?.ptr;
                        r.programs.rms_norm_3072.launch(
                            &[r.scratch.delta.buffer.ptr, ppn, ppn, ppn, ppn, ppn, r.scratch.delta_norm.buffer.ptr],
                            &s_rows, r.stream.raw,
                        )?;
                        r.programs.ihc_post_pre.launch(
                            &[
                                r.scratch.delta_norm.buffer.ptr,
                                r_in.ptr,
                                r.scratch.post.buffer.ptr,
                                r.weights.buf(&format!("{p}.hc_mlp_layer.hc_pre.hc_fn"))?.ptr,
                                r.weights.buf(&format!("{p}.hc_mlp_layer.hc_pre.hc_scale"))?.ptr,
                                r.weights.buf(&format!("{p}.hc_mlp_layer.hc_pre.hc_base"))?.ptr,
                                r.weights.buf(&format!("{p}.post_attention_layernorm.weight"))?.ptr,
                                r_out.ptr,
                                r.scratch.post_ffn.buffer.ptr,
                                r.scratch.y.buffer.ptr,
                            ],
                            &s_rows, r.stream.raw,
                        )?;

                        if self.cfg.dense_layer[layer] {
                            launch_proj_out(
                                r,
                                &format!("{p}.mlp.gate_proj"),
                                &r.programs.proj_dense_gate_up,
                                rows,
                                r.scratch.y.buffer,
                                r.scratch.ffn_gate.buffer,
                            )?;
                            launch_proj_out(
                                r,
                                &format!("{p}.mlp.up_proj"),
                                &r.programs.proj_dense_gate_up,
                                rows,
                                r.scratch.y.buffer,
                                r.scratch.ffn_up.buffer,
                            )?;
                            r.programs.post_silu_split_dense.launch(
                                &[r.scratch.ffn_gate.buffer.ptr, r.scratch.ffn_up.buffer.ptr, r.scratch.ffn_act.buffer.ptr],
                                &s_rows, r.stream.raw,
                            )?;
                            launch_proj_split(
                                r,
                                gpus,
                                &format!("{p}.mlp.down_proj"),
                                &r.programs.proj_dense_down,
                                rows,
                                r.scratch.ffn_act.buffer,
                                r.scratch.delta.buffer,
                                r.scratch.delta_f32.buffer,
                            )?;
                        } else {
                            // LatentMoE: router + shared expert + routed experts (EXL3 K4 MUL1 onboarded or skipped)
                            r.programs.proj_router.launch(
                                &[r.scratch.y.buffer.ptr, r.weights.weight(&format!("{p}.mlp.gate"))?.ptr, r.scratch.router_logits.buffer.ptr],
                                &s_rows, r.stream.raw,
                            )?;
                            let bias = r.weights.buf(&format!("{p}.mlp.gate.e_score_correction_bias"))?.ptr;
                            r.device.library.copy_d2d_async(
                                r.scratch.router_select.buffer,
                                r.scratch.router_logits.buffer,
                                rows * EXPERTS * 4,
                                r.stream.raw,
                            )?;
                            r.device.library.router_select(
                                r.scratch.router_select.buffer.ptr,
                                bias,
                                std::ptr::null(),
                                std::ptr::null(),
                                r.scratch.route_ids.buffer.ptr,
                                r.scratch.route_weights.buffer.ptr,
                                rows,
                                EXPERTS,
                                TOPK,
                                self.cfg.routed_scaling_factor as f32,
                                true,
                                r.stream.raw,
                            )?;
                            r.programs.route_weights.launch(
                                &[r.scratch.router_logits.buffer.ptr, bias, r.scratch.route_ids.buffer.ptr, r.scratch.route_weights.buffer.ptr],
                                &s_rows, r.stream.raw,
                            )?;
                            launch_proj_out(
                                r,
                                &format!("{p}.mlp.routed_expert_down_proj"),
                                &r.programs.proj_latent_down,
                                rows,
                                r.scratch.y.buffer,
                                r.scratch.latent_in.buffer,
                            )?;
                            let grid = r.quantize_grid.blocks(rows, LATENT_H);
                            r.programs.expert_input_quant.launch(
                                &[
                                    r.scratch.latent_in.buffer.ptr,
                                    r.scratch.wire_in.buffer.ptr,
                                    r.scratch.wire_in.buffer.ptr.wrapping_byte_add(LATENT_H),
                                    r.scratch.wire_in.buffer.ptr,
                                ],
                                &[Scalar::I32(rows as i32), Scalar::I32(grid as i32)],
                                r.stream.raw,
                            )?;

                            if !self.skip_routed_experts {
                                if let Some(experts) = &mut r.onboarded_experts {
                                    experts.enqueue(
                                        layer,
                                        rows,
                                        ExpertInput::Fp8K32(r.scratch.wire_in.buffer.ptr),
                                        Routes {
                                            ids: r.scratch.route_ids.buffer.ptr,
                                            weights: r.scratch.route_weights.buffer.ptr,
                                        },
                                        r.stream.raw,
                                        &mut || (),
                                    )?;
                                    let exp_out = CuteafdDeviceBuffer {
                                        ptr: experts.output(),
                                        bytes: rows * LATENT_H * 4,
                                        device_id: r.device.id,
                                        flags: 0,
                                    };
                                    if gpus == 1 {
                                        r.device.library.cuda_f32_to_bf16_async(
                                            exp_out,
                                            r.scratch.routed_latent.buffer,
                                            rows * LATENT_H,
                                            r.stream.raw,
                                        )?;
                                    } else {
                                        r.device.library.copy_d2d_async(
                                            r.scratch.delta_f32.buffer,
                                            exp_out,
                                            rows * LATENT_H * 4,
                                            r.stream.raw,
                                        )?;
                                    }
                                } else {
                                    anyhow::bail!(
                                        "layer {layer}: no onboarded hy4f:exl3-k45-mul1 expert package loaded (pass --expert-package or --skip-routed-experts for non-expert diagnostics)"
                                    );
                                }
                            } else {
                                r.device.library.cuda_zero_bytes_async(r.scratch.routed_latent.buffer, rows * LATENT_H * 2, r.stream.raw)?;
                                if gpus == 2 {
                                    r.device.library.cuda_zero_bytes_async(r.scratch.delta_f32.buffer, rows * LATENT_H * 4, r.stream.raw)?;
                                }
                            }

                            // Shared expert
                            launch_proj_out(
                                r,
                                &format!("{p}.mlp.shared_experts.gate_proj"),
                                &r.programs.proj_shared_gate_up,
                                rows,
                                r.scratch.y.buffer,
                                r.scratch.ffn_gate.buffer,
                            )?;
                            launch_proj_out(
                                r,
                                &format!("{p}.mlp.shared_experts.up_proj"),
                                &r.programs.proj_shared_gate_up,
                                rows,
                                r.scratch.y.buffer,
                                r.scratch.ffn_up.buffer,
                            )?;
                            r.programs.post_silu_split_shared.launch(
                                &[r.scratch.ffn_gate.buffer.ptr, r.scratch.ffn_up.buffer.ptr, r.scratch.ffn_act.buffer.ptr],
                                &s_rows, r.stream.raw,
                            )?;
                            launch_proj_split(
                                r,
                                gpus,
                                &format!("{p}.mlp.shared_experts.down_proj"),
                                &r.programs.proj_shared_down,
                                rows,
                                r.scratch.ffn_act.buffer,
                                r.scratch.shared_out.buffer,
                                r.scratch.shared_f32.buffer,
                            )?;
                        }
                    }
                    Ok(())
                })?;
            }
            use_a = !use_a;

            if self.cfg.dense_layer[layer] {
                self.reduce_tp2_f32_to_bf16(rows, H, Some(&format!("{p}.mlp.down_proj")))?;
            } else {
                // On TP2, reduce routed latent (1536) and shared expert (3072) across ranks before latent_up + combine.
                if gpus == 2 {
                    self.reduce_tp2_f32_to_bf16(rows, LATENT_H, None)?;
                    for r in &self.ranks {
                        r.device.run(|| unsafe {
                            r.device.library.copy_d2d_async(r.scratch.routed_latent.buffer, r.scratch.delta.buffer, rows * LATENT_H * 2, r.stream.raw)?;
                            r.device.library.copy_d2d_async(r.scratch.delta_f32.buffer, r.scratch.shared_f32.buffer, rows * H * 4, r.stream.raw)
                        })?;
                    }
                    self.reduce_tp2_f32_to_bf16(rows, H, Some(&format!("{p}.mlp.shared_experts.down_proj")))?;
                    for r in &self.ranks {
                        r.device.run(|| unsafe {
                            r.device.library.copy_d2d_async(r.scratch.shared_out.buffer, r.scratch.delta.buffer, rows * H * 2, r.stream.raw)
                        })?;
                    }
                }
                for r in &self.ranks {
                    r.device.run(|| {
                        launch_proj_out(
                            r,
                            &format!("{p}.mlp.routed_expert_up_proj"),
                            &r.programs.proj_latent_up,
                            rows,
                            r.scratch.routed_latent.buffer,
                            r.scratch.routed_out.buffer,
                        )?;
                        // SAFETY: routed_out and shared_out finished in stream order into disjoint delta.
                        unsafe {
                            r.bf16_add.launch(r.scratch.routed_out.buffer, r.scratch.shared_out.buffer, r.scratch.delta.buffer, rows * H, r.stream.raw)
                        }
                    })?;
                }
            }

            // (D) Post-FFN norm (`post_fc2_norm` on dense layer 0, `combine_output_norm` on MoE) + FFN iHC post
            let ffn_norm_name = if self.cfg.dense_layer[layer] {
                format!("{p}.mlp.post_fc2_norm.weight")
            } else {
                format!("{p}.mlp.combine_output_norm.weight")
            };
            for r in &self.ranks {
                let (r_in, r_out) = if use_a {
                    (r.scratch.r_a.buffer, r.scratch.r_b.buffer)
                } else {
                    (r.scratch.r_b.buffer, r.scratch.r_a.buffer)
                };
                let s_rows = [Scalar::I32(rows as i32)];
                r.device.run(|| unsafe {
                    let nw = r.weights.buf(&ffn_norm_name)?.ptr;
                    r.programs.rms_norm_3072.launch(
                        &[r.scratch.delta.buffer.ptr, nw, nw, nw, nw, nw, r.scratch.delta_norm.buffer.ptr],
                        &s_rows, r.stream.raw,
                    )?;
                    r.programs.ihc_post.launch(
                        &[r.scratch.delta_norm.buffer.ptr, r_in.ptr, r.scratch.post_ffn.buffer.ptr, r_out.ptr],
                        &s_rows, r.stream.raw,
                    )
                })?;
            }
            use_a = !use_a;
            if dump_rows > 0 {
                let r_cur = if use_a { self.ranks[0].scratch.r_a.buffer } else { self.ranks[0].scratch.r_b.buffer };
                self.dump_bf16(&format!("l{layer:02}_out"), r_cur, dump_rows * HC * H)?;
                if layer <= 3 {
                    let s0 = &self.ranks[0].scratch;
                    self.dump_bf16(&format!("l{layer:02}_ffn_y"), s0.y.buffer, dump_rows * H)?;
                    self.dump_bf16(&format!("l{layer:02}_ffn_delta"), s0.delta.buffer, dump_rows * H)?;
                    self.dump_bf16(&format!("l{layer:02}_ffn_delta_norm"), s0.delta_norm.buffer, dump_rows * H)?;
                    if !self.cfg.dense_layer[layer] {
                        self.dump_bf16(&format!("l{layer:02}_ffn_routed_latent"), s0.routed_latent.buffer, dump_rows * LATENT_H)?;
                        self.dump_bf16(&format!("l{layer:02}_ffn_shared_out"), s0.shared_out.buffer, dump_rows * H)?;
                    }
                }
            }
        }

        self.committed_pos = pos_end;

        if last_layer + 1 < self.cfg.layers {
            for r in &self.ranks {
                r.stream.drain()?;
            }
            return Ok(Vec::new());
        }

        // Terminal hc_head + final RMSNorm on rank 0 (head_rank = 0), replicated GPU-to-GPU to rank 1
        // on TP2, followed by FP32 vocabulary head sharded across ranks.
        let vocab = self.cfg.vocab_size;
        let mut all_logits = vec![0f32; rows * vocab];
        let mut r_host_tiles: Vec<Vec<f32>> = vec![Vec::new(); gpus];
        {
            let r0 = &mut self.ranks[0];
            let r_cur = if use_a { r0.scratch.r_a.buffer } else { r0.scratch.r_b.buffer };
            r0.device.run(|| unsafe {
                r0.programs.hc_head.launch(
                    &[
                        r_cur.ptr,
                        r0.weights.buf("model.hc_head.hc_head_fn")?.ptr,
                        r0.weights.buf("model.hc_head.hc_head_scale")?.ptr,
                        r0.weights.buf("model.hc_head.hc_head_base")?.ptr,
                        r0.weights.buf("model.norm.weight")?.ptr,
                        r0.scratch.delta_norm.buffer.ptr,
                        r0.scratch.y.buffer.ptr,
                    ],
                    &[Scalar::I32(rows as i32)],
                    r0.stream.raw,
                )
            })?;
            r0.peer_ready.record(&r0.stream)?;
        }
        if dump_rows > 0 {
            self.dump_bf16("hc_head_y", self.ranks[0].scratch.y.buffer, dump_rows * H)?;
        }
        if gpus == 2 {
            let bytes = rows * H * 2;
            let ev0 = self.ranks[0].peer_ready.raw;
            let src0 = CuteafdDeviceBuffer { bytes, ..self.ranks[0].scratch.y.buffer };
            self.ranks[1].device.run(|| unsafe {
                self.ranks[1].device.library.cuda_stream_wait_event(self.ranks[1].stream.raw, ev0)?;
                self.ranks[1].device.library.copy_peer_async(
                    CuteafdDeviceBuffer { bytes, ..self.ranks[1].scratch.y.buffer },
                    src0,
                    bytes,
                    self.ranks[1].stream.raw,
                )
            })?;
        }
        for (rank_idx, r) in self.ranks.iter().enumerate() {
            let local_v = r.weights.local_vocab;
            let mut rank_logits = vec![0f32; rows * local_v];
            for start in (0..rows).step_by(64) {
                let count = (rows - start).min(64);
                let x_ptr = r.scratch.y.buffer.ptr.cast::<u8>().wrapping_add(start * H * 2).cast();
                let bytes = count * local_v * 4;
                if let Some(native_head) = r.weights.native.get("lm_head") {
                    let ns = r.scratch.native.as_ref().context("Hy4 native lm_head missing scratch")?;
                    let x_buf = CuteafdDeviceBuffer {
                        ptr: x_ptr,
                        bytes: count * H * 2,
                        ..r.scratch.y.buffer
                    };
                    let out_buf = CuteafdDeviceBuffer {
                        bytes,
                        ..r.scratch.logits_tile.buffer
                    };
                    r.device.run(|| native_head.launch(count, x_buf, out_buf, &ns.scratch, r.stream.raw))?;
                } else {
                    let head = r.weights.head.as_ref().context("missing VocabularyHead")?;
                    let lm_head_w = r.weights.weight("lm_head")?;
                    r.device.run(|| unsafe {
                        target_head::launch(
                            TargetHead::configured(),
                            head,
                            x_ptr,
                            lm_head_w.ptr,
                            r.scratch.logits_tile.buffer.ptr.cast(),
                            count,
                            r.stream.raw,
                        )
                    })?;
                }
                r.stream.drain()?;
                let mut tile_bytes = vec![0u8; bytes];
                r.device.run(|| {
                    r.device.library.copy_d2h(&mut tile_bytes, CuteafdDeviceBuffer { bytes, ..r.scratch.logits_tile.buffer })
                })?;
                for (i, chunk) in tile_bytes.chunks_exact(4).enumerate() {
                    rank_logits[start * local_v + i] = f32::from_le_bytes(chunk.try_into().unwrap());
                }
            }
            r_host_tiles[rank_idx] = rank_logits;
        }
        for row in 0..rows {
            for (rank_idx, r) in self.ranks.iter().enumerate() {
                let local_v = r.weights.local_vocab;
                let dst_off = row * vocab + rank_idx * local_v;
                let src_off = row * local_v;
                all_logits[dst_off..dst_off + local_v]
                    .copy_from_slice(&r_host_tiles[rank_idx][src_off..src_off + local_v]);
            }
        }
        Ok(all_logits)
    }

    /// Return the active 4-stream residual `[rows, 4, H]` F32 from rank 0 (for layer-0 boundary checks).
    pub(crate) fn read_residual_f32(&self, rows: usize) -> Result<Vec<f32>> {
        let r0 = &self.ranks[0];
        r0.stream.drain()?;
        let bytes = rows * HC * H * 2;
        let mut raw = vec![0u8; bytes];
        r0.device.run(|| {
            r0.device.library.copy_d2h(&mut raw, CuteafdDeviceBuffer { bytes, ..r0.scratch.r_a.buffer })
        })?;
        Ok(raw
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect())
    }
}
