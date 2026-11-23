//! Rank-local decode/verify graph executor (`glm5_flash.decode_local`, `GRAPH_EXACT_ROWS`).
//!
//! Every `PeerExchange` and `ContextExchange` push/wait runs on the host outside
//! `replay_on`, pushing both ranks before waiting on either. Each captured CUDA
//! graph is strictly single-device (`Scope::Local`, `spans_devices = false`), so
//! runtime lazy capture and cold-donor `cudaGraphExecUpdate` stay safe after
//! readiness (#1878, #2460) and match eager arithmetic bit for bit.

use super::*;

impl<'a> GlmfEngine<'a> {
    fn local_rank_layer(&self, rank: usize, index: usize) -> Result<&GlmfLayer<'a>> {
        if rank == 0 {
            Ok(&self.weights.layers[index])
        } else {
            Ok(&self.peer()?.layers[index])
        }
    }

    fn local_tap(&self, previous: usize, w0: &Workspace<'_>, t: usize) -> Result<()> {
        if let Some(drafter) = &self.drafter {
            drafter.tap_streams(previous, w0.streams[0].buffer.ptr, HC, 0, t)?;
        }
        Ok(())
    }

    /// Push both ranks' attention partials outside capture before waiting on either.
    /// Returns each rank's merged-attention source pointer (rank 0, rank 1) for `post_pre_on(ffn)`
    /// when `complete_output_shard` or `core_attention_meet` already produced it, or `None` when
    /// standard BF16/FP32 `PeerExchange` awaits the local `add` inside the next section.
    fn local_exchange_attention(&self, index: usize, w0: &Workspace<'_>, w1: &Workspace<'_>,
        t: usize, cap: &str) -> Result<Option<[*mut c_void; 2]>> {
        let layer0 = self.local_rank_layer(0, index)?;
        let layer1 = self.local_rank_layer(1, index)?;
        let att_slot = slot(index, false, 0);
        let exchange = self.exchange()?;
        let h = self.cfg.hidden;

        if core::stage_attention(layer0) {
            // Core K6 attention meet: push both ranks before waiting on either.
            let out0 = self.core_attention_meet(0, w0, layer0, att_slot, t)?
                .context("rank 0 core attention meet")?;
            let out1 = self.core_attention_meet(1, w1, layer1, att_slot, t)?
                .context("rank 1 core attention meet")?;
            return Ok(Some([out0, out1]));
        }
        if self.output_shard_attention(layer0) {
            // Output-shard KDA: push both ranks' heads first, then finish each rank's output shard.
            let out0 = self.complete_output_shard(0, w0, layer0, att_slot, t, cap)?;
            let out1 = self.complete_output_shard(1, w1, layer1, att_slot, t, cap)?;
            return Ok(Some([out0, out1]));
        }
        let precise = self.precise_attention(layer0);
        let bytes = t * h * if precise { 4 } else { 2 };
        // Push BOTH ranks before waiting on EITHER so neither stream can deadlock behind a peer wait.
        self.exchange_push(exchange, "main", 0, att_slot, w0.delta.buffer.ptr, bytes)?;
        self.exchange_push(exchange, "main", 1, att_slot, w1.delta.buffer.ptr, bytes)?;
        self.exchange_wait(exchange, "main", 0, att_slot)?;
        self.exchange_wait(exchange, "main", 1, att_slot)?;
        Ok(None)
    }

    /// Rank-local attention partial reduction in identical operand order (rank 0 then rank 1).
    fn local_add_attention(&self, rank: usize, w: &Workspace<'_>, index: usize, t: usize) -> Result<*mut c_void> {
        let exchange = self.exchange()?;
        let layer = self.local_rank_layer(rank, index)?;
        let att_slot = slot(index, false, 0);
        let precise = self.precise_attention(layer);
        let recv = exchange.recv(rank, att_slot)?;
        let (a, b) = if rank == 0 { (w.delta.buffer.ptr, recv) } else { (recv, w.delta.buffer.ptr) };
        if precise {
            self.run_on(rank, true, "add_fp32",
                &[("a", a), ("b", b), ("out", w.sum_ptr()?)], &[Scalar::I32(t as i32)])?;
        } else {
            exchange.add(rank, a, b, w.sum_ptr()?, t * self.cfg.hidden)?;
        }
        w.sum_ptr()
    }

    /// Push both ranks' previous-layer FFN partials outside capture before waiting on either.
    fn local_exchange_prev_ffn(&self, previous: usize, layers: usize, w0: &Workspace<'_>,
        w1: &Workspace<'_>, t: usize) -> Result<()> {
        let exchange = self.exchange()?;
        let ffn_slot = slot(previous, true, 0);
        let bytes = t * self.cfg.hidden * 2;
        let last = previous + 1 == layers;
        // Rank 0 pushes to rank 1 unless this is the final layer (rank 1 stops at the last FFN).
        if !last {
            self.exchange_push(exchange, "main", 0, ffn_slot, w0.delta.buffer.ptr, bytes)?;
        }
        // Rank 1's FFN push: TP2 layers already pushed via `tp2_publish`; dense and non-TP2
        // layers push here before any wait.
        if !self.tp2_layer(previous) {
            let layer1 = self.local_rank_layer(1, previous)?;
            let out1 = self.peer_ffn_out(previous, w1, layer1)?;
            self.exchange_push(exchange, "main", 1, ffn_slot, out1, bytes)?;
        }
        // Wait after BOTH pushes are queued.
        self.exchange_wait(exchange, "main", 0, ffn_slot)?;
        if !last {
            self.exchange_wait(exchange, "main", 1, ffn_slot)?;
        }
        Ok(())
    }

    /// Rank-local FFN partial reduction in rank-0-then-rank-1 operand order.
    fn local_add_prev_ffn(&self, rank: usize, w: &Workspace<'_>, previous: usize, t: usize) -> Result<*mut c_void> {
        let Some(exchange) = &self.exchange else { return Ok(w.delta.buffer.ptr) };
        let ffn_slot = slot(previous, true, 0);
        let recv = exchange.recv(rank, ffn_slot)?;
        let (a, b) = if rank == 0 {
            (w.delta.buffer.ptr, recv)
        } else {
            let layer1 = self.local_rank_layer(1, previous)?;
            (recv, self.peer_ffn_out(previous, w, layer1)?)
        };
        exchange.add(rank, a, b, w.sum_ptr()?, t * self.cfg.hidden)?;
        w.sum_ptr()
    }

    /// Rank-local dense FFN execution on rank `rank` (handles whole-on-rank-0 dense fallback).
    fn local_dense_ffn(&self, rank: usize, w: &Workspace<'_>, index: usize, t: usize, cap: &str) -> Result<()> {
        let layer = self.local_rank_layer(rank, index)?;
        let rows = Scalar::I32(t as i32);
        if rank == 0 {
            self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)
        } else if layer.has("w_gate_up_fp8") || core::native_ffn(layer) || native::dense_ffn(layer) {
            let out = self.peer_ffn_out(index, w, layer)?;
            self.ffn_on(1, w, layer, self.cfg.dense_intermediate, cap, out, rows)
        } else {
            Ok(())
        }
    }

    /// Run rank-local decode/verify step under `glm5_flash.decode_local` (`Scope::Local`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_graphed_local(&self, w: &Workspace<'_>, w1: Option<&Workspace<'_>>,
        tables: &StepTables, tables1: &StepTables, t: usize, rows: Scalar, logit_rows: usize)
        -> Result<Option<DeviceLogits>> {
        let layers = &self.weights.layers;
        let head = layers.len() == self.cfg.layers && logit_rows == t;
        let gather = self.embedding.device_gather();
        let cap = decode_cap(t);

        // Single-GPU path: already 100% rank-local; capture one segment per layer + head.
        let Some(w1) = w1 else {
            for index in 0..=layers.len() {
                let key = GraphKey::new_sub(index, 0, t, tables)
                    .normalized(layers.get(index).map(|layer| layer.attention));
                let segment = || -> Result<()> {
                    let Some(layer) = layers.get(index) else {
                        self.run("mhc_post", &[("x", w.delta.buffer.ptr), ("residual", w.streams[1].buffer.ptr),
                            ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                            ("out", w.streams[0].buffer.ptr)], &[rows])?;
                        if let Some(prev) = index.checked_sub(1) { self.local_tap(prev, w, t)?; }
                        if head {
                            self.run("head", &[("streams", w.streams[0].buffer.ptr),
                                ("weight", self.weights.norm.buffer.ptr), ("out", w.x.buffer.ptr)], &[rows])?;
                            self.logits(w, t, t, true)?;
                            self.select_greedy(w, t)?;
                        }
                        return Ok(());
                    };
                    if index == 0 {
                        if gather { self.gather_streams(w, t)?; }
                        self.pre(w, &w.streams[0], layer, rows)?;
                    } else {
                        self.post_pre_on(0, w, w.delta.buffer.ptr, 1, layer, "attn", "input_norm", rows, cap)?;
                        self.local_tap(index - 1, w, t)?;
                    }
                    self.attention(0, w, index, layer, rows, cap, tables, None)?;
                    self.post_pre_on(0, w, w.delta.buffer.ptr, 0, layer, "ffn", "post_norm", rows, cap)?;
                    if layer.dense {
                        self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)
                    } else {
                        Ok(())
                    }
                };
                if index == layers.len() && !head { segment()?; } else { self.replay(key, segment)?; }
                if layers.get(index).is_some_and(|layer| !layer.dense) {
                    if self.warming_graphs.get() {
                        // SAFETY: masked warm-up owns live delta storage; no router ids are consumed.
                        unsafe { self.library.cuda_zero_bytes_async(w.delta.buffer, t * self.cfg.hidden * 2, self.stream)?; }
                    } else {
                        crate::shared::decode_graph::real_row_moe(tables.real_rows, t, |real| {
                            self.moe(w, index, &layers[index], real, Scalar::I32(real as i32), cap, true,
                                &[Span { first: 0, rows: real }], None)
                        }, |tail| {
                            let offset = tail.start * self.cfg.hidden * 2;
                            let bytes = tail.len() * self.cfg.hidden * 2;
                            ensure!(offset + bytes <= w.delta.buffer.bytes, "MoE tail exceeds delta");
                            // SAFETY: subrange lies within live delta allocation on the active stream.
                            unsafe {
                                self.library.cuda_zero_bytes_async(cuteafd_ffi::CuteafdDeviceBuffer {
                                    ptr: w.delta.buffer.ptr.cast::<u8>().add(offset).cast(), bytes, ..w.delta.buffer
                                }, bytes, self.stream)
                            }
                        })?;
                    }
                }
                if index < layers.len() { crate::shared::console::layer_mark(index); }
            }
            if layers.len() < self.cfg.layers { self.synchronize()?; return Ok(None); }
            if !head {
                let timer = std::time::Instant::now();
                self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                    ("out", w.x.buffer.ptr)], &[rows])?;
                self.logits(w, t, logit_rows, tables.decode)?;
                self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
            }
            return Ok(Some(self.device_logits(w, logit_rows, head)));
        };

        // Two-GPU rank-local path: all PeerExchange & ContextExchange push/wait run outside capture.
        let workspaces = [w, w1];
        let rank_tables = [tables, tables1];
        if !layers.is_empty() {
            if gather { self.gather_streams(w, t)?; }
            self.exchange_direct(w.streams[0].buffer.ptr, w1.streams[0].buffer.ptr,
                t * HC * self.cfg.hidden * 2)?;
            self.record_direct(1, w1.streams[0].buffer.ptr, false);
            self.exchange()?.wait(1, DIRECT)?;
        }

        for index in 0..layers.len() {
            let layer0 = &layers[index];
            let layer1 = self.local_rank_layer(1, index)?;
            let is_context_mla = self.context.is_some() && layer0.attention == GlmNextAttention::Mla;

            // 1. Previous layer FFN exchange outside capture (push both, then wait both).
            if let Some(previous) = index.checked_sub(1) {
                self.local_exchange_prev_ffn(previous, layers.len(), w, w1, t)?;
            }

            // 2. Layer attention + local graph capture (strictly single-device per rank).
            if is_context_mla {
                // Phase 0 (sub=0): prev-FFN add + post_pre(attn) + MLA/index produce + (short: expand + own partial; long: topk).
                for rank in 0..2 {
                    let key = GraphKey::new_sub(index, 0, t, rank_tables[rank]).normalized(Some(layer0.attention));
                    let wr = workspaces[rank];
                    let lr = if rank == 0 { layer0 } else { layer1 };
                    let tr = rank_tables[rank];
                    self.replay_on(rank, key, || {
                        if index == 0 {
                            self.pre_on(rank, wr, &wr.streams[0], lr, rows)?;
                        } else {
                            let prev_out = self.local_add_prev_ffn(rank, wr, index - 1, t)?;
                            self.post_pre_on(rank, wr, prev_out, 1, lr, "attn", "input_norm", rows, cap)?;
                            if rank == 0 { self.local_tap(index - 1, wr, t)?; }
                        }
                        self.context_local_produce_score(rank, wr, index, lr, rows, cap, tr)
                    })?;
                }

                // ContextExchange queries (and candidates on long) outside capture: push both, then wait both.
                let slot0 = self.context_local_slot(0, index)?;
                let slot1 = self.context_local_slot(1, index)?;
                let slots = [slot0, slot1];
                for rank in 0..2 {
                    self.context_local_push_query(rank, workspaces[rank], slots[rank], t)?;
                }
                let mut peer_candidates = [(std::ptr::null_mut(), std::ptr::null_mut()); 2];
                if tables.long {
                    for rank in 0..2 {
                        self.context_local_push_candidates(rank, slots[rank], t)?;
                    }
                    for rank in 0..2 {
                        peer_candidates[rank] = self.context_local_wait_candidates(rank, slots[rank])?;
                    }
                }
                let peer_queries = [
                    self.context_local_wait_query(0, slots[0])?,
                    self.context_local_wait_query(1, slots[1])?,
                ];

                // Phase 1: long captures merge + expand + peer partial + own partial (4 kernels >= 3);
                // short already ran own partial in Phase 0 and runs peer partial directly (1 kernel).
                if tables.long {
                    for rank in 0..2 {
                        let key = GraphKey::new_sub(index, 1, t, rank_tables[rank]).normalized(Some(layer0.attention));
                        let wr = workspaces[rank];
                        let tr = rank_tables[rank];
                        let (ps, pi) = peer_candidates[rank];
                        let pq = peer_queries[rank];
                        self.replay_on(rank, key, || {
                            self.context_local_merge_and_partials(rank, wr, index, rows, tr, ps, pi, pq)
                        })?;
                    }
                } else {
                    for rank in 0..2 {
                        self.context_local_partial(rank, workspaces[rank], index, rows, t,
                            peer_queries[rank], 1 - rank)?;
                    }
                }

                // ContextExchange partials outside capture: push both, then wait both.
                for rank in 0..2 {
                    self.context_local_push_partial(rank, slots[rank], t)?;
                }
                let received = [
                    self.context_local_wait_partial(0, slots[0])?,
                    self.context_local_wait_partial(1, slots[1])?,
                ];

                // Combine + o_proj (2 kernels: outside capture to avoid SmallGraphCapture <= 2).
                for rank in 0..2 {
                    let lr = if rank == 0 { layer0 } else { layer1 };
                    self.context_local_combine_out(rank, workspaces[rank], lr, rows, cap, t, received[rank])?;
                }
            } else {
                // KDA or head-split MLA: Phase 0 (sub=0) captures prev-FFN add + post_pre(attn) + local attention.
                for rank in 0..2 {
                    let key = GraphKey::new_sub(index, 0, t, rank_tables[rank]).normalized(Some(layer0.attention));
                    let wr = workspaces[rank];
                    let lr = if rank == 0 { layer0 } else { layer1 };
                    let tr = rank_tables[rank];
                    self.replay_on(rank, key, || {
                        if index == 0 {
                            self.pre_on(rank, wr, &wr.streams[0], lr, rows)?;
                        } else {
                            let prev_out = self.local_add_prev_ffn(rank, wr, index - 1, t)?;
                            self.post_pre_on(rank, wr, prev_out, 1, lr, "attn", "input_norm", rows, cap)?;
                            if rank == 0 { self.local_tap(index - 1, wr, t)?; }
                        }
                        self.attention(rank, wr, index, lr, rows, cap, tr, None)
                    })?;
                }
            }

            // 3. Attention PeerExchange outside capture (push both, then wait both).
            let merged_ptrs = self.local_exchange_attention(index, w, w1, t, cap)?;

            // 4. Post-attention + FFN.
            if layer0.dense {
                // Dense layers (0..3): capture add_attn + post_pre(ffn) + dense FFN (sub=1, >= 7 kernels).
                for rank in 0..2 {
                    let key = GraphKey::new_sub(index, 1, t, rank_tables[rank]).normalized(Some(layer0.attention));
                    let wr = workspaces[rank];
                    let lr = if rank == 0 { layer0 } else { layer1 };
                    let pre_merged = merged_ptrs.map(|p| p[rank]);
                    self.replay_on(rank, key, || {
                        let attended = match pre_merged {
                            Some(ptr) => ptr,
                            None => self.local_add_attention(rank, wr, index, t)?,
                        };
                        self.post_pre_on(rank, wr, attended, 0, lr, "ffn", "post_norm", rows, cap)?;
                        self.local_dense_ffn(rank, wr, index, t, cap)
                    })?;
                }
            } else {
                // MoE layers: 2-kernel add_attn + post_pre(ffn) directly outside capture, then MoE.
                for rank in 0..2 {
                    let wr = workspaces[rank];
                    let lr = if rank == 0 { layer0 } else { layer1 };
                    let attended = match merged_ptrs {
                        Some(ptrs) => ptrs[rank],
                        None => self.local_add_attention(rank, wr, index, t)?,
                    };
                    self.post_pre_on(rank, wr, attended, 0, lr, "ffn", "post_norm", rows, cap)?;
                }
                if self.warming_graphs.get() {
                    // SAFETY: masked warm-up owns live delta/shared storage; no router ids are consumed.
                    unsafe { self.library.cuda_zero_bytes_async(w.delta.buffer, t * self.cfg.hidden * 2, self.stream)?; }
                    let buffer = if self.tp2_layer(index) { w1.delta.buffer } else { w1.shared.buffer };
                    self.on(1, || unsafe {
                        self.library.cuda_zero_bytes_async(buffer, t * self.cfg.hidden * 2, self.stream_of(1))
                    })?;
                } else {
                    if !self.tp2_layer(index) {
                        let out1 = self.peer_ffn_out(index, w1, layer1)?;
                        let real = tables.real_rows;
                        ensure!(real > 0 && real <= t, "peer shared FFN real rows exceed step");
                        self.ffn_on(1, w1, layer1, self.cfg.moe_intermediate, cap, out1, Scalar::I32(real as i32))?;
                        if real < t {
                            let offset = real * self.cfg.hidden * 2;
                            let bytes = (t - real) * self.cfg.hidden * 2;
                            self.on(1, || unsafe {
                                self.library.cuda_zero_bytes_async(cuteafd_ffi::CuteafdDeviceBuffer {
                                    ptr: out1.wrapping_byte_add(offset), bytes, ..w1.shared.buffer
                                }, bytes, self.stream_of(1))
                            })?;
                        }
                    }
                    crate::shared::decode_graph::real_row_moe(tables.real_rows, t, |real| {
                        self.moe(w, index, layer0, real, Scalar::I32(real as i32), cap, true,
                            &[Span { first: 0, rows: real }], Some(w1))
                    }, |tail| {
                        let offset = tail.start * self.cfg.hidden * 2;
                        let bytes = tail.len() * self.cfg.hidden * 2;
                        ensure!(offset + bytes <= w.delta.buffer.bytes, "MoE tail exceeds delta");
                        // SAFETY: subrange belongs to live delta/shared storage on each rank's stream.
                        unsafe {
                            self.library.cuda_zero_bytes_async(cuteafd_ffi::CuteafdDeviceBuffer {
                                ptr: w.delta.buffer.ptr.cast::<u8>().add(offset).cast(), bytes, ..w.delta.buffer
                            }, bytes, self.stream)?;
                            if self.tp2_layer(index) {
                                self.on(1, || self.library.cuda_zero_bytes_async(cuteafd_ffi::CuteafdDeviceBuffer {
                                    ptr: w1.shared.buffer.ptr.cast::<u8>().add(offset).cast(), bytes, ..w1.shared.buffer
                                }, bytes, self.stream_of(1)))?;
                            }
                            Ok(())
                        }
                    })?;
                }
                if self.tp2_layer(index) {
                    self.tp2_publish(index, 0, t, Some(w1))?;
                }
            }
            crate::shared::console::layer_mark(index);
        }

        // 5. Final layer FFN exchange -> rank-0 local Tail (mhc_post + tap + head + logits + greedy).
        if let Some(last) = layers.len().checked_sub(1) {
            self.local_exchange_prev_ffn(last, layers.len(), w, w1, t)?;
        }
        let tail_key = GraphKey::new_sub(layers.len(), 0, t, tables).normalized(None);
        let tail_segment = || -> Result<()> {
            let out = match layers.len().checked_sub(1) {
                Some(last) => self.local_add_prev_ffn(0, w, last, t)?,
                None => w.delta.buffer.ptr,
            };
            self.run("mhc_post", &[("x", out), ("residual", w.streams[1].buffer.ptr),
                ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                ("out", w.streams[0].buffer.ptr)], &[rows])?;
            if let Some(last) = layers.len().checked_sub(1) {
                self.local_tap(last, w, t)?;
            }
            if head {
                self.run("head", &[("streams", w.streams[0].buffer.ptr),
                    ("weight", self.weights.norm.buffer.ptr), ("out", w.x.buffer.ptr)], &[rows])?;
                self.logits(w, t, t, true)?;
                self.select_greedy(w, t)?;
            }
            Ok(())
        };
        if !head { tail_segment()?; } else { self.replay(tail_key, tail_segment)?; }
        self.replicate_terminal(w, w1, 0, t)?;
        if layers.len() < self.cfg.layers {
            self.synchronize()?;
            return Ok(None);
        }
        if !head {
            let timer = std::time::Instant::now();
            self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                ("out", w.x.buffer.ptr)], &[rows])?;
            self.logits(w, t, logit_rows, tables.decode)?;
            self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        }
        Ok(Some(self.device_logits(w, logit_rows, head)))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn local_decode_captures_contain_no_peer_or_context_exchanges() {
        let source = include_str!("local_graph.rs");
        let body = source.split_once("pub(super) fn decode_graphed_local(").unwrap().1
            .split_once("#[cfg(test)]").unwrap().0;
        // Every replay_on / replay closure in decode_graphed_local must be strictly rank-local.
        for part in body.split("self.replay_on(").skip(1) {
            let closure = part.split_once("|| {").unwrap().1.split_once("})?;").unwrap().0;
            for forbidden in ["exchange_push", "exchange_wait", "exchange_direct", "push_query",
                "wait_query", "push_candidates", "wait_candidates", "push_partial", "wait_partial",
                "replicate_terminal", "tp2_publish"] {
                assert!(!closure.contains(forbidden),
                    "rank-local capture closure unexpectedly contains {forbidden}");
            }
        }
    }
}
