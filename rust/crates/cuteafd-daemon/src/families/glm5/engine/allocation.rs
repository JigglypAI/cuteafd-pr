//! Independent retained-field census for the target buffer subrecipe. This is
//! not a complete family census and cannot by itself authorize Ready.
use super::*;
use cuteafd_core::memory_layout::Category;
use cuteafd_loader::placement::planner::{Allocation, Lifetime, Location, Phase, Receipt, View};
use std::collections::BTreeMap;

struct Census {
    prefix: String,
    receipts: Vec<Receipt>,
}
impl Census {
    fn buffer(
        &mut self,
        id: &str,
        buffer: cuteafd_ffi::CuteafdDeviceBuffer,
        lifetime: Lifetime,
        category: Category,
    ) -> Result<()> {
        let device = u8::try_from(buffer.device_id).context("GLM buffer physical device")?;
        self.receipts.push(Receipt {
            allocation: Allocation::buffer(
                format!("{}.{id}", self.prefix),
                Location::coordinator(device),
                buffer.bytes as u64,
                lifetime,
                category,
                "GLM target buffers",
            ),
            address: buffer.ptr as u64,
        });
        Ok(())
    }
    fn rows(
        &mut self,
        kind: &str,
        slots: &[Workspace<'_>],
        decode: bool,
        library: &NativeLibrary,
        stream: *mut c_void,
    ) -> Result<()> {
        let lifetime = Lifetime {
            first: Phase::Warmup,
            last: Phase::Growth,
        };
        let mut owners = BTreeMap::new();
        let mut uploads = BTreeMap::new();
        for (slot, w) in slots.iter().enumerate() {
            ensure!(w.lane == slot, "GLM retained slot ordinal changed");
            let mut rows = vec![
                ("h", &w.h),
                ("x", &w.x),
                ("query", &w.query),
                ("q_resid", &w.q_resid),
                ("q_fp8", &w.q_fp8),
                ("head_weights", &w.head_weights),
                ("indices", &w.indices),
                ("lengths", &w.lengths),
                ("attn", &w.attn),
                ("delta", &w.delta),
                ("positions", &w.positions),
                ("slots", &w.slots),
                ("cache_lengths", &w.cache_lengths),
                ("router_logits", &w.router_logits),
                ("route_ids", &w.route_ids),
                ("route_weights", &w.route_weights),
                ("wire", &w.wire),
                ("shared", &w.shared),
                ("ids", &w.ids),
                ("select", &w.select),
            ];
            if let Some(payload) = &w.payload {
                rows.push(("payload", payload));
            }
            if let Some(core) = &w.core_rows {
                rows.extend([
                    ("core.attention", &core[0]),
                    ("core.down", &core[1]),
                    ("core.reduction", &core[2]),
                ]);
            }
            for (field, buf) in rows {
                let id = format!("{kind}.rows.{field}");
                let owner = buf._owner.buffer;
                if let Some(previous) = owners.insert(id.clone(), owner) {
                    ensure!(
                        previous.ptr == owner.ptr
                            && previous.bytes == owner.bytes
                            && previous.device_id == owner.device_id,
                        "GLM row field {id} changed backing between slots"
                    );
                } else {
                    self.buffer(&id, owner, lifetime, Category::Workspace)?;
                }
                if !decode {
                    let offset = (buf.buffer.ptr as u64)
                        .checked_sub(owner.ptr as u64)
                        .context("GLM slot before its retained backing")?;
                    ensure!(
                        offset
                            .checked_add(buf.buffer.bytes as u64)
                            .is_some_and(|end| end <= owner.bytes as u64),
                        "GLM slot past its retained backing"
                    );
                    self.buffer(
                        &format!("{kind}.slot.{slot}.{field}"),
                        buf.buffer,
                        lifetime,
                        Category::Workspace,
                    )?;
                    let spec = &mut self
                        .receipts
                        .last_mut()
                        .context("GLM slot receipt")?
                        .allocation;
                    spec.view = Some(View {
                        backing: format!("{}.{id}", self.prefix),
                        offset,
                    });
                    // A cloned zero-width placeholder is shared, not independent slot storage.
                    spec.slot = if slots.len() == 1 || buf.buffer.bytes == owner.bytes {
                        None
                    } else {
                        Some(u32::try_from(slot).context("GLM slot ordinal")?)
                    };
                } else {
                    ensure!(
                        buf.buffer.ptr == owner.ptr && buf.buffer.bytes == owner.bytes,
                        "GLM decode row is not its whole retained owner"
                    );
                }
            }
            for (field, buf) in [
                ("logits", &w.logits),
                ("scratch", &w.scratch),
                ("topk", &w.topk_scratch),
            ]
            .into_iter()
            .chain(w._head_workspace.as_ref().map(|b| ("head", b)))
            {
                let id = format!("{kind}.{field}");
                let buffer = buf._owner.buffer;
                ensure!(
                    field == "logits"
                        || (buffer.ptr == buf.buffer.ptr
                            && buffer.bytes == buf.buffer.bytes
                            && buffer.device_id == buf.buffer.device_id),
                    "GLM hook field {id} is not its whole owner"
                );
                if let Some(previous) = owners.insert(id.clone(), buffer) {
                    ensure!(
                        previous.ptr == buffer.ptr
                            && previous.bytes == buffer.bytes
                            && previous.device_id == buffer.device_id,
                        "GLM hook field {id} changed owner"
                    );
                } else {
                    self.buffer(&id, buffer, lifetime, Category::Workspace)?;
                }
                if !decode && field == "logits" {
                    self.buffer(
                        &format!("{kind}.slot.{slot}.logits"),
                        buf.buffer,
                        lifetime,
                        Category::Workspace,
                    )?;
                    self.receipts
                        .last_mut()
                        .context("GLM logits receipt")?
                        .allocation
                        .view = Some(View {
                        backing: format!("{}.{id}", self.prefix),
                        offset: (buf.buffer.ptr as u64)
                            .checked_sub(buffer.ptr as u64)
                            .context("GLM logits view before backing")?,
                    });
                }
            }
            ensure!(
                w.page_table.buffer.ptr == w.page_table._owner.buffer.ptr
                    && w.page_table.buffer.bytes == w.page_table._owner.buffer.bytes
                    && w.page_table.buffer.device_id == w.page_table._owner.buffer.device_id,
                "GLM page table is not its whole retained owner"
            );
            self.buffer(
                &format!("{kind}.page_table.{slot}"),
                w.page_table._owner.buffer,
                lifetime,
                Category::Workspace,
            )?;
            for (field, buf) in [
                ("positions", &w.positions),
                ("slots", &w.slots),
                ("page_table", &w.page_table),
                ("cache_lengths", &w.cache_lengths),
                ("lengths", &w.lengths),
                ("ids", &w.ids),
            ] {
                uploads
                    .entry((buf.buffer.device_id, buf.buffer.ptr as usize))
                    .or_insert((format!("{kind}.upload.{slot}.{field}"), buf.buffer));
            }
        }
        // Query the stores that actually exist; an unprimed destination is absent,
        // not a receipt synthesized from the forecast's expected staging size.
        for (_, (id, dst)) in uploads {
            if library.ordered_h2d_staging_bytes(dst, stream)? == 0 {
                continue;
            }
            let actual = library.ordered_h2d_staging_buffer(dst, stream)?;
            self.receipts.push(Receipt {
                allocation: Allocation::buffer(
                    format!("{}.{id}", self.prefix),
                    Location::PINNED,
                    actual.bytes as u64,
                    lifetime,
                    Category::Staging,
                    "GLM target buffers",
                ),
                address: actual.ptr as u64,
            });
        }
        Ok(())
    }
}

impl GlmEngine<'_> {
    pub(crate) fn target_buffer_receipts(&self) -> Result<Vec<Receipt>> {
        let mut out = Vec::new();
        for rank in 0..self.ranks() {
            let (kv, index, rope, decode, prefill) = if rank == 0 {
                (
                    &self.kv,
                    &self.index,
                    &self.cos_sin,
                    &self.decode_workspace,
                    &self.prefill_slots,
                )
            } else {
                let peer = self.peer()?;
                (
                    &peer.kv,
                    &peer.index,
                    &peer.cos_sin,
                    &peer.decode_workspace,
                    &peer.prefill_slots,
                )
            };
            let device =
                u8::try_from(rope.buffer.device_id).context("GLM target physical device")?;
            let mut census = Census {
                prefix: format!("glm.target.gpu.{device}"),
                receipts: Vec::new(),
            };
            for (layer, buffer) in kv.iter().enumerate() {
                census.buffer(
                    &format!("kv.{layer}"),
                    buffer.buffer,
                    Lifetime::TARGET,
                    Category::Kv,
                )?;
            }
            for (layer, buffer) in index.iter().enumerate() {
                if let Some(buffer) = buffer {
                    census.buffer(
                        &format!("index.{layer}"),
                        buffer.buffer,
                        Lifetime::TARGET,
                        Category::Kv,
                    )?;
                }
            }
            census.buffer("rope", rope.buffer, Lifetime::TARGET, Category::Kv)?;
            if let Some(workspace) = decode.borrow().as_ref() {
                census.rows(
                    "decode",
                    std::slice::from_ref(workspace),
                    true,
                    self.library,
                    self.stream_of(rank),
                )?;
            }
            census.rows(
                "prefill",
                &prefill.borrow(),
                false,
                self.library,
                self.stream_of(rank),
            )?;
            out.extend(census.receipts);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
