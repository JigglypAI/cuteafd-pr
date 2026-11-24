//! Persistent rank-local decode sections on the shared bounded device graph cache (#2591 C2).
//!
//! Host joins (Engram staging H2D/P2P, DSA indexer pending D2D + score/topk,
//! TP2 peer reductions, Spark expert exchange, and TP2 head gather) stay eager
//! outside capture so every captured section is strictly rank-local and depends
//! only on `(layer, rows, section)`.
use crate::shared::{
    decode_graph::{
        device::{self, Basis, Cycle, Preference, Site},
        GraphBank, GraphOwner,
    },
    memory::device::Stream,
};
use anyhow::Result;
use cuteafd_loader::serving_capacity::hy4_graphs::{self, Hy4GraphLayout};
use std::{
    cell::{Cell, RefCell},
    collections::HashSet,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Section {
    DraftFusion,
    DraftLayer,
    DraftCorrection,
    /// TP1 local SWA layer: optional Engram compute + MLA SWA + post-attn + Dense/MoE+EXL3+post-FFN.
    SwaLocal,
    /// SWA attention section (TP2 before peer reduce, or Spark-routed before FFN frontend).
    SwaAttn,
    /// Multi-row SWA pre-attention before eager `mla_swa_extend`.
    SwaExtendPre,
    /// Multi-row SWA post-attention after eager `mla_swa_extend`.
    SwaExtendPost,
    /// DSA pre-indexer: optional Engram compute + MLA Q/KV/absorb + indexer_wk + rms_norm_128.
    DsaPre,
    /// DSA >2048 indexer query projections + RoPE (3 kernels).
    DsaIndexQ,
    /// 2-GPU ContextSplit DSA dual partials (h0 + h16 over local pages).
    DsaPartial,
    /// TP1 local DSA post-indexer: `mla_dsa_decode` + sink/UV + o_proj + post-attn + MoE+EXL3+post-FFN.
    DsaLocalPost,
    /// DSA attention output only (TP2 before peer reduce, or Spark-routed): `mla_dsa_decode` + sink/UV + o_proj.
    DsaAttnOut,
    /// Post-attn iHC + Dense/MoE frontend (TP2 or Spark-routed; includes TP2 local EXL3 when onboarded).
    FfnFrontend,
    /// Post-FFN combine + norm + iHC post (TP2 or Spark-routed).
    FfnFinish,
    /// Rank-0 final layer post-FFN combine + norm + iHC post + terminal `hc_head`.
    Tail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Key {
    pub(crate) layer: usize,
    pub(crate) rows: usize,
    pub(crate) section: Section,
}

pub(crate) struct Hy4Graphs<'a> {
    bank: RefCell<GraphBank<Key, GraphOwner<'a, ()>>>,
    // A section found to contain fewer than two native nodes always stays eager.
    eager: RefCell<HashSet<Key>>,
    touched: RefCell<HashSet<Key>>,
    device: i32,
    first_touch_pending: Cell<bool>,
}

impl<'a> Hy4Graphs<'a> {
    pub(crate) fn new() -> Self {
        Self {
            bank: RefCell::new(GraphBank::with_family("hy4_flash")),
            eager: RefCell::new(HashSet::new()),
            touched: RefCell::new(HashSet::new()),
            device: -1,
            first_touch_pending: Cell::new(false),
        }
    }

    pub(crate) fn attach(
        &mut self,
        library: &'a cuteafd_ffi::NativeLibrary,
        device: i32,
        rank: usize,
        layout: Option<Hy4GraphLayout>,
    ) -> Result<()> {
        let policy = hy4_graphs::reserve_policy(layout, rank)?;
        self.bank.get_mut().attach_device(
            library,
            device,
            policy.bytes,
            Site {
                name: "hy4_flash.section",
                cycle: Cycle::Other,
                preference: Preference::Adaptive,
                bootstrap: 2 << 20,
                basis: Basis::Bytes,
                spans_devices: super::graph_sites::spans("hy4_flash.section", false),
            },
        )?;
        device::set_reserve_policy(device, policy)?;
        self.device = device;
        Ok(())
    }

    pub(crate) fn attached(&self) -> bool {
        self.bank.borrow().cache_attached()
    }

    pub(crate) fn cycle(&self, cycle: Cycle) {
        self.bank
            .borrow_mut()
            .cache_cycle(cycle);
    }

    pub(crate) fn clear_drained(&self) {
        self.bank.borrow_mut().cache_clear_drained();
    }

    pub(super) fn retained_draft_rows(&self, rows: usize) -> usize {
        self.bank.borrow().count(|key| key.rows == rows && matches!(key.section,
            Section::DraftFusion | Section::DraftLayer | Section::DraftCorrection))
    }

    pub(crate) fn retained_rows(&self, rows: usize) -> usize {
        self.bank.borrow().count(|key| key.rows == rows)
    }

    pub(crate) fn first_touch_pending(&self) -> bool {
        self.first_touch_pending.get()
    }

    pub(crate) fn observed(&self) {
        self.first_touch_pending.set(false);
    }

    pub(crate) fn visit(&self, visit: &mut device::GraphVisitor<'_>) -> Result<()> {
        visit(self.device, "hy4_flash.section", false, &mut *self.bank.borrow_mut())
    }

    pub(super) fn draft(
        &self, rows: usize, stage: usize, section: Section, stream: &Stream<'a>,
        work: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        self.run(Key { layer: stage, rows, section }, stream, work)
    }

    pub(crate) fn run(
        &self,
        key: Key,
        stream: &Stream<'a>,
        mut work: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        if !self.attached() || self.eager.borrow().contains(&key) {
            return work();
        }
        // First use initializes modules/workspaces eagerly; the outer step drains
        // before a later use can reserve capture growth for this geometry.
        if self.touched.borrow_mut().insert(key) {
            self.first_touch_pending.set(true);
            return work();
        }
        if let Some(raw) = self.bank.borrow_mut().cached_launch(&key).map(|graph| graph.raw) {
            return stream.device.run(|| {
                // SAFETY: containing Hy4Engine retains weights, cache, scratch, and EXL3 owners;
                // all rank streams drain before terminal clear or storage drop.
                unsafe { crate::shared::decode_graph::launch(&stream.device.library, raw, stream.raw) }
            });
        }
        let Some(mut ticket) = self
            .bank
            .borrow_mut()
            .cache_ticket(&key, false, None)
        else {
            return work();
        };
        stream.device.run(|| {
            let library = stream.device.library;
            // SAFETY: sections contain only ordered rank-local launches on this stream over persistent storage.
            if let Err(error) = unsafe { crate::shared::decode_graph::begin_capture(&library, stream.raw, &ticket) } {
                if self.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, ?key, "Hy4-flash capture unavailable; running eagerly");
                }
                return work();
            }
            let result = work();
            // SAFETY: always end this stream's capture, including a failed section.
            let captured = unsafe { crate::shared::decode_graph::end_capture(&library, stream.raw, &mut ticket) };
            if let Err(error) = result {
                self.bank
                    .borrow_mut()
                    .cache_ticket_failed(ticket, captured.as_ref().err().unwrap_or(&error));
                if let Ok((raw, ..)) = captured {
                    // SAFETY: failed capture has never launched.
                    unsafe {
                        let _ = library.cuda_graph_exec_destroy(raw);
                    }
                }
                return Err(error);
            }
            let (raw, nodes, width, spans_devices, inspection_error) = match captured {
                Ok(topology) => topology,
                Err(error) => {
                    if self.bank.borrow_mut().cache_ticket_failed(ticket, &error) {
                        tracing::warn!(%error, ?key, "Hy4-flash instantiate unavailable; running eagerly");
                    }
                    return work();
                }
            };
            // SAFETY: exclusive executable ownership; all captured allocations belong to Hy4Engine,
            // not the cache entry, and terminate() drains + clears banks before dropping storage.
            let graph = unsafe { GraphOwner::new(library, stream.device.id, raw, ())? };
            if !retain(nodes, inspection_error.is_none()) {
                self.eager.borrow_mut().insert(key);
                drop(graph);
                drop(ticket);
                return work();
            }
            self.bank
                .borrow_mut()
                .cache_commit(key, graph, ticket, nodes as u64, width, spans_devices);
            // SAFETY: committed executable retains the same persistent pointer contract.
            unsafe { crate::shared::decode_graph::launch(&library, raw, stream.raw) }
        })
    }
}

fn retain(nodes: usize, inspected: bool) -> bool {
    inspected && nodes >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_never_retain_single_node_or_unknown_topology() {
        assert!(!retain(0, true));
        assert!(!retain(1, true));
        assert!(!retain(10, false));
        assert!(retain(2, true));
    }

    #[test]
    fn untraced_zero_reserve_runs_every_section_eagerly_without_capture() {
        let policy = cuteafd_loader::placement::graph_reserve::resolve(None, 0, false, 0, 0).unwrap();
        assert_eq!(policy.bytes, 0);
        let mut bank: GraphBank<Key, ()> = GraphBank::with_family("hy4_flash");
        bank.attach_device_for_test(
            -2591,
            policy.bytes,
            Site {
                name: "hy4_flash.section",
                cycle: Cycle::Decode,
                preference: Preference::Adaptive,
                bootstrap: 2 << 20,
                basis: Basis::Bytes,
                spans_devices: false,
            },
        )
        .unwrap();
        for cycle in [Cycle::Decode, Cycle::Verify, Cycle::Prefill] {
            bank.cache_cycle(cycle);
            for layer in 0..42 {
                for section in [
                    Section::SwaLocal,
                    Section::SwaAttn,
                    Section::SwaExtendPre,
                    Section::SwaExtendPost,
                    Section::DsaPre,
                    Section::DsaIndexQ,
                    Section::DsaPartial,
                    Section::DsaLocalPost,
                    Section::DsaAttnOut,
                    Section::FfnFrontend,
                    Section::FfnFinish,
                ] {
                    for rows in [1, 4, 16, 64] {
                        assert!(bank
                            .cache_ticket(&Key { layer, rows, section }, false, None)
                            .is_none());
                    }
                }
            }
            bank.cache_tick();
            assert!(!bank.cache_needs_boundary());
        }
        assert!(bank
            .cached_launch(&Key {
                layer: 0,
                rows: 1,
                section: Section::SwaLocal,
            })
            .is_none());
    }

    #[test]
    fn keys_are_geometry_only() {
        let key = Key {
            layer: 8,
            rows: 16,
            section: Section::SwaLocal,
        };
        assert_eq!(
            key,
            Key {
                layer: 8,
                rows: 16,
                section: Section::SwaLocal,
            }
        );
        assert_ne!(key, Key { rows: 4, ..key });
        assert_ne!(
            key,
            Key {
                section: Section::DsaPre,
                ..key
            }
        );
    }

    #[test]
    fn observe_completed_advances_serial_epoch_for_cold_donor_reuse() -> Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        // Unique ordinal for this fixture; device cache registry is process-global.
        const DEV: i32 = 9;
        fixture.configure_memory_usage(&library, DEV, 256 << 20, 0)?;
        let mut bank: GraphBank<Key, GraphOwner<'_, ()>> = GraphBank::with_family("hy4_flash");
        bank.attach_device(
            &library,
            DEV,
            4 << 20,
            Site {
                name: "hy4_flash.section",
                cycle: Cycle::Verify,
                preference: Preference::Adaptive,
                bootstrap: 2 << 20,
                basis: Basis::Bytes,
                spans_devices: false,
            },
        )?;
        let k1 = Key { layer: 0, rows: 1, section: Section::DsaPartial };
        let mut t1 = bank.cache_ticket(&k1, false, None).expect("initial capture admitted");
        // SAFETY: CPU fixture records no device work or captured allocations.
        let src1 = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
        let (raw1, nodes1, width1, spans1, _) = t1.complete_source(&library, src1, std::ptr::null_mut())?;
        // SAFETY: fixture graph owns no device storage.
        let owner1 = unsafe { GraphOwner::new(&library, DEV, raw1, ())? };
        bank.cache_commit(k1, owner1, t1, nodes1 as u64, width1, spans1);
        // Simulate soft-cap pressure (2 MiB observed growth during k2 against 4 MiB reserve / 2 MiB soft cap).
        let k2 = Key { layer: 0, rows: 2, section: Section::DsaPartial };
        let mut t2 = bank.cache_ticket(&k2, false, None).expect("second capture crosses soft cap");
        // SAFETY: CPU fixture records no device work.
        let src2 = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
        let (raw2, nodes2, width2, spans2, _) = t2.complete_source(&library, src2, std::ptr::null_mut())?;
        let owner2 = unsafe { GraphOwner::new(&library, DEV, raw2, ())? };
        fixture.configure_memory_usage(&library, DEV, 256 << 20, 3 << 20)?;
        bank.cache_commit(k2, owner2, t2, nodes2 as u64, width2, spans2);
        assert!(!bank.cache_needs_boundary(), "cache_commit already settled measurement");
        let after_k2 = device::snapshots().into_iter().find(|s| s.device == DEV).unwrap();
        assert!(after_k2.charged >= after_k2.soft.limit, "expected soft cap crossed: {after_k2:?}");

        // Two hit-only steps completed through observe_completed must advance the serial epoch
        // even though cache_needs_boundary() is false, aging k1 into an eligible cold donor.
        for _ in 0..2 {
            // SAFETY: fixture owners hold no device allocations.
            unsafe {
                device::observe_completed(&library, |visit| {
                    visit(DEV, "hy4_flash.section", false, &mut bank)
                })?;
            }
        }
        let k3 = Key { layer: 0, rows: 3, section: Section::DsaPartial };
        let mut t3 = bank.cache_ticket(&k3, false, None).expect("cold donor update admitted");
        let src3 = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
        let (raw3, nodes3, width3, spans3, _) = t3.complete_source(&library, src3, std::ptr::null_mut())?;
        let owner3 = unsafe { GraphOwner::new(&library, DEV, raw3, ())? };
        bank.cache_commit(k3, owner3, t3, nodes3 as u64, width3, spans3);

        let snap = device::snapshots().into_iter().find(|s| s.device == DEV).unwrap();
        let verify = snap.cycles[&Cycle::Verify];
        assert!(verify.updates_soft >= 1, "expected cold donor update, got {verify:?}");
        assert_eq!(verify.eager_soft_room, 0, "expected no eager_soft_room fallback, got {verify:?}");
        assert_eq!(verify.no_cold_donor_soft, 0, "expected no donor starvation, got {verify:?}");
        bank.cache_clear_drained();
        Ok(())
    }

    #[test]
    fn graph_completed_always_advances_serial_boundary() {
        let src = include_str!("execution.rs");
        let start = src.find("pub(super) fn graph_completed(").expect("graph_completed present");
        let body = &src[start..start + 900];
        assert!(body.contains("observe_completed("), "graph_completed must call observe_completed");
        assert!(!body.contains("r.graphs.tick()"), "graph_completed must not bypass serial_boundary via tick()");
    }
}
