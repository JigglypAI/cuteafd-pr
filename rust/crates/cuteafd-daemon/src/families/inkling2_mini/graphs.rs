//! Monolithic 48-visit exact-row decode/verify graphs on the shared #2697 soft-cap device graph bank.
//!
//! All 48 loop visits (draft taps, attention, routing, local NVFP4/MXFP8 experts via
//! `DeviceRowCopy`, and residual/conv join) capture into one kernel-only graph per
//! `(rows, mode, taps)`, keeping metadata H2D uploads and head projection outside.
use crate::shared::{
    decode_graph::{
        self,
        device::{self, Basis, Cycle, Preference, Site},
        GraphBank, GraphOwner,
    },
    memory::device::Stream,
};
use anyhow::Result;
use cuteafd_loader::placement::graph_reserve::Resolved;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Mode {
    Decode,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    rows: usize,
    mode: Mode,
    taps: bool,
}

pub(super) struct DecodeGraphs<'a> {
    bank: GraphBank<Key, GraphOwner<'a, ()>>,
    first_touched: HashSet<Key>,
    first_touch_pending: bool,
    unsupported: HashSet<Key>,
    device: i32,
}

impl<'a> DecodeGraphs<'a> {
    pub(super) fn new(stream: &Stream<'a>, policy: Resolved) -> Result<Option<Self>> {
        if policy.bytes == 0 {
            return Ok(None);
        }
        let mut bank = GraphBank::with_family("inkling2_mini");
        let device = stream.device.id;
        bank.attach_device(
            stream.device.library,
            device,
            policy.bytes,
            Site {
                name: "inkling2_mini.step",
                cycle: Cycle::Decode,
                preference: Preference::Adaptive,
                bootstrap: 2 << 20,
                basis: Basis::Bytes,
                spans_devices: super::graph_sites::spans("inkling2_mini.step", false),
            },
        )?;
        device::set_reserve_policy(device, policy)?;
        Ok(Some(Self {
            bank,
            first_touched: HashSet::new(),
            first_touch_pending: false,
            unsupported: HashSet::new(),
            device,
        }))
    }

    pub(super) fn cycle(&mut self, cycle: Cycle) {
        self.bank.cache_cycle(cycle);
    }
    pub(super) fn needs_boundary(&self) -> bool {
        self.first_touch_pending || self.bank.cache_needs_boundary()
    }
    pub(super) fn observed(&mut self) {
        self.first_touch_pending = false;
    }
    pub(super) fn tick(&mut self) {
        self.bank.cache_tick();
    }
    pub(super) fn visit(&mut self, visit: &mut device::GraphVisitor<'_>) -> Result<()> {
        visit(self.device, "inkling2_mini.step", false, &mut self.bank)
    }
    pub(super) fn clear_drained(&mut self) {
        self.bank.cache_clear_drained();
    }

    fn run(&mut self, key: Key, stream: &Stream<'a>, mut work: impl FnMut() -> Result<()>) -> Result<()> {
        if self.unsupported.contains(&key) {
            return work();
        }
        if self.first_touched.insert(key) {
            self.first_touch_pending = true;
            return work();
        }
        let library = stream.device.library;
        if let Some(raw) = self.bank.cached_launch(&key).map(|owner| owner.raw) {
            // SAFETY: retained TP1 stream, geometry, and persistent owners outlive replay.
            return stream
                .device
                .run(|| unsafe { decode_graph::launch(library, raw, stream.raw) });
        }
        let Some(mut ticket) = self.bank.cache_ticket(&key, false, None) else {
            return work();
        };
        stream.device.run(|| {
            // SAFETY: rank-local stream operations on persistent admitted owners.
            if let Err(error) = unsafe { decode_graph::begin_capture(library, stream.raw, &ticket) } {
                if self.bank.cache_ticket_failed(ticket, &error) {
                    tracing::warn!(%error, ?key, "Inkling 2 Mini graph capture unavailable; running eagerly");
                }
                return work();
            }
            let result = work();
            // SAFETY: always close capture we opened.
            let source = unsafe { library.cuda_graph_capture_source(stream.raw) };
            if let Err(error) = result {
                self.bank.cache_ticket_failed(ticket, source.as_ref().err().unwrap_or(&error));
                return Err(error);
            }
            let captured = source.and_then(|source| {
                if source.spans_devices || source.inspection_error.is_some() {
                    self.unsupported.insert(key);
                    anyhow::bail!("Inkling 2 Mini graph must have inspected rank-local topology");
                }
                ticket.complete_source(library, source, stream.raw)
            });
            let (raw, nodes, width, spans, _) = match captured {
                Ok(topology) => topology,
                Err(error) => {
                    let warn = self.bank.cache_ticket_failed(ticket, &error);
                    if warn && !cuteafd_ffi::is_small_graph(&error) {
                        tracing::warn!(%error, ?key, "Inkling 2 Mini graph unavailable; running eagerly");
                    }
                    return work();
                }
            };
            // SAFETY: execution owns all captured persistent storage and drains before drop.
            let graph = unsafe { GraphOwner::new(library, self.device, raw, ())? };
            self.bank.census_capture_key("step", 0, None, &key, raw);
            self.bank.cache_commit(key, graph, ticket, nodes as u64, width, spans);
            // SAFETY: bank owns executable through this ordered launch.
            unsafe { decode_graph::launch(library, raw, stream.raw) }
        })
    }
}

pub(super) fn step<'a>(
    graphs: Option<&mut DecodeGraphs<'a>>,
    stream: &Stream<'a>,
    rows: usize,
    mode: Mode,
    taps: bool,
    mut work: impl FnMut() -> Result<()>,
) -> Result<()> {
    let Some(graphs) = graphs else {
        return work();
    };
    graphs.run(Key { rows, mode, taps }, stream, work)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mini_monolithic_graph_key_distinguishes_rows_verify_and_taps() {
        let base = Key {
            rows: 4,
            mode: Mode::Decode,
            taps: false,
        };
        assert_eq!(
            HashSet::from([
                base,
                Key { rows: 8, ..base },
                Key { mode: Mode::Verify, ..base },
                Key { taps: true, ..base },
                Key {
                    mode: Mode::Verify,
                    taps: true,
                    ..base
                },
            ])
            .len(),
            5
        );
    }
}
