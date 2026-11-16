"""Kimi graph banks retain the reviewed host-ordering classification."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "rust/crates/cuteafd-daemon/src/families/kimi_k3"


def test_kimi_site_flags_match_checked_in_classification():
    classification = (FAMILY / "graph_spanning.rs").read_text()
    classes = dict(re.findall(r'\("([^"]+)", Scope::(ContextLayout|Spanning),', classification))
    assert classes == {"kimi_k3.step": "ContextLayout", "kimi_k3.context": "Spanning"}
    graphs = (FAMILY / "graphs.rs").read_text().split("#[cfg(test)]\nmod tests", 1)[0]
    assert 'spans_devices: super::graph_spanning::spans(name, self.spans_devices)?' in graphs
    assert 'self.spans_devices = ranks > 1' not in graphs
    assert 'self.spans_devices = context;' in graphs
    for name, bank in (("kimi_k3.step", "bank"), ("kimi_k3.context", "context_bank")):
        assert f'self.{bank}.attach_device(library, device, reserve, self.site("{name}", ranks, rank)?)' in graphs
        assert f'super::graph_spanning::spans("{name}", self.spans_devices)?, &mut self.{bank}' in graphs
    assert 'if matches!(op, Op::Context { .. }) { &mut self.context_bank } else { &mut self.bank }' in graphs
    assert 'self.bank_mut(key.op).cache_commit(key, owner, ticket, nodes as u64, width, spanning)' in graphs
    device = (FAMILY / "device.rs").read_text()
    assert 'rank, reserve, self.context.is_some())' in device
    assert 'lane.visit_graphs(visit)?' in (FAMILY / "engine.rs").read_text()
    assert 'unclassified Kimi graph site {name}' in classification


def test_kimi_peer_publication_and_waits_stay_outside_local_steps():
    layers = (FAMILY / "layers.rs").read_text()
    heads = layers.split('fn heads(&mut self, layer:', 1)[1].split('fn context(', 1)[0]
    assert heads.index('for rank in 0..2 { self.attention_rank') < heads.index('self.publish(') < heads.index('self.meet(')
    begin = layers.split('fn begin_ffn(&mut self, layer:', 1)[1].split('fn finish_ffn(', 1)[0]
    assert begin.index('for local in self.local.iter_mut() { local.ffn') < begin.index('self.publish_ffn(')
    spark = (FAMILY / "spark.rs").read_text()
    begin = spark.split('pub(crate) fn begin(', 1)[1].split('pub(crate) fn has_link(', 1)[0]
    assert begin.index('exchange.send(') < begin.index('local.ffn(')
    assert 'exchange.land(' not in begin
    device = (FAMILY / "device.rs").read_text()
    context = device.split('fn context_step(', 1)[1].split('pub(crate) fn context_decode(', 1)[0]
    assert 'crate::shared::peer_split::attention::context_decode(' in context
