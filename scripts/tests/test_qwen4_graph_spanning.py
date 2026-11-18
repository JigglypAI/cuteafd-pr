"""Qwen4 graph attachment and observation use the reviewed site classification."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "rust/crates/cuteafd-daemon/src/families/qwen4"


def test_qwen4_site_flags_match_checked_in_classification():
    classification = (FAMILY / "graph_spanning.rs").read_text()
    classes = dict(re.findall(r'\("([^"]+)", Scope::(Local|Declared|Spanning),', classification))
    assert classes == {"qwen4.segment": "Local"}
    engine = (FAMILY / "engine.rs").read_text()
    configure = engine.split("pub fn configure_graph_cache(", 1)[1].split("fn visit_graphs(", 1)[0]
    observer = engine.split("fn visit_graphs(", 1)[1].split("fn graph_completed(", 1)[0]
    assert 'spans_devices: super::graph_spanning::spans("qwen4.segment"),' in configure
    assert 'super::graph_spanning::spans("qwen4.segment"), bank)?' in observer
    assert "self.gdn_banks.len() > 1" not in configure + observer
    assert '("qwen4.segment", Scope::Local,' in classification
    assert 'assert!(super::spans("new-unreviewed-site"))' in classification


def test_qwen4_peer_operations_stay_outside_replay():
    engine = (FAMILY / "engine.rs").read_text()
    placed = engine.split("fn decode_placed(", 1)[1].split("fn mix_head(", 1)[0]
    attention = placed.split("self.replay(key(index), || {", 1)[1].split("})?;", 1)[0]
    assert not any(peer in attention for peer in ("hops.", "moe_experts", "wave_send", "wave_land"))
    assert placed.index("hops.send(0,") < placed.index("hops.land(ticket)") < placed.index("self.replay(key(index)")
    assert placed.index("self.replay(key(index)") < placed.index("self.moe_experts(")
    assert placed.index("hops.send(1,") < placed.rindex("hops.land(ticket)") < placed.index("self.replay(key(n)")
    hop = (FAMILY.parents[1] / "shared/peer_split/hop.rs").read_text()
    send = hop.split("pub fn send_into(", 1)[1].split("pub fn land(", 1)[0]
    assert "peer_push_signal" in send and "peer_wait(" not in send
    experts = (FAMILY.parents[1] / "shared/experts/rtx/owner_tp2.rs").read_text()
    enqueue = experts.split("unsafe fn enqueue_live(", 1)[1].split("impl Drop", 1)[0]
    assert enqueue.index("self.exchange.push(owner, 0") < enqueue.index("self.exchange.wait(peer, 0")
    assert enqueue.index("self.experts[peer].enqueue(") < enqueue.index("self.exchange.push(peer, 1") < enqueue.index("self.exchange.wait(owner, 1")
