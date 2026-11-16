"""Every V4.1 graph visitor must match its reviewed capture-site classification."""
import re
from pathlib import Path
from test_resolved_report_settings import production

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "rust/crates/cuteafd-daemon/src/families/deepseek_v41"


def test_v41_site_flags_match_checked_in_classification():
    policy = (FAMILY / "graph_policy.rs").read_text()
    classification = (FAMILY / "graph_policy/spanning.rs").read_text()
    classes = dict(re.findall(r'\("([^"]+)", Scope::(Local|Declared|Spanning),', classification))
    found = set()
    for path in FAMILY.rglob("*.rs"):
        if path.name == "spanning.rs" or path.name.endswith("tests.rs"):
            continue
        source, _ = production(path.read_text())
        for match in re.finditer(r'\bf\([^;\n]*?,\s*"([^"]+)",\s*(.+?),\s*&mut\s+', source):
            name, flag = match.groups()
            assert name in classes, (path, name, "unclassified site")
            found.add(name)
            if classes[name] == "Local":
                assert flag == "false", (path, name, flag)
            elif classes[name] == "Spanning":
                assert flag == "true", (path, name, flag)
            else:
                assert flag in ("self.spans_devices", "self.spans_devices()", "self.peer_stream().is_some()"), (path, name, flag)
    # These four site names are selected by Kind rather than a string literal.
    projection = (FAMILY / "v41_projection_tp2.rs").read_text()
    for direction in ("query", "output"):
        rank = f"v41.projection.{direction}.rank"
        gather = f"v41.projection.{direction}.gather"
        assert classes[rank] == "Local" and classes[gather] == "Spanning"
        assert rank in projection and gather in projection
        found.update((rank, gather))
    # Router chooses the two literal names through current/other variables.
    router = (FAMILY / "v41_backbone_router.rs").read_text()
    for name in ("v41.router.local", "v41.router.remote"):
        assert name in router and classes[name] == "Local"
        found.add(name)
    assert ", rank_site, false, &mut rank.graphs)" in projection
    assert ", gather_site, true, &mut self.gather_graphs)" in projection
    assert "f(device, current, false," in router and "f(device, other, false," in router
    assert found == set(classes), (found - set(classes), set(classes) - found)
    assert "spans_devices: spanning::spans(name, spans)?" in policy
    assert "spans || ranks > 1" not in policy


def test_v41_peer_wrappers_and_commit_metadata_preserve_scope():
    query = (FAMILY / "v41_attention_query.rs").read_text()
    assert "width, spans || prefix" not in query
    assert "width, spans)" in query
    projection = (FAMILY / "v41_projection_tp2.rs").read_text()
    assert "self.graphs.cache_insert(layer,weights,rows,graph,ticket,nodes as u64,width,spans)" in projection
    assert "self.gather_graphs.cache_insert(0,weights,rows,graph,ticket,nodes as u64,width,true)" in projection
    assert projection.index("owner.storage.stream.complete().await?") < projection.index("rank.capture_ready(layer,rows)?")
    assert "f(device, site, true, bank)" in (FAMILY / "v41_experts/dspark/chain/distributed.rs").read_text()
    dual = (FAMILY / "v41_sparse_attention/dual.rs").read_text()
    assert "half.get_mut().spans_devices = true" not in dual
    assert "exec,ticket,nodes as u64,width,spans)" in dual
    terminal = (FAMILY / "v41_experts/dspark/terminal/distributed.rs").read_text()
    assert "self.terminal.get_mut().visit_graph_banks(f)?" in terminal
    poll = terminal.split("pub fn poll(", 1)[1]
    assert poll.index("cuda_stream_query") < poll.index("self.capture_stage")
    assert poll.index("poll_copy_logits()?") < poll.index("self.enqueue_stage(requests, true)")
    lane = (FAMILY / "v41_backbone_lane.rs").read_text()
    completion = lane.split("impl<'s, 'w, 'a> PendingLaneFfn", 1)[1]
    assert completion.index("finish_tp2_then") < completion.index("}).await?") < completion.index("graphs.capture_ready")
