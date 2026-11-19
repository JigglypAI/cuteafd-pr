"""Issue reporting reuses the exact production receipts and recovers partial publication."""
from datetime import datetime
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy4_flash import report_gates as reporting


def prepare(tmp_path):
    (tmp_path / "gates").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "identity.json").write_text(json.dumps(dict(schema="cuteafd-hy4-causal-mc-v2", codebook="mul1")))
    row = dict(module="model.layers.0.test", bits=8, heldout_nmse=.00001, heldout_rows=3,
               effective_bits_per_weight=8.1, kept_native=False, heldout_passed=True)
    gate = dict(schema="hy4-production-gate-v1", stage="target-000", status="passed", seconds=16,
                size_estimate=dict(view_bytes={"original": 8}), tp2_even_weight_bytes={"original": 4},
                projections=[row])
    (tmp_path / "gates/target-000.json").write_text(json.dumps(gate))
    (tmp_path / "metrics/target-000.json").write_text(json.dumps([row | dict(extra=42)]))
    return gate


def test_dedup_after_partial_post_and_lost_response(tmp_path, monkeypatch):
    prepare(tmp_path)
    remote = {}
    monkeypatch.setattr(reporting, "posted_parts", lambda: dict(remote))
    def post(*args, data):
        marker = json.loads(data)["body"].splitlines()[0]
        assert marker not in remote
        remote[marker] = 123 + len(remote)
        if len(remote) == 1:
            raise RuntimeError("HTTP response lost after create")
        return remote[marker]
    monkeypatch.setattr(reporting, "gh", post)
    reporting.report(tmp_path)  # layer group still unsealed
    assert not remote
    (tmp_path / "events.jsonl").write_text('{"phase":"target_layer_complete","current_layer":0,"layer_seconds":20}\n')
    with pytest.raises(RuntimeError):
        reporting.report(tmp_path)
    reporting.report(tmp_path)
    reporting.report(tmp_path)
    assert len(remote) == 2
    proof = json.loads((tmp_path / "gate-posts/target-000.json").read_text())
    assert proof["comment_ids"] == [123, 124]


def test_mismatched_projection_fails_before_any_remote_reads(tmp_path, monkeypatch):
    prepare(tmp_path)
    (tmp_path / "metrics/target-000.json").write_text("[]")
    (tmp_path / "events.jsonl").write_text('{"phase":"group_frontier_recovered","group":"target-000"}\n')
    monkeypatch.setattr(reporting, "gh", lambda *a, **k: pytest.fail("remote access"))
    with pytest.raises(ValueError, match="disagree"):
        reporting.report(tmp_path)


def test_timing_excludes_dense_and_replayed_moe_layers():
    starts = [dict(phase="target_layer", current_layer=i) for i in range(4)]
    complete = [dict(phase="target_layer_complete", current_layer=i, layer_seconds=sec)
                for i, sec in enumerate([50., 1., 60., 120.])]
    events = starts + complete + [dict(phase="expert_group_replayed", current_layer=1)]
    as_of = datetime(2026,11,19,15,0,tzinfo=reporting.TAIPEI)
    text = reporting.forecast(events, "target-003", as_of)
    assert "38 left" in text and "0.95 h" in text and "15:57" in text
    assert "2 uninterrupted" in text and "0.63–1.27 h" in text


def test_projection_failure_before_first_layer_seals(tmp_path, monkeypatch):
    (tmp_path / "identity.json").write_text('{"schema":"cuteafd-hy4-causal-mc-v2","codebook":"mul1"}')
    (tmp_path / "status.json").write_text(json.dumps(dict(status="blocked", current_layer=0,
        error="ValueError: model.layers.0.test K6 training error 0.1 outside [0, 0.001]")))
    posts = []
    def post(*args, data):
        posts.append(json.loads(data)["body"])
        return 42
    monkeypatch.setattr(reporting, "posted_parts", lambda: {})
    monkeypatch.setattr(reporting, "gh", post)
    reporting.report(tmp_path)
    reporting.report(tmp_path)
    assert len(posts) == 1 and "stopped" in posts[0] and "0.1" in posts[0]
