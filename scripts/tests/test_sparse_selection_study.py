import collections
import importlib.util
import struct
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "bench/sparse_selection_study.py"


def load():
    spec = importlib.util.spec_from_file_location("sparse_selection_study", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, layer, width, unit_tokens, heads, rows):
    with path.open("wb") as f:
        f.write(b"SSEL0001" + struct.pack("<6I", layer, width, unit_tokens, layer, heads, 0) + struct.pack("<Q", len(rows)))
        for step, (position, ids) in enumerate(rows):
            ids = np.unique(np.asarray(ids, dtype=np.int32))
            pad = np.full(width, -1, np.int32)
            pad[:len(ids)] = ids
            f.write(struct.pack("<QQii", step, position, position + 1, len(ids)) + pad.tobytes())


def test_stack_distances_match_their_definition_and_bound_a_plain_lru():
    study = load()
    rng = np.random.default_rng(1)
    sets = [np.unique(rng.integers(0, 300, 40)).astype(np.int32) for _ in range(120)]
    resets = np.zeros(120, dtype=bool)
    resets[0] = resets[60] = True
    dist = study.stack_distances(sets, resets, None, 300)
    for t, ids in enumerate(sets):
        start = 60 if t >= 60 else 0
        for k, d in zip(ids, dist[t]):
            earlier = [s for s in range(start, t) if k in sets[s]]
            if not earlier:
                assert np.isinf(d)
            else:
                # distinct units referenced from the unit's last step up to t-1, minus itself
                assert d == len(set(np.concatenate(sets[earlier[-1]:t]).tolist())) - 1
    # Pinned keys are excluded from the accounting.
    pins = [ids[:5] for ids in sets]
    pinned = study.stack_distances(sets, resets, pins, 300)
    assert all(len(p) == len(s) - 5 for p, s in zip(pinned, sets))


def test_tidy_reads_dumps_and_reports_every_table(tmp_path):
    study = load()
    rows = [(p, list(range(0, 50)) + [p - 1, p]) for p in range(1000, 1100)] + \
           [(p, list(range(0, 50)) + [p]) for p in range(1500, 1520)]
    write(tmp_path / "layer-003.bin", 3, 64, 1, 1, rows)
    write(tmp_path / "layer-007.bin", 7, 64, 1, 1, rows)
    write(tmp_path / "layer-009.bin", 9, 32, 128, 4, [(p, [0, 1, 2, 3, 4 * (p // 128)]) for p in range(1000, 1050)])
    out = tmp_path / "out"
    assert study.main(["tidy", "--family", "t", "--dump", str(tmp_path), "--out", str(out),
                       "--index-bytes-per-token", "132", "--kv-bytes-per-token", "656", "--min-position", "0"]) == 0
    for name in ("working.csv", "churn.csv", "cluster.csv", "cache.csv", "cross_layer.csv", "resident.csv",
                 "summary.json", "provenance.json", "summary.png"):
        assert (out / name).exists(), name
    import json
    summary = json.loads((out / "summary.json").read_text())
    s = summary["3|0-16K"]
    assert s["boundaries"] == 1 and s["working_1"] < s["working_16"] <= s["working_session"]
    assert 0.9 < s["jaccard_step"] < 1 and s["share_1"] > 0 and s["amp_1"] == 1.0
    assert summary["7|0-16K"]["lru_pin1k_2048_source_covers_misses"] == 1.0
    cross = (out / "cross_layer.csv").read_text().splitlines()
    assert "3,7,120,120,1.0" in cross
    # MiniMax-style ids fold 4 heads: 128-token units, pages per head.
    assert summary["9|0-16K"]["amp_1"] == 1.0 and summary["9|0-16K"]["amp_128"] >= 1.0
    provenance = json.loads((out / "provenance.json").read_text())
    assert set(provenance["inputs"]) == {"layer-003.bin", "layer-007.bin", "layer-009.bin"}


def test_cycles_split_tags_records_by_the_sessions_cycle_starts(tmp_path):
    import json
    study = load()
    rows = [(p, list(range(0, 50)) + [p]) for p in range(1000, 1200)]
    write(tmp_path / "layer-003.bin", 3, 64, 1, 1, rows)
    out = tmp_path / "out"
    assert study.main(["tidy", "--family", "t", "--dump", str(tmp_path), "--out", str(out),
                       "--index-bytes-per-token", "1", "--kv-bytes-per-token", "1", "--min-position", "0"]) == 0
    session = tmp_path / "session.json"
    turns = [{"prompt_tokens": n} for n in (900, 1100, 1150)]
    session.write_text(json.dumps({"sessions": [{"turns": turns, "chain": [
        {"task": "a", "cycle": 0, "first_turn": 0}, {"task": "b", "cycle": 1, "first_turn": 1}]}]}))
    assert study.main(["cycles", "--tidy", str(out), "--session", str(session)]) == 0
    import csv
    parts = {(r["band"], r["part"]): r for r in csv.DictReader((out / "cycles.csv").open())}
    assert set(parts) == {("0-16K", "cycle0"), ("0-16K", "later")}
    assert int(parts[("0-16K", "cycle0")]["steps"]) == 100 and int(parts[("0-16K", "later")]["steps"]) == 100
    assert json.loads((out / "cycles.json").read_text())["cycle_starts"] == [0, 1100]
