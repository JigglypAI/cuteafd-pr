"""Production gate isolation, serialized projection decisions, resume and cold held-out experts."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hy4_flash import gates, run
from hy4_flash.forward import Layer
from hy4_flash.storage import Routes, Stages


@pytest.mark.parametrize("fail", [False, True])
def test_production_gate_freezes_native_before_causal_replay(tmp_path, monkeypatch, fail):
    runner = run.Runner.__new__(run.Runner)
    runner.device = torch.device("cpu")
    runner.calibration_count = 2
    runner.records = [0, 1, 2]
    runner.native_modules = set()
    runner.stages = Stages(tmp_path, {"test": True})
    runner.progress = SimpleNamespace(event=lambda *args, **kwargs: None)
    name = "test.gate_proj"
    weight = torch.eye(2)
    layer = Layer(0, "test.", {"gate_proj.weight": weight})
    runner.core = {name: dict(out_features=2, in_features=2)}
    train = [torch.tensor([[1., 0.]]), torch.tensor([[0., 2.]])]
    heldout = torch.tensor([[300., 500.]]).expand(32, 2)
    invoked = []
    def encode(key, matrix, covariance, widths, device, metrics):
        assert torch.equal(covariance["H"], torch.diag(torch.tensor([1., 4.])))
        assert covariance["count"] == 2
        invoked.append(1)
        metrics.extend({"module":name, "bits":width} for width in widths)
        return {"candidate_8": matrix, "candidate_6": matrix * (2 if fail else 1)}
    monkeypatch.setattr(run, "encode", encode)
    monkeypatch.setattr(run, "reconstruction", lambda vs, n, b, shape, device: vs[f"candidate_{b}"])
    overrides = {}
    samples = lambda: ((xs,) for xs in [*train, heldout])
    runner.core_fit("target-000-attn", layer, [(name,)], samples, overrides)
    assert runner.native_modules == ({name} if fail else set())
    assert set(overrides) == (set() if fail else {name})
    assert set(runner.selected_core()) == set(overrides)
    rows = runner.stages.load("target-000-attn")["metadata"]["metrics"]
    assert [r["heldout_nmse"] for r in rows] == [0., float(fail)]
    assert all(row["heldout_rows"] == 16 for row in rows)
    assert all(row["kept_native"] == fail for row in rows)
    runner.native_modules.clear()
    overrides.clear()
    runner.core_fit("target-000-attn", layer, [(name,)], lambda: pytest.fail("repeated encode"), overrides)
    assert set(overrides) == (set() if fail else {name})
    assert runner.native_modules == ({name} if fail else set())
    assert invoked == [1]


def test_invalid_training_k6_aborts_before_install(monkeypatch):
    monkeypatch.setattr(run.numerical, "quantize_projection", lambda *a, **k: (
        {}, {"hessian_weighted_relative_error": .00101}, {}))
    with pytest.raises(ValueError, match="K6 training error"):
        run.encode("foo", torch.eye(2), {}, (6,), "cpu", [])


def test_heldout_experts_use_natural_rows_or_labeled_cold_probe():
    latent = torch.tensor([[1., 2.], [-1., 3.]])
    routes = Routes([(0, dict(ids=torch.zeros((2, 1), dtype=torch.long),
                              gamma=torch.ones(2, 1), latent=latent))], 2, 2)
    witness = gates.ExpertWitness(routes, 1, 10.)
    weights = [torch.eye(2)] * 3
    names = ["gate", "up", "down"]
    for expert, natural in [(0, 2), (1, 0)]:
        rows = [dict(module=n, bits=4) for n in names]
        witness.add(expert, weights, names=names, native=weights, metrics=rows)
        assert all(row["heldout_nmse"] == 0 and row["heldout_passed"] for row in rows)
        assert all(r["heldout_natural_rows"] == natural for r in rows)
        assert all(("no-natural" in row["heldout_selection"]) == (natural == 0) for row in rows)
    assert torch.equal(witness.combined, torch.nn.functional.silu(latent) * latent)
    bad = [2 * torch.eye(2), torch.eye(2), torch.eye(2)]
    with pytest.raises(ValueError, match="K4 held-out NMSE"):
        witness.add(1, bad, names=names, native=weights, metrics=[dict(module=n, bits=4) for n in names])


def test_engram_row_coverage_counts_no_table_payload(monkeypatch):
    monkeypatch.setitem(run.recipe.ENGRAM_ROWS, 1, 5)
    layer = Layer(1, "", {"engram_layer.lookup_table": torch.ones(1)})
    monkeypatch.setattr(run.reference, "engram_rows", lambda tokens, *args: tokens)
    counts = gates.engram_coverage({}, layer, [torch.tensor([0, 0, 1]), torch.tensor([1, 3])])
    assert (counts["unique_rows"], counts["repeated_lookups"], counts["untouched_rows"]) == (3, 2, 2)
