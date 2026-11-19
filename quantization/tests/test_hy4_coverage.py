"""Rank, weighting, identity and causal-isolation witnesses for Hy4 recovery."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy4_flash import run, storage


def test_grouped_recovery_replaces_last_rank_and_uses_unbiased_sigmoids():
    c = SimpleNamespace(n_group=2, topk_group=1, num_experts_per_tok=2,
                        norm_topk_prob=True, routed_scaling_factor=3.998)
    x = torch.tensor([[1.], [-1.]])
    weights = {"mlp.gate.weight": torch.tensor([[.5], [-2.], [.2], [-.5], [.4], [.3], [.2], [-.1]]),
               "mlp.gate.e_score_correction_bias": torch.tensor([0., .9, 0., 0., -.8, -.8, -.8, -.8])}
    scores, choice = run.reference.routing_scores(x, weights, c)
    ids, _ = run.reference.native_routes(scores, choice, c)
    assert set(ids[0].tolist()) == {0, 1}
    # Pass the weakest admitted expert *first* both times. Slot order is
    # unspecified for native topk(sorted=False), rank is specified by score.
    ids = ids.gather(1, choice.gather(1, ids).argsort())
    near, gamma = storage.candidate_routes(scores, choice, ids, c)
    for row in range(2):
        admitted = sorted(range(4), key=lambda expert: float(choice[row, expert]), reverse=True)
        assert near[row].tolist() == admitted[2:]
        for j, expert in enumerate(admitted[2:]):
            expected = scores[row, expert] / (scores[row, admitted[0]] + scores[row, expert]) * 3.998
            assert torch.equal(gamma[row, j], expected)


def test_rank_major_recovery_deduplicates_tied_live_routes_and_respects_corpus_order():
    c = SimpleNamespace(n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=1.)
    scores = torch.full((3, 5), .5)
    ids = torch.tensor([[1, 4], [2, 3], [4, 0]])
    near, _ = storage.candidate_routes(scores, scores, ids, c)
    assert all(not set(selected.tolist()).intersection(candidate.tolist())
               for selected, candidate in zip(ids, near, strict=True))
    neighbors = torch.tensor([[1, 3], [3, 0], [0, 3], [0, 1], [3, 1]], dtype=torch.int32)
    gamma = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    rec = storage.select_recovery(neighbors, gamma, [0, 5, 0, 1], first_rank=3, minimum=4)
    assert set(rec) == {0, 3}
    assert [v.tolist() for v in rec[3]] == [[1, 4, 0], [2., 8., 1.], [3, 3, 4]]
    assert [v.tolist() for v in rec[0]] == [[2, 3, 1], [4., 6., 3.], [3, 3, 4]]


def test_expert_hessians_weight_down_and_exclude_recovery_from_causal_sum(monkeypatch):
    monkeypatch.setattr(storage, "MINIMUM_EXPERT_ROWS", 4)
    monkeypatch.setattr(run, "EXPERT_ROWS", 2)
    latent = torch.tensor([[1., 2.], [3., 4.], [-2., 1.]])
    batch = dict(latent=latent, ids=torch.tensor([[0], [1], [1]]), gamma=torch.tensor([[.2], [.3], [.6]]),
                 candidate_ids=torch.tensor([[1], [0], [2]], dtype=torch.int32),
                 candidate_gamma=torch.tensor([[.8], [.4], [.1]]))
    spool = storage.Routes([(0, batch)], experts=4, tokens=3, recover=True)
    layer = SimpleNamespace(prefix="layer.", index=1)
    runner = run.Runner.__new__(run.Runner)
    runner.device = torch.device("cpu")
    runner.model = SimpleNamespace(config=SimpleNamespace(swiglu_limit=10.))
    runner.reader = SimpleNamespace(expert_weights=lambda *_: (torch.eye(2),) * 3)
    covariances = []

    def encode(name, weight, covariance, bits, device, metrics):
        covariances.append(covariance)
        metrics.append(dict(module=name))
        return {name: weight}

    monkeypatch.setattr(run, "encode", encode)
    monkeypatch.setattr(run, "reconstruction", lambda values, name, *args: values[name])
    for expert in (0, 2, 3):
        combined = torch.zeros_like(latent)
        metrics = []
        covariances.clear()
        runner.expert(layer, spool, expert, metrics, combined)
        selected, coefficients, proof = spool.calibration(expert)
        x = latent[selected]
        padding = proof["identity_rows"] * torch.eye(2)
        expected_down = torch.nn.functional.silu(x) * x * coefficients[:, None]
        assert torch.allclose(covariances[0]["H"], x.T @ x + padding)
        assert covariances[1] is covariances[0]
        assert torch.allclose(covariances[2]["H"], expected_down.T @ expected_down + padding)
        assert all(c["count"] == 4 for c in covariances)
        assert [m["selection_xxh3_128"] for m in metrics] == [proof["selection_xxh3_128"]] * 3
        assert [proof[k] for k in ("natural_rows", "augmented_rows", "identity_rows")] == {
            0: [1, 1, 2], 2: [0, 1, 3], 3: [0, 0, 4]}[expert]
        expected = torch.zeros_like(latent)
        if expert == 0:
            expected[0] = torch.nn.functional.silu(latent[0]) * latent[0] * .2
        assert torch.equal(combined, expected)


def test_duplicate_candidate_or_missing_evidence_fails_closed():
    b = dict(latent=torch.ones(2, 1), ids=torch.tensor([[0], [1]]), gamma=torch.ones(2, 1))
    with pytest.raises(ValueError, match="duplicate"):
        storage.Routes([(0, b | dict(candidate_ids=b["ids"].int(), candidate_gamma=b["gamma"]))],
                       experts=3, tokens=2, recover=True)
    with pytest.raises(ValueError, match="missing candidate"):
        storage.Routes([(0, b)], experts=3, tokens=2).calibration(0)
