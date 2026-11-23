"""A host-input compatibility receipt can authorize only its reviewed producer."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy4 import host_input_recovery as recovery


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(recovery, "REPO", checkout)
    git = recovery.git
    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    old = (*recovery.CHANGED, "quantization/hy4/forward.py")
    files = {}
    for name in old:
        p = checkout / name
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_text("original\n")
        files[name] = recovery.digest(p.read_bytes())
    git("add", ".")
    git("commit", "-qm", "original producer")
    monkeypatch.setattr(recovery, "BASE_COMMIT", git("rev-parse", "HEAD").decode().strip())
    root = tmp_path / "production"
    root.mkdir()
    frozen = {"code": {"files": files, "fork": "unchanged"}, "sigma": .025}
    recovery.atomic_json(root / "identity.json", frozen)
    recovery.atomic_json(root / "frontier.json", recovery.ANCHOR | {"identity": frozen})
    monkeypatch.setattr(recovery, "FROZEN_DIGEST", recovery.digest(recovery.canonical(frozen).encode()))
    for name in (*recovery.CHANGED, recovery.ADDED, *recovery.ORACLES):
        p = checkout / name
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_text("host bank\n")
    git("add", ".")
    git("commit", "-qm", "reviewed host bank")
    commit = git("rev-parse", "HEAD").decode().strip()
    evidence = tmp_path / "oracle.log"
    evidence.write_text("".join(f"{p}::{n} PASSED [ 50%]\n" for p, n in recovery.ORACLES.items())
                        + "==================== 2 passed in 1.23s ====================\n")
    files, _ = recovery.reviewed_files(frozen, commit)
    candidate = frozen | {"code": frozen["code"] | {"files": files}}
    return root, frozen, commit, evidence, candidate


def test_receipt_preserves_identity_and_advances(stopped):
    root, frozen, commit, evidence, candidate = stopped
    receipt = recovery.create(root, commit, evidence)
    assert recovery.create(root, commit, evidence) == receipt
    assert recovery.checkpoint_identity(root, candidate) == frozen
    recovery.atomic_json(root / "frontier.json", recovery.ANCHOR | {
        "identity": frozen, "target": 10, "stage": "target-009-next", "encodes": 7092})
    assert recovery.checkpoint_identity(root, candidate) == frozen
    for changed in (candidate | {"sigma": .5}, candidate | {"code": frozen["code"]}):
        with pytest.raises(ValueError, match="reviewed producer"):
            recovery.checkpoint_identity(root, changed)
    evidence.write_text("2 passed in 1.23s\n")
    with pytest.raises(ValueError, match="lacks a passing"):
        recovery.checkpoint_identity(root, candidate)


def test_rejects_skipped_or_unreviewed_producer(stopped):
    root, frozen, commit, evidence, candidate = stopped
    evidence.write_text(evidence.read_text().replace("2 passed", "1 passed, 1 skipped"))
    with pytest.raises(ValueError, match="passed-only"):
        recovery.create(root, commit, evidence)
    (recovery.REPO / "quantization/hy4/forward.py").write_text("changed forward")
    recovery.git("commit", "-qam", "different producer")
    with pytest.raises(ValueError, match="unexpected producer changes"):
        recovery.reviewed_files(frozen, recovery.git("rev-parse", "HEAD").decode().strip())
