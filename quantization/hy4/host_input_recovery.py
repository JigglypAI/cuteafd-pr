"""Record and verify full Hy4's switch to host-resident calibration inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from kimi_k3.uniform_k2.storage import atomic_json, canonical


REPO = Path(__file__).resolve().parents[2]
BASE_COMMIT = "a51d6bb6256a38a2b854fa602c7bc20f8c04ff9a"
FROZEN_DIGEST = "70828a5c5a4952cce27fbdbbb1257ec99e4172faf517b1c1be83c4bc5151c468"
CHANGED = ("quantization/hy4/run.py", "quantization/hy4_flash/run.py",
           "quantization/hy4_flash/storage.py", "quantization/hy4_flash/gates.py")
ADDED = "quantization/hy4/host_input_recovery.py"
ORACLES = {
    "quantization/tests/test_hy4_host_input_oracle.py": "test_host_inputs_match_resident_gpu_encode_and_replay",
    "quantization/tests/test_hy4_heldout_workspace.py": "test_heldout_workspace_matches_original",
}
ANCHOR = {"target": 9, "stage": "target-008-next", "encodes": 6306, "drafter": 0, "head": False}
RECEIPT = "host-input-recovery.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO)


def read(path):
    return json.loads(path.read_bytes())


def reviewed_files(frozen, commit):
    if digest(canonical(frozen).encode()) != FROZEN_DIGEST:
        raise ValueError("host-input recovery requires the original full Hy4 run identity")
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("supply the full reviewed commit ID")
    git("merge-base", "--is-ancestor", BASE_COMMIT, commit)
    original = frozen["code"]["files"]
    files = dict(original)
    paths = sorted(set(original) | {ADDED})
    changed = set(git("diff", "--name-only", BASE_COMMIT, commit, "--", *paths).decode().splitlines())
    if ADDED not in changed or not changed <= set(CHANGED) | {ADDED}:
        raise ValueError(f"unexpected producer changes in host-input recovery: {sorted(changed)}")
    for path in sorted(changed):
        if path != ADDED and digest(git("show", f"{BASE_COMMIT}:{path}")) != original[path]:
            raise ValueError(f"original producer differs at {path}")
        files[path] = digest(git("show", f"{commit}:{path}"))
    return files, {p: {"before": original.get(p), "after": files[p]} for p in sorted(changed)}


def oracle_evidence(root, path, commit):
    path = path.resolve()
    relative = path.relative_to(root.parent.resolve())
    data = path.read_bytes()
    log = data.decode().strip()
    if not log or not re.fullmatch(r"=* *[1-9]\d* passed(?:, \d+ warnings?)? in [^\n]+", log.splitlines()[-1]):
        raise ValueError("recovery requires a passed-only verbose CUDA oracle summary")
    tests = {}
    for filename, name in ORACLES.items():
        # Pytest can display node IDs relative to the tests directory.
        node = f"{Path(filename).name}::{name}"
        if not re.search(rf"(?m)^(?:[^\s]+/)?{re.escape(node)} PASSED\b", log):
            raise ValueError(f"host-input recovery lacks a passing {node}")
        tests[f"{filename}::{name}"] = digest(git("show", f"{commit}:{filename}"))
    return {"path": str(relative), "sha256": digest(data), "tests": tests}


def frontier(root, frozen):
    state = read(root / "frontier.json")
    if state.pop("identity") != frozen:
        raise ValueError("rolling frontier differs from the frozen Hy4 identity")
    return state


def create(root, commit, evidence):
    if (git("rev-parse", "HEAD").decode().strip() != commit
            or git("status", "--porcelain", "--", *CHANGED, ADDED, *ORACLES)):
        raise ValueError("record host-input recovery from the clean reviewed commit")
    frozen = read(root / "identity.json")
    _, changes = reviewed_files(frozen, commit)
    anchor = frontier(root, frozen)
    if anchor != ANCHOR:
        raise ValueError("host-input cutover requires the stopped target-008 frontier")
    receipt = dict(schema="cuteafd-hy4-host-input-recovery-v1", original_commit=BASE_COMMIT,
                   reviewed_commit=commit, frozen_identity_sha256=FROZEN_DIGEST, changes=changes,
                   frontier=anchor, oracle=oracle_evidence(root, evidence, commit))
    path = root / RECEIPT
    if path.exists():
        if read(path) != receipt:
            raise ValueError("a different host-input recovery receipt already exists")
    else:
        atomic_json(path, receipt)
    return path


def checkpoint_identity(root, candidate):
    path = root / RECEIPT
    if not path.exists():
        from hy4_flash.run import checkpoint_identity as exact_identity
        return exact_identity(root, candidate)
    record = read(path)
    frozen = read(root / "identity.json")
    commit = record["reviewed_commit"]
    files, changes = reviewed_files(frozen, commit)
    if (record.get("schema") != "cuteafd-hy4-host-input-recovery-v1"
            or record.get("original_commit") != BASE_COMMIT
            or record.get("frozen_identity_sha256") != FROZEN_DIGEST
            or record.get("changes") != changes
            or candidate != frozen | {"code": frozen["code"] | {"files": files}}):
        raise ValueError("host-input execution identity differs from the reviewed producer")
    saved = record["oracle"]
    if saved != oracle_evidence(root, root.parent / saved["path"], commit):
        raise ValueError("host-input parity evidence changed")
    anchor = record["frontier"]
    current = frontier(root, frozen)
    if (anchor != ANCHOR
            or current["target"] < anchor["target"] or current["encodes"] < anchor["encodes"]
            or (current["target"] == anchor["target"] and current != anchor)):
        raise ValueError("host-input recovery frontier differs from its sealed starting point")
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reviewed-commit", required=True)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    print(create(args.root.resolve(), args.reviewed_commit, args.evidence))
