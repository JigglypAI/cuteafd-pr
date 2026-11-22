"""Bind any operator-directed recovery to the exact failed full-Hy4 gate."""
from __future__ import annotations

from hashlib import sha256
import json


def check_recovery(root, label):
    gate_path = root / "gates" / f"{label}.json"
    raw = gate_path.read_bytes()
    gate = json.loads(raw)
    misses = [r for r in gate["projections"] if r["heldout_passed"] is False]
    k6_fallback = gate.get("published_k6_drafter_bits") == 8
    if not misses and not k6_fallback:
        return
    path = root / "recoveries" / f"{label}.json"
    if path.is_file():
        decision = json.loads(path.read_bytes())
        if (decision.get("gate_sha256") == sha256(raw).hexdigest()
                and decision.get("action") == "apply-recorded-fallback"
                and isinstance(decision.get("diagnosis"), str) and decision["diagnosis"].strip()):
            return
    raise ValueError(f"Hy4 production gate {label} needs review before committing its fallback; see {gate_path}")
