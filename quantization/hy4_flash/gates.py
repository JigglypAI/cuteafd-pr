"""Held-out evaluation of the serialized production encodes on the K8 causal stream."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from kimi_k3.uniform_k2.storage import atomic_json
from . import recipe
from .forward import native_projection, reference

BARS = {4: .01, 6: .001, 8: .0001}
TAIL = 16
TARGET_GATES = frozenset({0, 1, min(recipe.SWA_LAYERS), min(i for i in recipe.SPARSE_LAYERS if i), 41})
DRAFT_STAGES = ("attn-first", "attn-expand", "attn-out", "mlp-first", "mlp-down")


def nmse(inputs, weight, restored, *, projection=native_projection):
    """Aggregate energy over full output columns, bounding peak logits for the head."""
    sums = torch.zeros(2, dtype=torch.float64, device=restored.device)
    weight = weight.to(device=restored.device, dtype=torch.float32)
    if inputs.ndim != 2 or not len(inputs) or weight.shape != restored.shape:
        raise ValueError("invalid held-out projection geometry")
    for chunk in inputs.split(128):
        expected = projection("", chunk, weight)
        actual = projection("", chunk, restored)
        sums[0] += (actual - expected).square().sum(dtype=torch.float64)
        sums[1] += expected.square().sum(dtype=torch.float64)
    numerator, denominator = sums.tolist()
    if not (math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0):
        raise ValueError("nonfinite or zero-energy held-out projection")
    return numerator / denominator


def packed_bpw(values, name, width, shape):
    packed = [values[f"k{width}.{name}.{suffix}"] for suffix in recipe.SUFFIXES]
    return sum(t.element_size() * t.numel() for t in packed) * 8 / math.prod(shape)


def evaluated(row, inputs, weight, restored, selection, *, projection=native_projection):
    error = nmse(inputs, weight, restored, projection=projection)
    row.update(heldout_nmse=error, heldout_rows=len(inputs), heldout_bar=BARS[row["bits"]],
               heldout_selection=selection, heldout_passed=error <= BARS[row["bits"]])


class ExpertWitness:
    """Independent natural-route combine for held-out prompts, including across group recovery."""

    def __init__(self, routes, layer, limit, *, target_gates=TARGET_GATES, projection=native_projection):
        self.projection = projection
        self.target_gates = target_gates
        self.routes = routes
        self.combined = torch.zeros_like(routes.latent)
        self.layer, self.limit = layer, limit

    def add(self, expert, matrices, *, names=None, native=None, metrics=None):
        rows, gamma = self.routes.expert(expert)
        rows = rows.to(self.combined.device)
        inputs = self.routes.latent.index_select(0, rows)
        check = self.layer in self.target_gates and metrics is not None
        selection = "heldout-natural-routes"
        if not len(rows):
            if not check:
                return
            # An unrouted expert still has a real projection to evaluate. Use
            # disjoint held-out latent rows; these probes never enter the sum.
            inputs = self.routes.latent[-128:]
            selection = "heldout-latent-tail-no-natural-routes"
        gate, up, down = matrices
        gated = self.projection("", inputs, gate).clamp_max(self.limit)
        lifted = self.projection("", inputs, up).clamp(-self.limit, self.limit)
        active = F.silu(gated) * lifted
        if len(rows):
            output = self.projection("", active, down)
            self.combined.index_copy_(0, rows, self.combined.index_select(0, rows)
                                      + output * gamma.to(output.device)[:, None])
        if check:
            down_inputs = active * gamma.to(active.device)[:, None] if len(rows) else active
            for row, name, xs, weight, reconstruction in zip(metrics[-3:], names, (inputs, inputs, down_inputs),
                                                             native, matrices, strict=True):
                if row["module"] != name or row["bits"] != 4:
                    raise ValueError("held-out expert metric ordering differs from encode order")
                evaluated(row, xs, weight, reconstruction, selection, projection=self.projection)
                row["heldout_natural_rows"] = len(rows)
                if not row["heldout_passed"]:
                    raise ValueError(f"{name} K4 held-out NMSE {row['heldout_nmse']:.6g} exceeds {BARS[4]}")


def engram_coverage(config, layer, tokens):
    """Measure unique, repeated and untouched physical rows without reading the tables."""
    if layer.index not in recipe.ENGRAM_ROWS:
        return None
    table_rows = recipe.ENGRAM_ROWS[layer.index]
    lookup = layer.weights["engram_layer.lookup_table"].cpu()
    seen = torch.zeros(table_rows, dtype=torch.bool, device="cpu")
    probes = 0
    for record in tokens:
        rows = reference.engram_rows(record.cpu(), lookup, config, layer.index).reshape(-1)
        seen[rows] = True
        probes += len(rows)
    unique = sum(int(chunk.sum()) for chunk in seen.split(1 << 20))
    return dict(table_rows=table_rows, lookups=probes, unique_rows=unique,
                repeated_lookups=probes-unique, untouched_rows=table_rows-unique,
                split="calibration", definition="physical row reuse, independent of OS page-cache state")


def gate_report(root, label, metrics, sizes, seconds, *, recovery=None, engram=None, extra=None):
    directory = Path(root) / "gates"
    directory.mkdir(exist_ok=True)
    projections = [{key: row[key] for key in (
        "module", "bits", "effective_bits_per_weight", "heldout_nmse", "heldout_rows",
        "heldout_bar", "heldout_selection", "heldout_passed", "kept_native", "heldout_natural_rows",
    ) if key in row} for row in metrics]
    if not projections or any("heldout_nmse" not in row for row in projections):
        raise ValueError(f"production gate {label} lacks independent measurements")
    receipt = dict(schema="hy4-production-gate-v1", stage=label, seconds=seconds,
                   status="passed-with-native-fallback" if any(r.get("kept_native") for r in projections)
                   else "passed", heldout="agentic heldout split", projections=projections,
                   size_estimate=sizes,
                   tp2_even_weight_bytes={k: (v+1)//2 for k, v in sizes["view_bytes"].items()},
                   tp2_scope="even weight split excluding cache and workspace",
                   recovery=recovery, engram=engram)
    if extra:
        receipt.update(extra)
    path = directory / f"{label}.json"
    if path.exists():
        old = json.loads(path.read_bytes())
        if any(old.get(k) != v for k,v in receipt.items() if k != "seconds"):
            raise ValueError(f"production gate {label} changed on replay")
    else:
        atomic_json(path, receipt)


def recovery_distribution(metrics):
    experts = [m for m in metrics if m["bits"] == 4 and m["module"].endswith(".gate_proj")]
    if not experts:
        return None
    return dict(experts=len(experts), augmented=sum(m["augmented_rows"] > 0 for m in experts),
                identity=sum(m["identity_rows"] > 0 for m in experts),
                zero_natural=sum(m["natural_rows"] == 0 for m in experts),
                min_natural=min(m["natural_rows"] for m in experts),
                min_effective=min(m["hessian_rows"] for m in experts),
                augmented_rows=sum(m["augmented_rows"] for m in experts),
                identity_rows=sum(m["identity_rows"] for m in experts))
