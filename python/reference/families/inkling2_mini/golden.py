#!/usr/bin/env python3
"""Score Inkling 2 Mini with native mixed precision and save F32 logits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))

import numpy as np
import torch

from families.inkling2.golden import configure_f32, digest
from families.inkling2.model import head_forward
from families.inkling2_mini.model import CHECKPOINT, REVISION, TEXT, Config, Weights, forward
from fidelity_windows import finish_golden, load_set, verify_snapshot, write_scored_logits
from shape_invariant import ROWS, qualify

PINS = {
    "config.json": "a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2",
    "hf_quant_config.json": "7c656207d3ce37879c7be209a036b0dde43bf01f78159bff06880b868ed57143",
    "model.safetensors.index.json": "c1631065b7b00787993522dddc66a68b3abee869192977fb3dcb796e63bc5838",
}
HF = {"repository": "https://github.com/huggingface/transformers", "revision": "89c632fd826f68ecadd176320e9b2c070918f4f2",
      "version": "5.24.0"}
ROOT_ARITHMETIC = {
    "source": "Independent PyTorch mixed-precision oracle; Inkling 2 Mini checkpoint and Transformers 5.24.0 fixed-loop semantics",
    "arithmetic": "BF16 embeddings, dense projections, attention QK/PV, controller down projections and layer-state boundaries. F32 weighted RMS/L2 and log scale, block/message attention residuals and mixing, controller forget/gain and blending, short convolution, router, joint attention/sink softmax and output sigmoid/multiply. Native NVFP4 per-token fouroversix_mae or MXFP8 group32 QDQ around expert GEMMs, with F32 reduction and BF16 model boundary. F32 head projection from BF16 checkpoint weights and F32 softcap. Distinct causal state per visit; no MTP or vision.",
}


def expected_source_identity(runtime):
    if runtime:
        raise ValueError("Mini reference has no dynamic model dependencies")
    return {"revision": REVISION, "transformers": HF, "snapshot": PINS,
            "vllm": {"revision": "a0c1f162f4ded00f0f40928698b4b5816f4d3bd7", "files": {
                "vllm/models/inkling2/nvidia/loop.py": "94d0d7221c5709b5c4cf8fa96e4be1c82d254d510bb002f860aee6c3b96e896c",
                "vllm/models/inkling2/nvidia/ops/nvfp4.py": "7830385a15c4b57f12aaf4b84103724de0b4a0ba3cf3bafef16bb1e42c9151bd"}},
            "sglang": {"revision": "daba437630ff7c1e02a9b1d88a7d8414376acff7", "files": {
                "python/sglang/srt/models/inkling_common/inkling2_loop.py": "70d8797b523b0fcdaad6e709e14bfc4bbdcc735a4b49824b430997ff1018dd78"}},
            "reference_adapter_sha256": digest(Path(__file__).with_name("model.py")),
            "reference_runner_sha256": digest(__file__),
            "inkling_primitives_sha256": digest(ROOT / "python/reference/families/inkling2/model.py"),
            "shape_invariant_sha256": digest(ROOT / "python/reference/shape_invariant.py")}


def open_snapshot(snapshot, device):
    if snapshot.name != REVISION or any(digest(snapshot / name) != expected for name, expected in PINS.items()):
        raise ValueError("not the pinned official Inkling 2 Mini snapshot")
    cfg = Config.from_dict(json.loads((snapshot / "config.json").read_bytes()))
    return cfg, Weights(snapshot, cfg, device)


def start_f32():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = configure_f32()
    return {key: cfg[key] for key in ("matmul_precision", "torch_matmul_tf32", "cudnn_tf32", "deterministic_algorithms")}


@torch.inference_mode()
def execute(a):
    started = time.monotonic()
    arithmetic = {**start_f32(), "execution": "checkpoint mixed precision", "logits_dtype": "float32"}
    cfg, weights = open_snapshot(a.snapshot, a.device)
    if a.check_only:
        print(f"verified {len(weights.formats)} packed Mini expert projections; {len(weights.contracts)} physical layers")
        return
    manifest = load_set(a.windows, "inkling2_mini")
    if manifest["checkpoint"] != CHECKPOINT or any(w.get("media") for w in manifest["windows"]):
        raise ValueError("Mini golden accepts only its text panel")
    if any(t >= cfg.unpadded_vocab_size for w in manifest["windows"] for t in w["tokens"]):
        raise ValueError("panel token exceeds the unpadded vocabulary")
    snapshot = verify_snapshot(manifest, a.snapshot)
    a.out.mkdir(parents=True, exist_ok=True)
    proof = qualify(a, manifest, execute)
    windows = manifest["windows"]
    hidden = forward(weights, cfg, [torch.tensor(w["tokens"], device=a.device) for w in windows])
    head = weights.tensor(TEXT + "unembed.weight").float()
    weights.close()
    entries = []
    for window, state in zip(windows, hidden, strict=True):
        start, end = window["score_from"], len(window["tokens"])
        logits = np.empty((end - start, cfg.unpadded_vocab_size), dtype="<f4")
        for first in range(start, end, ROWS):
            last = min(first + ROWS, end)
            logits[first-start:last-start] = head_forward(state[first-1:last-1], head, cfg).cpu().numpy()
        entries.append(write_scored_logits(a.out, window, logits))
    rows = sum(entry["scored_tokens"] for entry in entries)
    platform = ({"capability": list(torch.cuda.get_device_capability(a.device)),
                 "gpu": torch.cuda.get_device_name(a.device), "cuda": torch.version.cuda}
                if torch.device(a.device).type == "cuda" else {"device": "cpu"})
    evidence = {"official_forward": json.loads(a.hf_evidence.read_text())} if a.hf_evidence else {}
    finish_golden(a.out, manifest, entries, snapshot_identity=snapshot,
                  source_pin=expected_source_identity({}), source_identity_provider="families.inkling2_mini.golden",
                  harness_commit=os.environ.get("CUTEAFD_REFERENCE_COMMIT", ""), **evidence,
                  nll=sum(e["nll"] * e["scored_tokens"] for e in entries) / rows, scored_tokens=rows,
                  seconds=time.monotonic() - started, platform=platform, torch=torch.__version__,
                  tokenizers=importlib.metadata.version("tokenizers"), transformers="not used in reference forward",
                  attention="eager", mtp=False, use_cache=False, prefix_qualification=proof, fixed_rows=ROWS,
                  arithmetic=arithmetic, root_arithmetic=ROOT_ARITHMETIC,
                  native_fp32_parameters="Declared F32 tensors preserved; BF16 checkpoint tensors used with official F32 islands.")
    print(f"Mini reference complete: {rows} scored rows in {time.monotonic()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hf-evidence", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and (args.windows is None or args.out is None):
        parser.error("--windows and --out are required for scoring")
    execute(args)


if __name__ == "__main__":
    main()
