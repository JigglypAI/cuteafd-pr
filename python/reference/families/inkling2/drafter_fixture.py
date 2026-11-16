"""Real-weight oracle fixture producer, run inside the native GPU test harness.

This only writes reference data; it never builds or launches a server. Synthetic
features are deliberately deterministic but all 84 drafter operands are real.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from families.inkling2.drafter import Config, Drafter
from families.inkling2.sources import VLLM, SGLANG


def create(snapshot, output, device):
    raw = json.loads((snapshot / "config.json").read_text())
    config = Config.from_dict(raw)
    with safe_open(str(snapshot / "mtp.safetensors"), framework="pt", device=str(device)) as file:
        weights = {name: file.get_tensor(name) for name in file.keys()}
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]

    def target(name):
        with safe_open(str(snapshot / index[name]), framework="pt", device=str(device)) as file:
            return file.get_tensor(name)

    draft = Drafter(config, weights, target("model.llm.embed.weight"),
                    target("model.llm.embed_norm.weight"), target("model.llm.unembed.weight"))
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    features = ((torch.arange(2048*len(config.taps)*config.hidden, device=device, dtype=torch.int64)
                 .reshape(2048, len(config.taps), config.hidden)*17 % 251).float()/128-1).bfloat16()
    # Includes a wrap beyond the local window and the real full/global layer.
    for rows in (3, 1031, 2048):
        result = draft.forward(101, features[:rows])
        tensors = {"features": features[:rows], "hidden": result["hidden"], "logits": result["logits"],
                   "states": result["states"], "tokens": result["tokens"].to(torch.int32)}
        for layer, (key, value) in enumerate(result["contexts"]):
            tensors[f"context.{layer}.key"] = key
            tensors[f"context.{layer}.value"] = value
        save_file({name: value.contiguous().cpu() for name, value in tensors.items()},
                  output / f"rows-{rows}.safetensors")
    receipt = {"schema": "cuteafd.inkling2.recurrent-oracle/1", "checkpoint": str(snapshot),
               "anchor": 101, "rows": [3, 1031, 2048], "reference_vllm": VLLM["revision"],
               "reference_sglang": SGLANG["revision"], "features": "deterministic arange*17%251/128-1 BF16",
               "real_operands": len(weights), "predictions": config.block-1, "valid_vocab": config.valid_vocab,
               "reference_sha256": hashlib.sha256(Path(__file__).with_name("drafter.py").read_bytes()).hexdigest()}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    create(args.snapshot, args.output, torch.device(args.device))
