"""Maxq-only quantization container with a task-owned runtime and build cache."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess

from kimi_k3.core_k6.container import command as sm120_command

REPO = Path(__file__).resolve().parents[2]
TASK = Path.home() / ".cache/cuteafd/quantization/hy4-flash-k4-mc"
BUILD = Path.home() / ".cache/cuteafd/builds/hy4-flash-k4-mc"
SOURCE = Path.home() / (".cache/huggingface/hub/models--tencent--Hy4-flash/snapshots/"
                        "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959")


def command(arguments, phase):
    if socket.gethostname().split(".")[0] != "maxq":
        raise ValueError("Hy4 quantization must run on maxq")
    if phase not in ("build", "pilot", "production"):
        raise ValueError("unrecognized Hy4 container phase")
    python = TASK / "environment/bin/python"
    argv = sm120_command(arguments, name="cuteafd-hy4-flash-" + phase,
                         memory_gib=80, build_root=BUILD / "exllamav3")
    argv[argv.index("--workdir") + 1] = str(REPO)
    argv[argv.index("--entrypoint") + 1] = str(python)
    for i, value in enumerate(argv):
        if value.startswith("LD_LIBRARY_PATH="):
            torch_lib = TASK / "environment/lib/python3.12/site-packages/torch/lib"
            argv[i] = f"LD_LIBRARY_PATH={torch_lib}:/usr/local/cuda-13.4/lib64"
    image = next(i for i, arg in enumerate(argv) if arg.startswith("cuteafd-coordinator-dev:"))
    argv[image:image] = [
        "-e", f"PYTHONPATH={REPO}/quantization:{REPO}/python/reference",
        "-e", f"TORCH_EXTENSIONS_DIR={BUILD}/torch-extensions",
        "-e", f"CUDA_CACHE_PATH={BUILD}/cuda-cache",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "PYTHONUNBUFFERED=1",
    ]
    if phase == "production":
        argv[image:image] = ["-e", "GPTQMODEL_EXL3_SPAN_PAUSE_MS=1"]
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "pilot", "production"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    name = "cuteafd-hy4-flash-" + args.phase
    child = None

    def interrupted(signum, frame):
        raise InterruptedError(f"Hy4 container launcher received signal {signum}")

    for value in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(value, interrupted)
    try:
        child = subprocess.Popen(command(args.arguments, args.phase))
        return child.wait()
    finally:
        if child is not None and child.poll() is None:
            subprocess.run(["docker", "stop", "--time", "20", name], timeout=35,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            child.wait(timeout=20)


if __name__ == "__main__":
    raise SystemExit(main())
