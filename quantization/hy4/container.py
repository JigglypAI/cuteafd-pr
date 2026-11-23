"""Run the full Hy4 quantizer on maxq with its task-owned environment and CUDA cache."""
from pathlib import Path

from hy4_flash.container import main

if __name__ == "__main__":
    raise SystemExit(main(family="hy4", task=Path.home() / "Developer/quantization-runs/hy4-k4-mc",
                         build=Path.home() / ".cache/cuteafd/builds/hy4-k4-mc"))
