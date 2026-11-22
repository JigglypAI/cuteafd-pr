"""Audit the complete pinned Tencent Hy4 source through its index and tensor headers."""
from hy4_flash.preflight import run_cli
from .source import HeaderSource

if __name__ == "__main__":
    run_cli(HeaderSource, __doc__)
