#!/usr/bin/env python3
"""Reject AVX-512 in portable x86-64 ELF code, not in embedded GPU data."""

import argparse
from pathlib import Path
import re
import subprocess
import sys


# Only these runtime-selected variants may contain AVX-512. In particular,
# neither an export nor a library nested under one of these directories is exempt.
VARIANTS = {
    Path("bin-variants/x86-64-v4/cuteafd"),
    Path("lib-variants/x86-64-v4/libcuteafd_native.so"),
}

# GNU objdump -dw prints the complete instruction on one line. EVEX can use
# XMM/YMM without an opmask. Conversely, some opmask instructions use VEX.
INSTRUCTION = re.compile(rb"^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)\s+(.+)")
EVEX = re.compile(rb"^(?:(?:26|2e|36|3e|64|65|66|67|f0|f2|f3) )*62 ")
REGISTERS = re.compile(rb"%(?:zmm[0-9]+|k[0-7])\b")


def x86_elf(path):
    with path.open("rb") as handle:
        header = handle.read(20)
    return header[:6] == b"\x7fELF\x02\x01" and header[18:20] == b"\x3e\x00"


def forbidden_instruction(path):
    with subprocess.Popen(["objdump", "-dw", "--", path], stdout=subprocess.PIPE) as command:
        for line in command.stdout:
            match = INSTRUCTION.match(line)
            if match and (EVEX.match(match[1]) or REGISTERS.search(match[2])):
                command.terminate()
                return line.decode("ascii", errors="replace").strip()
        if command.wait():
            raise RuntimeError(f"objdump could not disassemble {path}")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()
    checked = failed = 0
    try:
        for root in args.roots:
            if not root.is_file() and not root.is_dir():
                parser.error(f"missing artifact path: {root}")
            paths = (root,) if root.is_file() else sorted(root.rglob("*"))
            for path in paths:
                if not path.is_file() or (path in (root / p for p in VARIANTS)) or not x86_elf(path):
                    continue
                checked += 1
                if instruction := forbidden_instruction(path):
                    failed += 1
                    print(f"{path}: AVX-512/EVEX: {instruction}", file=sys.stderr)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"host ISA check failed: {exc}\n")
    print(f"Host ISA: checked {checked} x86-64 ELF files; {failed} rejected")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
