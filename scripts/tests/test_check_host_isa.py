"""Exercise the release ISA guard against actual ELF instruction encodings."""

from pathlib import Path
import platform
import subprocess
import sys

import pytest


CHECK = Path(__file__).resolve().parents[1] / "build/check-host-isa.py"


@pytest.fixture
def build_elf(tmp_path):
    if platform.machine() != "x86_64":
        pytest.skip("requires the host x86-64 assembler")

    def build(relative, code):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["as", "--64", "-o", path, "-"], check=True, text=True,
                       input=f".text\n.globl test\ntest:\n{code}\nret\n")
        return path

    return build


def run(root):
    return subprocess.run([sys.executable, CHECK, root], capture_output=True, text=True)


@pytest.mark.parametrize("code, diagnostic", [
    ("vmovaps %zmm0, %zmm1", "%zmm0"),
    ("vpternlogd $0xff, %xmm0, %xmm1, %xmm2", "%xmm0"),
    ("kmovw %k0, %k1", "%k0"),
])
def test_rejects_avx512_even_without_zmm(build_elf, code, diagnostic):
    result = run(build_elf("wrapper.o", code))
    assert result.returncode == 1
    assert "wrapper.o: AVX-512/EVEX" in result.stderr
    assert diagnostic in result.stderr


def test_accepts_avx2_and_ignores_embedded_data_and_device_elf(tmp_path, build_elf):
    path = build_elf("wrapper.o", "vmovaps %ymm0, %ymm1\n.section .rodata\n.byte 0x62,0xf1,0x7c,0x48,0x28,0xc8")
    device = bytearray(path.read_bytes())
    device[18:20] = b"\xbe\x00"  # EM_CUDA
    (tmp_path / "device.cubin").write_bytes(device)
    result = run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "checked 1 x86-64 ELF files" in result.stdout


def test_only_exact_runtime_variants_are_exempt(tmp_path, build_elf):
    for path in ("bin-variants/x86-64-v4/cuteafd", "lib-variants/x86-64-v4/libcuteafd_native.so"):
        build_elf(path, "vmovaps %zmm0, %zmm1")
    build_elf("fp8/valid.so", "vpxor %ymm0, %ymm0, %ymm0")
    assert run(tmp_path).returncode == 0
    build_elf("lib-variants/x86-64-v4/fp8/wrapper.so", "vmovaps %zmm0, %zmm1")
    result = run(tmp_path)
    assert result.returncode == 1
    assert "lib-variants/x86-64-v4/fp8/wrapper.so" in result.stderr


def test_missing_root_fails(tmp_path):
    assert run(tmp_path / "missing").returncode == 2
