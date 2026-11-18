"""#127: AOT exports pin CuTe DSL's host target to a portable baseline instead
of the build host's CPU (python/tools/lib/_pinned_sparkinfer.py)."""
import ast
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / 'python/tools/lib/_pinned_sparkinfer.py').read_text()


def test_every_exporter_imports_the_shared_host_pin():
    for path in sorted((ROOT / 'python/tools/aot').glob('*.py')):
        text = path.read_text()
        if 'cute.compile' in text or 'b12x' in text:
            assert '_pinned_sparkinfer' in text, path.name
    assert 'from b12x._lib.aot_host import pin_aot_host_target' in SOURCE
    assert '\n    pin_aot_host_target()\n' in SOURCE


def test_manifests_record_the_compiled_target_not_the_export_hosts_gpu_name():
    pin = types.ModuleType("pin")
    tree = ast.parse(SOURCE)
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'export_target')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'pin', 'exec'), pin.__dict__)
    props = types.SimpleNamespace(major=12, minor=0, name='NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition')
    assert pin.export_target(props) == 'sm120'
    v41 = (ROOT / 'python/tools/aot/export_b12x_v41_experts_aot.py').read_text()
    assert 'from export_b12x_slices_aot import export as export_slices' in v41
    for name in ('export_b12x_slices_aot.py', 'export_b12x_v41_nvfp4_aot.py', 'export_b12x_exl3_aot.py',
                 'export_b12x_v41_fp8_aot.py'):
        text = (ROOT / 'python/tools/aot' / name).read_text()
        assert 'props.name' not in text and 'properties.name' not in text, name
        assert '_pinned_sparkinfer.export_target(' in text, name


def test_cuda_objects_are_seeded_per_translation_unit():
    # nvcc embedded tmpxft_<pid> names and random per-TU IDs: two builds of the
    # same source never matched. Every .cu compile gets -objtemp and its own seed.
    cmake = (ROOT / 'native/CMakeLists.txt').read_text()
    block = cmake[cmake.index('function(cuteafd_reproducible_cuda_objects)'):]
    assert '"-objtemp" "--frandom-seed=${target}/${relative}"' in block
    assert 'file(RELATIVE_PATH relative "${CMAKE_CURRENT_SOURCE_DIR}"' in block
    assert 'cmake_language(DEFER CALL cuteafd_reproducible_cuda_objects)' in cmake


def test_export_manifests_never_iterate_directories_unsorted():
    # Directory order is filesystem-specific: an unsorted iterdir/glob into a
    # manifest dict made identical exports differ between hosts (#127).
    import re
    for path in sorted((ROOT / 'python/tools/aot').glob('export_*.py')):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'for \w+ in [\w.]+\.(iterdir|glob|rglob)\(', line):
                assert 'sorted(' in line, f'{path.name}:{number}: {line.strip()}'
