"""Target verifier exports and flattened drafter covers are independent."""
import importlib.util
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_v41_export_capacity_contract():
    path = ROOT / "python/tools/lib/v41_capacities.py"
    spec = importlib.util.spec_from_file_location("v41_capacities", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (ROOT / "rust/crates/cuteafd-core/src/dspark_verify.rs").read_text()
    assert re.search(r"V41_DECODE_CAPACITIES: \[usize; 3\] = \[1, 16, V41_VERIFY_ROWS\]", source)
    assert module.DECODE_CAPACITIES == (1, 16, module.VERIFY_ROWS)
    assert module.TARGET_CAPACITIES[:3] == module.DECODE_CAPACITIES
    assert module.DRAFT_LEGACY_ROWS not in module.DECODE_CAPACITIES
    assert module.DRAFT_WIDE_ROWS not in module.DECODE_CAPACITIES
    assert module.DRAFT_LEGACY_ROWS == 16 * 5
    assert 32 * 7 <= module.DRAFT_WIDE_ROWS
    cmake = (ROOT / "native/CMakeLists.txt").read_text()
    assert "V41_${name}: usize" in cmake
    assert "set(CUTEAFD_V41_TARGET_ROWS 1 16 ${CUTEAFD_V41_VERIFY_ROWS}" in cmake
    for name in ("v41_tp2_experts", "v41_local_experts", "v41_spark_tp_experts"):
        text = (ROOT / f"native/cmake/families/deepseek_v41/{name}.cmake").read_text()
        assert "CUTEAFD_V41_TARGET_ROWS" in text
        assert "1,16,80,256" not in text


@pytest.mark.parametrize("seed_legacy_cache", [False, True])
@pytest.mark.parametrize("override", [None, "128", "1:128,16:192,64:128,256:192,1024:192,4096:192"])
def test_spark_width_maps_follow_reconfigured_capacities(tmp_path, seed_legacy_cache, override):
    source, build = tmp_path / "source", tmp_path / "build"
    source.mkdir()
    spark = ROOT / "native/cmake/families/deepseek_v41/v41_spark_tp_experts.cmake"
    adjacent = ROOT / "native/cmake/shared/mxfp4_widths.cmake"
    (source / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.24)
project(capacity_reconfigure NONE)
set(CUTEAFD_ENABLE_V41_EXPERT_AOT ON)
set(CUTEAFD_V41_EXPERT_ROLE spark)
set(CUTEAFD_SPARK_TP_ROLES tp2 tp3 tp5 tp6 tp8)
foreach(world 2 3 6 8)
  set(CUTEAFD_NATIVE_ARTIFACTS_v41_${{world}} spark_tp${{world}})
endforeach()
set(CUTEAFD_NATIVE_ARTIFACTS_v41_5 spark_w448)
set(CUTEAFD_NATIVE_ARTIFACTS_dsv4f_5 spark_w448)
set(CUTEAFD_NATIVE_ARTIFACTS_dsv4p_5 spark_w448)
set(CUTEAFD_EXPERT_FAMILIES dsv4f:spark_tp5 dsv4p:spark_tp5)
set(CUTEAFD_V41_TARGET_ROWS 1 16 ${{CUTEAFD_V41_VERIFY_ROWS}} 256 1024 4096)
list(JOIN CUTEAFD_V41_TARGET_ROWS "," CUTEAFD_V41_TARGET_ROWS_ARG)
set(Python3_EXECUTABLE python3)
add_custom_target(cuteafd_verify_sparkinfer_source)
include("{spark.as_posix()}")
include("{adjacent.as_posix()}")
''')

    def configure(rows, *settings):
        result = subprocess.run(["cmake", "-G", "Unix Makefiles", "-S", str(source), "-B", str(build),
                                 f"-DCUTEAFD_V41_VERIFY_ROWS={rows}", *settings],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr

    legacy = "1:192,16:192,80:192,256:192,1024:192,4096:192"
    seed = ([f"-DCUTEAFD_V41_SPARK_TP{world}_SLICE_WIDTH:STRING={legacy}"
             for world in (2, 3, 6, 8)] if seed_legacy_cache else [])
    configure(80, *seed)
    for world in (2, 3, 6, 8):
        assert f"width={legacy}\n" in (build / f"v41_spark_tp{world}_width.stamp").read_text()
    settings = [] if override is None else [f"-DCUTEAFD_V41_SPARK_TP6_SLICE_WIDTH_OVERRIDE={override}"]
    configure(64, *settings)
    expected = legacy.replace("80:192", "64:192")
    for world in (2, 3, 6, 8):
        width = override if world == 6 and override is not None else expected
        assert f"width={width}\n" in (build / f"v41_spark_tp{world}_width.stamp").read_text()
    commands = [line for makefile in build.glob("CMakeFiles/*.dir/build.make")
                for line in makefile.read_text().splitlines()
                if line.startswith("\t") and "export_b12x_slices_aot.py" in line]
    assert len(commands) == 7
    for command in commands:
        rows = "1,16,80,256,1024,4096" if "--geometry dsv4" in command else "1,16,64,256,1024,4096"
        assert f"--rows {rows} " in command
        if "--role spark_tp" in command:
            width = override if "--role spark_tp6 " in command and override is not None else expected
            assert f"--width {width} " in command
    cache = (build / "CMakeCache.txt").read_text()
    if seed_legacy_cache:
        # The effective normal variable must shadow old defaults, not rewrite the cache.
        for world in (2, 3, 6, 8):
            assert f"CUTEAFD_V41_SPARK_TP{world}_SLICE_WIDTH:STRING={legacy}" in cache
    else:
        assert not re.search(r"^CUTEAFD_V41_SPARK_TP\d_SLICE_WIDTH:", cache, re.MULTILINE)


def test_native_manifest_decode_contract_matches_shared_planner_recipe():
    import ast
    path = ROOT / "python/tools/lib/mxfp4_layouts.py"
    spec = importlib.util.spec_from_file_location("capacity_recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    source = ast.parse((ROOT / "python/tools/aot/export_b12x_slices_aot.py").read_text())
    export = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "export")
    record = next(n for n in ast.walk(export) if isinstance(n, ast.keyword) and n.arg == "decode_capacities")
    expression = compile(ast.Expression(record.value), "manifest-decode-capacities", "eval")
    for family in ("v41", "dsv4f", "dsv4p"):
        for role in ("spark", "spark_tp2", "spark_tp3", "spark_tp6", "spark_tp8", "spark_w320", "rtx_backbone", "rtx_tp2", "coordinator", "dspark_tp2"):
            capacities, decode = recipe.native_export_capacities(family, role)
            actual = eval(expression, {"native_export_capacities": recipe.native_export_capacities,
                                      "capacities": capacities, "family": family, "role": role})
            assert actual == list(decode)
            assert decode[-1] == (64 if family == "v41" and role not in ("coordinator", "dspark_tp2") else 80)
    # V4.1 hidden 5120 inherits m80's tile/split recipe at m64 (not measured/tuned).
    # Follow-up: screen E384/top6 on kakapo/maxq.
    for row in recipe.layouts()["native_slices"]:
        if row["hidden"] == 5120 and 80 in row["capacities"]:
            assert recipe.native_slice_width(5120, row["width"], row["sm"], 64, 192) == recipe.native_slice_width(5120, row["width"], row["sm"], 80, 192)
            assert recipe.native_fc1_k_splits(5120, row["width"], row["sm"], 64) == recipe.native_fc1_k_splits(5120, row["width"], row["sm"], 80)


def test_v41_terminal_head_bound_is_derived_for_wide_lanes():
    source = (ROOT / "rust/crates/cuteafd-core/src/dspark_verify.rs").read_text()
    assert "V41_VOCAB_ROWS: usize = V41_MAX_LANE_MEMBERS * (MAX_DSPARK_PROPOSALS + 1)" in source
    cmake = (ROOT / "native/CMakeLists.txt").read_text()
    assert 'CUTEAFD_V41_VOCAB_ROWS "${CUTEAFD_V41_MAX_LANE_MEMBERS} * (${CMAKE_MATCH_1} + 1)"' in cmake
    native = (ROOT / "native/families/deepseek_v41/cuda/v41_dspark.cu").read_text()
    assert "rows > CUTEAFD_V41_VOCAB_ROWS" in native
    assert "vocabulary_merge_greedy<<<1, CUTEAFD_V41_VOCAB_ROWS" in native
    assert "output, 5120, 128" not in native


def test_nvfp4_wave_capacity_metadata_is_format_specific():
    local = (ROOT / "rust/crates/cuteafd-daemon/src/shared/experts/local.rs").read_text()
    rank = (ROOT / "rust/crates/cuteafd-daemon/src/shared/experts/rtx/native.rs").read_text()
    assert local.count("if nvfp4 { library.nvfp4_expert_exports(2)? }") == 2
    assert "if nvfp4 { Self::capacities(capacity)? }" not in local
    assert "if nvfp4 { library.nvfp4_expert_exports(3)? }" in rank
    assert "if nvfp4 { device.library.nvfp4_expert_exports(3)? }" in rank
