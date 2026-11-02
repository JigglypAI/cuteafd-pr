"""Target verifier exports and flattened drafter covers are independent."""
import importlib.util
from pathlib import Path
import re

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
