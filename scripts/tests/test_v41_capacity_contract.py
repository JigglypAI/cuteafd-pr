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
