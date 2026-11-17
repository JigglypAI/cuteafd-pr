"""Serving captures require shared-cache tickets; raw CUDA entry points stay local."""
from pathlib import Path
import re

from test_resolved_report_settings import production

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "rust/crates/cuteafd-daemon/src"
GRAPH = SOURCE / "shared/decode_graph"


def test_raw_cuda_graph_calls_stay_in_shared_cache_or_test_oracles():
    violations = []
    for path in SOURCE.rglob("*.rs"):
        if path.is_relative_to(GRAPH) or path == GRAPH.with_suffix(".rs"):
            continue
        # These modules are included only behind cfg(test) in their parents.
        if path.stem == "tests" or path.stem.endswith("_tests"):
            continue
        _, masked = production(path.read_text())
        for match in re.finditer(r"\b(?:cuda_graph_begin_capture|cuda_graph_launch)\s*\(", masked):
            line = masked.count("\n", 0, match.start()) + 1
            violations.append(f"{path.relative_to(ROOT)}:{line}")
    assert not violations, "raw serving graph calls bypass shared cache: " + ", ".join(violations)


def test_graph_adapters_have_no_unattached_insert_or_startup_policies():
    for name in ("bank.rs", "native.rs", "binding.rs"):
        _, masked = production((GRAPH / name).read_text())
        for removed in ("GraphPolicy", "declare_startup", "finish_startup", "seal_startup",
                        "retire_all", "drain_retired", "enable_small_shapes", "would_retire"):
            assert not re.search(rf"\b{removed}\b", masked), (name, removed)
        assert not re.search(r"(?:pub\s+)?(?:unsafe\s+)?fn\s+insert\s*\(", masked), name


def test_capture_funnel_requires_a_live_ticket():
    source = GRAPH.with_suffix(".rs").read_text()
    begin = source.split("unsafe fn begin_capture(", 1)[1].split("\n}", 1)[0]
    assert "ticket: &CaptureTicket" in begin
    assert begin.index("ticket.require_active()?") < begin.index("cuda_graph_begin_capture")
    bank = (GRAPH / "bank.rs").read_text()
    assert 'expect("graph capture requires an attached device cache")' in bank
    assert "Arc::ptr_eq(&binding.cache, &ticket.cache)" in bank


def test_sparse_inline_and_deferred_records_share_host_rollback():
    family = SOURCE / "families/deepseek_v41"
    source = (family / "v41_sparse_attention.rs").read_text()
    # Both miss paths must enter the same transaction, never record directly.
    finish = source.split("pub async unsafe fn finish_prepare(", 1)[1].split("unsafe fn capture_plan(", 1)[0]
    inline = source.split("let graph = if let Some(graph) = cached", 1)[1]
    for path in (finish, inline):
        assert "self.capture_plan(&mut plan, &mut tail)" in path
        assert "self.record_plan(" not in path
    capture = source.split("unsafe fn capture_plan(", 1)[1].split("unsafe fn record_plan(", 1)[0]
    assert capture.index("tail.checkpoint()") < capture.index("capture_transaction(")
    assert "tail.rollback(checkpoint)" in capture


def test_dual_tail_rolls_back_before_rejected_record_eager_retry():
    source = (SOURCE / "families/deepseek_v41/v41_sparse_attention/dual.rs").read_text()
    tail = source.split("pub async unsafe fn complete_owned_tail(", 1)[1].split("pub unsafe fn enqueue_cached", 1)[0]
    assert tail.index("tail.restore_warmup()?") < tail.index("tail.checkpoint()")
    record = tail.split("let recorded =", 1)[1].split("let (graph,nodes,width,spans,inspection) = match recorded", 1)[0]
    assert "capture_transaction(" in record
    assert "tail.enqueue(&attention,stream)" in record
    assert "tail.rollback(checkpoint)" in record
    assert tail.index("tail.rollback(checkpoint)") < tail.index("=> return tail.enqueue(&attention, stream)")
