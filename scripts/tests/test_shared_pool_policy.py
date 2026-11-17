"""Families declare pool facts; only shared placement resolves policy."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CRATES = ROOT / "rust/crates"
CORE = Path("cuteafd-loader/src/placement")
EXCEPTIONS = ()
FIELDS = "context|compiled_context|requested|target|floor|ceiling|max|unit_rows"


def test_family_pool_policy_has_no_private_resolver_or_mutation():
    assert EXCEPTIONS == ()
    violations = []
    for path in CRATES.rglob("*.rs"):
        relative = path.relative_to(CRATES)
        # Shared placement implementation and its direct tests are the core.
        in_core = relative.parent == CORE or relative.is_relative_to(CORE / "planner")
        if in_core:
            continue
        text = path.read_text()
        patterns = [r"PoolPolicy\s*::\s*resolve\s*\(",
                    rf"\b(?:\w+\.)?pool\.(?:{FIELDS})\s*=(?!=)",
                    r"\bPoolPolicy\s*\{[^}]*\b(?:target|floor|ceiling)\s*:"]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.S):
                violations.append(f"{relative}:{text.count(chr(10), 0, match.start()) + 1}: {match.group(0)[:100]}")
    assert not violations, "Private pool policy outside shared placement:\n" + "\n".join(violations)


def test_memory_policy_bounds_have_one_placement_call_site():
    calls = []
    for path in (CRATES / "cuteafd-loader/src").rglob("*.rs"):
        for line, text in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\b(?:policy|memory_policy)\.kv\s*\(", text):
                calls.append((path.relative_to(CRATES).as_posix(), line))
    assert len(calls) == 1, calls
    assert calls[0][0] == "cuteafd-loader/src/placement/pool.rs", calls
    source = (CRATES / "cuteafd-loader/src/placement/pool.rs").read_text()
    assert "MemoryPolicy::current()" in source
    assert "pub(super) target:" in source
    assert "pub(super) floor:" in source
    assert "pub(super) max:" in source
