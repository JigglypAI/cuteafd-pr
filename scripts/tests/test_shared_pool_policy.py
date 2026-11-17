"""Families declare pool facts; only shared placement resolves policy."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
CRATES = ROOT / "rust/crates"
CORE = Path("cuteafd-loader/src/placement")
EXCEPTIONS = ()
FIELDS = "context|compiled_context|requested|target|floor|ceiling|max|unit_rows"


def violations(text):
    """Follow typed policy fields, inferred preparation bindings and their aliases."""
    names = set(re.findall(r"\b(\w+)\s*:\s*(?:&\s*(?:mut\s+)?)?(?:[\w:]+::)?PoolPolicy\b", text))
    names.update(re.findall(r"\blet\s+(?:mut\s+)?(\w+)\s*(?::[^=;]+)?=\s*(?:[\w:]+::)?prepare_pool\s*\(", text))
    # Match/unwrap wrappers can separate the binding from the preparation call.
    names.update(re.findall(r"\blet\s+(?:mut\s+)?(\w+)\s*=\s*match\s+(?:[\w:]+::)?prepare_pool\s*\(", text))
    changed = True
    while changed:
        changed = False
        for alias, source in re.findall(r"\blet\s+(?:mut\s+)?(\w+)\s*=\s*&?\s*(?:mut\s+)?([\w.]+)\s*;", text):
            if source.split(".")[-1] in names and alias not in names:
                names.add(alias)
                changed = True
    found = []
    patterns = [r"PoolPolicy\s*::\s*resolve\s*\(",
                r"\bPoolPolicy\s*\{[^}]*\b(?:target|floor|ceiling)\s*:"]
    if names:
        bindings = "|".join(re.escape(name) for name in names)
        patterns.append(rf"\b(?:[\w]+\.)?(?:{bindings})\.(?:{FIELDS})\s*(?:=(?!=)|\+=|-=|\*=|/=)")
        patterns.append(rf"&\s*mut\s+(?:[\w]+\.)?(?:{bindings})\.(?:{FIELDS})\b")
    for pattern in patterns:
        found.extend(match.start() for match in re.finditer(pattern, text, re.S))
    return sorted(set(found))


def test_family_pool_policy_has_no_private_resolver_or_mutation():
    assert EXCEPTIONS == ()
    found = []
    for path in CRATES.rglob("*.rs"):
        relative = path.relative_to(CRATES)
        if relative.parent == CORE or relative.is_relative_to(CORE / "planner"):
            continue
        text = path.read_text()
        for offset in violations(text):
            found.append(f"{relative}:{text.count(chr(10), 0, offset) + 1}")
    assert not found, "Private pool policy outside shared placement:\n" + "\n".join(found)


@pytest.mark.parametrize("source", [
    "fn bad(mut renamed: PoolPolicy) { renamed.target = 1; }",
    "let mut renamed = prepare_pool(facts)?; renamed.ceiling = 1;",
    "let renamed = prepare_pool(facts)?; let alias = &mut renamed; alias.floor = 1;",
    "fn bad(renamed: &mut placement::PoolPolicy) { renamed.max += 1; }",
    "fn bad(renamed: PoolPolicy) { change(&mut renamed.target); }",
    "let owner = PoolPolicy { target: 1, floor: 1, ceiling: 1 };",
    "let owner = PoolPolicy::resolve(cards, context, None, 64);",
])
def test_source_audit_rejects_private_policy_variants(source):
    assert violations(source), source


def test_source_audit_accepts_fact_inputs_and_read_only_policy():
    assert not violations("let mut facts = PoolFacts::new(cards, context, None, 64); facts.target = Some(1);")
    assert not violations("fn read(policy: PoolPolicy) { assert_eq!(policy.target(), 1); }")


def test_memory_policy_bounds_have_one_placement_call_site():
    calls = []
    for path in (CRATES / "cuteafd-loader/src").rglob("*.rs"):
        for line, text in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\.kv\s*\(|MemoryPolicy\s*::\s*kv\s*\(", text):
                calls.append((path.relative_to(CRATES).as_posix(), line))
    assert len(calls) == 1, calls
    assert calls[0][0] == "cuteafd-loader/src/placement/pool.rs", calls
    source = (CRATES / "cuteafd-loader/src/placement/pool.rs").read_text()
    for field in ["target", "floor", "max", "ceiling"]:
        assert f"pub(super) {field}:" in source
