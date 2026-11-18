//! Per-site rank-local/spanning classification and the `GRAPH_PREWARM=off`
//! startup scope (#2303).
//!
//! A site spans devices when its captured graph contains a peer wait, a peer
//! copy or a kernel on a second device. Spanning sites are unevictable, never
//! donate and are sealed after readiness (#2251, #1878), so they always
//! prewarm. `off` skips startup capture only for rank-local sites; those
//! capture lazily on first use through the same bounded cache and reserve.

use std::cell::Cell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    /// Rank-local operands only; a composite caller may still promote it.
    Local,
    /// The caller's layout flag (rank count, peer stream, context layout) is the dependency.
    Declared,
    /// Always contains a peer dependency.
    Spanning,
}

/// (visitor/site name, scope, capture and host-issue-order evidence).
pub(crate) type Table = [(&'static str, Scope, &'static str)];

/// An unclassified site counts as spanning: it stays prewarmed and sealed.
pub(crate) fn spans(table: &Table, name: &str, declared: bool) -> bool {
    table.iter().find(|(site, ..)| *site == name)
        .is_none_or(|&(_, scope, _)| declared || scope == Scope::Spanning)
}

thread_local! {
    // Startup prewarm runs on the engine thread that owns every capture.
    static STARTUP: Cell<u32> = const { Cell::new(0) };
}

/// Marks this thread's startup prewarm. Under `off` the bank declines tickets
/// for rank-local sites inside it; runtime misses outside it capture lazily.
pub(crate) struct Startup(());
impl Drop for Startup {
    fn drop(&mut self) { STARTUP.with(|depth| depth.set(depth.get() - 1)); }
}

/// `None` when prewarm is off and this layout has no spanning site: the family
/// skips startup capture, marks its graphs ready and captures on first use.
pub(crate) fn startup(spanning: bool) -> Option<Startup> {
    if !keeps(spanning) { return None; }
    STARTUP.with(|depth| depth.set(depth.get() + 1));
    Some(Startup(()))
}

/// Whether startup prewarm keeps a site of this class.
pub(crate) fn keeps(spanning: bool) -> bool { spanning || super::prewarm_enabled() }

/// Startup under `off`: rank-local tickets are deferred to first use.
pub(crate) fn deferring() -> bool {
    !super::prewarm_enabled() && STARTUP.with(|depth| depth.get() > 0)
}
pub(crate) fn defers(spans_devices: bool) -> bool { !spans_devices && deferring() }

#[cfg(test)]
pub(crate) mod testing {
    use super::{Scope, Table};
    use std::collections::BTreeSet;

    fn literal(rest: &str) -> Option<String> {
        let rest = rest.trim_start().strip_prefix('"')?;
        let end = rest.find('"')?;
        let name = &rest[..end];
        (!name.is_empty() && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'.'))
            .then(|| name.to_owned())
    }

    /// Top-level arguments of a call whose opening parenthesis precedes `rest`.
    fn arguments(rest: &str) -> Option<Vec<&str>> {
        let (mut depth, mut from, mut args, mut quoted) = (0i32, 0, Vec::new(), false);
        for (index, ch) in rest.char_indices() {
            if quoted { if ch == '"' { quoted = false; } continue; }
            match ch {
                '"' => quoted = true,
                '(' | '[' | '{' => depth += 1,
                ')' if depth == 0 => { args.push(rest[from..index].trim()); return Some(args); }
                ')' | ']' | '}' => depth -= 1,
                ';' => return None,
                ',' if depth == 0 => { args.push(rest[from..index].trim()); from = index + 1; }
                _ => {}
            }
        }
        None
    }

    /// Site names that reach a bank attach (`Site { name: "..." }`, a `.site("...")`
    /// helper) or a graph visitor call `(device, "name", spans, &mut bank)`.
    pub(crate) fn site_names(source: &str) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        for (at, _) in source.match_indices("Site {") {
            let body = &source[at..];
            let body = &body[..body.find('}').unwrap_or(body.len())];
            if let Some(name) = body.find("name: ").and_then(|name| literal(&body[name + 6..])) { names.insert(name); }
        }
        for (at, _) in source.match_indices(".site(") {
            if let Some(name) = literal(&source[at + 6..]) { names.insert(name); }
        }
        for (at, _) in source.match_indices('(') {
            let before = source[..at].trim_end().chars().next_back();
            if !before.is_some_and(|ch| ch.is_alphanumeric() || ch == '_') { continue; }
            let Some(args) = arguments(&source[at + 1..]) else { continue };
            if args.len() == 4 && args[3].starts_with("&mut") {
                if let Some(name) = literal(args[1]) { names.insert(name); }
            }
        }
        names
    }

    /// Drop every item annotated `#[cfg(test)]`, skipping strings while matching braces.
    pub(crate) fn without_tests(text: &str) -> String {
        let bytes = text.as_bytes();
        let (mut out, mut at) = (String::new(), 0);
        while let Some(found) = text[at..].find("#[cfg(test)]") {
            let start = at + found;
            out.push_str(&text[at..start]);
            let mut index = start + "#[cfg(test)]".len();
            let mut depth = 0usize;
            while index < bytes.len() {
                match bytes[index] {
                    b'r' if text[index..].starts_with("r#\"") => {
                        index += text[index + 3..].find("\"#").map_or(bytes.len(), |end| end + 5);
                        continue;
                    }
                    b'"' => {
                        index += 1;
                        while index < bytes.len() && bytes[index] != b'"' {
                            index += if bytes[index] == b'\\' { 2 } else { 1 };
                        }
                    }
                    b'\'' if bytes.get(index + 2) == Some(&b'\'') => index += 2,
                    b';' if depth == 0 => { index += 1; break; }
                    b'{' => depth += 1,
                    // A statement attribute ends at its enclosing block's close.
                    b'}' if depth == 0 => break,
                    b'}' => { depth -= 1; if depth == 0 { index += 1; break; } }
                    _ => {}
                }
                index += 1;
            }
            at = index.min(bytes.len());
        }
        out.push_str(&text[at..]);
        out
    }

    /// Production source of one family directory, without test modules or test files.
    pub(crate) fn family_source(family: &str) -> String {
        fn walk(dir: &std::path::Path, out: &mut String) {
            let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().map(|entry| entry.unwrap().path()).collect();
            entries.sort();
            for path in entries {
                if path.is_dir() { walk(&path, out); continue; }
                let file = path.file_name().unwrap().to_string_lossy();
                if !file.ends_with(".rs") || file.contains("test") { continue; }
                let text = std::fs::read_to_string(&path).unwrap();
                out.push_str(&without_tests(&text));
                out.push('\n');
            }
        }
        let mut source = String::new();
        walk(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/families").join(family), &mut source);
        source
    }

    /// `GRAPH_PREWARM=off` state transitions for one site on one layout: a
    /// spanning bank is warmed at startup, a rank-local bank leaves no startup
    /// record, and its first serving miss captures. Returns whether it spans.
    pub(crate) fn assert_off_transitions(table: &Table, name: &'static str, declared: bool) -> bool {
        use crate::shared::decode_graph::{GraphBank, device::{Basis, Cycle, Preference, Site}};
        static NEXT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-23_030_000);
        let spanning = super::spans(table, name, declared);
        let site = Site { name, cycle: Cycle::Decode, preference: Preference::Adaptive,
            bootstrap: 10, basis: Basis::Bytes, spans_devices: spanning };
        let device = NEXT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        let mut bank: GraphBank<u32, ()> = GraphBank::new();
        bank.attach_device_for_test(device, 1000, site).unwrap();
        let _off = crate::shared::decode_graph::testing::prewarm_off_on_this_thread();
        let startup = super::startup(spanning);
        assert_eq!(startup.is_some(), spanning, "{name}: only a spanning layout keeps a startup pass");
        if let Some(_startup) = startup {
            let ticket = bank.cache_ticket(&1, false, None).unwrap_or_else(|| panic!("{name}: spanning startup capture"));
            bank.cache_commit(1, (), ticket, 4, 1, true);
            assert!(bank.contains(&1), "{name}: spanning startup shape is resident at readiness");
        }
        let lazy = if spanning { 2 } else { 1 };
        let ticket = bank.cache_ticket(&lazy, false, None)
            .unwrap_or_else(|| panic!("{name}: first-use capture after startup"));
        bank.cache_commit(lazy, (), ticket, 4, 1, spanning);
        assert!(bank.contains(&lazy), "{name}: first-use capture is retained");
        spanning
    }

    /// Every production site of `family` is classified with source evidence.
    pub(crate) fn assert_classified(family: &str, table: &Table) {
        let mut seen = BTreeSet::new();
        for &(name, scope, evidence) in table {
            assert!(seen.insert(name), "{family}: duplicate site {name}");
            assert!(evidence.contains(".rs:"), "{family}: missing source evidence for {name}");
            assert!(super::spans(table, name, true), "{family}: a declared peer dependency must promote {name}");
            assert_eq!(super::spans(table, name, false), scope == Scope::Spanning);
        }
        let used = site_names(&family_source(family));
        assert!(!used.is_empty(), "{family}: no graph sites found in production source");
        let missing: Vec<_> = used.iter().filter(|name| !seen.contains(name.as_str())).collect();
        assert!(missing.is_empty(), "{family}: unclassified graph sites {missing:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &Table = &[
        ("local", Scope::Local, "fixture.rs:1 local"),
        ("declared", Scope::Declared, "fixture.rs:2 layout"),
        ("spanning", Scope::Spanning, "fixture.rs:3 peer"),
    ];

    #[test]
    fn unclassified_sites_are_spanning_and_declared_flags_promote() {
        assert!(spans(TABLE, "new-unreviewed-site", false));
        assert!(!spans(TABLE, "local", false) && spans(TABLE, "local", true));
        assert!(!spans(TABLE, "declared", false) && spans(TABLE, "declared", true));
        assert!(spans(TABLE, "spanning", false));
    }

    #[test]
    fn off_skips_all_local_layouts_and_defers_local_sites_only_inside_startup() {
        let _off = crate::shared::decode_graph::testing::prewarm_off_on_this_thread();
        assert!(startup(false).is_none(), "an all-local layout skips startup capture");
        assert!(!deferring(), "runtime misses outside startup capture lazily");
        {
            let _scope = startup(true).expect("spanning layouts still prewarm");
            assert!(defers(false) && !defers(true));
        }
        assert!(!defers(false));
    }

    #[test]
    fn spanning_layout_startup_defers_its_rank_local_banks_to_first_use() {
        use crate::shared::decode_graph::{GraphBank, device::{Basis, Cycle, Preference, Site}};
        let site = |name, spans_devices| Site { name, cycle: Cycle::Decode, preference: Preference::Adaptive,
            bootstrap: 10, basis: Basis::Bytes, spans_devices };
        let mut local: GraphBank<u32, ()> = GraphBank::new();
        local.attach_device_for_test(-23_031_001, 1000, site("local", false)).unwrap();
        let mut peer: GraphBank<u32, ()> = GraphBank::new();
        peer.attach_device_for_test(-23_031_002, 1000, site("spanning", true)).unwrap();
        let _off = crate::shared::decode_graph::testing::prewarm_off_on_this_thread();
        {
            let _startup = startup(true).unwrap();
            assert!(local.cache_ticket(&1, false, None).is_none(), "rank-local startup shape is deferred");
            let ticket = peer.cache_ticket(&1, false, None).expect("spanning startup shape still captures");
            peer.cache_commit(1, (), ticket, 4, 1, true);
        }
        let ticket = local.cache_ticket(&1, false, None).expect("rank-local first use captures");
        local.cache_commit(1, (), ticket, 4, 1, false);
        assert!(local.contains(&1) && peer.contains(&1));
    }

    #[test]
    fn unclassified_site_is_prewarmed_under_off() {
        assert!(testing::assert_off_transitions(TABLE, "new-unreviewed-site", false));
    }

    #[test]
    fn on_never_defers() {
        let _scope = startup(false).expect("default prewarm on");
        assert!(!defers(false) && !defers(true));
    }

    #[test]
    fn test_items_are_removed_but_later_production_code_is_kept() {
        let text = "a();\n#[cfg(test)]\nuse x::y;\nb();\n#[cfg(test)]\nmod tests { fn f() { let s = \"}\"; let c = '}'; } }\nc();\n";
        let kept = testing::without_tests(text);
        assert!(kept.contains("a();") && kept.contains("b();") && kept.contains("c();"));
        assert!(!kept.contains("x::y") && !kept.contains("fn f"));
    }

    #[test]
    fn site_scanner_reads_attach_and_visitor_names() {
        let source = r#"
            bank.attach(lib, device, reserve, Site { name: "fam.attach", cycle });
            f(self.b(0).device_id, "fam.child", false, &mut self.graphs)?;
            visit(self.device,
                "fam.multi", spans, &mut self.bank)?;
            self.site("fam.site", ranks, rank)?;
            call(x, "not.a.site", y);
            other(x, "not.either", y, z);
            let site = Site { cycle, name: "fam.later" };
        "#;
        let names: Vec<_> = testing::site_names(source).into_iter().collect();
        assert_eq!(names, ["fam.attach", "fam.child", "fam.later", "fam.multi", "fam.site"]);
    }
}
