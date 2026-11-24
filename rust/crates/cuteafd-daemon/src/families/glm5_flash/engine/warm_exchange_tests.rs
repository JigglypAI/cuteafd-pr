//! Structural coverage of native decode tracing.

fn body<'a>(source: &'a str, first: &str, next: &str) -> &'a str {
    source.split_once(first).unwrap().1.split_once(next).unwrap().0
}

#[test]
fn native_mla_trace_reads_live_stages_with_rank_local_extents() {
    let source = include_str!("../engine.rs");
    let trace = body(source, "fn mla_over(", "fn moe(");
    let native = body(trace, "if native {", "} else {");
    assert!(native.contains("slot.stage(\"q\")?"));
    assert!(native.contains("slot.stage(\"k\")?"));
    assert!(!native.contains("w.scratch"));
    assert!(trace.contains("?.stage(\"values\")?"));
    assert!(trace.contains("t * heads * self.cfg.v_head_dim * 2"));
    assert!(!trace.contains("t * self.cfg.heads"));
}
