//! Integration tests for `cuteafd-core`, merged into one test binary.
//!
//! Each `mod` below was a top-level `tests/<stem>.rs` binary; keeping the old
//! file stem as the module name keeps ids readable (`<stem>::<test>`) while
//! linking once instead of once per file.

mod draft_history_replay;
mod target_sampling_bench;
mod upstream_spec_acceptance;
