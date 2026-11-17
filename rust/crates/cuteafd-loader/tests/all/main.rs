//! Integration tests for `cuteafd-loader`, merged into one test binary.
//!
//! Each `mod` below was a top-level `tests/<stem>.rs` binary; keeping the old
//! file stem as the module name keeps ids readable (`<stem>::<test>`) while
//! linking once instead of once per file.

mod ownership_replay;
mod replicated_tp_staging;
mod upstream_container_invariants;
