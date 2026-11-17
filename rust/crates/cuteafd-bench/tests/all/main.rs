//! Integration tests for `cuteafd-bench`, merged into one test binary.
//!
//! Each `mod` below was a top-level `tests/<stem>.rs` binary; keeping the old
//! file stem as the module name keeps ids readable (`<stem>::<test>`) while
//! linking once instead of once per file.

mod common;
mod sustained;
mod sweep_window;
