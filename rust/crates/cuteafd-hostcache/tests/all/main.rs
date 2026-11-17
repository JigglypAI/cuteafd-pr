//! Integration tests for `cuteafd-hostcache`, merged into one test binary.
//!
//! Each `mod` below was a top-level `tests/<stem>.rs` binary; keeping the old
//! file stem as the module name keeps ids readable (`<stem>::<test>`) while
//! linking once instead of once per file.

mod common;
mod common_hc_5;
mod concurrency_copy;
mod concurrency_hc2;
mod concurrency_hc8;
mod concurrency_hc_5;
mod concurrency_hc_7;
mod concurrency_hc_9;
mod concurrency_pool;
mod concurrency_sim;
mod functional_copy;
mod functional_hc2;
mod functional_hc8;
mod functional_hc_5;
mod functional_hc_7;
mod functional_hc_9;
mod functional_pool;
mod functional_sim;
mod soak_hc_6;
mod stability_hc_5;
mod upstream_eviction_invariants;
