//! Inkling 2 Mini (`inkling2_mini`) TP1 serving & golden family runtime (#2701).
//!
//! 24 physical layers unrolled into 48 visits (prelude 0..6, core 6..18 x 3 passes,
//! coda 18..24) with 48 distinct KV + short-conv cache slots, native W4A4 NVFP4
//! (0..20) + W8A8 MXFP8 (20..24) routed experts, bundled 3-layer recurrent dSpark
//! drafter, and bundled 23-layer PE-Lang vision tower on 1x RTX PRO 6000 (0 Sparks).
//!
//! Authoritative precision contract: reference commit `f1940f40eb`.
pub(crate) mod admission;
pub(crate) mod backend;
pub(crate) mod cache;
pub(crate) mod draft;
pub(crate) mod draft_taps;
pub(crate) mod engine;
pub(crate) mod experts;
pub(crate) mod golden;
pub(crate) mod graph_sites;
pub(crate) mod graphs;
pub(crate) mod head;
pub(crate) mod layers;
pub(crate) mod owners;
pub(crate) mod prefill;
pub(crate) mod serve;
pub(crate) mod startup;
pub(crate) mod verify;
pub(crate) mod verify_target;
pub(crate) mod weights;
