//! Hy4-flash (`hy_v4_flash`, `HYV4FlashForConditionalGeneration`) coordinator engine (#2478).
//!
//! Architecture: 42 layers (H=3072, 4-stream iHC + hc_head), 12 DSA layers (32 heads +
//! 16-head compressed indexer C4 top-512/2048) + 30 SWA-512 layers (64 heads), absorbed
//! 576/512 MLA with learnable F32 sinks and sigmoid linear gate, layer 0 dense FFN (13824),
//! layers 1..=41 LatentMoE (3072->1536, 512 EXL3 K4 MUL1 experts top-16, swiglu_limit=10,
//! 1 shared expert), 4 host-resident Engram FP8 tables (layers 1, 11, 21, 31), FP32 LM head.

pub(crate) mod cache;
pub(crate) mod execution;
pub(crate) mod golden;
pub(crate) mod native;
pub(crate) mod programs;
pub(crate) mod weights;
