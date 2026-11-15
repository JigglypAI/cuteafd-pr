//! Canonical owner-reduced TP2 row and persistent transport layout.
use crate::placement::Demand;
use cuteafd_core::memory_layout::{Basis, Category};

/// Exact TP2 row ABI shared by admission and the owner executor. EXL3 sends
/// E4M3/K32 wire rows and produces FP32 partials; other native packages can
/// independently select BF16 input or BF16 partials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerTp2Rows {
    pub input: u64,
    pub ids: u64,
    pub weights: u64,
    pub bytes: u64,
    pub partial_bytes: u64,
    pub reduction_bytes: u64,
    pub exchange_slot_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_partial_precision_preserves_legacy_plans() -> anyhow::Result<()> {
        for hidden in [32u64, 2560, 4096, 4608, 6144, 7168] {
            for topk in [1u64, 4, 8, 10] {
                for rows in [1u64, 16, 64, 80, 2048, 2752, 4096] {
                    for wire in [false, true] {
                        let legacy = OwnerTp2Rows::new(hidden, topk, rows, wire)?;
                        assert_eq!(legacy, OwnerTp2Rows::with_partial(hidden, topk, rows, wire,
                            if wire { 4 } else { 2 })?);
                        let native = OwnerTp2Rows::with_partial(hidden, topk, rows, wire, 2)?;
                        assert_eq!((native.input, native.ids, native.weights, native.bytes),
                            (legacy.input, legacy.ids, legacy.weights, legacy.bytes));
                        assert_eq!(native.partial_bytes, rows * hidden * 2);
                        assert_eq!(native.reduction_bytes, legacy.reduction_bytes);
                        assert!(native.exchange_slot_bytes >= native.bytes.max(native.partial_bytes));
                    }
                }
            }
        }
        for bytes in [0, 1, 3, 8, u64::MAX] {
            assert!(OwnerTp2Rows::with_partial(4608, 8, 64, true, bytes).is_err());
        }
        Ok(())
    }
}

impl OwnerTp2Rows {
    pub fn new(hidden: u64, topk: u64, rows: u64, wire: bool) -> anyhow::Result<Self> {
        Self::with_partial(hidden, topk, rows, wire, if wire { 4 } else { 2 })
    }

    /// Native input encoding and output partial precision are independent.
    /// The legacy constructor retains every existing family's exact layout.
    pub fn with_partial(hidden: u64, topk: u64, rows: u64, wire: bool, partial_bytes: u64) -> anyhow::Result<Self> {
        anyhow::ensure!(matches!(partial_bytes, 2 | 4), "owner TP2 partial must be BF16 or F32");
        anyhow::ensure!(hidden > 0 && topk > 0 && rows > 0 && (!wire || hidden % 32 == 0),
            "invalid owner TP2 row geometry");
        let overflow = || anyhow::anyhow!("owner TP2 row extent overflow");
        let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(overflow);
        let align = |bytes: u64| bytes.checked_add(15).map(|n| n / 16 * 16).ok_or_else(overflow);
        let stride = if wire { hidden.checked_add(hidden / 32).ok_or_else(overflow)? }
            else { mul(hidden, 2)? };
        let input = mul(rows, stride)?;
        let ids = align(input)?;
        let route_bytes = align(mul(mul(rows, topk)?, 4)?)?;
        let weights = ids.checked_add(route_bytes).ok_or_else(overflow)?;
        let bytes = weights.checked_add(route_bytes).ok_or_else(overflow)?;
        let partial_bytes = mul(mul(rows, hidden)?, partial_bytes)?;
        Ok(Self { input, ids, weights, bytes, partial_bytes,
            reduction_bytes: mul(mul(rows, hidden)?, 2)?,
            exchange_slot_bytes: align(bytes.max(partial_bytes))? })
    }

    /// Persistent buffers of the abortable two-slot expert exchange. Native
    /// publisher modules and streams belong to the context/module baseline.
    pub fn demands(self) -> anyhow::Result<Vec<Demand>> {
        let receive = self.exchange_slot_bytes.max(256).checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("owner TP2 receive extent overflow"))?;
        Ok((0..2).flat_map(|gpu| [
            Demand::new(gpu, Category::Transport, "expert send", self.bytes, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert receive", receive, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert reduction", self.reduction_bytes, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert peer control", 256, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert abort word", 256, Basis::Formula),
        ]).collect())
    }
}

