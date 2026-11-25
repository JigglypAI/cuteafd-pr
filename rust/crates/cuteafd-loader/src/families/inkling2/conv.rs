//! Contiguous FIR state geometry shared by replay producers and admission.
use anyhow::{ensure, Context, Result};
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvLayout {
    pub slots: usize,
    pub layers: usize,
    pub hidden: usize,
    pub width: usize,
    pub lag_stride: usize,
    pub layer_stride: usize,
    pub slot_stride: usize,
}
impl ConvLayout {
    pub fn new(slots: usize, layers: usize, hidden: usize, kv: usize) -> Result<Self> {
        ensure!(
            slots > 0 && layers > 0 && layers <= 48 && matches!(hidden, 4608 | 2304) && matches!(kv, 1024 | 512),
            "Inkling FIR release geometry"
        );
        let width = hidden
            .checked_mul(2)
            .and_then(|n| n.checked_add(2 * kv))
            .context("FIR width overflow")?;
        let layer_stride = width.checked_mul(3).context("FIR layer stride overflow")?;
        let slot_stride = layer_stride.checked_mul(layers).context("FIR slot stride overflow")?;
        Ok(Self {
            slots,
            layers,
            hidden,
            width,
            lag_stride: width,
            layer_stride,
            slot_stride,
        })
    }
    pub fn bytes(self) -> Result<usize> {
        self.slots
            .checked_mul(self.slot_stride)
            .and_then(|n| n.checked_mul(4))
            .context("FIR backing bytes overflow")
    }
    pub fn field(self, slot: usize, layer: usize, name: &str) -> Result<(Range<usize>, usize)> {
        ensure!(slot < self.slots && layer < self.layers, "FIR field outside rank slots/layers");
        let h = self.hidden;
        let kv = (self.width - 2 * h) / 2;
        let (offset, width) = match name {
            "attn_conv_history" => (0, h),
            "mlp_conv_history" => (h, h),
            "k_conv_history" => (2 * h, kv),
            "v_conv_history" => (2 * h + kv, kv),
            _ => anyhow::bail!("unknown FIR field {name}"),
        };
        let start = (slot * self.slot_stride + layer * self.layer_stride + offset) * 4;
        Ok((start..start + (2 * self.lag_stride + width) * 4, width))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_fir_bank_is_one_backing_with_exact_strided_aliases() -> Result<()> {
        for (hidden, kv) in [(4608, 1024), (4608, 512), (2304, 1024)] {
            let l = ConvLayout::new(16, 48, hidden, kv)?;
            assert_eq!(l.width, 2 * hidden + 2 * kv);
            assert_eq!(l.bytes()?, 16 * 48 * 3 * l.width * 4);
            let mut channels = std::collections::BTreeSet::new();
            for name in ["attn_conv_history", "mlp_conv_history", "k_conv_history", "v_conv_history"] {
                let (field, width) = l.field(0, 0, name)?;
                for channel in field.start / 4..field.start / 4 + width {
                    assert!(channels.insert(channel));
                }
                assert_eq!(field.len(), (2 * l.width + width) * 4);
            }
            assert_eq!(channels.len(), l.width);
            assert!(l.field(16, 0, "k_conv_history").is_err());
        }
        Ok(())
    }
}
