//! Inkling 2 Mini configuration and loop schedule reader (`config.json`).
//!
//! Loop and cache-slot semantics match HF Transformers commit
//! `d8f10a82213a3a9db48385aba3a4b57ab535de50` (`modular_inkling2_loop.py:491-494, 535-553`):
//! - 48 visits: prelude layers 0..6 (pass 0), core layers 6..18 x 3 passes, coda layers 18..24 (pass 0).
//! - 48 distinct KV + short-conv cache slots: slots 0..24 hold physical layers 0..24
//!   (front 0..6, core pass 0 6..18, tail 18..24); core pass 1 uses slots 24..36;
//!   core pass 2 uses slots 36..48.
//! - SWA vs global follows the physical layer: every pass has 10 local SWA-1024 slots
//!   and 2 global paged slots, yielding 40 local + 8 global slots overall
//!   (global slots: 5, 11, 17, 23, 29, 35, 41, 47).
//! - Passes 1 and 2 restart hidden state and AttnRes block history from pre-core state
//!   and mix cross-pass messages from the previous pass's completed core blocks plus final partial.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::families::inkling2::Inkling2Config;
use crate::plan::checkpoint::read_json;

/// `text_config.loop_config` parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inkling2LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    pub is_varloop: bool,
}

impl Inkling2LoopConfig {
    pub fn core_layers(&self) -> usize {
        self.core_end - self.core_start
    }

    pub fn is_core_layer(&self, layer: usize) -> bool {
        (self.core_start..self.core_end).contains(&layer)
    }

    pub fn from_hf(text: &Value, layers: usize, res_block: usize) -> Result<Self> {
        let v = text
            .get("loop_config")
            .filter(|v| v.is_object())
            .context("Inkling 2 Mini requires text_config.loop_config")?;
        let read_usize = |key: &str| -> Result<usize> {
            let n = v[key]
                .as_u64()
                .with_context(|| format!("Inkling 2 Mini loop_config lacks {key}"))?;
            ensure!(n <= 1 << 20, "invalid Inkling 2 Mini loop_config.{key}: {n}");
            Ok(n as usize)
        };
        let core_start = read_usize("core_start")?;
        let core_end = read_usize("core_end")?;
        let num_passes = read_usize("num_passes")?;
        let gate_rank = read_usize("gate_rank")?;
        let is_varloop = v
            .get("is_varloop")
            .and_then(Value::as_bool)
            .context("Inkling 2 Mini loop_config lacks boolean is_varloop")?;
        ensure!(
            core_start < core_end && core_end <= layers,
            "Inkling 2 Mini loop core [{core_start}..{core_end}) is outside {layers} physical layers"
        );
        ensure!(
            num_passes >= 2,
            "Inkling 2 Mini loop_config.num_passes must be >= 2, found {num_passes}"
        );
        ensure!(
            gate_rank > 0,
            "Inkling 2 Mini loop_config.gate_rank must be positive"
        );
        ensure!(
            !is_varloop,
            "Inkling 2 Mini variable-depth looping (is_varloop=true) is not supported"
        );
        ensure!(
            core_start % res_block == 0
                && (core_end - core_start) % res_block == 0
                && (layers - core_end) % res_block == 0,
            "Inkling 2 Mini loop boundaries [{core_start}, {core_end}, {layers}) must align to AttnRes block size {res_block}"
        );
        Ok(Self {
            core_start,
            core_end,
            num_passes,
            gate_rank,
            is_varloop,
        })
    }
}

/// One execution step in the unrolled looped forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopVisit {
    /// Zero-based sequential visit index (`0..total_visits`, matching MTP `target_layer_ids`).
    pub visit_idx: usize,
    /// Physical weight layer index (`0..num_hidden_layers`).
    pub layer: usize,
    /// Zero-based pass index (`0` for prelude, core pass 0, and coda; `1..num_passes` for extra core passes).
    pub pass: usize,
    /// Dedicated KV + short-conv cache slot (`0..cache_slots`).
    pub cache_slot: usize,
    /// Whether this visit uses SWA (`true`) or global attention (`false`).
    pub local: bool,
    /// Whether this visit is inside the looped core (`core_start..core_end`).
    pub in_core: bool,
}

/// Full Inkling 2 Mini text + loop configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2MiniConfig {
    pub text: Inkling2Config,
    pub loop_cfg: Inkling2LoopConfig,
}

impl Inkling2MiniConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        ensure!(
            root["model_type"] == "inkling2_loop",
            "Inkling 2 Mini requires model_type=inkling2_loop, found {}",
            root["model_type"]
        );
        let text = Inkling2Config::parse_text_config(root)?;
        let raw_text = root
            .get("text_config")
            .filter(|v| v.is_object())
            .context("Inkling 2 Mini requires text_config")?;
        if let Some(sub_type) = raw_text.get("model_type").and_then(Value::as_str) {
            ensure!(
                sub_type == "inkling2_loop_text",
                "Inkling 2 Mini text_config.model_type must be inkling2_loop_text, found {sub_type}"
            );
        }
        let loop_cfg = Inkling2LoopConfig::from_hf(raw_text, text.layers, text.res_block)?;
        Ok(Self { text, loop_cfg })
    }

    /// Validates the shipped `thinkingmachines/Inkling-2-Mini` geometry.
    pub fn release_geometry(&self) -> Result<()> {
        let t = &self.text;
        let attention = (
            t.hidden,
            t.layers,
            t.heads,
            t.kv_heads,
            t.head_dim,
            t.d_rel,
            t.rel_extent,
            t.window,
            t.sconv_kernel,
            t.res_block,
        );
        ensure!(
            attention == (2304, 24, 24, 8, 128, 16, 1024, 1024, 4, 6),
            "Inkling 2 Mini attention geometry {attention:?} does not match the Inkling-2-Mini release"
        );
        let moe = (
            t.experts,
            t.topk,
            t.intermediate,
            t.vocab_size,
            t.unpadded_vocab_size,
        );
        ensure!(
            moe == (128, 8, 1152, 201024, 200058),
            "Inkling 2 Mini MoE/head geometry {moe:?} does not match the Inkling-2-Mini release"
        );
        ensure!(
            t.global_layers() == vec![5, 11, 17, 23],
            "Inkling-2-Mini attends globally on physical layers [5, 11, 17, 23]"
        );
        let lp = &self.loop_cfg;
        ensure!(
            (lp.core_start, lp.core_end, lp.num_passes, lp.gate_rank, lp.is_varloop)
                == (6, 18, 3, 64, false),
            "Inkling 2 Mini loop_config {lp:?} does not match the Inkling-2-Mini release"
        );
        Ok(())
    }

    /// Total layer visits in one forward pass: `24 + 12 * (3 - 1) = 48`.
    pub fn total_visits(&self) -> usize {
        self.text.layers + self.loop_cfg.core_layers() * (self.loop_cfg.num_passes - 1)
    }

    /// Dedicated KV + short-conv cache slots (one per visit = 48).
    pub fn cache_slots(&self) -> usize {
        self.total_visits()
    }

    /// HF-exact cache slot for `(layer, pass)`:
    /// - pass 0 (prelude 0..6, core pass 0 6..18, coda 18..24): slot = `layer`
    /// - pass `p >= 1` (core 6..18): slot = `layers + (p - 1) * core_len + (layer - core_start)`
    pub fn cache_slot(&self, layer: usize, pass: usize) -> usize {
        if pass == 0 {
            layer
        } else {
            self.text.layers
                + (pass - 1) * self.loop_cfg.core_layers()
                + (layer - self.loop_cfg.core_start)
        }
    }

    /// Physical layer index corresponding to a cache slot `0..cache_slots`.
    pub fn physical_layer_for_slot(&self, slot: usize) -> usize {
        if slot < self.text.layers {
            slot
        } else {
            self.loop_cfg.core_start + ((slot - self.text.layers) % self.loop_cfg.core_layers())
        }
    }

    /// Whether cache slot `slot` uses sliding-window attention (follows physical layer).
    pub fn cache_slot_is_local(&self, slot: usize) -> bool {
        self.text.local[self.physical_layer_for_slot(slot)]
    }

    /// All global (paged KV) cache slot indices in ascending order:
    /// `[5, 11, 17, 23, 29, 35, 41, 47]`.
    pub fn global_cache_slots(&self) -> Vec<usize> {
        (0..self.cache_slots())
            .filter(|&slot| !self.cache_slot_is_local(slot))
            .collect()
    }

    /// Number of local SWA-1024 cache slots (`40` for Inkling 2 Mini).
    pub fn local_cache_slots(&self) -> usize {
        self.cache_slots() - self.global_cache_slots().len()
    }

    /// Unrolled visit schedule in exact execution order (`0..total_visits`).
    pub fn visits(&self) -> Vec<LoopVisit> {
        let mut out = Vec::with_capacity(self.total_visits());
        let mut push = |layer: usize, pass: usize, in_core: bool| {
            let visit_idx = out.len();
            let cache_slot = self.cache_slot(layer, pass);
            out.push(LoopVisit {
                visit_idx,
                layer,
                pass,
                cache_slot,
                local: self.text.local[layer],
                in_core,
            });
        };
        for layer in 0..self.loop_cfg.core_start {
            push(layer, 0, false);
        }
        for pass in 0..self.loop_cfg.num_passes {
            for layer in self.loop_cfg.core_start..self.loop_cfg.core_end {
                push(layer, pass, true);
            }
        }
        for layer in self.loop_cfg.core_end..self.text.layers {
            push(layer, 0, false);
        }
        out
    }

    pub fn activation(&self) -> cuteafd_core::FfnActivation {
        self.text.activation()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::families::inkling2::{Inkling2MtpConfig, Inkling2VisionConfig};

    pub(crate) fn release_config() -> Value {
        serde_json::from_str(include_str!("fixtures/config.json")).unwrap()
    }

    #[test]
    fn release_mini_config_and_48_slot_schedule() {
        let root = release_config();
        let cfg = Inkling2MiniConfig::from_hf(&root).unwrap();
        cfg.release_geometry().unwrap();
        assert_eq!(cfg.total_visits(), 48);
        assert_eq!(cfg.cache_slots(), 48);
        assert_eq!(cfg.global_cache_slots(), vec![5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(cfg.local_cache_slots(), 40);
        assert_eq!(cfg.text.log_scaling, Some((8192, 0.1)));
        assert_eq!(cfg.text.eos, vec![200_006]);

        let visits = cfg.visits();
        assert_eq!(visits.len(), 48);
        // Every slot 0..48 is visited exactly once.
        let mut slots: Vec<usize> = visits.iter().map(|v| v.cache_slot).collect();
        slots.sort_unstable();
        assert_eq!(slots, (0..48).collect::<Vec<_>>());
        // Pass 0 prelude (0..6), core pass 0 (6..18), core pass 1 (24..36), core pass 2 (36..48), coda (18..24).
        assert_eq!((visits[5].layer, visits[5].pass, visits[5].cache_slot, visits[5].local), (5, 0, 5, false));
        assert_eq!((visits[11].layer, visits[11].pass, visits[11].cache_slot, visits[11].local), (11, 0, 11, false));
        assert_eq!((visits[18].layer, visits[18].pass, visits[18].cache_slot, visits[18].local), (6, 1, 24, true));
        assert_eq!((visits[23].layer, visits[23].pass, visits[23].cache_slot, visits[23].local), (11, 1, 29, false));
        assert_eq!((visits[29].layer, visits[29].pass, visits[29].cache_slot, visits[29].local), (17, 1, 35, false));
        assert_eq!((visits[30].layer, visits[30].pass, visits[30].cache_slot, visits[30].local), (6, 2, 36, true));
        assert_eq!((visits[41].layer, visits[41].pass, visits[41].cache_slot, visits[41].local), (17, 2, 47, false));
        assert_eq!((visits[42].layer, visits[42].pass, visits[42].cache_slot, visits[42].local), (18, 0, 18, true));
        assert_eq!((visits[47].layer, visits[47].pass, visits[47].cache_slot, visits[47].local), (23, 0, 23, false));

        let vision = Inkling2VisionConfig::from_hf(&root).unwrap().unwrap();
        assert_eq!((vision.hidden, vision.heads, vision.layers, vision.intermediate), (1024, 16, 23, 4096));
        let mtp = Inkling2MtpConfig::from_hf_with_target(&root, cfg.text.hidden, cfg.total_visits())
            .unwrap()
            .unwrap();
        assert_eq!((mtp.layers, mtp.hidden, mtp.heads, mtp.kv_heads, mtp.intermediate), (3, 2304, 24, 8, 9216));
        assert_eq!(mtp.target_layers, vec![5, 23, 41, 48]);
        assert_eq!(mtp.local, vec![true, false, true]);
    }

    #[test]
    fn rejects_invalid_loop_configs() {
        let mut root = release_config();
        root["text_config"]["loop_config"]["is_varloop"] = true.into();
        assert!(Inkling2MiniConfig::from_hf(&root).unwrap_err().to_string().contains("is_varloop"));
        let mut root = release_config();
        root["text_config"]["loop_config"]["num_passes"] = 1.into();
        assert!(Inkling2MiniConfig::from_hf(&root).unwrap_err().to_string().contains("num_passes"));
        let mut root = release_config();
        root["text_config"]["loop_config"]["core_start"] = 5.into();
        assert!(Inkling2MiniConfig::from_hf(&root).unwrap_err().to_string().contains("AttnRes block"));
    }
}
