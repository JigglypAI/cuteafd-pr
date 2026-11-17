//! Typed physical layouts for Kimi's shared MLA and GQA draft executors.
use crate::formats::{block_dspark::BlockDsparkRuntimeLayout, dflash::MlaDraftRuntimeLayout};
use crate::families::glm5::draft_representation::GlmDraftCapacity;
use anyhow::{ensure, Result};

/// Automatic placement follows native kernel geometry, never checkpoint identity.
pub fn resolve_context_split(snapshot: &std::path::Path, requested: Option<bool>, eligible: bool) -> Result<bool> {
    ensure!(requested != Some(true) || eligible, "TP2_DFLASH requires two physical GPUs and DFlash2");
    if requested == Some(false) || !eligible && requested != Some(true) { return Ok(false) }
    let cfg = crate::formats::dflash::MlaDflashConfig::read(snapshot)?;
    let supported = cfg.kind == cuteafd_core::coordinator_programs::MlaDraftKind::Dflash2
        && cfg.mla.hidden == 7168 && cfg.mla.heads == 64;
    ensure!(requested != Some(true) || supported,
        "TP2_DFLASH requires missing context-split kernel kimik31_mla_draft_partial_causal_w1024_h96_m64 for causal MLA h96; use whole-drafter placement (omit TP2_DFLASH)");
    Ok(eligible && requested.unwrap_or(supported))
}

/// A checkpoint-declared target feature, not a user-selectable execution mode.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DraftHiddenStream { #[default] PlainResidual, AttnRes }
impl DraftHiddenStream {
    pub fn from_config(raw: &serde_json::Value) -> Result<Self> {
        match raw.get("dflash_config").and_then(|config| config.get("aux_hidden_stream")) {
            None => Ok(Self::PlainResidual),
            Some(serde_json::Value::String(stream)) if stream == "attn_res" => Ok(Self::AttnRes),
            Some(other) => anyhow::bail!("unsupported drafter dflash_config.aux_hidden_stream={other}; expected absent or \"attn_res\""),
        }
    }
    pub fn read(snapshot: &std::path::Path) -> Result<Self> {
        Self::from_config(&serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DraftLayout {
    Mla(MlaDraftRuntimeLayout),
    Block(BlockDsparkRuntimeLayout),
    BlockTp2 { ranks: Box<[BlockDsparkRuntimeLayout;2]>, lead:usize },
}

impl From<MlaDraftRuntimeLayout> for DraftLayout {
    fn from(value: MlaDraftRuntimeLayout) -> Self { Self::Mla(value) }
}

impl DraftLayout {
    pub fn hidden_stream(&self) -> DraftHiddenStream {
        match self { Self::Mla(layout) => layout.hidden_stream(), _ => DraftHiddenStream::PlainResidual }
    }
    pub fn needs_final_attn_res(&self, layers: usize) -> bool {
        self.hidden_stream() == DraftHiddenStream::AttnRes && layers.checked_sub(1)
            .is_some_and(|last| self.target_taps().contains(&last))
    }
    pub fn capacity(&self) -> GlmDraftCapacity {
        match self { Self::Mla(l) => l.capacity(), Self::Block(l) => l.capacity, Self::BlockTp2 { ranks,lead }=>ranks[*lead].capacity }
    }
    pub fn target_hidden(&self) -> usize {
        match self { Self::Mla(l) => l.target_hidden(), Self::Block(l) => l.config.hidden, Self::BlockTp2 { ranks,lead }=>ranks[*lead].config.hidden }
    }
    pub fn target_vocab(&self) -> usize {
        match self { Self::Mla(l) => l.target_vocab(), Self::Block(l) => l.config.vocab, Self::BlockTp2 { ranks,lead }=>ranks[*lead].config.vocab }
    }
    pub fn target_taps(&self) -> &[usize] {
        match self { Self::Mla(l) => l.target_taps(), Self::Block(l) => &l.config.taps, Self::BlockTp2 { ranks,lead }=>&ranks[*lead].config.taps }
    }
    pub fn context_split(&self) -> bool {
        matches!(self, Self::Mla(l) if l.context_split())
    }
    pub fn with_context_split(self, enabled: bool) -> Result<Self> {
        match self {
            Self::Mla(l) => Ok(Self::Mla(l.with_context_split(enabled)?)),
            Self::Block(l) => { ensure!(!enabled,"GQA block draft has no MLA Context split"); Ok(Self::Block(l)) }
            other @ Self::BlockTp2 { .. }=>{ ensure!(!enabled,"GQA head split is not MLA Context split"); Ok(other) }
        }
    }
    pub fn window_mark_bytes(&self) -> Result<usize> {
        match self { Self::Mla(l) => Ok(l.window_mark_bytes()?), Self::Block(l) => l.window_mark_bytes(),Self::BlockTp2 { ranks,.. }=>ranks.iter().try_fold(0usize,|sum,r|sum.checked_add(r.window_mark_bytes()?).ok_or_else(||anyhow::anyhow!("TP2 window mark overflow"))) }
    }
    pub fn window_mark_demands(&self,head:u8)->Result<Vec<crate::placement::Demand>> {
        use cuteafd_core::memory_layout::{Basis,Category};
        let extents=match self {
            Self::BlockTp2 { ranks,.. }=>ranks.iter().enumerate().map(|(rank,l)|Ok((rank as u8,l.window_mark_bytes()?)))
                .collect::<Result<Vec<_>>>()?,
            _=>vec![(head,self.window_mark_bytes()?)],
        };
        Ok(extents.into_iter().filter(|(_,bytes)|*bytes>0).map(|(gpu,bytes)|crate::placement::Demand::new(gpu,
            Category::Prefix,"Kimi exact draft windows",bytes as u64,Basis::Exact)).collect())
    }
    pub fn weight_arena_requests(&self) -> Vec<(String, usize)> {
        match self { Self::Mla(l) => l.weight_arena_requests(), Self::Block(l) => l.weight_requests.clone(),Self::BlockTp2 { ranks,lead }=>ranks[*lead].weight_requests.clone() }
    }
    pub fn with_pool_tokens(&self, tokens: usize) -> Result<Self> {
        Ok(match self { Self::Mla(l) => Self::Mla(l.with_pool_tokens(tokens)?), Self::Block(l) => Self::Block(l.clone()),Self::BlockTp2 { ranks,lead }=>Self::BlockTp2 { ranks:ranks.clone(),lead:*lead } })
    }
    pub fn validate_capacity(&self, slots: usize, rows: usize, context: usize,
        _target_cache: cuteafd_core::MlaCacheFormat) -> Result<()> {
        ensure!(self.capacity().context_slots == slots && self.capacity().max_batch_sequences <= rows.min(64),
            "Kimi drafter capacity differs from target slots/rows");
        match self {
            Self::Mla(l) => ensure!(l.program().context == context
                && l.cache().layers.iter().all(|layer| layer.slots() == slots),
                "Kimi MLA drafter cache differs from target capacity"),
            Self::Block(l) => ensure!(l.config.window.is_some(), "block drafter requires window-bounded KV"),
            Self::BlockTp2 { ranks,lead }=>ensure!(*lead<2 && ranks.iter().all(|r|r.config.window.is_some() && r.capacity.context_slots==slots),"invalid TP2 block capacity"),
        }
        Ok(())
    }
    pub fn demands(&self, gpu: u8) -> Result<(crate::placement::draft::DrafterDemand, crate::placement::Demand)> {
        use cuteafd_core::memory_layout::{Basis, Category};
        use crate::placement::{draft::DrafterDemand, Demand, ResidentWeights};
        match self {
            Self::Mla(l) => Ok(l.demands(gpu)?),
            Self::Block(l) => {
                let mut owned = l.demands(gpu);
                owned.push(Demand::new(gpu,Category::Drafter,"block dSpark owned weights",l.weights.payload_bytes() as u64,
                    Basis::Exact).with_resident_weights(ResidentWeights(l.weights.payload_bytes() as u64)));
                let peak = owned.iter().try_fold(0u64, |sum,d| sum.checked_add(d.bytes))
                    .ok_or_else(|| anyhow::anyhow!("block draft total overflow"))?;
                Ok((DrafterDemand::itemized(owned,gpu),Demand::new(gpu,Category::Drafter,
                    "block dSpark load peak including owners",peak,Basis::Exact)))
            }            Self::BlockTp2 { ranks,lead }=>{
                ensure!(usize::from(gpu)==*lead,"TP2 block lead differs from placement");
                let owned=ranks.iter().enumerate().flat_map(|(rank,l)|l.demands(rank as u8)).collect::<Vec<_>>();
                let peak=owned.iter().try_fold(0u64,|sum,d|sum.checked_add(d.bytes))
                    .ok_or_else(||anyhow::anyhow!("TP2 block owners overflow"))?;
                Ok((DrafterDemand::itemized(owned,gpu),Demand::new(gpu,Category::Drafter,
                    "block dSpark load peak including owners",peak,Basis::Exact)))
            }

        }
    }
}

#[cfg(test)]
mod hidden_stream_tests {
    use super::DraftHiddenStream;
    use serde_json::json;

    #[test]
    fn checkpoint_hidden_stream_is_absent_attn_res_or_a_clear_refusal() {
        for raw in [json!({"architectures":["DSparkDraftModel"]}),
            json!({"architectures":["DFlash2DraftModel"], "dflash_config":{"attention_mode":"mla"}})] {
            assert_eq!(DraftHiddenStream::from_config(&raw).unwrap(), DraftHiddenStream::PlainResidual);
        }
        assert_eq!(DraftHiddenStream::from_config(&json!({"dflash_config":{"aux_hidden_stream":"attn_res"}})).unwrap(),
            DraftHiddenStream::AttnRes);
        for value in [json!("prefix_only"), json!("other"), json!(null), json!(false), json!(3)] {
            let error = DraftHiddenStream::from_config(&json!({"dflash_config":{"aux_hidden_stream":value}})).unwrap_err();
            assert!(error.to_string().contains("dflash_config.aux_hidden_stream"));
        }
    }
}

/// The target-side staging/bridge is shared with the MLA binding.
pub fn bridge_demands(layout: &DraftLayout, rows: usize, ranks: usize)
    -> Result<Vec<crate::placement::Demand>> {
    super::draft_bridge_inventory(layout.target_hidden() as u64,layout.target_vocab() as u64,
        layout.target_taps().len() as u64,layout.capacity().block_rows as u64,rows,ranks)
}
