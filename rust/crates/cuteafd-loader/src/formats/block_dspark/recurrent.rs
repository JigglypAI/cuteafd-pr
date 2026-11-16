//! Recurrent-hidden block drafters: explicit context mixtures and head-parallel
//! operands, separate from the vanilla head's feature projector/confidence head.
pub mod loading;
pub mod runtime;

use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::{collections::BTreeMap, ops::Range};

use crate::{families::{glm5::draft_physical::ShardAxis,
    inkling2::{Inkling2Config, Inkling2MtpConfig}}, serving_capacity::WeightArenaLayout};

/// Tap sites are arithmetic facts, not a conversion to ordinary post-layer taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapSite { PartialAtLayerEntry(usize), FinalNorm }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextPolicy { BlockAnchoredWindow(usize), Full }

#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentBlockConfig {
    pub transformer: Inkling2MtpConfig,
    pub taps: Vec<TapSite>,
    pub context: Vec<ContextPolicy>,
    pub padded_vocab: usize,
    pub valid_vocab: usize,
    pub softcap: f64,
}

impl RecurrentBlockConfig {
    pub fn from_hf(root: &Value) -> Result<Self> {
        let target = Inkling2Config::from_hf(root)?;
        let transformer = Inkling2MtpConfig::from_hf(root, &target)?
            .context("checkpoint has no bundled recurrent block drafter")?;
        let taps = transformer.target_layers.iter().map(|&id| {
            if id == target.layers { TapSite::FinalNorm } else { TapSite::PartialAtLayerEntry(id) }
        }).collect();
        let context = transformer.local.iter().map(|&local| {
            if local { ContextPolicy::BlockAnchoredWindow(transformer.window) } else { ContextPolicy::Full }
        }).collect();
        Ok(Self { transformer, taps, context, padded_vocab: target.vocab_size,
            valid_vocab: target.unpadded_vocab_size, softcap: target.final_softcap })
    }

    pub fn predictions(&self) -> usize { self.transformer.block - 1 }

    /// Each layer's query sees the entire draft block. The local lower bound
    /// is anchored at block start, independent of the query's index in it.
    pub fn visible_context(&self, layer: usize, block_start: usize) -> Result<Range<usize>> {
        Ok(match self.context.get(layer).context("draft context layer outside checkpoint")? {
            ContextPolicy::Full => 0..block_start,
            ContextPolicy::BlockAnchoredWindow(window) => block_start.saturating_sub(*window)..block_start,
        })
    }

    pub fn rank(&self, rank: usize, ranks: usize, lead: usize) -> Result<RecurrentRank> {
        let c = &self.transformer;
        ensure!(matches!(ranks, 1 | 2) && rank < ranks && lead < ranks
            && c.heads % ranks == 0 && c.kv_heads % ranks == 0 && c.intermediate % ranks == 0,
            "recurrent draft rank split cuts checkpoint heads or FFN columns");
        Ok(RecurrentRank { rank, ranks, lead, heads: c.heads/ranks,
            kv_heads: c.kv_heads/ranks, intermediate: c.intermediate/ranks })
    }

    pub fn recurrent_input_width(&self) -> Result<usize> {
        self.transformer.hidden.checked_add(self.transformer.markov_rank)
            .and_then(|n| n.checked_mul(2)).context("recurrent input width overflow")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentRank {
    pub rank: usize, pub ranks: usize, pub lead: usize,
    pub heads: usize, pub kv_heads: usize, pub intermediate: usize,
}

pub struct RecurrentLayer<T, M> {
    pub input_norm: T, pub post_norm: T, pub dflare_norm: T,
    pub qkv: M, pub q_norm: T, pub k_norm: T,
    pub context_kv: M, pub context_k_norm: T,
    pub output: M, pub gate_up: M, pub down: M,
}

pub struct RecurrentHead<T, M> {
    pub token_embedding: T, pub projection: M, pub projection_bias: T,
    pub hidden_out: M, pub hidden_out_bias: T, pub dflare_norm: T,
}

pub struct RecurrentWeights<T, M> {
    pub layers: Vec<RecurrentLayer<T, M>>, pub norm: T, pub fuse: T,
    pub mask_embedding: T, pub head: Option<RecurrentHead<T, M>>,
}

/// One recipe for rank admission and runtime loads. Names are relative to the
/// bundled MTP root: native unindexed files use bare names, MC uses `mtp.*`.
/// `matrix` sees full source shapes and the sharding axis, as in vanilla DSpark.
pub fn rank_recipe<T, M>(config: &RecurrentBlockConfig, rank: usize, ranks: usize, lead: usize,
    mut tensor: impl FnMut(&str, &[usize]) -> Result<T>,
    mut matrix: impl FnMut(&[(&str, usize)], usize, Option<ShardAxis>) -> Result<M>)
    -> Result<RecurrentWeights<T, M>> {
    config.rank(rank, ranks, lead)?;
    let c = &config.transformer;
    let (h, inter, r) = (c.hidden, c.intermediate, c.markov_rank);
    let q = c.heads.checked_mul(c.head_dim).context("draft Q width overflow")?;
    let kv = c.kv_heads.checked_mul(c.head_dim).context("draft KV width overflow")?;
    let layers = (0..c.layers).map(|layer| {
        let p = format!("layers.{layer}");
        let a = format!("{p}.self_attn");
        Ok(RecurrentLayer {
            input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
            post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
            dflare_norm: tensor(&format!("{p}.dflare_norm.weight"), &[h])?,
            qkv: matrix(&[(&format!("{a}.q_proj.weight"), q), (&format!("{a}.k_proj.weight"), kv),
                (&format!("{a}.v_proj.weight"), kv)], h, Some(ShardAxis::Rows))?,
            q_norm: tensor(&format!("{a}.q_norm.weight"), &[c.head_dim])?,
            k_norm: tensor(&format!("{a}.k_norm.weight"), &[c.head_dim])?,
            context_kv: matrix(&[(&format!("{a}.k_proj_ctx.weight"), kv),
                (&format!("{a}.v_proj_ctx.weight"), kv)], h, Some(ShardAxis::Rows))?,
            context_k_norm: tensor(&format!("{a}.k_norm_ctx.weight"), &[c.head_dim])?,
            output: matrix(&[(&format!("{a}.o_proj.weight"), h)], q, Some(ShardAxis::Columns))?,
            gate_up: matrix(&[(&format!("{p}.mlp.gate_proj.weight"), inter),
                (&format!("{p}.mlp.up_proj.weight"), inter)], h, Some(ShardAxis::Rows))?,
            down: matrix(&[(&format!("{p}.mlp.down_proj.weight"), h)], inter, Some(ShardAxis::Columns))?,
        })
    }).collect::<Result<Vec<_>>>()?;
    let head = if rank == lead {
        Some(RecurrentHead {
            token_embedding: tensor("markov_head.markov_w1.weight", &[config.padded_vocab, r])?,
            projection: matrix(&[("markov_head.rnn_proj.weight", 3*r)], config.recurrent_input_width()?, None)?,
            projection_bias: tensor("markov_head.rnn_proj.bias", &[3*r])?,
            hidden_out: matrix(&[("markov_head.hidden_out.weight", h)], r, None)?,
            hidden_out_bias: tensor("markov_head.hidden_out.bias", &[h])?,
            dflare_norm: tensor("markov_head.dflare_norm.weight", &[h])?,
        })
    } else { None };
    Ok(RecurrentWeights { layers, head, norm: tensor("norm.weight", &[h])?,
        fuse: tensor("fuse_weight", &[c.layers+1, config.taps.len()])?,
        mask_embedding: tensor("mask_embed", &[h])? })
}

/// Native BF16 source views and derived folded-head operands have one backing
/// per physical rank. Borrowed target embeddings/head are deliberately absent.
pub fn native_rank_arena(config: &RecurrentBlockConfig, rank: usize, ranks: usize, lead: usize)
    -> Result<WeightArenaLayout> {
    let views = std::cell::RefCell::new(Vec::new());
    rank_recipe(config, rank, ranks, lead, |name, shape| {
        let bytes = shape.iter().try_fold(2usize, |n, dim| n.checked_mul(*dim))
            .context("draft tensor bytes overflow")?;
        views.borrow_mut().push((name.to_owned(), bytes));
        Ok(())
    }, |parts, columns, axis| {
        for &(name, rows) in parts {
            let (rows, cols) = super::block_dspark_part_ranges(rows, columns, axis, rank, ranks)?;
            let bytes = rows.len().checked_mul(cols.len()).and_then(|n| n.checked_mul(2))
                .context("draft projection bytes overflow")?;
            views.borrow_mut().push((name.to_owned(), bytes));
        }
        Ok(())
    })?;
    let mut views = views.into_inner();
    views.push(("derived.fuse_alpha".into(), (config.transformer.layers+1)*config.taps.len()*2));
    if rank == lead {
        views.push(("derived.fold_weight".into(), config.padded_vocab.checked_mul(config.transformer.markov_rank)
            .and_then(|n| n.checked_mul(2)).context("folded draft head bytes overflow")?));
        views.push(("derived.fold_bias".into(), config.padded_vocab.checked_mul(2).context("folded draft bias bytes overflow")?));
    }
    WeightArenaLayout::new(format!("block-dspark/recurrent/weights/{rank}"), views)
}

/// Full source inventory, for strict header validation before any device work.
pub fn source_shapes(config: &RecurrentBlockConfig) -> Result<BTreeMap<String, Vec<usize>>> {
    let shapes = std::cell::RefCell::new(BTreeMap::new());
    rank_recipe(config, 0, 1, 0, |name, shape| {
        ensure!(shapes.borrow_mut().insert(name.to_owned(), shape.to_vec()).is_none(), "duplicate drafter tensor");
        Ok(())
    }, |parts, columns, _| {
        for &(name, rows) in parts {
            ensure!(shapes.borrow_mut().insert(name.to_owned(), vec![rows, columns]).is_none(), "duplicate drafter projection");
        }
        Ok(())
    })?;
    Ok(shapes.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn release() -> RecurrentBlockConfig {
        RecurrentBlockConfig::from_hf(&serde_json::from_str(include_str!(
            "../../families/inkling2/fixtures/config.json")).unwrap()).unwrap()
    }

    #[test]
    fn inkling_recurrent_descriptor_preserves_native_taps_and_block() -> Result<()> {
        let c = release();
        assert_eq!(c.predictions(), 7);
        assert_eq!(c.recurrent_input_width()?, 9728);
        assert_eq!((c.padded_vocab, c.valid_vocab), (201024, 200058));
        assert_eq!(c.taps, [TapSite::PartialAtLayerEntry(5), TapSite::PartialAtLayerEntry(11),
            TapSite::PartialAtLayerEntry(17), TapSite::PartialAtLayerEntry(23), TapSite::PartialAtLayerEntry(29),
            TapSite::PartialAtLayerEntry(35), TapSite::PartialAtLayerEntry(41), TapSite::FinalNorm]);
        for layer in [0, 1, 3, 4] {
            assert_eq!(c.visible_context(layer, 100)?, 0..100);
            assert_eq!(c.visible_context(layer, 5000)?, 3976..5000);
        }
        assert_eq!(c.visible_context(2, 5000)?, 0..5000);
        assert!(c.visible_context(5, 100).is_err());
        Ok(())
    }

    #[test]
    fn inkling_recurrent_recipe_covers_all_84_checkpoint_operands() -> Result<()> {
        let shapes = source_shapes(&release())?;
        assert_eq!(shapes.len(), 84);
        assert_eq!(shapes["fuse_weight"], [6, 8]);
        assert_eq!(shapes["markov_head.rnn_proj.weight"], [768, 9728]);
        assert_eq!(shapes["markov_head.hidden_out.bias"], [4608]);
        assert_eq!(shapes["layers.2.self_attn.k_proj_ctx.weight"], [1024, 4608]);
        assert!(!shapes.contains_key("fc.weight"));
        assert!(!shapes.keys().any(|name| name.starts_with("confidence_head")));
        Ok(())
    }

    #[test]
    fn inkling_recurrent_tp2_splits_heads_ffn_and_keeps_one_head_owner() -> Result<()> {
        let c = release();
        for lead in 0..2 {
            for rank in 0..2 {
                let local = c.rank(rank, 2, lead)?;
                assert_eq!((local.heads, local.kv_heads, local.intermediate), (24, 4, 9216));
                let arena = native_rank_arena(&c, rank, 2, lead)?;
                assert_eq!(arena.view("layers.0.self_attn.q_proj.weight").unwrap().bytes(), 3072*4608*2);
                assert_eq!(arena.view("layers.2.self_attn.k_proj_ctx.weight").unwrap().bytes(), 512*4608*2);
                assert_eq!(arena.view("layers.0.self_attn.o_proj.weight").unwrap().bytes(), 4608*3072*2);
                assert_eq!(arena.view("layers.0.mlp.down_proj.weight").unwrap().bytes(), 4608*9216*2);
                assert_eq!(arena.view("markov_head.rnn_proj.weight").is_some(), rank == lead);
                assert_eq!(arena.view("derived.fold_weight").is_some(), rank == lead);
                assert!(arena.views().values().all(|view| view.offset()%256 == 0));
            }
        }
        assert!(c.rank(0, 3, 0).is_err());
        assert!(c.rank(0, 2, 2).is_err());
        Ok(())
    }
}
