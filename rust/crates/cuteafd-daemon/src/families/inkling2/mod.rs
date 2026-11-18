//! Inkling arithmetic and operands on shared ownership/execution mechanisms.
pub(crate) mod weights;
pub(crate) mod owners;
pub(crate) mod cache;
pub(crate) mod experts;
pub(crate) mod layers;
pub(crate) mod head;
pub(crate) mod admission;
pub(crate) mod golden;
pub(crate) mod width_invariance;
pub(crate) mod prefill;
pub(crate) mod startup;
pub(crate) mod engine;
pub(crate) mod backend;
pub(crate) mod serve;
pub(crate) mod draft_taps;
pub(crate) mod verify;
pub(crate) mod draft;
pub(crate) mod verify_target;

fn vocab_span(cfg: &cuteafd_loader::families::inkling2::Inkling2Config, rank: usize) -> anyhow::Result<std::ops::Range<usize>> {
    let rows = cuteafd_loader::shared::vocabulary::shard(cfg.vocab_size, 128, rank)?;
    anyhow::ensure!(rows.start < cfg.unpadded_vocab_size, "Inkling rank owns no valid tokens");
    Ok(rows.start..rows.end.min(cfg.unpadded_vocab_size))
}
