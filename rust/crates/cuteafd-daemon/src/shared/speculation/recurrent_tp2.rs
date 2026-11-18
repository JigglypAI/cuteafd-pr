//! Shared head-parallel block-drafter execution; no NCCL or host payload hop.
use super::{recurrent_forward::Rank,recurrent_weights::ProjectionWeight,gqa_tp2::GqaHeadSplit,dflash::DraftSeq};
use crate::shared::peer_split::DIRECT;
use anyhow::{ensure,Context,Result};

fn reduce_output(lead: &Rank<'_>, split: &GqaHeadSplit<'_, Rank<'_>>, layer: usize, rows: usize,
    attention: bool) -> Result<()> {
    let ranks = [lead, &**split.peer];
    let weights = ranks.map(|rank| {
        let layer = &rank.weights.tensors.layers[layer];
        if attention { &layer.output[0] } else { &layer.down[0] }
    });
    let partials = [ranks[0].row_partial(weights[0])?, ranks[1].row_partial(weights[1])?];
    let elements = rows * lead.config.transformer.hidden;
    match weights {
        [ProjectionWeight::Trellis(a), ProjectionWeight::Trellis(b)] => {
            use cuteafd_loader::formats::dense_trellis::TrellisSide;
            ensure!(a.side == TrellisSide::Input && b.side == a.side && b.shape == a.shape
                && b.bits == a.bits && b.codebook == a.codebook, "recurrent trellis rank partials differ");
            let name = if attention { format!("layers.{layer}.self_attn.o_proj.weight") }
                else { format!("layers.{layer}.mlp.down_proj.weight") };
            let output = [split.reduced.buffer, split.peer_reduced.buffer];
            split.reduce_f32(partials, elements, |rank, [rank0, rank1]| {
                let (projection, scratch) = ranks[rank].programs.trellis_parts(&name)?;
                let weight = if rank == 0 { a } else { b };
                let stream = &ranks[rank].stream;
                let add = ranks[rank].programs.trellis_add.as_ref().context("recurrent TP2 trellis add not bound")?;
                let sum = *scratch.buffers.get("accumulator").context("recurrent TP2 F32 sum not admitted")?;
                projection.reduced_f32(add, rows, rank0, rank1, sum, weight, output[rank], stream.raw)
            })
        }
        [ProjectionWeight::Bf16(_), ProjectionWeight::Bf16(_)]
        | [ProjectionWeight::Fp8 { .. }, ProjectionWeight::Fp8 { .. }] => split.reduce(partials, elements),
        _ => anyhow::bail!("recurrent TP2 projection representations differ between ranks"),
    }
}

pub(crate) fn forward(lead:&Rank<'_>,split:&GqaHeadSplit<'_,Rank<'_>>,sequences:&[DraftSeq],pages:&[Vec<i32>])->Result<()> {
    let count=sequences.len();let rows=count*lead.config.transformer.block;
    let peer=&split.peer;
    ensure!(lead.layout.rank.ranks==2 && peer.layout.rank.ranks==2
        && lead.layout.rank.rank==split.lead && peer.layout.rank.rank==1-split.lead
        && lead.config==peer.config,"recurrent TP2 body layouts differ");
    let bytes=rows*lead.config.transformer.hidden*2;
    split.exchange.push_to(split.lead,DIRECT,lead.buffer("step.residual")?.ptr,
        peer.buffer("step.residual")?.ptr,bytes)?;
    split.exchange.wait(1-split.lead,DIRECT)?;
    peer.stream.device.run(||peer.norm_first(rows))?;
    for index in 0..lead.config.transformer.layers {
        let keys=lead.attention_metadata(index,sequences,pages)?;
        peer.stream.device.run(||peer.attention_metadata(index,sequences,pages).map(|_|()))?;
        lead.attention_partial(index,count,keys)?;
        peer.stream.device.run(||peer.attention_partial(index,count,keys))?;
        reduce_output(lead, split, index, rows, true)?;
        lead.ffn_partial(index,count,split.reduced.buffer)?;
        peer.stream.device.run(||peer.ffn_partial(index,count,split.peer_reduced.buffer))?;
        reduce_output(lead, split, index, rows, false)?;
        lead.finish_layer(index,count,split.reduced.buffer)?;
        peer.stream.device.run(||peer.finish_layer(index,count,split.peer_reduced.buffer))?;
    }
    split.exchange.signal(1-split.lead,DIRECT)?;
    split.exchange.wait(split.lead,DIRECT)
}
