//! Shared head-parallel block-drafter execution; no NCCL or host payload hop.
use super::{recurrent_forward::Rank,gqa_tp2::GqaHeadSplit,dflash::DraftSeq};
use crate::shared::peer_split::DIRECT;
use anyhow::{ensure,Result};

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
        split.reduce([lead.buffer("step.partial")?,peer.buffer("step.partial")?],rows*lead.config.transformer.hidden)?;
        lead.ffn_partial(index,count,split.reduced.buffer)?;
        peer.stream.device.run(||peer.ffn_partial(index,count,split.peer_reduced.buffer))?;
        split.reduce([lead.buffer("step.partial")?,peer.buffer("step.partial")?],rows*lead.config.transformer.hidden)?;
        lead.finish_layer(index,count,split.reduced.buffer)?;
        peer.stream.device.run(||peer.finish_layer(index,count,split.peer_reduced.buffer))?;
    }
    split.exchange.signal(1-split.lead,DIRECT)?;
    split.exchange.wait(split.lead,DIRECT)
}
