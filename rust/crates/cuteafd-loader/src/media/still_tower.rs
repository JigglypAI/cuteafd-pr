//! Exact header-native arenas and phase scratch for independent still towers.
//! This inventory does not qualify native execution or choose a placement.
use crate::{plan::Checkpoint, serving_capacity::WeightArenaLayout};
use anyhow::{ensure, Context, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StillTowerKind { Kimi, MiniMax }

#[derive(Clone, Debug)]
pub struct StillTowerInventory {
    pub kind: StillTowerKind,
    pub arena: WeightArenaLayout,
    pub scratch: u64,
    pub max_output_rows: usize,
}

impl StillTowerInventory {
    pub fn from_checkpoint(checkpoint: &Checkpoint, max_output_rows: usize) -> Result<Self> {
        let tensors=checkpoint.tensors.iter().map(|tensor| &tensor.meta);
        let (kind,id,regions,constant_bytes,limit)=match checkpoint.model_type() {
            Some("kimi_k3"|"kimi_k31")=>{
                let contract=crate::families::kimi_k3::vision::VisionContract::from_config(&checkpoint.config)?;
                let packed=contract.pack(tensors)?;
                let regions=packed.tensors.into_iter().map(|tensor| {
                    Ok((tensor.metadata.name,usize::try_from(tensor.metadata.byte_length)?))
                }).collect::<Result<Vec<_>>>()?;
                (StillTowerKind::Kimi,"kimi/vision",regions,128,
                    crate::families::kimi_k3::vision::MAX_IMAGE_TOKENS as usize)
            },
            Some("minimax_m3_vl")=>{
                let contract=crate::families::minimax_m3::vision::VisionContract::from_config(&checkpoint.config)?;
                let packed=contract.pack(tensors)?;
                let regions=packed.tensors.into_iter().map(|tensor| {
                    Ok((tensor.metadata.name,usize::try_from(tensor.resident_bytes)?))
                }).collect::<Result<Vec<_>>>()?;
                (StillTowerKind::MiniMax,"minimax/vision",regions,52,
                    crate::families::minimax_m3::vision::MAX_IMAGE_TOKENS)
            },
            other=>anyhow::bail!("no independent still tower for {other:?}"),
        };
        ensure!((1..=limit).contains(&max_output_rows),"still tower output rows outside 1..={limit}");
        let arena=WeightArenaLayout::new(id,regions.into_iter().chain([
            ("rotary.inv_freq".into(),constant_bytes)]))?;
        let scratch=scratch_bytes(kind,max_output_rows)?;
        Ok(Self {kind,arena,scratch,max_output_rows})
    }
}

/// Mirror the production phase liveness in scratch_generic_tower, with every
/// individually aligned request and the same peak union (not a payload sum).
pub fn scratch_bytes(kind: StillTowerKind, rows: usize) -> Result<u64> {
    let limit=match kind {StillTowerKind::Kimi=>16698,StillTowerKind::MiniMax=>576};
    ensure!((1..=limit).contains(&rows),"still tower scratch capacity outside native range");
    let (hidden,attention,heads,intermediate,rotated,temporal,output,projector_hidden)=match kind {
        StillTowerKind::Kimi=>(1024u64,1536u64,12u64,4096u64,128u64,1u64,7168u64,4096u64),
        StillTowerKind::MiniMax=>(1280,1280,16,5120,78,2,6144,6144),
    };
    let units=u64::try_from(rows)?;
    let n=units.checked_mul(4).context("tower patch extent overflow")?;
    let align=|bytes:u64|->Result<u64>{Ok(bytes.checked_add(255).context("tower scratch alignment overflow")?/256*256)};
    let phase=|requests:&[u64]|->Result<u64>{requests.iter().try_fold(0u64,|sum,&bytes|
        sum.checked_add(align(bytes)?).context("tower scratch phase overflow"))};
    let persistent=phase(&[n*hidden*2,n*hidden*2,n*rotated/2*4,n*rotated/2*4,8])?;
    let mut phases=vec![
        phase(&[n*14*14*3,768*4,n*3*temporal*14*14*2])?,
        phase(&[n*hidden*2])?,
        phase(&[n*3*attention*2,n*hidden*2])?,
        phase(&[n*3*attention*2,n*attention*2,n*attention*2,n*attention*2])?,
        phase(&[n*attention*2,n*heads*4])?,
        phase(&[n*hidden*2,n*intermediate*2])?,
    ];
    if kind==StillTowerKind::MiniMax {
        phases.push(phase(&[n*6144*2,n*6144*2])?);
        // Grouping is a view of the first projector result: it cannot be
        // charged a second time, nor overlap the final projector activation.
        phases.push(phase(&[units*output*2,units*projector_hidden*2])?);
    } else {
        phases.push(phase(&[units*output*2,units*4*hidden*2,units*projector_hidden*2])?);
    }
    phases.push(phase(&[units*output*2,units*output*2])?);
    persistent.checked_add(*phases.iter().max().unwrap()).context("tower scratch total overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_scratch_matches_native_capacity_receipts() {
        assert_eq!(scratch_bytes(StillTowerKind::Kimi,16698).unwrap(),1_538_887_936);
        assert_eq!(scratch_bytes(StillTowerKind::Kimi,4096).unwrap(),377_487_616);
        assert_eq!(scratch_bytes(StillTowerKind::MiniMax,4).unwrap(),480_512);
        assert_eq!(scratch_bytes(StillTowerKind::MiniMax,144).unwrap(),17_284_864);
        for kind in [StillTowerKind::Kimi,StillTowerKind::MiniMax] {
            assert!(scratch_bytes(kind,0).is_err());
            assert!(scratch_bytes(kind,16699).is_err());
        }
    }
}
