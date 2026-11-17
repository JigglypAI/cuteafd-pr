//! Arithmetic-free accepted-prefix short-convolution state reconstruction.
//! Shared verify contract provides input-row spans; no target forward reruns.
use anyhow::{ensure,Result};

#[derive(Debug,Clone,PartialEq,Eq)]
pub enum Source { Input(usize), Prior(usize) }

/// Newest-first history after exactly `accepted` inputs. Checkpoint rows and
/// staged per-layer F32 convolution inputs are copied byte for byte.
pub fn accepted_history(lags:usize,available:usize,accepted:usize)->Result<Vec<Source>> {
    ensure!(lags>0 && accepted<=available,"conv accepted-prefix count outside staged inputs");
    Ok((0..lags).map(|lag|if accepted>lag {Source::Input(accepted-1-lag)}else{Source::Prior(lag-accepted)}).collect())
}

/// The entire proposed append fits in the ring's dead-row allowance. Thus even
/// keep=0 cannot expose an overwritten live sliding-window entry.
pub fn verify_ring(window:usize,ring:usize,maximum_append:usize)->Result<()> {
    ensure!(window>0 && ring>window && maximum_append>0
        && maximum_append<=ring-window,"speculative append exceeds exact ring allowance");Ok(())
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn all_kept_prefixes_reconstruct_exact_newest_first_lag_bytes()->Result<()> {
        let old=[100,99,98];let inputs=[101,102,103,104,105,106,107,108];
        for keep in 0..=8 {
            let got=accepted_history(3,8,keep)?.iter().map(|s|match s{Source::Input(i)=>inputs[*i],Source::Prior(i)=>old[*i]}).collect::<Vec<_>>();
            let mut serial=old.to_vec();for token in &inputs[..keep]{serial.insert(0,*token);serial.truncate(3);}
            assert_eq!(got,serial);
        }
        assert!(accepted_history(3,8,9).is_err());Ok(())
    }
    #[test]
    fn decoded_rows_never_exceed_inkling_ring_dead_horizon()->Result<()> {
        let cap=crate::placement::families::inkling2::DECODE_ROWS as usize;
        verify_ring(1024,1088,cap)?;
        assert!(verify_ring(1024,1088,cap+1).is_err());
        assert!(verify_ring(1024,1087,cap).is_err());
        for frontier in 0..2176 {
            for keep in 0..=cap {
                let live=(frontier+keep).saturating_sub(1024)..frontier+keep;
                for proposed in frontier+keep..frontier+cap {
                    if let Some(old)=proposed.checked_sub(1088){assert!(!live.contains(&old));}
                }
            }
        }
        Ok(())
    }
}
