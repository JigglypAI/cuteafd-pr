//! Independent still-tower startup and checked prompt expansion, shared by families.
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::media::MediaPreparer;
use cuteafd_engine::media::{EncodeJob, MediaKeys, RequestMedia};
use cuteafd_loader::{media::{ImageFamily, PreparedImageGroup, ProcessorConfig},
    plan::{encoder::{EncoderKind, EncoderPlacement}, MediaMode}};
use std::{net::SocketAddr, path::Path, sync::Arc};

#[derive(Clone, Debug, clap::Args)]
pub(crate) struct Options {
    #[arg(long, default_value="auto")] pub vision: MediaMode,
    #[arg(long, value_delimiter=',')] pub vision_peers: Vec<SocketAddr>,
    #[arg(long)] pub encoder_revision: Option<String>,
    #[arg(long)] pub encoder_plan_hash: Option<String>,
    #[arg(long, default_value_t=1)] pub vision_replicas: usize,
    #[arg(long)] pub media_cache_bytes: Option<crate::shared::memory::budget::ByteSize>,
}
impl Default for Options {
    fn default() -> Self { Self { vision:MediaMode::Auto,vision_peers:vec![],encoder_revision:None,
        encoder_plan_hash:None,vision_replicas:1,media_cache_bytes:None } }
}
pub(crate) struct Ready {
    pub push: super::push::Contribution,
    pub encoder: crate::shared::qwen::Encoder,
    pub preparer: Option<Arc<MediaPreparer>>,
    pub cache_bytes: usize,
}
impl Ready {
    pub fn load(options:&Options,snapshot:&Path,library_path:&Path,library:&cuteafd_ffi::NativeLibrary,
        devices:&[i32],placement:Option<EncoderPlacement>,family:ImageFamily,cache_bytes:u64) -> Result<Self> {
        use super::{EncoderService,TowerSpec,local::LocalEncoder,remote::RemoteEncoder};
        use crate::shared::qwen::Encoder;
        let Some(placement)=placement.filter(|p|p.kind!=EncoderKind::Off) else {
            ensure!(options.vision_peers.is_empty(),"disabled still tower has remote peers");
            return Ok(Self {push:Default::default(),encoder:Encoder::Off,preparer:None,cache_bytes:0});
        };
        let processor=ProcessorConfig::from_snapshot(snapshot,family)?;
        let rows=processor.max_image_tokens;
        let spec=TowerSpec::from_snapshot(snapshot,rows)?;
        let ledger=cuteafd_ffi::vision::NativeVision::required(library_path,&spec.native)?;
        ensure!(ledger.total_bytes()==placement.admitted_bytes(),"still tower ledger differs from admission");
        let width=spec.native.output_width as usize;
        let mut push=super::push::Contribution::default();
        let (id,encoder)=match placement.kind {
            EncoderKind::Spark {..}=>{
                ensure!(options.vision_peers.len()==placement.spark_ranks().len(),"still tower peer replicas differ from admission");
                let revision=options.encoder_revision.as_deref().context("remote still tower revision")?;
                let id=spec.encoder_id(revision,121);
                let hash=super::worker::parse_plan_hash(options.encoder_plan_hash.as_deref().context("remote still tower plan hash")?)?;
                let handshake=super::worker::image_handshake(&spec,&processor,id,hash);
                (id,Encoder::Remote(RemoteEncoder::connect(options.vision_peers.clone(),handshake,std::time::Duration::from_secs(60))?))
            },
            EncoderKind::Rtx {gpu}=>{
                ensure!(options.vision_peers.is_empty(),"local still tower has remote peers");
                let device=*devices.get(gpu).context("still tower GPU rank")?;
                let info=library.cuda_device_info(device)?;
                let sm=u32::try_from(info.compute_capability_major*10+info.compute_capability_minor)?;
                let revision=snapshot.file_name().and_then(|s|s.to_str()).context("still tower revision")?;
                let id=spec.encoder_id(revision,sm);
                let (service,contribution)=EncoderService::start_coordinator(snapshot,Some((spec,placement.admitted_bytes())),library_path.to_owned(),device,None)?;
                push=contribution;
                (id,Encoder::Local(LocalEncoder::new(service,&processor,width,rows)))
            },
            _=>anyhow::bail!("still tower has no runnable owner"),
        };
        let preparer=Arc::new(MediaPreparer::for_loaded_encoder(processor,id,4)?);
        Ok(Self {push,encoder,preparer:Some(preparer),cache_bytes:usize::try_from(cache_bytes)?})
    }
}

/// Families assemble native fragments (including grouped local/global crops)
/// before this single checked expansion and scheduler binding boundary.
pub(crate) struct Binding;
impl Binding {
    pub fn prepare(mut job:cuteafd_api::openai::NativeRequest,tokens:Vec<u32>,vocabulary:u32,hidden:usize,
        context:usize,source_placeholder:u32,groups:Vec<PreparedImageGroup>)
        ->Result<(crate::shared::qwen::scheduler::Prompt,RequestMedia,Vec<EncodeJob>)> {
        ensure!(job.audio.is_empty(),"still tower has no audio owner");
        let expanded=PreparedImageGroup::expand_prompt(&tokens,source_placeholder,&groups,vocabulary,context)?;
        // The expanded fragment order, not source-image count, owns encoder jobs.
        job.media=groups.iter().flat_map(|group|group.keys_in_prompt_order().map(|key|
            group.images.iter().find(|image|image.key==key).expect("validated group key").clone())).collect();
        let keys=MediaKeys::new(&expanded.tokens,vocabulary,&expanded.media)?;
        let media=RequestMedia::new(expanded.media.clone(),hidden,expanded.tokens.len())?;
        let jobs=job.media.iter().map(|i|EncodeJob::image_with_pixels(i.key,[i.grid.t,i.grid.h,i.grid.w],i.rgb8.clone(),i.pixels,i.tokens,hidden)).collect();
        crate::shared::probe::image_echo(&job,&expanded.media)?;
        Ok((crate::shared::qwen::scheduler::Prompt {job,tokens:expanded.tokens,keys},media,jobs))
    }
}
