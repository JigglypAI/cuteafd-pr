//! Independent still-tower startup and checked prompt expansion, shared by families.
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::media::MediaPreparer;
use cuteafd_engine::media::{EncodeJob, MediaKeys, RequestMedia};
use cuteafd_loader::{media::{ImageFamily, ImageFragment, PreparedImageGroup, ProcessorConfig},
    plan::{encoder::{EncoderKind, EncoderPlacement}, MediaMode}};
use std::{net::SocketAddr, path::{Path, PathBuf}, sync::Arc};

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

pub(crate) struct Binding { snapshot:PathBuf, family:ImageFamily, enabled:bool }
impl Binding {
    pub fn new(snapshot:&Path,family:ImageFamily,enabled:bool)->Self {Self {snapshot:snapshot.into(),family,enabled}}
    fn marker(&self,name:&str,vocabulary:u32)->Result<u32> {
        let config:serde_json::Value=cuteafd_loader::read_json_file(&self.snapshot.join("tokenizer_config.json"))?;
        let mut ids=config["added_tokens_decoder"].as_object().context("still tokenizer added tokens")?.iter()
            .filter(|(_,v)|v["content"].as_str()==Some(name));
        let id:u32=ids.next().with_context(||format!("missing image marker {name}"))?.0.parse()?;
        ensure!(ids.next().is_none()&&id<vocabulary,"invalid still image marker");Ok(id)
    }
    pub fn prepare(&self,job:cuteafd_api::openai::NativeRequest,tokens:Vec<u32>,vocabulary:u32,hidden:usize,context:usize)
        ->Result<(crate::shared::qwen::scheduler::Prompt,RequestMedia,Vec<EncodeJob>)> {
        ensure!(job.audio.is_empty(),"still tower has no audio owner");
        if job.probe.as_ref().is_some_and(|p|!p.spec.media.is_empty()) {
            return crate::shared::probe::still::prepare(job,tokens,vocabulary,hidden,context,&self.snapshot,self.family);
        }
        ensure!(self.enabled||job.media.is_empty(),"vision is disabled");
        if job.media.is_empty() {
            let keys=MediaKeys::new(&tokens,vocabulary,&[])?;
            let media=RequestMedia::new(vec![],hidden,tokens.len())?;
            return Ok((crate::shared::qwen::scheduler::Prompt {job,tokens,keys},media,vec![]));
        }
        let placeholder=self.marker(if self.family==ImageFamily::Kimi {"<|kimi_image_placeholder|>"}else{"]<]image[>["},vocabulary)?;
        let pad=self.marker(if self.family==ImageFamily::Kimi {"<|media_pad|>"}else{"]<]image[>["},vocabulary)?;
        let tokenizer=cuteafd_loader::cached_tokenizer(&self.snapshot)?;
        let mut groups=Vec::new();
        for image in &job.media {
            let mut group=PreparedImageGroup::single(image.clone(),pad);
            let (prefix,suffix)=if self.family==ImageFamily::Kimi {
                let [w,h]=image.pixels.source;
                (tokenizer.encode_ids(&format!("<|media_begin|>image {w}x{h}<|media_content|>"))?,self.marker("<|media_end|>",vocabulary)?)
            }else{(vec![self.marker("]<]start of image[>[",vocabulary)?],self.marker("]<]end of image[>[",vocabulary)?)};
            group.fragments.insert(0,ImageFragment::Tokens(prefix));
            group.fragments.push(ImageFragment::Tokens(vec![suffix]));groups.push(group);
        }
        let expanded=PreparedImageGroup::expand_prompt(&tokens,placeholder,&groups,vocabulary,context)?;
        let keys=MediaKeys::new(&expanded.tokens,vocabulary,&expanded.media)?;
        let media=RequestMedia::new(expanded.media.clone(),hidden,expanded.tokens.len())?;
        let jobs=job.media.iter().map(|i|EncodeJob::image_with_pixels(i.key,[i.grid.t,i.grid.h,i.grid.w],i.rgb8.clone(),i.pixels,i.tokens,hidden)).collect();
        crate::shared::probe::image_echo(&job,&expanded.media)?;
        Ok((crate::shared::qwen::scheduler::Prompt {job,tokens:expanded.tokens,keys},media,jobs))
    }
}
