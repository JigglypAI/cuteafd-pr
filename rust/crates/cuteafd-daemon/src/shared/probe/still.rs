//! Cold fidelity feature replacement for still towers that are not production-qualified.
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{ModelProfile, NativeRequest};
use cuteafd_core::MediaSpan;
use cuteafd_engine::media::{EncodeJob, EmbeddingCache, MediaKeys, RequestMedia};
use cuteafd_loader::media::{ImageFamily, ProcessorConfig};
use std::{path::Path, sync::Arc};

pub(crate) fn install(profile: ModelProfile, snapshot: &Path, family: ImageFamily) -> Result<ModelProfile> {
    let config = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
    if config.get("vision_config").is_none_or(serde_json::Value::is_null) { return Ok(profile); }
    let processor = ProcessorConfig::from_snapshot(snapshot, family)?;
    let spec = crate::shared::vision::TowerSpec::from_snapshot(snapshot, processor.max_image_tokens)?;
    let revision = snapshot.file_name().and_then(|s| s.to_str()).context("probe snapshot revision")?;
    // No GPU tower exists on this route. The zero architecture identifies the
    // processor-only source; feature overrides have their own cache identity.
    let preparer = cuteafd_api::openai::media::MediaPreparer::for_loaded_encoder(
        processor, spec.encoder_id(revision, 0), 2)?;
    Ok(profile.with_probe_media(Arc::new(preparer)))
}

pub(crate) fn admission<T>() -> cuteafd_engine::media::MediaAdmission<T, crate::shared::qwen::Encoder> {
    cuteafd_engine::media::MediaAdmission::new(EmbeddingCache::new(8 << 30), crate::shared::qwen::Encoder::Off, 16)
}

fn marker(snapshot: &Path, name: &str, vocabulary: u32) -> Result<u32> {
    let config: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("tokenizer_config.json"))?;
    let mut ids = config["added_tokens_decoder"].as_object().context("still probe tokenizer added tokens")?
        .iter().filter(|(_, token)| token["content"].as_str() == Some(name));
    let id: u32 = ids.next().with_context(|| format!("still probe tokenizer lacks {name}"))?.0.parse()?;
    ensure!(ids.next().is_none() && id < vocabulary, "still probe marker duplicate/outside vocabulary");
    Ok(id)
}

pub(crate) fn spans(job: &NativeRequest, tokens: &[u32], vocabulary: u32, context: usize,
    snapshot: &Path, family: ImageFamily) -> Result<Vec<MediaSpan>> {
    if job.media.is_empty() { return Ok(Vec::new()); }
    let probe = job.probe.as_ref().filter(|p| p.spec.feature_scoring())
        .context("bundled vision is not production-qualified; images require a cold fidelity feature override")?;
    ensure!(job.audio.is_empty() && tokens.len() <= context && tokens.iter().all(|&id| id < vocabulary),
        "still probe token/context/audio extent");
    ensure!(probe.spec.prompt_ids.as_deref() == Some(tokens) && probe.spec.media.len() == job.media.len(),
        "still probe expanded ids/image count differs");
    probe.spec.validate_media()?;
    let placeholder = marker(snapshot, if family == ImageFamily::Kimi { "<|media_pad|>" } else { "]<]image[>[" }, vocabulary)?;
    let tokenizer = cuteafd_loader::cached_tokenizer(snapshot)?;
    let mut covered = std::collections::BTreeSet::new();
    let mut spans = Vec::new();
    for (span, image) in probe.spec.media.iter().zip(&job.media) {
        span.verify_image(image)?;
        let end = span.start.checked_add(span.len).context("still probe span overflow")?;
        ensure!(end <= tokens.len() && tokens[span.start..end].iter().all(|&id| id == placeholder),
            "still probe feature placeholders differ");
        if family == ImageFamily::Kimi {
            let [width, height] = image.pixels.source;
            let prefix = tokenizer.encode_ids(&format!("<|media_begin|>image {width}x{height}<|media_content|>"))?;
            let suffix = marker(snapshot, "<|media_end|>", vocabulary)?;
            ensure!(span.start >= prefix.len() && tokens[span.start-prefix.len()..span.start] == prefix
                && tokens.get(end) == Some(&suffix), "Kimi probe source-dimension markers differ");
        }
        covered.extend(span.start..end);
        spans.push(MediaSpan { start: span.start, len: span.len, key: image.key.into() });
    }
    ensure!(tokens.iter().enumerate().all(|(i, &id)| id != placeholder || covered.contains(&i)),
        "unbound still probe placeholders");
    Ok(spans)
}

pub(crate) fn prepare(job: NativeRequest, tokens: Vec<u32>, vocabulary: u32, hidden: usize,
    context: usize, snapshot: &Path, family: ImageFamily)
    -> Result<(crate::shared::qwen::scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
    let spans = spans(&job, &tokens, vocabulary, context, snapshot, family)?;
    let keys = MediaKeys::new(&tokens, vocabulary, &spans)?;
    let media = RequestMedia::new(spans.clone(), hidden, tokens.len())?;
    let jobs = job.media.iter().map(|image| EncodeJob::image_with_pixels(image.key,
        [image.grid.t, image.grid.h, image.grid.w], image.rgb8.clone(), image.pixels, image.tokens, hidden)).collect();
    crate::shared::probe::image_echo(&job, &spans)?;
    Ok((crate::shared::qwen::scheduler::Prompt { job, tokens, keys }, media, jobs))
}

pub(crate) fn snapshot_identity(snapshot: &Path, family: ImageFamily) -> Result<serde_json::Value> {
    use sha2::{Digest, Sha256};
    let mut identity = serde_json::json!({"snapshot_revision": snapshot.file_name().and_then(|s| s.to_str())
        .context("probe snapshot revision")?, "family": format!("{family:?}"), "source": "sglang@a9bf8c76"});
    for name in ["config.json", "tokenizer_config.json", "tokenizer.json", "tiktoken.model", "model.safetensors.index.json"] {
        let path = snapshot.join(name);
        if path.is_file() { identity[format!("{name}_sha256")] = format!("{:x}", Sha256::digest(std::fs::read(path)?)).into(); }
    }
    Ok(identity)
}

pub(crate) fn features(prompt: &crate::shared::qwen::scheduler::Prompt, media: &mut RequestMedia,
    cache: &mut EmbeddingCache, snapshot: &Path, family: ImageFamily) -> Result<()> {
    if media.spans().is_empty() { return Ok(()); }
    ensure!(prompt.job.probe.as_ref().is_some_and(|p| p.spec.feature_scoring()), "still media is fidelity-only");
    let root = std::env::var_os("CUTEAFD_MEDIA_FEATURES_DIR").context("fidelity image feature payload directory absent")?;
    super::features::apply_probe_features(&prompt.job, media, cache, &root.into(),
        &snapshot_identity(snapshot, family)?, "a9bf8c76")
}

/// Diagnostic path only: reuse admitted embedding storage, never allocate GPU
/// scratch or introduce a captured pointer. Normal text calls return immediately.
pub(crate) fn inject(stream: &crate::shared::memory::device::Stream<'_>,
    output: cuteafd_ffi::CuteafdDeviceBuffer, media: &RequestMedia, start: usize, rows: usize,
    width: usize, fp32: bool) -> Result<()> {
    let end = start.checked_add(rows).context("still probe injection extent")?;
    if media.needed(start, end).next().is_none() { return Ok(()); }
    let mut chunk = cuteafd_engine::media::MediaChunk::default();
    media.write_chunk(start, end, &mut chunk)?;
    ensure!(media.row_bytes() == width * 2 && output.bytes >= rows * width * if fp32 { 4 } else { 2 },
        "still probe injection output extent");
    stream.drain()?;
    for (&row, bytes) in chunk.indices.iter().zip(chunk.features.chunks_exact(width * 2)) {
        let converted;
        let data = if fp32 {
            converted = bytes.chunks_exact(2).flat_map(|b|
                (u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).to_le_bytes()).collect::<Vec<_>>();
            &converted[..]
        } else { bytes };
        let offset = row as usize * data.len();
        let target = crate::shared::prefix::view(output, offset, data.len())?;
        stream.device.library.copy_h2d(target, data)?;
    }
    Ok(())
}
