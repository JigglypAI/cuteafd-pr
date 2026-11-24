//! Hy4-flash image request state: checked span expansion of the chat template's
//! `<image start> <image pad> <image end>` triple into the tower's `h/2 * (w/2+1)`
//! rows, exact prefix-cache media keys and interleaved M-RoPE coordinates.
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::media::{EncodeJob, MediaKeys, RequestMedia};
use cuteafd_loader::{families::hy4_flash::vision::{ImageSpan, RopePositions, VisionContract},
    media::{MediaSpan, SpanExpander}};
use std::sync::Arc;

/// Expand native images (or bind a probe's already-expanded ids) into spans.
pub(super) fn prepare(job: NativeRequest, tokens: Vec<u32>, config: &serde_json::Value, contract: Option<&VisionContract>,
    vocabulary: usize, hidden: usize, context: usize) -> Result<(crate::shared::qwen::scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
    ensure!(job.audio.is_empty(), "Hy4 has no bundled audio tower");
    let vocabulary = u32::try_from(vocabulary)?;
    let (tokens, spans) = match contract {
        None => {
            ensure!(job.media.is_empty() && job.image_groups.is_empty(), "vision is disabled (VISION=off)");
            (tokens, Vec::new())
        }
        Some(contract) if job.probe.as_ref().is_some_and(|p| p.spec.prompt_ids.is_some() && !p.spec.media.is_empty()) => {
            let probe = job.probe.as_ref().expect("checked probe");
            probe.spec.validate_media()?;
            ensure!(probe.spec.media.len() == job.media.len() && tokens.len() <= context
                && tokens.iter().all(|&id| id < vocabulary), "Hy4 expanded probe media/context differs");
            let mut spans = Vec::with_capacity(job.media.len());
            for (span, image) in probe.spec.media.iter().zip(&job.media) {
                span.verify_image(image)?;
                let end = span.start.checked_add(span.len).context("Hy4 probe media extent")?;
                ensure!(span.start > 0 && end < tokens.len() && tokens[span.start - 1] == contract.image_start
                    && tokens[end] == contract.image_end
                    && tokens[span.start..end].iter().all(|&id| id == contract.image_token),
                    "Hy4 probe image rows/marker boundaries differ");
                spans.push(MediaSpan { start: span.start, len: span.len, key: image.key.into() });
            }
            ensure!(tokens.iter().filter(|&&id| id == contract.image_token).count()
                == spans.iter().map(|s| s.len).sum::<usize>(), "unbound Hy4 probe image placeholders");
            (tokens, spans)
        }
        Some(_) => {
            let expander = SpanExpander::from_config(config, vocabulary)?;
            let images = job.media.iter().map(|image| image.as_ref().clone()).collect::<Vec<_>>();
            let expanded = expander.expand(&tokens, &images, context)?;
            (expanded.tokens, expanded.media)
        }
    };
    if job.probe.is_some() { crate::shared::probe::image_echo(&job, &spans)?; }
    let keys = MediaKeys::new(&tokens, vocabulary, &spans)?;
    let media = RequestMedia::new(spans, hidden, tokens.len())?;
    let jobs = job.media.iter().map(|image| EncodeJob::image_with_pixels(image.key,
        [image.grid.t, image.grid.h, image.grid.w], image.rgb8.clone(), image.pixels, image.tokens, hidden)).collect();
    Ok((crate::shared::qwen::scheduler::Prompt { job, tokens, keys }, media, jobs))
}

/// Interleaved M-RoPE for the bound spans (equal axes for text-only requests).
pub(super) fn rope_positions(tokens: &[u32], job: &NativeRequest, media: &RequestMedia,
    contract: Option<&VisionContract>) -> Result<Arc<RopePositions>> {
    let Some(contract) = contract else {
        ensure!(media.spans().is_empty(), "Hy4 media spans without a vision contract");
        return Ok(Default::default());
    };
    ensure!(media.spans().len() == job.media.len() && media.prompt_len() == tokens.len(),
        "Hy4 rope span count/extent differs");
    let images = media.spans().iter().zip(&job.media).map(|(span, image)| {
        ensure!(image.grid.t == 1, "Hy4 images have one temporal frame");
        Ok(ImageSpan { start: span.start, grid: [image.grid.h as usize, image.grid.w as usize] })
    }).collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(RopePositions::new(tokens, &images, contract)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::media::{ImageGrid, ImageKey, PreparedImage};

    fn contract() -> (serde_json::Value, VisionContract) {
        let cfg: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/hy4_flash/fixtures/vision_config.json"))).unwrap();
        let contract = VisionContract::from_config(&cfg).unwrap();
        (cfg, contract)
    }
    fn request(media: Vec<Arc<PreparedImage>>) -> NativeRequest {
        NativeRequest { image_groups: Vec::new(), prefix_cache_source: Default::default(), finish_detail: Default::default(),
            literal_spans: Vec::new(), prompt: String::new(), constraint: None, images: Vec::new(), media, audio: Vec::new(),
            max_tokens: 8, sampling: Default::default(), stop_token_ids: Vec::new(),
            events: tokio::sync::mpsc::unbounded_channel().0, usage: None, probe: None }
    }
    fn image(key: u8, h: u32, w: u32) -> Arc<PreparedImage> {
        Arc::new(PreparedImage { key: ImageKey([key; 32]), encoder: cuteafd_loader::media::EncoderId([0; 32]),
            processor: cuteafd_loader::media::PreprocessId([0; 32]), grid: ImageGrid { t: 1, h, w },
            rgb8: Arc::from(vec![0; (h * w * 768) as usize]), pixels: cuteafd_core::media::ImagePixels::full(w * 16, h * 16),
            tokens: (h as usize / 2) * (w as usize / 2 + 1) })
    }

    #[test]
    fn native_images_expand_to_newline_spans_with_mrope_and_exact_keys() {
        let (cfg, c) = contract();
        let (s, pad, e) = (c.image_start, c.image_token, c.image_end);
        let images = || vec![image(1, 4, 6), image(2, 2, 2)];
        let tokens = vec![10, s, pad, e, 11, s, pad, e, 12];
        let (prompt, media, jobs) = prepare(request(images()), tokens.clone(), &cfg, Some(&c), 120832, 3072, 64).unwrap();
        assert_eq!(jobs.len(), 2);
        // 4x6 patches -> 2 rows x (3+1) columns = 8 rows; 2x2 -> 1 x 2 = 2 rows.
        assert_eq!(media.spans().iter().map(|s| (s.start, s.len)).collect::<Vec<_>>(), [(2, 8), (13, 2)]);
        assert_eq!(prompt.tokens.len(), tokens.len() + 7 + 1);
        let rope = rope_positions(&prompt.tokens, &prompt.job, &media, Some(&c)).unwrap();
        assert_eq!(rope.at(0).unwrap(), [0; 3]);
        assert_eq!(rope.at(2).unwrap(), [2, 2, 2]);
        assert_eq!(rope.at(9).unwrap(), [2, 3, 5]);
        // After the first span text resumes at 2 + max(2, 4) = 6.
        assert_eq!(rope.at(10).unwrap(), [6; 3]);
        assert_eq!(rope.at(13).unwrap(), [9, 9, 9]);
        assert_eq!(rope.at(14).unwrap(), [9, 9, 10]);
        assert_eq!(rope.at(15).unwrap(), [11; 3]);
        // A different image changes the prefix-cache keys from the span onward.
        let other = request(vec![image(3, 4, 6), image(2, 2, 2)]);
        let (changed, _, _) = prepare(other, tokens.clone(), &cfg, Some(&c), 120832, 3072, 64).unwrap();
        assert_eq!(changed.keys.tokens()[..2], prompt.keys.tokens()[..2]);
        assert_ne!(changed.keys.tokens()[2..], prompt.keys.tokens()[2..]);
        assert!(prepare(request(images()), tokens.clone(), &cfg, None, 120832, 3072, 64).is_err());
        assert!(prepare(request(images()), tokens, &cfg, Some(&c), 120832, 3072, 12).is_err());
    }

    #[test]
    fn text_only_requests_keep_equal_axes() {
        let (cfg, c) = contract();
        let (prompt, media, jobs) = prepare(request(Vec::new()), vec![1, 2, 3], &cfg, Some(&c), 120832, 3072, 64).unwrap();
        assert!(jobs.is_empty() && media.spans().is_empty());
        let rope = rope_positions(&prompt.tokens, &prompt.job, &media, Some(&c)).unwrap();
        assert!(rope.is_text());
        assert_eq!(rope.at(2).unwrap(), [2; 3]);
        assert!(prepare(request(Vec::new()), vec![1, 2], &cfg, None, 120832, 3072, 64).is_ok());
    }
}
