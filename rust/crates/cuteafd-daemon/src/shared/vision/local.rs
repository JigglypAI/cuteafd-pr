//! Nonblocking adapter: the scheduler queues host descriptors, the resident owner
//! encodes one image at a time. The small CUDA-owner queue is never a history limit.
use super::{normalization_lut, EncodeJob, EncoderService, EncoderTicket};
use cuteafd_engine::media::{self, EncodeInput, EncodeOutput, EncoderClient, MediaError};
use cuteafd_loader::media::ProcessorConfig;
use std::{collections::VecDeque, sync::Arc};

struct Pending {
    id: media::EncoderTicket,
    job: media::EncodeJob,
}
struct Running {
    pending: Pending,
    ticket: EncoderTicket,
}
pub struct LocalEncoder {
    service: EncoderService,
    lut: Arc<[f32; 768]>,
    width: usize,
    limits: cuteafd_core::media::ImageEncodeLimits,
    patch_size: usize,
    queue: VecDeque<Pending>,
    running: Option<Running>,
    next: u64,
}
impl LocalEncoder {
    pub fn new(
        service: EncoderService,
        config: &ProcessorConfig,
        width: usize,
        max_tokens: usize,
    ) -> Self {
        let limits = cuteafd_core::media::ImageEncodeLimits {
            max_patches: u32::try_from(max_tokens.saturating_mul(config.merge.pow(2) as usize))
                .unwrap_or(u32::MAX),
            max_output_rows: u32::try_from(max_tokens).unwrap_or(u32::MAX),
            max_canvas_pixels: (max_tokens as u64)
                .saturating_mul(u64::from(config.merge.pow(2)))
                .saturating_mul(u64::from(config.patch.pow(2))),
            geometry: cuteafd_core::media::ImageTokenGeometry {
                merge: config.merge,
                row_markers: u32::from(
                    config.family == cuteafd_loader::media::ImageFamily::Mistral,
                ),
            },
        };
        Self::with_limits(service, config, width, limits)
    }
    /// Towers with a padded/source canvas declare its physical limit explicitly.
    pub fn with_limits(
        service: EncoderService,
        config: &ProcessorConfig,
        width: usize,
        limits: cuteafd_core::media::ImageEncodeLimits,
    ) -> Self {
        Self {
            service,
            lut: normalization_lut(config),
            width,
            limits,
            patch_size: config.patch as usize,
            queue: VecDeque::new(),
            running: None,
            next: 0,
        }
    }
    pub(crate) fn gpu_turn(&self) -> Option<Arc<crate::shared::gpu_turn::GpuTurn>> {
        self.service.gpu_turn()
    }
    pub fn healthy(&self) -> bool {
        self.service.healthy()
    }
    pub fn health_handle(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.service.health_handle()
    }
    fn validate(&self, job: &media::EncodeJob) -> Result<(), MediaError> {
        job.validate()?;
        if job.hidden_width != self.width {
            return Err(MediaError::Features);
        }
        let EncodeInput::Image {
            grid: [t, h, w],
            rgb8,
            ..
        } = &job.input
        else {
            return if matches!(&job.input, EncodeInput::Audio { geometry, .. } if Some(*geometry) == self.service.audio_geometry())
                && self.service.audio_ledger.is_some()
            {
                Ok(())
            } else {
                Err(MediaError::Encoder("audio encoder not loaded".into()))
            };
        };
        let (t, h, w) = (*t, *h, *w);
        let pixels = job.pixels(self.patch_size as u32)?;
        self.limits
            .validate([t, h, w], pixels, job.tokens as u64, rgb8.len() as u64)
            .map_err(|_| MediaError::Features)?;
        job.feature_bytes()?;
        Ok(())
    }
}
impl EncoderClient for LocalEncoder {
    fn submit(&mut self, job: media::EncodeJob) -> Result<media::EncoderTicket, MediaError> {
        self.validate(&job)?;
        // MediaAdmission bounds requests and reserves feature bytes before this call.
        let id = media::EncoderTicket(self.next);
        self.next = self.next.checked_add(1).ok_or(MediaError::QueueFull)?;
        self.queue.push_back(Pending { id, job });
        Ok(id)
    }
    fn poll(&mut self, id: media::EncoderTicket) -> Option<Result<EncodeOutput, MediaError>> {
        if self.running.is_none() {
            let pending = self.queue.pop_front()?;
            let output = vec![0; pending.job.tokens * self.width];
            let submitted = match &pending.job.input {
                EncodeInput::Image { grid, rgb8, .. } => self.service.submit(EncodeJob {
                    rgb: rgb8.clone(),
                    grid: [grid[1] as usize, grid[2] as usize],
                    pixels: Some(
                        pending
                            .job
                            .pixels(self.patch_size as u32)
                            .expect("validated image geometry"),
                    ),
                    lut: self.lut.clone(),
                    output,
                }),
                EncodeInput::Audio { pcm, .. } => {
                    self.service.submit_audio(super::audio::AudioEncodeJob {
                        pcm: pcm.clone(),
                        fp32_scratch: if self.service.audio_geometry().is_some_and(|g| g.is_mimo())
                        {
                            vec![0.0; output.len()]
                        } else {
                            Vec::new()
                        },
                        output,
                    })
                }
            };
            match submitted {
                Ok(ticket) => self.running = Some(Running { pending, ticket }),
                Err(super::VisionError::QueueFull) => {
                    // Cancelled work still drains on the owner. Backpressure is not a failed encode.
                    self.queue.push_front(pending);
                    return None;
                }
                Err(error) => {
                    // Keep errors keyed: MediaAdmission may poll tickets in any order.
                    self.queue.push_front(pending);
                    if self.queue.front()?.id == id {
                        self.queue.pop_front();
                        return Some(Err(MediaError::Encoder(error.to_string())));
                    }
                    return None;
                }
            }
        }
        let running = self.running.as_ref()?;
        if running.pending.id != id {
            return None;
        }
        let result = match running.ticket.poll() {
            Ok(None) => return None,
            Ok(Some(output)) => Ok(output),
            Err(error) => Err(MediaError::Encoder(error.to_string())),
        };
        let running = self.running.take().unwrap();
        Some(result.map(|output| {
            EncodeOutput {
                key: running.pending.job.key,
                features: output
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>()
                    .into(),
                elapsed_ms: running.ticket.encode_ms(),
            }
        }))
    }
    fn cancel(&mut self, id: media::EncoderTicket) {
        self.queue.retain(|pending| pending.id != id);
        if self
            .running
            .as_ref()
            .is_some_and(|running| running.pending.id == id)
        {
            // Dropping the ticket cancels, but the owner retains queued/running storage until drain.
            self.running.take();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::ImageKey;
    use std::sync::{atomic::Ordering, mpsc};
    use std::time::Instant;

    #[test]
    fn kimi_canvas_boundary_is_identical_on_local_and_remote_admission() {
        use cuteafd_core::media::{ImageEncodeLimits, ImagePixels, ImageTokenGeometry};
        let limits = ImageEncodeLimits {
            max_patches: 65536,
            max_output_rows: 16384,
            max_canvas_pixels: 65536 * 14 * 14,
            geometry: ImageTokenGeometry::MERGED,
        };
        let (_send, _jobs) = mpsc::sync_channel::<super::super::Work>(2);
        let service = EncoderService {
            queue: Some(_send),
            owner: None,
            ledger: Default::default(),
            audio_ledger: None,
            audio_geometry: None,
            healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            turn: None,
        };
        let config = ProcessorConfig::for_family(cuteafd_loader::media::ImageFamily::Qwen);
        let encoder = LocalEncoder::with_limits(service, &config, 1, limits);
        let handshake = super::super::remote::EncoderHandshake {
            encoder_id: cuteafd_loader::media::EncoderId([1; 32]),
            max_patches: limits.max_patches,
            max_output_rows: limits.max_output_rows,
            max_canvas_pixels: limits.max_canvas_pixels,
            output_width: 1,
            patch_size: 14,
            merge_size: 2,
            row_markers: 0,
            plan_hash: [2; 32],
        };
        assert_eq!(handshake.image_limits(), limits);
        let pixels = ImagePixels {
            source: [8192, 4096],
            canvas: [3584, 3584],
            content: [0, 0, 3583, 3584],
        };
        let bytes = (limits.max_canvas_pixels * 3) as usize;
        let job = media::EncodeJob::image_with_pixels(
            ImageKey([1; 32]),
            [1, 256, 256],
            vec![0; bytes].into(),
            pixels,
            16384,
            1,
        );
        assert!(encoder.validate(&job).is_ok());
        assert!(handshake
            .image_limits()
            .validate([1, 256, 256], pixels, 16384, bytes as u64)
            .is_ok());
        let mut over = pixels;
        over.canvas = [1, limits.max_canvas_pixels as u32 + 1];
        over.content = [0, 0, 1, over.canvas[1]];
        let over_job = media::EncodeJob::image_with_pixels(
            ImageKey([1; 32]),
            [1, 256, 256],
            vec![0; bytes + 3].into(),
            over,
            16384,
            1,
        );
        assert!(matches!(
            encoder.validate(&over_job),
            Err(MediaError::Features)
        ));
        assert!(handshake
            .image_limits()
            .validate([1, 256, 256], over, 16384, (bytes + 3) as u64)
            .is_err());
    }
    #[test]
    fn pcm_audio_uses_shared_owner_without_fp32_scratch() {
        use cuteafd_core::media::AudioTokenGeometry;
        let geometry = AudioTokenGeometry::PcmPatches {
            samples_per_token: 1920,
        };
        let (queue, jobs) = mpsc::sync_channel::<super::super::Work>(2);
        let owner = std::thread::spawn(move || {
            while let Ok(work) = jobs.recv() {
                let super::super::EncoderJob::Audio(mut job) = work.job else {
                    panic!("expected PCM job");
                };
                assert!(job.fp32_scratch.is_empty());
                assert_eq!(job.output.len(), job.pcm.len().div_ceil(1920) * 8192);
                job.output.fill(0x3f80);
                work.encode_ns.store(1_250_000, Ordering::Release);
                let _ = work.reply.send(Ok(job.output));
            }
        });
        let service = EncoderService {
            queue: Some(queue),
            owner: Some(super::super::push::OwnerThread::mock(owner)),
            ledger: Default::default(),
            audio_ledger: Some(Default::default()),
            audio_geometry: Some(geometry),
            healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            turn: None,
        };
        let config = ProcessorConfig::for_family(cuteafd_loader::media::ImageFamily::Mimo);
        let mut encoder = LocalEncoder::new(service, &config, 8192, 256);
        let wrong =
            media::EncodeJob::audio(media::AudioKey([0; 32]), vec![0.0; 1921].into(), 1, 8192);
        assert!(encoder.submit(wrong).is_err());
        let job = media::EncodeJob::audio_with_geometry(
            media::AudioKey([1; 32]),
            vec![0.0; 1921].into(),
            2,
            8192,
            geometry,
        );
        let id = encoder.submit(job).unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Some(output) = encoder.poll(id) {
                let output = output.unwrap();
                assert_eq!(
                    output.elapsed_ms, 1.25,
                    "must use native owner timing, not poll latency"
                );
                assert_eq!(output.features.len(), 2 * 8192 * 2);
                assert!(output.features.chunks_exact(2).all(|v| v == [0x80, 0x3f]));
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    }

    #[test]
    fn rgb_extent_uses_the_processor_patch_size() {
        use cuteafd_loader::media::ImageFamily;
        for family in [ImageFamily::Mimo, ImageFamily::Qwen, ImageFamily::GlmFlash] {
            let (queue, _jobs) = mpsc::sync_channel::<super::super::Work>(2);
            let service = EncoderService {
                queue: Some(queue),
                owner: None,
                ledger: Default::default(),
                audio_ledger: None,
                audio_geometry: None,
                healthy: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            turn: None,
            };
            let config = ProcessorConfig::for_family(family);
            let encoder = LocalEncoder::new(service, &config, 2, 256);
            let bytes = (config.patch * config.patch * 3 * 4) as usize;
            let mut job =
                media::EncodeJob::image(ImageKey([0; 32]), [1, 2, 2], vec![0; bytes].into(), 1, 2);
            assert!(encoder.validate(&job).is_ok());
            let wrong_patch = if config.patch == 14 { 16 } else { 14 };
            job.input = EncodeInput::Image {
                grid: [1, 2, 2],
                rgb8: vec![0; wrong_patch * wrong_patch * 3 * 4].into(),
                pixels: None,
            };
            assert!(matches!(encoder.validate(&job), Err(MediaError::Features)));
        }
    }

    #[test]
    fn local_health_rejects_cached_or_probe_admission_after_owner_exit() {
        let (queue, _jobs) = mpsc::sync_channel::<super::super::Work>(2);
        let health = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let owner_health = health.clone();
        let owner = std::thread::spawn(move || {
            let _health = super::super::OwnerHealth(owner_health);
        });
        owner.join().unwrap();
        let service = EncoderService {
            queue: Some(queue),
            owner: None,
            ledger: Default::default(),
            audio_ledger: None,
            audio_geometry: None,
            healthy: health,
            turn: None,
        };
        let config = ProcessorConfig::for_family(cuteafd_loader::media::ImageFamily::Qwen);
        let encoder = LocalEncoder::new(service, &config, 2, 256);
        assert!(!encoder.healthy());
        assert!(!encoder.health_handle().load(Ordering::Acquire));
    }
    #[test]
    fn cancelled_owner_queue_is_backpressure_not_an_encode_failure() {
        let (queue, jobs) = mpsc::sync_channel::<super::super::Work>(2);
        let (gate, wait) = mpsc::sync_channel(0);
        let owner = std::thread::spawn(move || {
            wait.recv().unwrap();
            while let Ok(work) = jobs.recv() {
                if work.cancelled.load(Ordering::Acquire) {
                    continue;
                }
                let super::super::EncoderJob::Vision(mut job) = work.job else {
                    panic!("expected vision job")
                };
                job.output.fill(0x3f80);
                let _ = work.reply.send(Ok(job.output));
            }
        });
        let service = EncoderService {
            queue: Some(queue),
            owner: Some(super::super::push::OwnerThread::mock(owner)),
            ledger: Default::default(),
            audio_ledger: None,
            audio_geometry: None,
            healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            turn: None,
        };
        let config = ProcessorConfig::for_family(cuteafd_loader::media::ImageFamily::Mimo);
        let mut encoder = LocalEncoder::new(service, &config, 2, 256);
        for n in 0..2 {
            let id = encoder
                .submit(media::EncodeJob::image(
                    ImageKey([n; 32]),
                    [1, 2, 2],
                    vec![0; 3072].into(),
                    1,
                    2,
                ))
                .unwrap();
            assert!(encoder.poll(id).is_none());
            encoder.cancel(id);
        }
        let id = encoder
            .submit(media::EncodeJob::image(
                ImageKey([2; 32]),
                [1, 2, 2],
                vec![0; 3072].into(),
                1,
                2,
            ))
            .unwrap();
        assert!(encoder.poll(id).is_none());
        assert_eq!(encoder.queue.len(), 1);
        gate.send(()).unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Some(output) = encoder.poll(id) {
                assert_eq!(&*output.unwrap().features, &[0x80, 0x3f, 0x80, 0x3f]);
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
    }

    #[test]
    fn long_history_is_queued_and_cancelled_without_owner_queue_overflow() {
        let (queue, jobs) = mpsc::sync_channel::<super::super::Work>(2);
        let owner = std::thread::spawn(move || {
            while let Ok(work) = jobs.recv() {
                if work.cancelled.load(Ordering::Acquire) {
                    continue;
                }
                let super::super::EncoderJob::Vision(mut job) = work.job else {
                    panic!("expected vision job")
                };
                job.output.fill(u16::from_le_bytes([job.rgb[0], 0x3f]));
                let _ = work.reply.send(Ok(job.output));
            }
        });
        let service = EncoderService {
            queue: Some(queue),
            owner: Some(super::super::push::OwnerThread::mock(owner)),
            ledger: Default::default(),
            audio_ledger: None,
            audio_geometry: None,
            healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            turn: None,
        };
        let config = ProcessorConfig::for_family(cuteafd_loader::media::ImageFamily::Mimo);
        let mut encoder = LocalEncoder::new(service, &config, 2, 256);
        let mut ids = (0..64u8)
            .map(|i| {
                encoder
                    .submit(media::EncodeJob::image(
                        ImageKey([i; 32]),
                        [1, 2, 2],
                        vec![i; 3072].into(),
                        1,
                        2,
                    ))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        encoder.cancel(ids.remove(20));
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        let mut completed = 0;
        while !ids.is_empty() {
            assert!(Instant::now() < deadline);
            ids.retain(|&id| {
                if let Some(output) = encoder.poll(id) {
                    let output = output.unwrap();
                    assert_eq!(
                        &*output.features,
                        &[output.key.bytes()[0], 0x3f, output.key.bytes()[0], 0x3f]
                    );
                    completed += 1;
                    false
                } else {
                    true
                }
            });
            std::thread::yield_now();
        }
        assert_eq!(completed, 63);
    }
}
