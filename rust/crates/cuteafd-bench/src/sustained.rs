//! Continuous-refill concurrency, shared by the in-server panel and external sweep.
use crate::client::Client;
use crate::panels::common::plain;
use crate::panels::speed::CONTENT;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub sustained: bool,
    pub multiplier: usize,
}
impl Default for Options {
    fn default() -> Self { Self { sustained: false, multiplier: 4 } }
}
impl Options {
    pub fn parse(value: &Value) -> Result<Self> {
        let options: Self = serde_json::from_value(value.clone()).context("concurrency panel-config")?;
        ensure!((2..=64).contains(&options.multiplier), "concurrency multiplier must be 2..=64");
        Ok(options)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Request {
    #[serde(default)]
    pub timing: crate::report::StreamTiming,
    pub index: usize,
    pub sent_s: f64,
    pub first_token_s: f64,
    pub last_token_s: f64,
    pub completed_s: f64,
    pub completion_tokens: u64,
    pub credited_tokens: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Window {
    /// All timestamps are seconds from the beginning of this refill run.
    pub start_s: f64,
    pub end_s: f64,
    pub elapsed_s: f64,
    pub initial_requests: usize,
    pub total_requests: usize,
    pub refill_requests: usize,
    pub completed_requests: usize,
    pub completions_in_window: usize,
    pub requests_overlapping_window: usize,
    pub credited_tokens: f64,
    pub token_accounting: String,
    pub pool_sha256: String,
    pub requests: Vec<Request>,
}
impl Window {
    pub fn tok_s(&self) -> f64 { self.credited_tokens / self.elapsed_s }
    /// Whole-request medians, distinct from the window's aggregate estimate.
    pub fn per_request_tok_s(&self) -> f64 {
        let mut rates: Vec<_> = self.requests.iter().map(|r| {
            let seconds = r.last_token_s - r.first_token_s;
            if seconds > 0.0 { r.completion_tokens.saturating_sub(1) as f64 / seconds } else { 0.0 }
        }).collect();
        crate::panels::common::median(&mut rates)
    }
    pub fn ttft_s(&self) -> f64 {
        crate::panels::common::median(&mut self.requests.iter().map(|r| r.first_token_s - r.sent_s).collect::<Vec<_>>())
    }
}

/// Usage gives exact token totals, not token timestamps. Do not count SSE text
/// chunks (one chunk can contain many tokens), or whole boundary-crossing turns.
fn credit(request: &Request, start: f64, end: f64) -> f64 {
    let duration = request.last_token_s - request.first_token_s;
    if duration > 0.0 {
        let overlap = (request.last_token_s.min(end) - request.first_token_s.max(start)).max(0.0);
        request.completion_tokens as f64 * overlap / duration
    } else if request.first_token_s >= start && request.first_token_s < end {
        request.completion_tokens as f64
    } else { 0.0 }
}

/// Precompute the indexed pool before dispatch. Worker finish order must not
/// decide prompts/nonces; independent arms get the same pool, including without
/// the optional process-wide nonce seed. The digest makes pairing auditable.
fn pool(c: usize, multiplier: usize) -> Vec<String> {
    let seed = std::env::var("CUTEAFD_BENCH_NONCE_SEED").unwrap_or_else(|_| "sustained-v1".into());
    (0..c * multiplier).map(|i| format!("[sustained-{seed}-c{c}-{i:010}]\n{}\n\n{}",
        crate::text::corpus(4242 + i as u64, 120), CONTENT[1].1)).collect()
}

/// C blocking workers immediately take the next indexed request on completion.
/// This leaves only normal HTTP dispatch latency between completion and refill,
/// not a closed-wave barrier. Errors invalidate the measurement, after joining
/// all workers, rather than silently reporting a lower effective concurrency.
pub fn run(client: &Client, c: usize, options: &Options, tokens: u64) -> Result<Window> {
    ensure!((1..=256).contains(&c), "sustained concurrency must be 1..=256");
    ensure!((2..=64).contains(&options.multiplier), "concurrency multiplier must be 2..=64");
    let prompts = pool(c, options.multiplier);
    let pool_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&prompts)?));
    let bodies: Vec<_> = prompts.iter().map(|text| plain(client, text, tokens)).collect();
    let ready = Arc::new(Barrier::new(c));
    let next = Mutex::new(0usize);
    let start = Instant::now();
    let requests = Mutex::new(Vec::with_capacity(bodies.len()));
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..c).map(|_| {
            let (ready, next, bodies, requests) = (ready.clone(), &next, &bodies, &requests);
            scope.spawn(move || -> Result<()> {
                ready.wait();
                loop {
                    client.check()?;
                    let job = {
                        let mut next = next.lock().unwrap();
                        if *next == bodies.len() { None } else {
                            let index = *next;
                            *next += 1;
                            Some((index, start.elapsed().as_secs_f64()))
                        }
                    };
                    let Some((index, sent_s)) = job else { return Ok(()); };
                    let chat = client.chat(bodies[index].clone(), None)
                        .with_context(|| format!("sustained C{c} request {index}"))?;
                    let completed_s = start.elapsed().as_secs_f64();
                    requests.lock().unwrap().push(Request { timing: chat.timing.clone(), index, sent_s,
                        first_token_s: sent_s + chat.timing.ttft_s,
                        last_token_s: sent_s + chat.timing.ttft_s + chat.timing.decode_s,
                        completed_s, completion_tokens: chat.timing.completion_tokens, credited_tokens: 0.0 });
                }
            })
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("sustained request thread panicked"))))
            .collect::<Vec<_>>()
    });
    results.into_iter().collect::<Result<Vec<_>>>()?;
    let mut requests = requests.into_inner().unwrap();
    requests.sort_by_key(|r| r.index);
    let start_s = requests.iter().map(|r| r.completed_s).fold(f64::INFINITY, f64::min);
    let end_s = requests.iter().map(|r| r.sent_s).fold(0.0, f64::max);
    ensure!(end_s > start_s, "sustained C{c}: no steady-state window (requests completed before refills)");
    for request in &mut requests { request.credited_tokens = credit(request, start_s, end_s); }
    Ok(Window { start_s, end_s, elapsed_s: end_s - start_s, initial_requests: c,
        total_requests: bodies.len(), refill_requests: bodies.len() - c, completed_requests: requests.len(),
        completions_in_window: requests.iter().filter(|r| r.completed_s >= start_s && r.completed_s <= end_s).count(),
        requests_overlapping_window: requests.iter().filter(|r| r.credited_tokens > 0.0).count(),
        credited_tokens: requests.iter().map(|r| r.credited_tokens).sum(),
        token_accounting: "usage_tokens_prorated_by_first_to_last_token_overlap_assuming_uniform_emission".into(),
        pool_sha256, requests })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_credits_only_overlap_not_warmup_or_drain() {
        let request = |first, last| Request { timing: Default::default(), index: 0, sent_s: 0.0, first_token_s: first,
            last_token_s: last, completed_s: last, completion_tokens: 100, credited_tokens: 0.0 };
        assert_eq!(credit(&request(0.0, 2.0), 2.0, 6.0), 0.0);
        assert_eq!(credit(&request(6.0, 8.0), 2.0, 6.0), 0.0);
        assert_eq!(credit(&request(0.0, 8.0), 2.0, 6.0), 50.0);
        assert_eq!(credit(&request(3.0, 5.0), 2.0, 6.0), 100.0);
        assert_eq!(credit(&request(3.0, 3.0), 2.0, 6.0), 100.0);
    }

    #[test]
    fn pool_is_paired_independent_of_request_finish_order() {
        assert_eq!(pool(8, 4), pool(8, 4));
        assert_eq!(pool(8, 4).len(), 32);
        assert_ne!(pool(8, 4)[0], pool(8, 4)[1]);
    }
}
