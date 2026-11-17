//! CPU-only HTTP tests for the shared continuous-refill runner.
use cuteafd_api::openai::{router_with_console, ConsoleHub, InferenceChunk, InferenceFinishReason,
    NativeLimits, NativeRequest, PromptUsage};
use cuteafd_bench::{client::Client, sustained};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use std::time::Duration;
use tokio::sync::mpsc;

fn begin(job: &NativeRequest) {
    job.events.send(Ok(InferenceChunk::Ready { system_fingerprint: None,
        prompt_usage: PromptUsage { prompt_tokens: 120, prompt_cache_hit_tokens: 0 } })).unwrap();
    // Multiple tokens per native chunk: SSE chunk count is not token count.
    job.events.send(Ok(InferenceChunk::Text { content: "a ".into(), content_tokens: 3 })).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refill_holds_c_until_pool_exhausted_and_records_only_steady_window() {
    const C: usize = 3;
    const N: usize = 12;
    let (queue, mut receive) = mpsc::channel::<NativeRequest>(32);
    let engine = tokio::spawn(async move {
        let mut active = VecDeque::new();
        let mut prompts = Vec::new();
        for _ in 0..C {
            let job = receive.recv().await.unwrap();
            prompts.push(job.prompt.clone());
            begin(&job);
            active.push_back(job);
        }
        let mut widths = Vec::new();
        for completed in 0..N {
            widths.push(active.len());
            // Finish exactly one request, keeping all the other requests live.
            tokio::time::sleep(Duration::from_millis(10)).await;
            let job = active.pop_front().unwrap();
            job.events.send(Ok(InferenceChunk::Text { content: "b ".into(), content_tokens: 7 })).unwrap();
            job.events.send(Ok(InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length })).unwrap();
            if completed < N - C {
                // A closed-wave implementation deadlocks here: the remaining C-1
                // streams are deliberately unfinished until the refill arrives.
                let job = tokio::time::timeout(Duration::from_secs(5), receive.recv()).await.unwrap().unwrap();
                prompts.push(job.prompt.clone());
                begin(&job);
                active.push_back(job);
                assert_eq!(active.len(), C);
                assert!(receive.try_recv().is_err(), "more than C requests in flight");
            }
        }
        (widths, prompts)
    });
    let router = router_with_console(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(5), ConsoleHub::disabled());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    let window = tokio::task::spawn_blocking(move || {
        let mut client = Client::new(&base, None, Arc::new(AtomicBool::new(false)));
        client.discover().unwrap();
        sustained::run(&client, C, &sustained::Options { sustained: true, multiplier: 4 }, 192).unwrap()
    }).await.unwrap();
    let (widths, prompts) = engine.await.unwrap();
    server.abort();
    assert_eq!(&widths[..N - C + 1], &[C; N - C + 1]);
    assert_eq!(&widths[N - C + 1..], &[2, 1]);
    assert_eq!(prompts.len(), N);
    assert_eq!(prompts.iter().collect::<std::collections::HashSet<_>>().len(), N);
    assert_eq!(window.initial_requests, C);
    assert_eq!(window.total_requests, N);
    assert_eq!(window.refill_requests, N - C);
    assert_eq!(window.completed_requests, N);
    assert_eq!(window.start_s, window.requests.iter().map(|r| r.completed_s).fold(f64::INFINITY, f64::min));
    assert_eq!(window.end_s, window.requests.iter().map(|r| r.sent_s).fold(0.0, f64::max));
    assert!(window.start_s > window.requests.iter().map(|r| r.first_token_s).fold(f64::INFINITY, f64::min));
    assert!(window.end_s < window.requests.iter().map(|r| r.completed_s).fold(0.0, f64::max));
    assert!(window.requests.iter().all(|r| r.completion_tokens == 10));
    assert!(window.credited_tokens > 0.0 && window.credited_tokens < (N * 10) as f64);
    assert_eq!(window.credited_tokens, window.requests.iter().map(|r| r.credited_tokens).sum::<f64>());
    assert!(window.token_accounting.contains("assuming_uniform_emission"));
    for r in &window.requests {
        let overlap = (r.last_token_s.min(window.end_s) - r.first_token_s.max(window.start_s)).max(0.0);
        assert!((r.credited_tokens - 10.0 * overlap / (r.last_token_s - r.first_token_s)).abs() < 1e-9);
    }
}

#[test]
fn closed_point_bytes_and_old_sweep_deserialization_are_unchanged() {
    let old = r#"{"c":8,"phase":"levels","aggregate_tok_s":100.0,"per_request_tok_s":12.5,"ttft_s":0.1,"errors":0}"#;
    let point: cuteafd_bench::sweep::Point = serde_json::from_str(old).unwrap();
    assert_eq!(serde_json::to_string(&point).unwrap(), old);
    assert!(point.sustained.is_none());
    let old_sweep = format!(r#"{{"points":[{old}]}}"#);
    let sweep: cuteafd_bench::sweep::Sweep = serde_json::from_str(&old_sweep).unwrap();
    assert_eq!(serde_json::to_string(&sweep).unwrap(), old_sweep);
}
