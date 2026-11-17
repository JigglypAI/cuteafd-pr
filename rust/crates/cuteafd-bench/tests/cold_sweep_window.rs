//! Separate binary: benchmark readiness and its loopback address are process-wide.
mod common;
use common::fake_engine;
use cuteafd_api::openai::{router_with_console, ConsoleHub, NativeLimits, NativeRequest};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

/// A cold purge runs before every prefill point's warm-up and again before its
/// timed request; the measurement window still opens exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prefill_purge_precedes_each_warm_up_and_timed_request() {
    let (queue, receive) = mpsc::channel::<NativeRequest>(16);
    let (counted, forward) = mpsc::channel::<NativeRequest>(16);
    let seen = Arc::new(AtomicUsize::new(0));
    let tally = seen.clone();
    tokio::spawn(async move {
        let mut receive = receive;
        while let Some(job) = receive.recv().await {
            tally.fetch_add(1, Ordering::SeqCst);
            if counted.send(job).await.is_err() {
                return;
            }
        }
    });
    fake_engine(forward);
    let router = router_with_console(queue, NativeLimits::default(), Arc::new(Mutex::new(Value::Null)),
        Duration::from_secs(5), ConsoleHub::disabled());
    let directory = tempfile::tempdir().unwrap();
    let bench = cuteafd_bench::Bench::new(cuteafd_bench::store::Store::open(directory.path()).unwrap());
    bench.set_api_key(cuteafd_api::openai::auth::ApiKey::new("sweep-test-key").unwrap());
    let router = cuteafd_bench::http::mount(common::with_fake_token_count(router), bench);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    cuteafd_bench::context::mark_ready(listener.local_addr().unwrap());
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, router.into_make_service_with_connect_info::<std::net::SocketAddr>()).await.unwrap();
    });
    let (purged_at, opened_at, sweep) = tokio::task::spawn_blocking(move || {
        let purged_at = Mutex::new(Vec::new());
        let opened_at = Arc::new(Mutex::new(Vec::new()));
        let (record, counter) = (opened_at.clone(), seen.clone());
        let purge = || { purged_at.lock().unwrap().push(seen.load(Ordering::SeqCst)); Ok(()) };
        let options = cuteafd_bench::sweep::PrefillOptions { repeat: true, purge: Some(&purge), idle_seconds: 0.03 };
        let started = std::time::Instant::now();
        let sweep = cuteafd_bench::sweep::run_windowed(&base, Some("sweep-test-key"), &[], false, false, &[256, 512], &options, false,
            cuteafd_bench::sweep::TOKENS, Some(Box::new(move || {
                record.lock().unwrap().push(counter.load(Ordering::SeqCst));
                Ok(())
            }))).unwrap();
        assert!(started.elapsed() >= Duration::from_millis(60));
        let purged = purged_at.lock().unwrap().clone();
        let opened = opened_at.lock().unwrap().clone();
        (purged, opened, sweep)
    }).await.unwrap();
    // Six prelude requests, two sizing requests, then purge/warm/purge/timed.
    assert_eq!(purged_at, vec![8, 9, 10, 11]);
    assert_eq!(opened_at, vec![9]);
    assert_eq!(sweep.prefill.len(), 2);
    assert!(sweep.prefill.iter().all(|point| point.idle_seconds == 0.03));
}

