//! Gate-private immutable sparse-header templates, shared by nextest processes.
use std::{fs, io::{Read, Write}, path::Path, time::{Duration, Instant}};
use sha2::{Digest, Sha256};

/// Each caller gets new inodes: mutation tests cannot alter a cached template.
/// Templates live with the test executables, so removing the task's Cargo target
/// also removes every cached header; no user cache or checkpoint is consulted.
pub(super) fn snapshot(destination: &Path, key: &str, build: impl FnOnce(&Path)) {
    if !cfg!(any(test, feature = "test-support")) {
        build(destination);
        return;
    }
    let started = Instant::now();
    let executable = std::env::current_exe().expect("fixture executable");
    let root = executable.parent().expect("fixture executable directory").join("header-templates");
    fs::create_dir_all(&root).expect("fixture template root");
    let mut hash = Sha256::new();
    static SOURCE_HASH: std::sync::LazyLock<[u8; 32]> = std::sync::LazyLock::new(|| {
        let mut source = Sha256::new();
        source.update(include_bytes!("../testing.rs"));
        source.update(include_bytes!("header_templates.rs"));
        source.update(include_bytes!("../families/step5/fixtures/official-fp8.json"));
        source.update(include_bytes!("../../families/deepseek_v41/official-v41-config.json"));
        source.update(include_bytes!("../../families/deepseek_v41/official-v41-tensors.json"));
        source.finalize().into()
    });
    hash.update(*SOURCE_HASH);
    hash.update(key.as_bytes());
    let key = format!("{:x}", hash.finalize());
    let ready = root.join(&key);
    let lock = fs::OpenOptions::new().create(true).truncate(false).write(true)
        .open(root.join(format!("{key}.lock"))).expect("fixture template lock");
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match rustix::fs::flock(&lock, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
            Ok(()) => break,
            Err(rustix::io::Errno::WOULDBLOCK) => {
                assert!(Instant::now() < deadline, "fixture template lock timed out");
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("fixture template lock: {error}"),
        }
    }
    let missing = !ready.is_dir();
    if missing {
        let staging = root.join(format!("{key}.building-{}", std::process::id()));
        if staging.exists() { fs::remove_dir_all(&staging).expect("stale fixture template staging"); }
        fs::create_dir(&staging).expect("fixture template staging");
        build(&staging);
        fs::rename(&staging, &ready).expect("publish fixture template atomically");
    }
    drop(lock);
    let built = started.elapsed();
    clone_headers(&ready, destination);
    eprintln!("fixture-template key={key} cold={missing} build_or_wait={built:?} clone={:?}", started.elapsed() - built);
}

fn clone_headers(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("fixture clone directory");
    for entry in fs::read_dir(source).expect("fixture template entries") {
        let entry = entry.expect("fixture template entry");
        let from = entry.path();
        let to = destination.join(entry.file_name());
        if from.is_dir() { clone_headers(&from, &to); }
        else if from.extension().is_some_and(|extension| extension == "safetensors") {
            // Payloads created by these builders are sparse zeros. Copy only the
            // header and restore the logical extent without reading the hole.
            let mut input = fs::File::open(&from).expect("template shard");
            let mut length = [0; 8];
            input.read_exact(&mut length).expect("template header length");
            let length_bytes = u64::from_le_bytes(length);
            assert!(length_bytes <= 256 << 20, "fixture header exceeds bounded size");
            let mut header = vec![0; length_bytes as usize];
            input.read_exact(&mut header).expect("template header");
            let mut output = fs::File::create(&to).expect("private fixture shard");
            output.write_all(&length).expect("clone header length");
            output.write_all(&header).expect("clone header");
            output.set_len(input.metadata().expect("template extent").len()).expect("clone sparse extent");
        } else { fs::copy(&from, &to).expect("clone fixture metadata"); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_templates_are_complete_sparse_and_mutation_isolated() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let key = format!("isolation-{}", std::process::id());
        snapshot(first.path(), &key, |path| {
            super::super::write_safetensors(&path.join("model.safetensors"),
                &[super::super::t("matrix", "BF16", &[128, 128])]);
            fs::write(path.join("config.json"), "{}").unwrap();
        });
        snapshot(second.path(), &key, |_| panic!("cache hit rebuilt headers"));
        assert_eq!(crate::read_safetensors_metadata(&second.path().join("model.safetensors")).unwrap().len(), 1);
        fs::write(first.path().join("config.json"), "changed").unwrap();
        fs::write(first.path().join("model.safetensors"), "changed").unwrap();
        assert_eq!(fs::read_to_string(second.path().join("config.json")).unwrap(), "{}");
        assert_eq!(crate::read_safetensors_metadata(&second.path().join("model.safetensors")).unwrap().len(), 1);
    }
}
