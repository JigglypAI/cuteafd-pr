"""CPU-only release builds take no hardware export locks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_build_takes_a_cpu_lock_and_runs_cpu_only_exports():
    """`build.sh` serializes only on its CPU build.lock and runs both
    coordinator and Spark AOT exports CPU-only without any GPU or lock."""
    build = (ROOT / 'build.sh').read_text()
    assert 'exec 9>"$release_build_lock_dir/build.lock"' in build
    assert 'release_build_lock_dir="$HOME/.cache/cuteafd"' in build
    # The only flock in the build is the build lock: no sparks.lock/gpu1.lock,
    # no hardware-lock helper, and therefore no wait behind or for a measurement.
    assert [line.strip() for line in build.splitlines() if 'flock' in line and not line.strip().startswith('release_need')] == [
        'flock -w "$release_build_lock_timeout" 9 ||'
    ]
    assert 'release_with_export_locks' not in build
    assert 'release-export-locks.sh' not in build
    assert build.index('if ((dry_run)); then') < build.index('exec 9>"$release_build_lock_dir/build.lock"')
    # Neither coordinator nor Spark export touches a GPU or runs nvidia-smi.
    for removed in (
        'release_select_idle_export_gpu',
        'release_watch_export_gpu',
        'cdi-gpus.sh',
        'CUTEAFD_CDI_ARGS',
        'nvidia-smi',
        'CUTEAFD_RELEASE_IDLE_GPU_LIMIT_MIB',
        'CUTEAFD_RELEASE_EXPORT_GPU_LIMIT_MIB',
        'CUTEAFD_RELEASE_SPARK_MIN_FREE_GIB',
        'allocatable_memory.py',
    ):
        assert removed not in build, f'stale GPU export symbol in build.sh: {removed}'
    assert build.count('-e "CUDA_VISIBLE_DEVICES="') == 2
    assert build.count('-e "NVIDIA_VISIBLE_DEVICES=void"') == 2
    assert 'timeout --foreground "$export_timeout" docker run --rm --name "$coordinator_export_container"' in build
    assert "trap 'exit 143' TERM" in build
    assert "trap 'release_export_cleanup; rm -rf \"$release_source_dir\"' EXIT" in build
    assert 'build_spark_release_leg export' in build
    assert 'timeout --foreground "$export_timeout" ssh "${release_ssh_opts[@]}" "$seed_host"' in build
    assert 'trap cleanup_spark_phase EXIT' in build
    assert "trap 'exit 129' HUP" in build
    # The queue-then-flock export helper had no caller left and is gone.
    assert not (ROOT / 'scripts/lib/release-export-locks.sh').exists()

