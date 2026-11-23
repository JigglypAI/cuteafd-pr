"""CPU-only release builds and legacy export lock helper."""
import fcntl
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_export(tmp_path, command='true', host=''):
    lock_dir = tmp_path / '.cache/cuteafd'
    lock_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / 'docker'
    docker.write_text('''#!/usr/bin/env bash
set -eu
# Cleanup must run before either lock is released.
flock -n "$HOME/.cache/cuteafd/sparks.lock" true && exit 80
flock -n "$HOME/.cache/cuteafd/gpu1.lock" true && exit 81
printf '%s\\n' "$*" >>"$HOME/cleanup.log"
if [[ ${MOCK_ALREADY_REMOVED:-0} == 1 ]]; then
  echo "Error response from daemon: No such container: housekeeping-export" >&2
  exit 1
fi
''')
    docker.chmod(0o755)
    ssh = bin_dir / 'ssh'
    ssh.write_text('#!/usr/bin/env bash\n[[ $1 == ostrich ]] || exit 90\nshift\nexec "$@"\n')
    ssh.chmod(0o755)
    script = f'''source "{ROOT}/scripts/lib/release-common.sh"
source "{ROOT}/scripts/lib/release-export-locks.sh"
release_ssh_opts=()
release_with_export_locks '{host}' housekeeping-export {command}
'''
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=10,
                          env={**os.environ, 'HOME': str(tmp_path), 'PATH': f'{bin_dir}:' + os.environ['PATH'],
                               'CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS': '1'})


@pytest.mark.parametrize('host', ['', 'ostrich'])
@pytest.mark.parametrize('command,status', [('true', 0), ('false', 1), ('timeout 0.1 sleep 5', 124)])
def test_export_always_cleans_container_under_both_locks(tmp_path, command, status, host):
    result = run_export(tmp_path, command, host)
    assert result.returncode == status, result.stderr
    assert (tmp_path / 'cleanup.log').read_text().strip() == 'rm -f housekeeping-export'
    for name in ('sparks', 'gpu1'):
        with (tmp_path / f'.cache/cuteafd/{name}.lock').open('w') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize('held', ['sparks', 'gpu1'])
def test_lock_contention_times_out_without_export_or_cleanup(tmp_path, held):
    lock_dir = tmp_path / '.cache/cuteafd'
    lock_dir.mkdir(parents=True)
    with (lock_dir / f'{held}.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        result = run_export(tmp_path)
    assert result.returncode != 0
    # The fair queue (scripts/hw/hwlock.py) waits for every lock at once.
    assert 'timed out waiting for a fair turn on sparks.lock and gpu1.lock' in result.stderr
    assert not (tmp_path / 'cleanup.log').exists()


def test_cleanup_accepts_auto_removed_container(tmp_path, monkeypatch):
    monkeypatch.setenv('MOCK_ALREADY_REMOVED', '1')
    result = run_export(tmp_path)
    assert result.returncode == 0, result.stderr


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
    helper = (ROOT / 'scripts/lib/release-export-locks.sh').read_text()
    assert helper.index('flock -w "$wait_seconds" 9') < helper.index('flock -w "$wait_seconds" 8')


def test_term_stops_the_export_before_releasing_locks(tmp_path):
    result = run_export(tmp_path, "bash -c 'kill -TERM \"$PPID\"'")
    assert result.returncode == 143, result.stderr
    assert (tmp_path / 'cleanup.log').read_text().strip() == 'rm -f housekeeping-export'
