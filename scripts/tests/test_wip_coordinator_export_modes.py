"""WIP coordinator and Spark native exports run CPU-only with no GPU placement."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIP = (ROOT / 'wip.sh').read_text()


def _function(name):
    opening = name + ('() (' if name == 'build_expert' else '() {')
    closing = '\n)\n' if name == 'build_expert' else '\n}\n'
    start = WIP.index(opening)
    return WIP[start:WIP.index(closing, start) + 3]


def test_wip_has_no_gpu_export_placement_or_remote_routing():
    for removed in (
        'export-placement.py',
        'export-remote.sh',
        'wip_capped_export',
        'wip_remote_export',
        'wip_seed_export_turn',
        'wip_serving_export_priority',
        'wip_coordinator_export_modes',
        'CUTEAFD_EXPORT_HOST',
        'CUTEAFD_WIP_EXPORT_LOCKS',
    ):
        assert removed not in WIP, f'stale GPU export symbol in wip.sh: {removed}'
    assert not (ROOT / 'scripts/build/export-placement.py').exists()
    assert not (ROOT / 'scripts/build/export-remote.sh').exists()


def test_coordinator_native_export_runs_cpu_only():
    function = _function('build_coordinator')
    assert '-e CUDA_VISIBLE_DEVICES=' in function
    assert '-e NVIDIA_VISIBLE_DEVICES=void' in function
    assert 'wip_hwlock' not in function


def test_spark_native_export_runs_cpu_only():
    function = _function('build_expert')
    assert "-e 'CUDA_VISIBLE_DEVICES='" in function
    assert "-e 'NVIDIA_VISIBLE_DEVICES=void'" in function
    assert 'export-request' not in function
    assert 'export-granted' not in function


def test_native_cache_hit_restores_without_rebuilding(tmp_path):
    function = _function('build_coordinator')
    script = """set -euo pipefail
sync_local_source() { :; }
docker() { echo "docker $*"; }
wip_exec() { "$@"; }
native_store_sync() { :; }
native_store_has() { return 0; }
release_die() { exit 9; }
slot=test coordinator_container=fixture COORDINATOR_DOCKER_DEV=fixture
bf16_families= audio_aot=OFF
""" + function + '\nbuild_coordinator native /frozen/source\n'
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'native cache hit' in result.stdout
    assert 'CUTEAFD_WIP_NATIVE_REQUIRE_HIT=1' in result.stdout
    assert '-e CUDA_VISIBLE_DEVICES=' in result.stdout
    assert '-e NVIDIA_VISIBLE_DEVICES=void' in result.stdout


def test_build_wip_artifacts_forces_cpu_only_and_nice():
    text = (ROOT / 'scripts/build/build-wip-artifacts.sh').read_text()
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert 'renice -n 19' in text
    assert 'CUTEAFD_EXPORT_HOST_CLASS' not in text
    assert 'CUTEAFD_WIP_EXPORT_LOCKS' not in text
