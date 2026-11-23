"""Boot-keyed idle Spark capacity, without GPU hardware or SSH."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('spark_memory_probe',
    Path(__file__).parents[1] / 'launch/spark-memory-probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def host(tmp_path):
    proc = tmp_path / 'proc'
    (proc / 'sys/kernel/random').mkdir(parents=True)
    (proc / 'sys/kernel/random/boot_id').write_text('boot-one\n')
    (proc / 'meminfo').write_text('MemTotal: 128000 kB\nMemAvailable: 120000 kB\nMemFree: 100 kB\n')
    (proc / 'uptime').write_text('300.5 0\n')
    return tmp_path, tmp_path / 'cache/idle.json'


def test_idle_refresh_uses_reclaimable_memavailable(tmp_path):
    root, cache = host(tmp_path)
    result = probe.sample(root, cache, gpu_processes='', now=1000)
    assert result['bytes'] == 120000 * 1024
    assert result['uptime_seconds'] == 300.5
    assert result['age_seconds'] == 0 and not result['cached']
    assert json.loads(cache.read_text())['boot_id'] == 'boot-one'
    (root / 'proc/meminfo').write_text('MemAvailable: 119000 kB\n')
    assert probe.sample(root, cache, gpu_processes='', now=1010)['bytes'] == 119000 * 1024


def test_busy_reuses_idle_not_busy_memavailable(tmp_path):
    root, cache = host(tmp_path)
    probe.sample(root, cache, gpu_processes='', now=1000)
    (root / 'proc/meminfo').write_text('MemAvailable: 1000 kB\n')
    result = probe.sample(root, cache, gpu_processes='1234', now=1025)
    assert result['bytes'] == 120000 * 1024
    assert result['cached'] and result['age_seconds'] == 25


def test_busy_missing_or_reboot_stale_fails_closed(tmp_path):
    root, cache = host(tmp_path)
    with pytest.raises(RuntimeError, match='stop workers first'):
        probe.sample(root, cache, gpu_processes='1234')
    probe.sample(root, cache, gpu_processes='', now=1000)
    (root / 'proc/sys/kernel/random/boot_id').write_text('boot-two\n')
    with pytest.raises(RuntimeError, match='stale after reboot'):
        probe.sample(root, cache, gpu_processes='1234', now=1010)


def test_missing_memavailable_never_falls_back(tmp_path):
    root, cache = host(tmp_path)
    (root / 'proc/meminfo').write_text('MemFree: 1000 kB\n')
    with pytest.raises(KeyError):
        probe.sample(root, cache, gpu_processes='')


def test_selected_host_probe_deduplicates_without_changing_rank_order(tmp_path):
    calls = tmp_path / 'calls'
    common = Path(__file__).parents[1] / 'lib/release-common.sh'
    result = probe.subprocess.run(['bash', '-c', r'''source "$1"
release_ssh() { while [[ "$1" == -o ]]; do shift 2; done; printf '%s\n' "$1" >> "$CALLS"; printf '{"bytes":123,"boot_id":"boot","age_seconds":0}\n'; }
release_spark_memory_samples rhea ostrich rhea
''', '_', str(common)], env={**probe.os.environ, 'CALLS': str(calls)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert calls.read_text().splitlines() == ['rhea', 'ostrich']
    assert [row['host'] for row in json.loads(result.stdout)] == ['rhea', 'ostrich', 'rhea']


def attic(tmp_path, *, state='ready', version='0.1.6', lent=0, mode=0o660, key_mode=0o640, updated=1000, gid=None):
    import os
    run = tmp_path / 'run/spark-attic'
    run.mkdir(parents=True, exist_ok=True)
    (run / 'status.json').write_text(json.dumps({
        'name': 'spark-attic', 'version': version, 'state': state, 'why': 'x', 'validated': state == 'ready',
        'size_mib': 2044, 'free_mib': 2044, 'largest_mib': 2044, 'lent_mib': lent, 'draining_mib': 0,
        'every_s': 10, 'updated': updated}))
    sock = run / 'attic.sock'
    sock.write_text('')
    sock.chmod(mode)
    key = tmp_path / 'etc/spark-attic/key'
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text('attic_0123\n')
    key.chmod(key_mode)
    return tmp_path


def test_carveout_credit_needs_ready_fresh_unlent_and_access(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(os, 'getuid', lambda: 4242)  # not the files' owner, not root
    root = attic(tmp_path)
    own_gid = (root / 'run/spark-attic/attic.sock').stat().st_gid
    # The launcher adds the socket's gid, so a group-readable key suffices.
    assert probe.carveout(root, now=1005, groups=[], env={}) == (2044 << 20, 'spark-attic 0.1.6 ready')
    assert probe.carveout(attic(tmp_path, key_mode=0o600), now=1005, groups=[], env={})[0] == 0
    _, reason = probe.carveout(attic(tmp_path, key_mode=0o600), now=1005, groups=[], env={})
    assert '--socket-group' in reason and 'INSTALL.md' in reason
    for case in [dict(state='refusing'), dict(version='0.2.0'), dict(lent=2), dict(updated=900)]:
        assert probe.carveout(attic(tmp_path, **case), now=1005, groups=[], env={})[0] == 0, case
    assert probe.carveout(tmp_path / 'absent', now=1005, env={}) == (0, 'not installed')
    del own_gid
