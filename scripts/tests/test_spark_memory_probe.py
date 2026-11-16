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
