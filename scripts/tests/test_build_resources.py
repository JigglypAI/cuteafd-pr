"""Build-only resources count leases; serving hosts remain exclusive."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python/tools/lib'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resource = load('build_resource', ROOT / 'scripts/build/build-resource.py')
export = load('export_jobs', ROOT / 'python/tools/lib/_export_jobs.py')


def test_seed_capacity_uses_cores_and_memory():
    assert resource.capacity(20, 124608) == 3
    assert resource.capacity(64, 65536) == 2
    assert resource.capacity(2, 4096) == 1


def test_grants_are_fsynced_before_atomic_publication(tmp_path, monkeypatch):
    path = tmp_path / 'granted'
    grant = {'host': 'kakapo', 'slots': 3, 'slot': 1}
    fsynced = []
    original_fsync, original_replace = os.fsync, os.replace

    def fsync(fd):
        assert not path.exists()
        original_fsync(fd)
        fsynced.append(fd)

    def replace(source, destination):
        assert fsynced
        assert source.parent == destination.parent == tmp_path
        assert json.loads(source.read_text()) == grant
        original_replace(source, destination)

    monkeypatch.setattr(resource.os, 'fsync', fsync)
    monkeypatch.setattr(resource.os, 'replace', replace)
    resource.publish_grant(path, grant)
    assert json.loads(path.read_text()) == grant
    assert list(tmp_path.iterdir()) == [path]


def test_grant_readers_never_see_partial_json(tmp_path):
    path = tmp_path / 'granted'
    stop, first_read = Event(), Event()
    payload = 'x' * 65536

    def read_grants():
        reads = 0
        while not stop.is_set():
            try:
                grant = json.loads(path.read_text())
            except FileNotFoundError:
                continue
            assert grant['payload'] == payload
            assert 0 <= grant['generation'] < 100
            reads += 1
            first_read.set()
        return reads

    with ThreadPoolExecutor(max_workers=1) as executor:
        reader = executor.submit(read_grants)
        try:
            for generation in range(100):
                resource.publish_grant(path, {'generation': generation, 'payload': payload})
                if generation == 0:
                    assert first_read.wait(timeout=5)
        finally:
            stop.set()
        assert reader.result(timeout=5) > 0
    assert json.loads(path.read_text())['generation'] == 99
    assert list(tmp_path.iterdir()) == [path]


def test_counting_slots_preserve_legacy_exclusion_and_release(tmp_path):
    leases = [resource.acquire(tmp_path, 'kakapo', 3) for _ in range(3)]
    try:
        assert all(leases)
        assert resource.acquire(tmp_path, 'kakapo', 3) is None
        assert resource.try_lock(tmp_path / 'kakapo.lock') is None
        for handle in leases.pop():
            handle.close()
        extra = resource.acquire(tmp_path, 'kakapo', 3)
        assert extra
        leases.append(extra)
    finally:
        for lease in leases:
            for handle in lease:
                handle.close()
    exclusive = resource.try_lock(tmp_path / 'kakapo.lock')
    assert exclusive
    try:
        assert resource.acquire(tmp_path, 'kakapo', 3) is None
    finally:
        exclusive.close()


def test_busy_kakapo_falls_back_only_under_serving_host_lock(tmp_path):
    held = resource.acquire(tmp_path, 'kakapo', 1)
    rhea = resource.try_lock(tmp_path / 'rhea.lock')
    lease = tmp_path / 'lease'
    lease.mkdir()
    os.mkfifo(lease / 'release')
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'),
                               '--slots', '1', '--fallback', 'rhea,moa', '--wait', '0',
                               '--lease-dir', str(lease)],
                              env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        deadline = time.monotonic() + 5
        while not (lease / 'granted').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert json.loads((lease / 'granted').read_text())['host'] == 'moa'
        assert resource.try_lock(tmp_path / 'moa.lock') is None
        with (lease / 'release').open('w') as handle:
            handle.write('x')
        assert process.wait(timeout=5) == 0
        released = resource.try_lock(tmp_path / 'moa.lock')
        assert released
        released.close()
    finally:
        process.terminate()
        process.wait(timeout=5)
        rhea.close()
        for handle in held:
            handle.close()


@pytest.mark.parametrize('cores,expected', [(64, 16), (20, 10), (4, 2)])
def test_export_budget(cores, expected, monkeypatch):
    monkeypatch.setattr(os, 'cpu_count', lambda: cores)
    monkeypatch.delenv('CUTEAFD_EXPORT_JOBS', raising=False)
    monkeypatch.delenv('CUTEAFD_EXPORT_HOST_WORKERS', raising=False)
    monkeypatch.delenv('CARGO_BUILD_JOBS', raising=False)
    assert export.export_jobs() == expected
    monkeypatch.setenv('CARGO_BUILD_JOBS', '7')
    assert export.export_jobs() == expected
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '3')
    assert export.export_jobs() == 3
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '0')
    with pytest.raises(ValueError):
        export.export_jobs()


def test_export_slot_released_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('CUTEAFD_EXPORT_SLOT_DIR', str(tmp_path))
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '1')
    with pytest.raises(RuntimeError):
        with export.export_slot():
            assert resource.try_lock(tmp_path / '0.lock') is None
            raise RuntimeError('compiler failure')
    with export.export_slot():
        assert resource.try_lock(tmp_path / '0.lock') is None


def test_build_slot_default_admits_five_and_keeps_nice(tmp_path):
    held = []
    try:
        for name in ['build.lock', 'build.lock.1', 'build.lock.2', 'build.lock.3']:
            held.append(resource.try_lock(tmp_path / name))
        result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'bash', '-c',
                                 'ps -o ni= -p $$'], capture_output=True, text=True, timeout=5,
                                env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path), 'CUTEAFD_BUILD_SLOT_WAIT': '0'})
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == '19'
        assert 'build slot 4:' in result.stderr
    finally:
        for handle in held:
            handle.close()


def test_failed_probe_is_unavailable(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(255, 'ssh')
    monkeypatch.setattr(resource.subprocess, 'run', fail)
    assert resource.probe('kakapo') is None


def test_owner_death_before_grant_does_not_acquire(tmp_path):
    held = resource.acquire(tmp_path, 'kakapo', 1)
    lease = tmp_path / 'lease'
    lease.mkdir()
    os.mkfifo(lease / 'release')
    owner = subprocess.Popen(['sleep', '30'])
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--slots', '1',
                                '--owner', str(owner.pid), '--wait', '30', '--lease-dir', str(lease)],
                               env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        time.sleep(0.1)
        owner.terminate()
        owner.wait(timeout=5)
        assert process.wait(timeout=5) == 0
        assert not (lease / 'granted').exists()
    finally:
        process.terminate()
        process.wait(timeout=5)
        for handle in held:
            handle.close()


def test_lease_has_no_elapsed_unlock_deadline_and_stops_before_unlock():
    text = (ROOT / 'scripts/build/build-resource.py').read_text()
    granted = text[text.index("publish_grant(args.lease_dir / 'granted'"):]
    assert 'deadline' not in granted and '7260' not in granted
    assert granted.index('docker stop -t 10') < granted.index('handle.close()')


def test_host_worker_admission_is_shared_across_builds(tmp_path, monkeypatch):
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_SLOT_DIR', str(tmp_path / 'host'))
    monkeypatch.setenv('CUTEAFD_EXPORT_HOST_WORKERS', '1')
    monkeypatch.setenv('CUTEAFD_EXPORT_JOBS', '8')
    monkeypatch.setenv('CUTEAFD_EXPORT_SLOT_DIR', str(tmp_path / 'build-a'))
    with export.export_slot():
        assert resource.try_lock(tmp_path / 'host/0.lock') is None
        assert resource.try_lock(tmp_path / 'build-a/0.lock') is None
    with export.export_slot():
        assert resource.try_lock(tmp_path / 'host/0.lock') is None


@pytest.mark.parametrize('signum', [1, 3])
def test_seed_hup_quit_stops_container_before_unlock(tmp_path, signum):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    marker = tmp_path / 'stopped'
    ssh = bin_dir / 'ssh'
    ssh.write_text(f'#!/bin/sh\nsleep 0.3\ntouch {marker}\n')
    ssh.chmod(0o755)
    lease = tmp_path / 'lease'
    lease.mkdir()
    os.mkfifo(lease / 'release')
    (lease / 'active').touch()
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--host', 'rhea',
                                '--slots', '1', '--container', 'owned', '--lease-dir', str(lease)],
                               env={**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}',
                                    'CUTEAFD_LOCK_DIR': str(tmp_path)})
    try:
        deadline = time.monotonic() + 5
        while not (lease / 'granted').exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (lease / 'granted').exists()
        process.send_signal(signum)
        assert resource.try_lock(tmp_path / 'rhea.lock') is None
        assert process.wait(timeout=5) == 0
        assert marker.exists()
        handle = resource.try_lock(tmp_path / 'rhea.lock')
        assert handle
        handle.close()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


_POOL_CHILD = """
import os, sys
sys.path.insert(0, sys.argv[1])
import package_exl3_aot as package
pids = package.compile_variants([None] * 9, 2)
# Each worker set up its session once and served several variants.
assert len(set(pids)) <= 2 and len(pids) == 9, pids
print(len(set(pids)))
"""


def test_exl3_variant_workers_persist_across_variants(tmp_path):
    # A variant stand-in that reports its worker pid: the real pool, spawn
    # context and initializer, without CUDA.
    shim = tmp_path / 'shim.py'
    shim.write_text('import os, time\n'
                    'def variant(payload):\n'
                    '    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
                    '    assert os.getsid(0) == os.getpid()\n'
                    '    time.sleep(0.05)\n'
                    '    return os.getpid()\n')
    child = _POOL_CHILD.replace('import package_exl3_aot as package\n',
                                'import package_exl3_aot as package\nimport shim\n'
                                'package.compile_variant = shim.variant\n')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    result = subprocess.run([sys.executable, '-c', child, str(ROOT / 'python/tools/aot')], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) in (1, 2)
    text = (ROOT / 'python/tools/aot/package_exl3_aot.py').read_text()
    assert 'max_tasks_per_child' not in text
    assert 'initializer=init_variant_worker' in text


@pytest.mark.parametrize('crash', [False, True])
def test_failed_variant_pool_kills_owned_sessions(tmp_path, crash):
    shim = tmp_path / 'shim.py'
    shim.write_text('import os\n'
                    'def variant(payload):\n'
                    '    if payload:\n'
                    '        os._exit(17)\n'
                    '    raise RuntimeError("variant failed")\n')
    child = '''
import os, sys
sys.path.insert(0, sys.argv[1])
import package_exl3_aot as package
import shim
package.compile_variant = shim.variant
kills = []
original = package.os.killpg
def killpg(pid, sig):
    kills.append((pid, sig))
    return original(pid, sig)
package.os.killpg = killpg
try:
    package.compile_variants([sys.argv[2] == '1'] * 8, 2)
except Exception:
    pass
else:
    raise AssertionError('pool should fail')
assert kills, 'owned worker sessions were not killed'
print('RELEASED')
'''
    env = {**os.environ, 'PYTHONPATH': str(tmp_path),
           'CUTEAFD_EXPORT_HOST_SLOT_DIR': str(tmp_path / 'slots')}
    result = subprocess.run([sys.executable, '-c', child, str(ROOT / 'python/tools/aot'), str(int(crash))],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'RELEASED' in result.stdout


def test_spark_native_builds_brake_ninja_on_load():
    host_isa = ROOT / 'scripts/build/host-isa.sh'
    for machine, expected in (('aarch64', ['-l', '10']), ('x86_64', [])):
        out = subprocess.run(['bash', '-c', f'uname() {{ echo {machine}; }}; source "$1"; cuteafd_performance_cores() {{ echo 10; }}; cuteafd_native_build_args',
                              'host-isa', str(host_isa)], capture_output=True, text=True, check=True).stdout
        assert out.split() == expected
    for script, root in (('build-wip-artifacts.sh', 'build_dir'), ('build-release-artifacts.sh', 'build_root')):
        text = (ROOT / 'scripts/build' / script).read_text()
        assert f'cmake --build "${root}/native" -- "${{native_build_args[@]}}"' in text
        assert text.index('cuteafd_native_build_args') < text.index('cmake --build')


# LAN-only seeds: counting CPU leases (AOT exports run CPU-only with no SEED-export lock).

def _lease(tmp_path, name, *args, ssh_ok=True):
    """A real build-resource.py lease with stubbed ssh (probe: 20 cores, 121 GiB)."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    ssh = bin_dir / 'ssh'
    if not ssh.exists():
        ssh.write_text('#!/bin/sh\n[ -e "$SSH_DOWN_MARK" ] && case "$*" in *cassowary*) exit 255;; esac\n'
                       'case "$*" in *nproc*) printf "20\\n123904\\n";; esac\nexit 0\n')
        ssh.chmod(0o755)
    lease = tmp_path / name
    lease.mkdir()
    os.mkfifo(lease / 'release')
    owner = subprocess.Popen(['sleep', '60'])
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'CUTEAFD_LOCK_DIR': str(tmp_path / 'locks'),
           'SSH_DOWN_MARK': str(tmp_path / ('ok' if ssh_ok else 'ssh-down'))}
    if not ssh_ok:
        (tmp_path / 'ssh-down').touch()
    process = subprocess.Popen(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--owner', str(owner.pid),
                                '--wait', '20', '--lease-dir', str(lease), *args], env=env)
    return lease, process, owner


def _wait_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f'{path.name} never appeared'
        time.sleep(0.02)


def _end(lease, process, owner):
    try:
        with (lease / 'release').open('w') as handle:
            handle.write('x')
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        owner.terminate()
        owner.wait(timeout=5)


@pytest.fixture
def locks(tmp_path):
    (tmp_path / 'locks').mkdir()
    return tmp_path / 'locks'


def test_counting_lease_survives_empty_release_fifo_with_open_writer(tmp_path, locks):
    lease, process, owner = _lease(tmp_path, 'open-writer', '--host', 'kakapo')
    writer = os.open(lease / 'release', os.O_RDWR | os.O_NONBLOCK)
    try:
        _wait_file(lease / 'granted')
        assert process.poll() is None
        os.write(writer, b'x')
        assert process.wait(timeout=10) == 0
    finally:
        os.close(writer)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        owner.terminate()
        owner.wait(timeout=5)


def test_two_builds_on_one_seed_share_cpu_slots_without_export_lock(tmp_path, locks):
    first = _lease(tmp_path, 'a', '--host', 'kakapo')
    second = _lease(tmp_path, 'b', '--host', 'kakapo')
    try:
        granted_slots = set()
        for lease, _, _ in (first, second):
            _wait_file(lease / 'granted')
            grant = json.loads((lease / 'granted').read_text())
            slot = grant.pop('slot')
            assert grant == {'host': 'kakapo', 'slots': 3}
            assert 0 <= slot < 3
            granted_slots.add(slot)
            assert resource.try_lock(locks / f'kakapo.build.{slot}.lock') is None
        assert len(granted_slots) == 2
        assert resource.held_slots(locks, 'kakapo', 3) == 2
        assert not (locks / 'kakapo-export.lock').exists()
    finally:
        _end(*first)
        _end(*second)


def test_balance_picks_the_less_loaded_lan_seed(tmp_path, locks):
    held = resource.acquire(locks, 'kakapo', 3)
    try:
        lease, process, owner = _lease(tmp_path, 'a', '--host', 'kakapo', '--balance')
        try:
            _wait_file(lease / 'granted')
            assert json.loads((lease / 'granted').read_text())['host'] == 'cassowary'
        finally:
            _end(lease, process, owner)
    finally:
        for handle in held:
            handle.close()
    # A tie keeps the named seed.
    lease, process, owner = _lease(tmp_path, 'b', '--host', 'kakapo', '--balance')
    try:
        _wait_file(lease / 'granted')
        assert json.loads((lease / 'granted').read_text())['host'] == 'kakapo'
    finally:
        _end(lease, process, owner)


def test_two_balanced_builds_arriving_together_land_on_both_seeds(tmp_path, locks):
    leases = [_lease(tmp_path, name, '--host', 'kakapo', '--balance') for name in ('a', 'b')]
    try:
        for lease, _, _ in leases:
            _wait_file(lease / 'granted')
        hosts = sorted(json.loads((lease / 'granted').read_text())['host'] for lease, _, _ in leases)
        assert hosts == ['cassowary', 'kakapo']
    finally:
        for lease in leases:
            _end(*lease)


def test_balance_skips_an_unreachable_seed_and_falls_back_when_all_are_full(tmp_path, locks):
    lease, process, owner = _lease(tmp_path, 'a', '--host', 'cassowary', '--balance', ssh_ok=False)
    try:
        _wait_file(lease / 'granted')
        assert json.loads((lease / 'granted').read_text())['host'] == 'kakapo'
    finally:
        _end(lease, process, owner)
    (tmp_path / 'ssh-down').unlink()
    held = [resource.acquire(locks, 'kakapo', 1), resource.acquire(locks, 'cassowary', 1)]
    rhea = resource.try_lock(locks / 'rhea.lock')
    try:
        lease, process, owner = _lease(tmp_path, 'b', '--host', 'kakapo', '--balance', '--slots', '1',
                                       '--fallback', 'rhea,moa')
        try:
            _wait_file(lease / 'granted')
            assert json.loads((lease / 'granted').read_text())['host'] == 'moa'
        finally:
            _end(lease, process, owner)
    finally:
        rhea.close()
        for lease_handles in held:
            for handle in lease_handles:
                handle.close()


def test_pick_names_the_balanced_seed_without_holding_anything(tmp_path, locks, monkeypatch):
    held = resource.acquire(locks, 'kakapo', 3)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'ssh').write_text('#!/bin/sh\ncase "$*" in *nproc*) printf "20\\n123904\\n";; esac\n')
    (bin_dir / 'ssh').chmod(0o755)
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'CUTEAFD_LOCK_DIR': str(locks)}
    try:
        result = subprocess.run(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--pick', '--balance',
                                 '--host', 'kakapo', '--fallback', 'rhea,moa', '--lease-dir', '/nonexistent'],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0 and result.stdout.strip() == 'cassowary'
        assert resource.held_slots(locks, 'cassowary', 3) == 0
    finally:
        for handle in held:
            handle.close()
    # Every LAN seed unreachable: the first reachable fallback.
    (bin_dir / 'ssh').write_text('#!/bin/sh\ncase "$*" in *kakapo*|*cassowary*) exit 255;; *nproc*) printf "20\\n123904\\n";; esac\n')
    result = subprocess.run(['python3', str(ROOT / 'scripts/build/build-resource.py'), '--pick', '--balance',
                             '--host', 'kakapo', '--fallback', 'rhea,moa', '--lease-dir', '/nonexistent'],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'rhea'
