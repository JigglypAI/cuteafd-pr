#!/usr/bin/env python3
"""Read idle GB10 capacity; a busy host may reuse only this boot's idle sample.

Executed on the Spark by release_ssh with this source on stdin. The cache is
host-local and atomic; no checkpoint, CUDA allocation or container is touched.
"""
import json
import os
from pathlib import Path
import subprocess
import time


def sample(root=Path('/'), cache=None, gpu_processes=None, now=None):
    boot = (root / 'proc/sys/kernel/random/boot_id').read_text().strip()
    now = time.time() if now is None else now
    cache = cache or Path.home() / '.cache/cuteafd/spark-idle-memory.json'
    if gpu_processes is None:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                check=True, capture_output=True, text=True, timeout=10)
        gpu_processes = result.stdout.strip()
    if gpu_processes:
        try:
            measured = json.loads(cache.read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError('Spark is busy without an idle sample; stop workers first') from error
        if measured.get('boot_id') != boot or measured.get('bytes', 0) <= 0:
            raise RuntimeError('Spark idle sample is stale after reboot; stop workers first')
    else:
        info = dict(line.split(':', 1) for line in (root / 'proc/meminfo').read_text().splitlines())
        available = int(info['MemAvailable'].split()[0]) * 1024
        if available <= 0:
            raise RuntimeError('missing positive MemAvailable')
        measured = {'bytes': available, 'boot_id': boot, 'measured_unix': now,
                    'uptime_seconds': float((root / 'proc/uptime').read_text().split()[0]),
                    'source': 'idle /proc/meminfo MemAvailable; no GPU compute processes'}
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + '.' + str(os.getpid()))
        temporary.write_text(json.dumps(measured) + '\n')
        os.replace(temporary, cache)
    return {**measured, 'age_seconds': max(0, now - measured['measured_unix']),
            'cached': bool(gpu_processes)}


ATTIC_STATUS = 'run/spark-attic/status.json'
SLICE = 2 << 20


def carveout(root=Path('/'), now=None, groups=None, env=None):
    """Display-carveout bytes a validated spark-attic would lend a serving
    container (cuteafd #2546), or 0 with the reason. Stock interface only:
    the keyless status file (STATUS-FILE.md schema 1), the socket and key at
    their own paths or SPARK_ATTIC_SOCK / SPARK_ATTIC_KEY_FILE. The container
    runs as this uid with these groups plus the socket's own gid, so access
    means the documented non-root step (INSTALL.md section 6) was applied."""
    env = os.environ if env is None else env
    now = time.time() if now is None else now
    sock = Path(env.get('SPARK_ATTIC_SOCK') or '/run/spark-attic/attic.sock')
    key = Path(env.get('SPARK_ATTIC_KEY_FILE') or '/etc/spark-attic/key')
    status_path = root / sock.relative_to('/').parent / 'status.json' if sock.is_absolute() else root / ATTIC_STATUS
    try:
        status = json.loads(status_path.read_text()[:8192])
    except (OSError, ValueError):
        return 0, 'not installed'
    if not isinstance(status, dict) or status.get('name') != 'spark-attic':
        return 0, 'status file is not spark-attic'
    version = str(status.get('version', ''))
    parts = version.split('.')
    if len(parts) < 2 or parts[0] != '0' or parts[1] != '1':
        return 0, f'spark-attic {version!r} outside the supported 0.1.x range'
    every = max(int(status.get('every_s') or 10), 2)
    if status.get('state') != 'ready' or status.get('validated') is not True:
        return 0, f"spark-attic {status.get('state')}: {status.get('why', '')}"
    if abs(now - float(status.get('updated') or 0)) > 3 * every:
        return 0, 'spark-attic status is stale'
    if int(status.get('lent_mib') or 0) or int(status.get('draining_mib') or 0):
        return 0, 'spark-attic carveout is lent or draining'
    try:
        sock_stat = (root / sock.relative_to('/')).stat()
        key_stat = (root / key.relative_to('/')).stat()
    except (OSError, ValueError):
        return 0, 'spark-attic socket or key missing'
    uid = os.getuid()
    groups = set(os.getgroups()) | {os.getgid()} if groups is None else set(groups)
    groups.add(sock_stat.st_gid)  # the launcher adds the socket's gid to the container
    def can(st, bits_user, bits_group):
        return uid == 0 or (st.st_uid == uid and st.st_mode & bits_user) or (st.st_gid in groups and st.st_mode & bits_group)
    if not (can(sock_stat, 0o200, 0o020) and can(key_stat, 0o400, 0o040)):
        return 0, ('spark-attic is root-only: give atticd --socket-group <group> and chgrp <group> '
                   '/etc/spark-attic/key; chmod 0640 (spark-attic INSTALL.md section 6)')
    largest = int(status.get('largest_mib') or 0) << 20
    return largest - largest % SLICE, f'spark-attic {version} ready'


if __name__ == '__main__':
    measured = sample()
    try:
        bytes_, reason = carveout()
    except Exception as error:  # fail open: the carveout is never required
        bytes_, reason = 0, f'carveout probe failed: {error}'
    print(json.dumps({**measured, 'carveout_bytes': bytes_, 'carveout': reason}))
