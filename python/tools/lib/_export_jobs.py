"""Shared CPU worker budget for independent AOT variants and nested Ninja jobs."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time

from allocatable_memory import host_allocatable_memory


# Conservative per-worker CPU RSS budget (MiB) used to bound concurrent export
# workers so parallel Ninja jobs never exhaust host memory.
WORKER_RAM_MIB = 3072


def host_worker_cap() -> int:
    """Host-wide concurrent CPU export workers across builds and Ninja jobs."""
    value = os.environ.get('CUTEAFD_EXPORT_HOST_WORKERS')
    cores = os.cpu_count() or 1
    if value:
        cap = int(value)
        if cap < 1:
            raise ValueError('CUTEAFD_EXPORT_HOST_WORKERS must be positive')
        return cap
    try:
        ram_cap = max(1, (host_allocatable_memory() // (1024 * 1024)) // WORKER_RAM_MIB)
    except OSError:
        ram_cap = cores
    return max(1, min(cores, ram_cap))


def export_jobs() -> int:
    """Concurrent CPU export workers for one package build."""
    value = os.environ.get('CUTEAFD_EXPORT_JOBS')
    cores = os.cpu_count() or 1
    if value:
        jobs = int(value)
    else:
        default = cores // 2 if cores <= 24 else cores // 4
        jobs = max(1, min(default, host_worker_cap()))
    if jobs < 1:
        raise ValueError('CUTEAFD_EXPORT_JOBS must be positive')
    return jobs


@contextmanager
def _slot(root, count):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 7200
    while True:
        for slot in range(count):
            handle = (root / f'{slot}.lock').open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield
            finally:
                handle.close()
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('AOT variant waited 7200s for an export slot')
        time.sleep(0.2)


@contextmanager
def export_slot():
    local = os.environ.get('CUTEAFD_EXPORT_SLOT_DIR',
                           str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'export-slots'))
    host = os.environ.get('CUTEAFD_EXPORT_HOST_SLOT_DIR', local)
    with _slot(local, export_jobs()):
        if host == local:
            yield
        else:
            with _slot(host, host_worker_cap()):
                yield
