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


if __name__ == '__main__':
    print(json.dumps(sample()))
