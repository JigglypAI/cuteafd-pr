"""Adopt the published Hy4-flash snapshot and verify native SparkNest placement."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from functools import partial
import json
from pathlib import Path
import re

from inkling2 import handoff as nest
from inkling2 import publish as hub
from kimi_k3.uniform_k2.storage import atomic_json, file_identity

from .publish import FAMILY, TARGET, TARGET_REPO, pinned_files

MODEL = 'models--' + TARGET_REPO.replace('/', '--')
PIN_HOST = 'raptor'
ADOPTION_HOST = 'maxq'
TARGETS = frozenset(nest.TARGETS)
NATIVE = partial(nest.native, repo_id=TARGET_REPO)


def _publication(root, output):
    record = hub.load(root / 'publication-complete.json')
    identity = {k: record[k] for k in ('repo_id', 'revision', 'manifest_sha256')}
    if (record.get('schema') != TARGET.schema('complete') or record.get('status') != 'passed'
            or identity['repo_id'] != TARGET_REPO or record.get('output') != str(output)
            or not re.fullmatch('[0-9a-f]{40,64}', identity['revision'])
            or hub.manifest_digest(record['files']) != identity['manifest_sha256']):
        raise ValueError('matching verified Hy4 publication required before handoff')
    return record, identity


def _inventory(call, managed, files, revision):
    """Use native namespace metadata and read only the 40-byte main ref."""
    repo = nest.NAMESPACE / MODEL
    snapshot = repo / 'snapshots' / revision
    snapshots = call('ls', snapshot.parent)['entries']
    if [(row['name'], row['kind']) for row in snapshots] != [(revision, 'directory')]:
        raise ValueError('managed checkpoint must contain only the published snapshot')
    result = {}
    for directory in sorted({Path(name).parent for name in files}):
        for row in call('ls', snapshot / directory)['entries']:
            if row.get('kind') == 'directory':
                continue
            name = (directory / row['name']).as_posix()
            hosts = row['hosts']
            if (name not in files or row.get('kind') != 'regular' or row['size'] != files[name]['bytes']
                    or row.get('sealed') is not True or row.get('writing_on') is not None
                    or not hosts or len(set(hosts)) != len(hosts) or name in result):
                raise ValueError('managed file inventory differs: ' + name)
            result[name] = set(hosts)
    if result.keys() != files.keys():
        raise ValueError('managed file inventory is incomplete')
    refs = call('ls', repo / 'refs')['entries']
    if (len(refs) != 1 or refs[0].get('name') != 'main' or refs[0].get('kind') != 'regular'
            or refs[0].get('size') != len(revision) or refs[0].get('writing_on') is not None
            or not refs[0].get('hosts')):
        raise ValueError('managed main ref metadata differs')
    if (Path(managed) / MODEL / 'refs/main').read_text() != revision:
        raise ValueError('managed main ref revision differs')
    result['refs/main'] = set(refs[0]['hosts'])
    return result


def _check_layout(holders, files, pins, *, allow_adoption=False):
    dist = dict.fromkeys(nest.TARGETS, 0)
    for name, hosts in holders.items():
        placed = hosts - {ADOPTION_HOST} if allow_adoption else hosts
        if name in pins:
            if placed != {PIN_HOST}:
                raise ValueError('FP8 Engram pin differs: ' + name)
        else:
            if len(placed) != 1 or not placed <= TARGETS:
                raise ValueError('managed one-copy placement differs: ' + name)
            if name.endswith('.safetensors'):
                dist[next(iter(placed))] += files[name]['bytes']
    largest = max(files[n]['bytes'] for n in files.keys() - pins if n.endswith('.safetensors'))
    if not all(dist.values()) or max(dist.values()) - min(dist.values()) > largest:
        raise ValueError('managed non-pin shards fail eight-way balance')
    return dist


def _check_rule(row, selector, size):
    if (row['selector'] != selector or row['hosts'] != [PIN_HOST] or row.get('auto') is not True
            or row.get('summary', {}).get('files') != 1
            or row['summary'].get('bytes') != size
            or row['summary'].get('models') != [TARGET_REPO]):
        raise ValueError('Engram copy rule differs from the published file: ' + row['name'])


def _pin(call, root, state_path, state, files, pins, revision, holders):
    snapshot = nest.NAMESPACE / MODEL / 'snapshots' / revision
    rules = {row['name']: row for row in call('rule', 'ls')['rules']}
    desired = {}
    for filename in sorted(pins):
        name = f'hy4-flash-{filename.removesuffix(".safetensors")}-{revision}'
        selector = str(snapshot / filename)
        if name in rules:
            _check_rule(rules[name], selector, files[filename]['bytes'])
        else:
            call('rule', 'set', name, selector, '--hosts', PIN_HOST, '--auto')
        desired[name] = filename
    rules = {row['name']: row for row in call('rule', 'ls')['rules']}
    for name, filename in desired.items():
        _check_rule(rules[name], str(snapshot / filename), files[filename]['bytes'])
    atomic_json(root / 'engram-pins.json', {name: rules[name] for name in desired})
    # Only reconcile the named rules that still need their one required copy.
    # Keep the asynchronous job ids so a resumed handoff waits on existing work.
    def one_job(*args):
        result = call(*args)
        if args[0] != 'reconcile':
            return result
        jobs = result.get('jobs')
        if not isinstance(jobs, list) or len(jobs) != 1:
            raise ValueError('unexpected native Engram reconcile job list')
        return {'job': jobs[0]}

    for name, filename in desired.items():
        key = name + '-job'
        if key not in state and key + '-submission' not in state and PIN_HOST in holders[filename]:
            continue
        nest.run_job(root, state_path, state, one_job, key, ('reconcile', name),
                     job_description=f'replicate {snapshot / filename}')


def _project(plan, holders):
    projected = {name: set(hosts) for name, hosts in holders.items()}
    seen = set()
    for step in plan['steps']:
        host = step['host']
        for row in step['copies']:
            if row.get('skip'):
                continue
            name = 'refs/main' if row['path'].endswith('/refs/main') else row['name']
            if (name, host) in seen:
                raise ValueError('native placement has duplicate operations on one file/place')
            seen.add((name, host))
            if step['kind'] == 'replicate':
                projected[name].add(host)
            else:
                projected[name].remove(host)
    return projected


def _place(call, root, state_path, state, key, revision, files, pins, inventory,
           *, offgrid=False):
    goal = ('--offgrid', 'hf:' + TARGET_REPO) if offgrid else (
        '--place', 'hf:' + TARGET_REPO, '--factor', '1/8',
        *(v for host in nest.TARGET_ARGS for v in ('--host', host)))
    evict_hosts = {ADOPTION_HOST} if offgrid else TARGETS
    proposed = root / (key + '-proposed.json')
    raw = hub.load(proposed) if proposed.exists() else call('plan', *goal)
    nest.check_plan(raw, revision, files, repo_id=TARGET_REPO,
                    offgrid=offgrid, evict_hosts=evict_hosts)
    if type(raw.get('id')) is not int or raw['id'] <= 0:
        raise ValueError('native placement plan id missing')
    if not proposed.exists():
        atomic_json(proposed, raw)
    job_key = key + '-job'
    reviewed = root / (key + '.json')
    expected = deepcopy(raw)
    skipped = set()
    if not offgrid:
        for step in expected['steps']:
            for row in step['copies']:
                if row['name'] in pins:
                    skipped.add(row['name'])
                    row['skip'] = True
    operations = [(s, c) for s in expected['steps'] for c in s['copies'] if not c.get('skip')]
    def reviewed_plan(*args):
        if args[:2] == ('plan', 'apply'):
            shown = call('plan', 'show', raw['id'], '--files')
            nest.check_plan(shown, revision, files, repo_id=TARGET_REPO, offgrid=offgrid,
                            skipped=skipped, evict_hosts=evict_hosts)
            if shown != expected:
                raise ValueError('native plan changes differ from the reviewed operations')
            _check_layout(_project(shown, inventory()), files, pins, allow_adoption=not offgrid)
            atomic_json(reviewed, shown)
            if not operations:
                return None
        return call(*args)

    # The first native submission will do its own immediate show and layout
    # check. Recovery only waits if the exact named job is already recorded.
    if reviewed.exists() and hub.load(reviewed) != expected:
        raise ValueError('persisted reviewed plan differs from the proposed operations')
    if skipped and not reviewed.exists():
        snapshot = nest.NAMESPACE / MODEL / 'snapshots' / revision
        call('plan', 'skip', raw['id'], *(snapshot / n for n in sorted(skipped)))
    command = ('plan', 'apply', raw['id'])
    if operations:
        eviction_count = len(operations) if all(s['kind'] == 'evict' for s, _ in operations) else None
        nest.run_job(root, state_path, state, reviewed_plan, job_key, command,
                     job_description=f"apply plan {raw['id']}", eviction_files=eviction_count)
    elif not reviewed.exists():
        reviewed_plan(*command)  # Verify placement already satisfies the goal.


def handoff(root, output, stage_root, *, call=NATIVE, managed=nest.MANAGED):
    root, output, stage_root = Path(root).resolve(), Path(output).absolute(), Path(stage_root).absolute()
    published, identity = _publication(root, output)
    files, revision = published['files'], identity['revision']
    pins = pinned_files(files)
    if not pins:
        raise ValueError('published FP8 Engram shard inventory is empty')
    with (root / 'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / 'handoff-complete.json'
        if marker.exists():
            complete = hub.load(marker)
            if (complete.get('status') != 'passed' or complete.get('identity') != identity
                    or complete.get('stage_root') != str(stage_root)):
                raise ValueError('Hy4 handoff receipt differs')
            return complete
        state_path = root / 'handoff-state.json'
        binding = dict(identity=identity, stage_root=str(stage_root))
        state = hub.load(state_path) if state_path.exists() else binding.copy()
        if any(state.get(k) != v for k, v in binding.items()):
            raise ValueError('Hy4 handoff recovery receipt differs')
        if not state_path.exists():
            atomic_json(state_path, state)
        if 'import-job' in state or 'import-job-submission' in state:
            repo = stage_root / MODEL
        else:
            repo = nest.stage_snapshot(published, stage_root, repo_id=TARGET_REPO, family=FAMILY)
        nest.run_job(root, state_path, state, call, 'import-job',
                     ('hf', 'import', repo, '--offline', '--no-copy'),
                     job_description=f'hf import {repo} -> {nest.NAMESPACE}',
                     importing_files=len(files))

        def inventory():
            return _inventory(call, managed, files, revision)

        _pin(call, root, state_path, state, files, pins, revision, inventory())
        holders = inventory()
        if any(PIN_HOST not in holders[n] or not holders[n] <= {PIN_HOST, ADOPTION_HOST} for n in pins):
            raise ValueError('FP8 Engram files are not exclusively pinned plus the adoption copy')
        _place(call, root, state_path, state, 'placement', revision, files, pins, inventory)
        _check_layout(inventory(), files, pins, allow_adoption=True)
        # Native --place only manages fabric stores; retire maxq via the off-grid goal.
        _place(call, root, state_path, state, 'offgrid', revision, files, pins, inventory, offgrid=True)
        distribution = _check_layout(inventory(), files, pins)
        record = dict(schema='cuteafd-hy4_flash-handoff-complete-v1', status='passed', identity=identity,
            repo_id=TARGET_REPO, revision=revision, files=len(files), copied_bytes=0,
            stage_root=str(stage_root), targets=list(nest.TARGETS), distribution_bytes=distribution,
            pinned_host=PIN_HOST, pinned_files=sorted(pins), pinned_bytes=sum(files[n]['bytes'] for n in pins),
            managed_snapshot=str(Path(managed) / MODEL / 'snapshots' / revision),
            one_copy_per_file=True, time=datetime.now(timezone.utc).isoformat())
        atomic_json(marker, record)
        return record


def _engram_staging(root, directory, files):
    stage = Path(directory).absolute()
    if stage != root.parent / 'output' or stage.is_symlink():
        raise ValueError('standalone Engram staging must be this task’s output directory')
    if not stage.exists():
        return stage
    expected = pinned_files(files) | {name for name in files
                if re.fullmatch(r'engram-bf16-layer-(?:01|11|21|31)\.safetensors', name)}
    for path in stage.iterdir():
        hub.regular(path)
        if path.suffix in ('.json', '.jsonl', '.csv', '.log'):
            continue  # These are archived before deletion.
        if (path.name not in expected or file_identity(path) != files[path.name]['file_identity']
                or path.stat().st_dev != files[path.name]['device']):
            raise ValueError('Engram stage has unowned or unadopted payload: ' + path.name)
    return stage


def retire(root, output, *, intermediates=(), build_root=None, engram_stage=None):
    root = Path(root).resolve()
    published, identity = _publication(root, Path(output).absolute())
    complete = hub.load(root / 'handoff-complete.json')
    if (complete.get('schema') != 'cuteafd-hy4_flash-handoff-complete-v1'
            or complete.get('pinned_host') != PIN_HOST
            or set(complete.get('pinned_files', [])) != pinned_files(published['files'])):
        raise ValueError('verified Hy4 Engram placement is required for cleanup')
    with (root / 'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / 'handoff-cleanup-complete.json'
        if marker.exists():
            record = hub.load(marker)
            if record.get('status') != 'passed' or record.get('identity') != identity:
                raise ValueError('Hy4 cleanup receipt differs')
            return record
        stage = _engram_staging(root, engram_stage, published['files']) if engram_stage else None
        return nest.retire(root, intermediates=intermediates, build_root=build_root, repo_id=TARGET_REPO,
                           family=FAMILY, build_prefix='hy4-flash-k4-mc', verified_staging=stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-root', type=Path)
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--engram-stage', type=Path, help='retire the task’s adopted Engram-only export')
    parser.add_argument('--cleanup-request', type=Path,
                        help='JSON containing explicit intermediates and optional build_root')
    args = parser.parse_args()
    if args.cleanup:
        request = hub.load(args.cleanup_request) if args.cleanup_request else {}
        if set(request) - {'intermediates', 'build_root'}:
            parser.error('unsupported cleanup request field')
        result = retire(args.root, args.output, engram_stage=args.engram_stage, **request)
    else:
        stage = args.stage_root or args.root / 'handoff-stage'
        result = handoff(args.root, args.output, stage)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
