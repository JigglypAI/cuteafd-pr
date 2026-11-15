"""Adopt a published original-MTP delta without copying or rereading weights.

Only newly published files enter a local HF staging cache. Existing managed
payloads receive metadata aliases; the managed main ref changes only after the
whole new snapshot passes native inventory verification. New blobs retain their
import placement. Any later placement must select only the added blobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

from kimi_k3.uniform_k2.storage import atomic_json, file_identity, fsync_directory

MAX_METADATA_BYTES = 64 << 20


def load(path):
    path = Path(path)
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError('handoff receipt exceeds metadata bound')
    return json.loads(path.read_bytes())


def relative(name):
    if not isinstance(name, str):
        raise ValueError('unsafe published relative path')
    p = PurePosixPath(name)
    if (p.is_absolute() or not p.parts or '\\' in name
            or any(x in ('', '.', '..') for x in name.split('/'))):
        raise ValueError('unsafe published relative path')
    return p


def below(path, root):
    return path == root or root in path.parents


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('task file must be regular without symlink parents')
    return path


def native(nest, *args, timeout=120):
    result = subprocess.run([str(nest), '--json', *map(str, args)], check=True,
                            capture_output=True, text=True, timeout=timeout)
    raw = result.stdout
    # The importer emits this exact harmless notice ahead of its JSON result.
    if args[:2] == ('hf', 'import'):
        model = Path(args[2]).name
        if model.startswith('models--'):
            notice = model[len('models--'):].replace('--', '/', 1) + ': mirrored\n'
            if raw.startswith(notice):
                raw = raw[len(notice):]
    return json.loads(raw)


def publication(path):
    row = load(path)
    identity = row.get('identity', {})
    if (row.get('schema') != 'cuteafd-original-mtp-publication-complete-v1'
            or row.get('status') != 'passed' or identity.get('schema') != 'cuteafd-original-mtp-repair-v1'
            or identity.get('repo') != row.get('repo_id')
            or identity.get('parent_revision') != row.get('parent_revision')
            or not re.fullmatch(r'wrldsuksgo2mars/[A-Za-z0-9._-]+', row.get('repo_id', ''))
            or any(not re.fullmatch('[0-9a-f]{40}', row.get(k, '')) for k in ('revision', 'parent_revision'))):
        raise ValueError('matching passed original-MTP publication required')
    files, changed, parent = row['files'], row['changed_files'], row['parent_files']
    if set(files) != set(parent) | set(changed):
        raise ValueError('publication inventory differs from parent plus delta')
    for name, info in files.items():
        relative(name)
        if (type(info.get('bytes')) is not int or info['bytes'] < 0
                or info.get('storage') not in ('lfs', 'git')
                or not re.fullmatch('[0-9a-f]{64}' if info['storage'] == 'lfs' else '[0-9a-f]{40}', info.get('oid', ''))):
            raise ValueError('invalid published size or OID')
        if name in changed:
            expected = changed[name]
            if info['bytes'] != expected['bytes'] or info['oid'] != expected['sha256' if info['storage'] == 'lfs' else 'git_blob']:
                raise ValueError('published delta differs from writer identity')
        elif info != parent[name]:
            if name != '.gitattributes' or info != row.get('normalized_attributes'):
                raise ValueError('inherited publication identity changed')
    return row


def _local_sources(published, receipt):
    sources = {}
    for name, row in published['changed_files'].items():
        path = regular(row['path'])
        if file_identity(path) != row['file_identity'] or path.stat().st_dev != row['device']:
            raise ValueError('published delta source changed')
        sources[name] = path
    if published.get('normalized_attributes') is not None:
        path = regular(Path(receipt).parent / 'published.gitattributes')
        if path.stat().st_size > 1 << 20:
            raise ValueError('normalized attributes exceed metadata bound')
        raw = path.read_bytes()
        expected = published['files']['.gitattributes']
        if len(raw) != expected['bytes'] or hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() != expected['oid']:
            raise ValueError('normalized attributes differ from publication')
        sources['.gitattributes'] = path
    return sources


def _parent_sources(published, hub, local_names):
    snapshot = hub / ('models--' + published['repo_id'].replace('/', '--')) / 'snapshots' / published['parent_revision']
    inherited = {}
    for name in published['files'].keys() - local_names:
        path = snapshot / name
        resolved = path.resolve(strict=True)
        md = path.stat()
        if (not below(resolved, hub) or not stat.S_ISREG(md.st_mode)
                or md.st_size != published['files'][name]['bytes']):
            raise ValueError('managed parent file missing or size differs: ' + name)
        inherited[name] = dict(path=str(path), file_identity=file_identity(path), device=md.st_dev)
    return inherited


def _stage(published, root, sources):
    stage = root / 'hf-stage'
    stamp = root / 'stage-owner.json'
    expected = dict(repo_id=published['repo_id'], revision=published['revision'], files=sorted(sources))
    if stamp.exists():
        if load(stamp) != expected:
            raise ValueError('HF stage ownership differs')
    elif stage.exists():
        raise ValueError('unowned HF stage exists')
    else:
        atomic_json(stamp, expected)
    stage.mkdir(exist_ok=True)
    device = stage.stat().st_dev
    for source in sources.values():
        if source.stat().st_dev != device:
            raise ValueError('delta staging crosses filesystems; copying is forbidden')
    repo = stage / ('models--' + published['repo_id'].replace('/', '--'))
    if (repo / 'refs').exists() or (repo / 'refs').is_symlink():
        raise ValueError('delta staging must never contain refs/main')
    blobs = repo / 'blobs'
    snapshot = repo / 'snapshots' / published['revision']
    if any(p.is_symlink() for p in (blobs, *blobs.parents, snapshot, *snapshot.parents)):
        raise ValueError('HF staging directories cannot traverse symlinks')
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)
    if ({p.name for p in repo.iterdir()} != {'blobs', 'snapshots'}
            or {p.name for p in (repo / 'snapshots').iterdir()} != {published['revision']}
            or {p.name for p in blobs.iterdir()} - {published['files'][name]['oid'] for name in sources}
            or any(p.is_symlink() or not p.is_file() for p in blobs.iterdir())):
        raise ValueError('unexpected file in owned delta HF cache')
    actual = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file() or p.is_symlink()}
    if not actual <= sources.keys():
        raise ValueError('unexpected delta staging file')
    for name, source in sources.items():
        blob = blobs / published['files'][name]['oid']
        if blob.exists():
            if not blob.samefile(source):
                raise ValueError('staging blob is not the published hardlink')
        else:
            os.link(source, blob)  # No copy, reflink or payload hash fallback.
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        link = os.path.relpath(blob, target.parent)
        if target.is_symlink():
            if os.readlink(target) != link:
                raise ValueError('staging snapshot link differs')
        elif target.exists():
            raise ValueError('unexpected regular staging snapshot entry')
        else:
            target.symlink_to(link)
    return repo


def _successful_import(result, count):
    job, progress = result.get('job', {}), result.get('import', {})
    counts = [progress.get(key, 0) for key in ('files', 'adopted', 'skipped')]
    if (job.get('finished') is not True or job.get('error') or job.get('cancelled')
            or job.get('pending', 0) or any(': failed' in str(n) for n in job.get('notes', []))
            or progress.get('finished') is not True
            or progress.get('errors') != [] or progress.get('cancelled')
            or progress.get('copied_bytes') != 0
            or any(type(n) is not int or n < 0 for n in counts) or sum(counts) != count):
        raise ValueError('native no-copy delta import failed or is incomplete; diagnose saved job')
    for host in job.get('hosts', {}).values():
        if (host.get('finished') is not True or host.get('failed') or host.get('cancelled')
                or host.get('done_files', 0) != host.get('total_files', 0)):
            raise ValueError('native delta import host counters failed')


def _import(repo, count, root, call):
    job_path, done = root / 'import-job.json', root / 'import-complete.json'
    if done.exists():
        result = load(done)
        _successful_import(result, count)
        return result
    if job_path.exists():
        job = load(job_path)['job']
    else:
        started = call('hf', 'import', repo, '--offline', '--no-copy', timeout=120)
        job = started['job']
        if type(job) is not int or job < 0:
            raise ValueError('native import returned invalid job ID')
        atomic_json(job_path, dict(job=job))
    result = call('jobs', str(job), '--wait', timeout=7200)
    atomic_json(root / 'import-result.json', result)
    _successful_import(result, count)
    atomic_json(done, result)
    return result


def _alias(published, hub, inherited):
    repo = hub / ('models--' + published['repo_id'].replace('/', '--'))
    snapshot = repo / 'snapshots' / published['revision']
    for name, source_row in sorted(inherited.items()):
        source = Path(source_row['path'])
        if (file_identity(source) != source_row['file_identity']
                or source.stat().st_dev != source_row['device']):
            raise ValueError('managed inherited source changed: ' + name)
        blob = repo / 'blobs' / published['files'][name]['oid']
        blob.parent.mkdir(parents=True, exist_ok=True)
        if blob.exists():
            if not blob.samefile(source):
                raise ValueError('inherited managed OID has a different inode: ' + name)
        else:
            os.link(source.resolve(strict=True), blob)
        if not blob.samefile(source):
            raise ValueError('native metadata hardlink identity differs')
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        link = os.path.relpath(blob, target.parent)
        if target.is_symlink():
            if os.readlink(target) != link:
                raise ValueError('existing inherited snapshot alias differs')
        elif target.exists():
            raise ValueError('existing inherited snapshot entry differs')
        else:
            target.symlink_to(link)
    return snapshot


def _verify(published, hub, namespace, inherited, call):
    repo_name = 'models--' + published['repo_id'].replace('/', '--')
    snapshot = hub / repo_name / 'snapshots' / published['revision']
    wanted = published['files']
    actual = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file() or p.is_symlink()}
    if actual != wanted.keys():
        raise ValueError('complete managed snapshot inventory differs')
    seen = set()
    for directory in sorted({str(Path(name).parent) for name in wanted}):
        selection = namespace / repo_name / 'snapshots' / published['revision'] / directory
        for row in call('ls', selection, timeout=120)['entries']:
            if row.get('kind') == 'directory':
                continue
            name = (Path(directory) / row['name']).as_posix()
            relative(name)
            if (name not in wanted or name in seen or row.get('kind') != 'regular'
                    or row.get('size') != wanted[name]['bytes'] or row.get('sealed') is not True
                    or row.get('writing_on') is not None or not row.get('hosts')):
                raise ValueError('native sealed file inventory differs: ' + name)
            path = snapshot / name
            if (path.stat().st_size != wanted[name]['bytes']
                    or not below(path.resolve(strict=True), hub)
                    or path.resolve(strict=True).name != wanted[name]['oid']):
                raise ValueError('managed file OID alias or size differs: ' + name)
            if name in inherited and not path.samefile(inherited[name]['path']):
                raise ValueError('inherited snapshot does not retain the same native inode')
            seen.add(name)
    if seen != wanted.keys():
        raise ValueError('native snapshot inventory is incomplete')
    return snapshot


def _read_ref(main):
    if main.is_symlink() or not main.is_file() or main.stat().st_size > 128:
        raise ValueError('managed main ref is missing or not a bounded regular file')
    return main.read_text().strip()


def handoff(publication_path, root, *, hub=Path('/mnt/sparknest/hf-home/hub'),
            namespace=Path('/hf-home/hub'), nest='nest', call=None):
    publication_path = regular(publication_path)
    published = publication(publication_path)
    root, hub, namespace = Path(root).absolute(), Path(hub).absolute(), Path(namespace)
    delta = Path(published['delta']).absolute()
    if (any(p.is_symlink() for p in (root, *root.parents))
            or any(below(a, b) or below(b, a) for a, b in ((root, hub), (root, delta), (root, publication_path.parent)))):
        raise ValueError('handoff root must be disjoint from managed hub, delta and publication receipts')
    root.mkdir(parents=True, exist_ok=True)
    call = call or (lambda *args, **kwargs: native(nest, *args, **kwargs))
    plan = dict(schema='cuteafd-original-mtp-handoff-v1', publication_sha256=hashlib.sha256(publication_path.read_bytes()).hexdigest(),
                repo_id=published['repo_id'], revision=published['revision'], hub=str(hub), namespace=str(namespace))
    with (root / 'handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path, completed = root / 'handoff-plan.json', root / 'handoff-complete.json'
        if plan_path.exists() and load(plan_path) != plan:
            raise ValueError('handoff recovery identity differs')
        if not plan_path.exists():
            atomic_json(plan_path, plan)
        model = hub / ('models--' + published['repo_id'].replace('/', '--'))
        main = model / 'refs/main'
        ready = root / 'snapshot-ready.json'
        current = _read_ref(main)
        if current not in (published['parent_revision'], published['revision']):
            raise ValueError('managed main ref moved outside this repair')
        if current == published['revision'] and not ready.exists():
            raise ValueError('managed main advanced without this handoff verification')
        # A finished handoff can be checked after its task-only delta was retired.
        if completed.exists():
            saved = load(completed)
            inherited = load(root / 'inherited-sources.json')
            if saved.get('identity') != plan or saved.get('status') != 'passed' or current != published['revision']:
                raise ValueError('completed handoff receipt differs')
            _verify(published, hub, namespace, inherited, call)
            return saved
        sources = _local_sources(published, publication_path)
        inherited = _parent_sources(published, hub, sources.keys())
        inherited_path = root / 'inherited-sources.json'
        if inherited_path.exists() and load(inherited_path) != inherited:
            raise ValueError('managed parent recovery identities differ')
        if not inherited_path.exists():
            atomic_json(inherited_path, inherited)
        stage_repo = _stage(published, root, sources)
        _local_sources(published, publication_path)
        _import(stage_repo, len(sources), root, call)
        _local_sources(published, publication_path)
        _alias(published, hub, inherited)
        snapshot = _verify(published, hub, namespace, inherited, call)
        verification = dict(identity=plan, status='passed', snapshot=str(snapshot), files=len(published['files']))
        if ready.exists() and load(ready) != verification:
            raise ValueError('snapshot verification receipt differs')
        atomic_json(ready, verification)
        # Native import cannot publish main early: the staged cache has no refs.
        current = _read_ref(main)
        if current not in (published['parent_revision'], published['revision']):
            raise ValueError('managed main changed before finalization')
        if current != published['revision']:
            temporary = main.with_name('.original-mtp-' + published['revision'])
            if temporary.is_symlink() or (temporary.exists() and (not temporary.is_file()
                    or temporary.stat().st_size > 40 or not published['revision'].startswith(temporary.read_text()))):
                raise ValueError('pending managed ref differs from this verified handoff')
            with temporary.open('w') as out:
                out.write(published['revision']); out.flush(); os.fsync(out.fileno())
            os.replace(temporary, main)
            fsync_directory(main.parent)
        if _read_ref(main) != published['revision']:
            raise ValueError('managed main ref finalization differs')
        result = dict(schema='cuteafd-original-mtp-handoff-complete-v1', status='passed', identity=plan,
            repo_id=published['repo_id'], revision=published['revision'], snapshot=str(snapshot),
            files=len(published['files']), imported_delta_files=len(sources), inherited_files=len(inherited),
            copied_bytes=0, fetched_bytes=0, placement='new blobs retain native import placement; inherited placement unchanged',
            verification='native sealed sizes/OID aliases and inherited same-inode links; no weight payload reads or hashes',
            time=datetime.now(timezone.utc).isoformat())
        atomic_json(completed, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--hub', type=Path, default=Path('/mnt/sparknest/hf-home/hub'))
    parser.add_argument('--namespace', type=Path, default=Path('/hf-home/hub'))
    args = parser.parse_args()
    print(json.dumps(handoff(args.publication, args.root, hub=args.hub, namespace=args.namespace)), flush=True)


if __name__ == '__main__':
    main()
