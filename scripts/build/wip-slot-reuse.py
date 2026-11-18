#!/usr/bin/env python3
"""Find and adopt a sealed WIP slot built from the same inputs (wip.sh build key).

Runs on the host that holds the slot (raptor for coordinator slots, the seed
Spark for expert slots; wip.sh pipes it over ssh). A candidate qualifies only
when its META records exactly the requested build key and its seal verifies:
FINGERPRINT == sha256(META.json), the source and artifact manifests match the
hashes META records, and every artifact matches its manifest line. Adoption
hardlinks the candidate into this instance's root and rewrites only the slot
identity in META (instance, root, slot), then reseals it; the source and
artifact manifests, and so every built byte, are the candidate's.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROLES = ('coordinator', 'spark-expert')
AT_FDCWD = -100
RENAME_EXCHANGE = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(slot: Path, key: str) -> dict:
    """The candidate's META if its seal and every artifact verify, else raise."""
    meta_bytes = (slot / 'META.json').read_bytes()
    if hashlib.sha256(meta_bytes).hexdigest() != (slot / 'FINGERPRINT').read_text().strip():
        raise ValueError('META seal mismatch')
    meta = json.loads(meta_bytes)
    if meta.get('build_key') != key:
        raise ValueError('build key differs')
    artifacts = slot / 'workspace/.cuteafd-wip'
    for relative, field in (('SOURCE_SHA256SUMS', 'source_manifest_sha256'),
                            ('workspace/.cuteafd-wip/ARTIFACT_SHA256SUMS', 'artifact_manifest_sha256')):
        if sha256(slot / relative) != meta[field]:
            raise ValueError(relative + ' seal mismatch')
    for line in (artifacts / 'ARTIFACT_SHA256SUMS').read_text().splitlines():
        expected, name = line.split(None, 1)
        if sha256(artifacts / name.lstrip('*')) != expected:
            raise ValueError('artifact mismatch: ' + name)
    # Packages are not in ARTIFACT_SHA256SUMS; their own manifests were verified
    # at build time, and the source manifest covers the workspace tree. Check the
    # source manifest against the files on disk as the final gate.
    checked = subprocess.run(['sha256sum', '--quiet', '--strict', '-c', str(slot / 'SOURCE_SHA256SUMS')],
                             cwd=slot / 'workspace', capture_output=True, text=True)
    if checked.returncode:
        raise ValueError('source tree mismatch: ' + (checked.stdout or checked.stderr).strip()[-200:])
    return meta


def find(args) -> int:
    builds = Path.home() / '.cache/cuteafd/builds'
    candidates = []
    # This instance's own slot first: a repeat build of the same key is a no-op.
    own = Path(args.exclude_root) / 'slots' / args.slot / args.role if args.exclude_root else None
    paths = ([own / 'META.json'] if own and (own / 'META.json').is_file() else [])
    paths += [p for p in builds.glob(f'wip-*/slots/*/{args.role}/META.json') if not own or p.parent.resolve() != own.resolve()]
    for meta_path in paths:
        try:
            if json.loads(meta_path.read_text()).get('build_key') != args.key:
                continue
        except (OSError, ValueError):
            continue
        candidates.append(meta_path.parent)
    # Own slot, then newest first; the first one that verifies wins.
    ordered = sorted(candidates, key=lambda p: (own is not None and p == own, p.joinpath('META.json').stat().st_mtime), reverse=True)
    for slot in ordered:
        try:
            verify(slot, args.key)
        except (OSError, ValueError, KeyError) as error:
            print(f'wip slot reuse: skipping {slot}: {error}', file=sys.stderr)
            continue
        print(slot)
        return 0
    return 1


def slot_lock(root, slot):
    directory = root / 'slot-owners'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / (slot + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def claim(root, slot, generation):
    with slot_lock(root, slot):
        owner = root / 'slot-owners' / slot
        pending = owner.with_name(f'.{slot}.{generation}.next')
        pending.write_text(generation + '\n')
        pending.replace(owner)


def publish(root, slot, role, generation, incoming):
    target = root / 'slots' / slot / role
    with slot_lock(root, slot):
        owner = root / 'slot-owners' / slot
        current = owner.read_text().strip() if owner.is_file() else '(none)'
        if generation != current:
            raise ValueError(f'slot {slot} was claimed by build {current}; this build\'s seal is discarded ({generation})')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # rename cannot replace a nonempty directory. Exchange entire trees
            # without making the prior seal disappear, even on a cancelled build.
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.renameat2(AT_FDCWD, os.fsencode(incoming), AT_FDCWD, os.fsencode(target), RENAME_EXCHANGE):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(target))
        else:
            os.rename(incoming, target)


def copy_slot(args, clone=False):
    source = Path(args.source).resolve()
    if clone:
        raw = (source / 'META.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != (source / 'FINGERPRINT').read_text().strip():
            raise ValueError('cloned role seal mismatch')
        meta = json.loads(raw)
    else:
        meta = verify(source, args.key)
    root = Path(args.root)
    target = root / 'slots' / args.slot / args.role
    if target.resolve() == source:
        print(f'wip slot reuse: {target} already holds the build')
        return 0
    incoming = root / 'incoming' / f'{args.slot}.{args.role}.{args.generation}.{os.getpid()}'
    incoming.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            subprocess.run(['cp', '-al', str(source), str(incoming)], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            shutil.rmtree(incoming, ignore_errors=True)
            subprocess.run(['cp', '-a', str(source), str(incoming)], check=True)
        for name in ('META.json', 'FINGERPRINT'):
            (incoming / name).unlink()
        if clone:
            meta.update(cloned_from=meta.get('cloned_from') or meta['slot'], slot=args.slot)
        else:
            meta.update(wip_instance=args.instance, wip_root=args.wip_root, slot=args.slot,
                        reused_from={'path': str(source), 'seal_sha256': (source / 'FINGERPRINT').read_text().strip(),
                                     'wip_instance': meta.get('wip_instance'), 'slot': meta.get('slot')})
        meta_bytes = (json.dumps(meta, indent=2, sort_keys=True) + '\n').encode()
        (incoming / 'META.json').write_bytes(meta_bytes)
        (incoming / 'FINGERPRINT').write_text(hashlib.sha256(meta_bytes).hexdigest() + '\n')
        publish(root, args.slot, args.role, args.generation, incoming)
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    print(f'WIP slot {"cloned" if clone else "reused"}: slot={args.slot} role={args.role}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    finder = sub.add_parser('find')
    finder.add_argument('--key', required=True)
    finder.add_argument('--role', required=True, choices=ROLES)
    finder.add_argument('--slot', required=True)
    finder.add_argument('--exclude-root', default='')
    for operation in ('adopt', 'clone', 'claim', 'publish'):
        writer = sub.add_parser(operation)
        writer.add_argument('--root', type=Path, required=True)
        writer.add_argument('--slot', required=True)
        writer.add_argument('--generation', required=True)
        if operation != 'claim':
            writer.add_argument('--role', required=True, choices=ROLES)
        if operation in ('adopt', 'clone'):
            writer.add_argument('--source', required=True)
        if operation == 'adopt':
            writer.add_argument('--key', required=True)
            writer.add_argument('--instance', default='')
            writer.add_argument('--wip-root', default='')
        if operation == 'publish':
            writer.add_argument('--incoming', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', args.slot):
        parser.error('invalid slot name')
    if hasattr(args, 'generation') and not re.fullmatch(r'[A-Za-z0-9-]{1,64}', args.generation):
        parser.error('invalid build generation')
    if hasattr(args, 'key') and (len(args.key) != 64 or any(c not in '0123456789abcdef' for c in args.key)):
        parser.error('--key must be a sha256 hex digest')
    try:
        if args.command == 'find':
            return find(args)
        if args.command == 'claim':
            claim(args.root, args.slot, args.generation)
        elif args.command == 'publish':
            publish(args.root, args.slot, args.role, args.generation, args.incoming)
        else:
            return copy_slot(args, clone=args.command == 'clone')
    except (OSError, ValueError) as error:
        sys.exit(f'wip slot: {error}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
