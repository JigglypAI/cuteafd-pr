"""CPU-only delta publication guards with real lazy Hub add operations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mc_repair_publish as repair
from kimi_k3.uniform_k2.storage import file_identity

REVISION = '1' * 40


def fixture(tmp_path):
    delta = tmp_path / 'delta'
    delta.mkdir()
    shard = 'original-mtp-0001.safetensors'
    files = {}
    for name in repair.METADATA | {shard}:
        path = delta / name
        raw = b'synthetic weight payload' if name == shard else json.dumps({'file': name}).encode()
        path.write_bytes(raw)
        files[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), file_identity=file_identity(path))
    repo, parent = next(iter(repair.PARENTS.items()))
    report = dict(identity=dict(schema='cuteafd-original-mtp-repair-v1', repo=repo,
        parent_revision=parent, source={'repo':'test/source','revision':'source-sha'}, modules=['mtp']),
        status='validated-delta-headers-and-streamed-source-bytes',
        compact_metadata_unchanged=True, new_shards={shard: deepcopy(files[shard])}, new_files=files)
    (delta / repair.REPORT).write_text(json.dumps(report))
    return delta, repair.delta_packet(delta)


class Hub:
    def __init__(self, packet):
        self.packet = packet
        self.parent = packet['parent_revision']
        self.head = self.parent
        self.private = False
        self.calls = []
        self.fail_after_commit = False
        self.attributes = b'*.safetensors filter=lfs diff=lfs merge=lfs -text\n'
        self.normalized = None
        self.trees = {self.parent: {name: dict(bytes=7, oid='a'*40, storage='git') for name in repair.METADATA}}
        self.trees[self.parent]['.gitattributes'] = dict(bytes=len(self.attributes),
            oid=hashlib.sha1(f'blob {len(self.attributes)}\0'.encode()+self.attributes).hexdigest(), storage='git')
        self.trees[self.parent]['existing.safetensors'] = dict(bytes=1000, oid='b'*64, storage='lfs')
        self.trees[self.parent]['model.k8.safetensors.index.json'] = dict(bytes=8, oid='c'*40, storage='git')

    def model_info(self, repo, revision=None):
        return SimpleNamespace(sha=revision or self.head, private=self.private)

    def list_repo_tree(self, repo, revision, recursive):
        assert recursive
        return [SimpleNamespace(path=n, size=r['bytes'], blob_id='d'*40 if r['storage']=='lfs' else r['oid'],
            lfs=SimpleNamespace(sha256=r['oid']) if r['storage']=='lfs' else None)
            for n,r in self.trees[revision].items()]

    def create_commit(self, **kwargs):
        assert kwargs['parent_commit'] == self.parent and kwargs['repo_id'] == self.packet['repo_id']
        assert kwargs['repo_type'] == 'model'
        operations = kwargs['operations']
        # Exercise the SDK's actual SHA scheduling against the prepared operations.
        from huggingface_hub._commit_api import _compute_missing_sha256s
        _compute_missing_sha256s(operations, num_threads=1)
        assert {op.path_in_repo for op in operations} == set(self.packet['files'])
        assert not any(op.path_in_repo == 'existing.safetensors' for op in operations)
        tree = deepcopy(self.trees[self.parent])
        for op in operations:
            row = self.packet['files'][op.path_in_repo]
            assert op.upload_info.is_hashed and op.upload_info.sha256.hex() == row['sha256']
            assert op._is_uploaded is False and op._is_committed is False
            assert op.upload_info.size == row['bytes']
            lfs = op.path_in_repo.endswith('.safetensors') or op.path_in_repo == 'quantize_config.json'
            tree[op.path_in_repo] = dict(bytes=row['bytes'], oid=row['sha256'] if lfs else row['git_blob'], storage='lfs' if lfs else 'git')
        self.calls.append(kwargs)
        if self.normalized:
            tree['.gitattributes'] = dict(bytes=len(self.normalized),
                oid=hashlib.sha1(f'blob {len(self.normalized)}\0'.encode()+self.normalized).hexdigest(), storage='git')
        self.trees[REVISION] = tree
        self.head = REVISION
        if self.fail_after_commit:
            raise OSError('commit response lost')
        return SimpleNamespace(oid=REVISION)


def deny_payload_reads(monkeypatch):
    real_open = Path.open
    def checked(path, *args, **kwargs):
        if path.suffix == '.safetensors':
            pytest.fail('publisher reread or rehashed a weight payload')
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', checked)
    import huggingface_hub.lfs as lfs
    monkeypatch.setattr(lfs, 'sha_fileobj', lambda *a, **kw: pytest.fail('extra SDK payload hash'))


def test_delta_only_commit_uses_writer_sha_and_pinned_parent(tmp_path, monkeypatch):
    delta, packet = fixture(tmp_path)
    hub, root = Hub(packet), tmp_path / 'receipts'
    deny_payload_reads(monkeypatch)
    result = repair.publish(delta, root, api=hub, anonymous_api=hub)
    assert result['status'] == 'passed' and result['revision'] == REVISION
    assert len(hub.calls) == 1
    assert result['files']['existing.safetensors'] == hub.trees[hub.parent]['existing.safetensors']
    assert set(result['changed_files']) == set(packet['files'])
    shutil.rmtree(delta)
    assert repair.publish(delta, root, api=hub, anonymous_api=hub) == result
    assert len(hub.calls) == 1


def test_commit_response_loss_recovers_without_second_commit(tmp_path, monkeypatch):
    delta, packet = fixture(tmp_path)
    hub, root = Hub(packet), tmp_path / 'receipts'
    deny_payload_reads(monkeypatch)
    hub.fail_after_commit = True
    with pytest.raises(OSError, match='response lost'):
        repair.publish(delta, root, api=hub, anonymous_api=hub)
    assert not (root / 'publication-complete.json').exists()
    assert repair.publish(delta, root, api=hub, anonymous_api=hub)['status'] == 'passed'
    assert len(hub.calls) == 1


def test_unrelated_new_head_never_gets_overwritten(tmp_path):
    delta, packet = fixture(tmp_path)
    hub = Hub(packet)
    hub.head = 'e' * 40
    with pytest.raises(ValueError, match='head moved'):
        repair.publish(delta, tmp_path/'receipts', api=hub, anonymous_api=hub)
    assert hub.calls == []


@pytest.mark.parametrize('invalid', [False, True])
def test_hub_attributes_normalization_is_tiny_literal_new_shard_only(tmp_path, monkeypatch, invalid):
    delta, packet = fixture(tmp_path)
    hub, root = Hub(packet), tmp_path/'receipts'
    hub.normalized = hub.attributes + ((b'*.py' if invalid else b'original-mtp-0001.safetensors')
                                      + b' filter=lfs diff=lfs merge=lfs -text\n')
    deny_payload_reads(monkeypatch)
    fetched = []
    def fetch(revision, name):
        assert name == '.gitattributes'
        fetched.append(revision)
        return hub.attributes if revision == 'parent' else hub.normalized
    if invalid:
        with pytest.raises(ValueError, match='attributes rules'):
            repair.publish(delta,root,api=hub,anonymous_api=hub,fetch_metadata=fetch)
        assert not (root/'publication-complete.json').exists()
    else:
        result = repair.publish(delta,root,api=hub,anonymous_api=hub,fetch_metadata=fetch)
        assert result['normalized_attributes'] == hub.trees[REVISION]['.gitattributes']
        assert (root/'published.gitattributes').read_bytes() == hub.normalized
        assert fetched == ['parent', REVISION]


@pytest.mark.parametrize('mutation', ['size', 'oid', 'extra', 'missing'])
def test_exact_remote_tree_rejects_inherited_changes(tmp_path, mutation):
    _, packet = fixture(tmp_path)
    hub = Hub(packet)
    original = hub.trees[hub.parent]
    expected = deepcopy(original)
    for n,r in packet['files'].items():
        expected[n] = dict(bytes=r['bytes'], oid=r['sha256'] if n.endswith('.safetensors') else r['git_blob'],
                           storage='lfs' if n.endswith('.safetensors') else 'git')
    if mutation == 'size': expected['existing.safetensors']['bytes'] += 1
    elif mutation == 'oid': expected['existing.safetensors']['oid'] = 'e'*64
    elif mutation == 'extra': expected['foreign'] = dict(bytes=1,oid='e'*40,storage='git')
    else: del expected['existing.safetensors']
    with pytest.raises(ValueError, match='inventory|inherited'):
        repair.verify_tree(expected, original, packet['files'])


@pytest.mark.parametrize('mutation', ['metadata', 'weight_stat', 'unowned', 'symlink'])
def test_delta_stat_metadata_and_allowlist_guards(tmp_path, mutation):
    delta, _ = fixture(tmp_path)
    if mutation == 'metadata': (delta/'README.md').write_text('changed')
    elif mutation == 'weight_stat':
        with (delta/'original-mtp-0001.safetensors').open('ab') as f: f.write(b'changed')
    elif mutation == 'unowned': (delta/'unowned').write_text('foreign')
    else:
        (delta/'README.md').unlink()
        (delta/'README.md').symlink_to(delta/'config.json')
    with pytest.raises(ValueError): repair.delta_packet(delta)


def test_wrong_sdk_and_private_repository_fail_closed(tmp_path, monkeypatch):
    delta, packet = fixture(tmp_path)
    hub = Hub(packet)
    hub.private = True
    with pytest.raises(ValueError, match='public'):
        repair.publish(delta, tmp_path/'private', api=hub, anonymous_api=hub)
    monkeypatch.setattr(repair.importlib.metadata, 'version', lambda name: '1.0.0')
    with pytest.raises(ValueError, match='2.2.0'):
        repair.publish(delta, tmp_path/'wrong-sdk', api=hub, anonymous_api=hub)
    assert hub.calls == []
