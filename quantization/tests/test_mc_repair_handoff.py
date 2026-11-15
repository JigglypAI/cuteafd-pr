"""Native delta adoption uses hardlinks and metadata, never weight rereads."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mc_repair_handoff as handoff
from kimi_k3.uniform_k2.storage import file_identity


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def row(raw, lfs=False):
    return dict(bytes=len(raw), oid=hashlib.sha256(raw).hexdigest() if lfs else hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest(),
                storage='lfs' if lfs else 'git')


@pytest.fixture
def setup(tmp_path):
    hub, root, delta = [tmp_path / name for name in ('hub', 'handoff', 'delta')]
    delta.mkdir()
    repo, old, new = 'wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC', 'a' * 40, 'b' * 40
    model = hub / ('models--' + repo.replace('/', '--'))
    parent = model / 'snapshots' / old
    parent.mkdir(parents=True)
    blobs = model / 'blobs'; blobs.mkdir()
    inherited = {'inherited.safetensors': (b'old weights', True), 'nested/tokenizer.json': (b'{}', False),
                 '.gitattributes': (b'*.safetensors filter=lfs diff=lfs merge=lfs -text\n', False)}
    old_files = {}
    for name, (raw, lfs) in inherited.items():
        info = row(raw, lfs); old_files[name] = info
        blob = blobs / info['oid']; blob.write_bytes(raw)
        path = parent / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(blob, path.parent))
    (model / 'refs').mkdir(); (model / 'refs/main').write_text(old)
    changed, files = {}, dict(old_files)
    for name, raw in [('original-mtp-0001.safetensors', b'original missing weights'), ('config.json', b'{"fixed":true}')]:
        path = delta / name; path.write_bytes(raw)
        info = row(raw, name.endswith('.safetensors')); files[name] = info
        changed[name] = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            git_blob=info['oid'] if info['storage'] == 'git' else None,
            file_identity=file_identity(path), device=path.stat().st_dev)
    published = dict(schema='cuteafd-original-mtp-publication-complete-v1', status='passed',
        identity=dict(schema='cuteafd-original-mtp-repair-v1', repo=repo, parent_revision=old),
        repo_id=repo, parent_revision=old, revision=new, delta=str(delta), parent_files=old_files,
        changed_files=changed, files=files, normalized_attributes=None)
    receipt = dump(tmp_path / 'publication/publication-complete.json', published)
    calls = []
    def call(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs['timeout'] > 0
        if args[:2] == ('hf', 'import'):
            stage = Path(args[2]); assert not (stage / 'refs').exists()
            assert '--offline' in args and '--no-copy' in args
            assert (model / 'refs/main').read_text() == old
            # Fake native adoption by linking each staged OID and snapshot alias.
            for blob in (stage / 'blobs').iterdir():
                target = blobs / blob.name
                if not target.exists():os.link(blob, target)
            target_snapshot = model / 'snapshots' / new; target_snapshot.mkdir(exist_ok=True)
            for path in (stage / 'snapshots' / new).rglob('*'):
                if path.is_symlink():
                    target = target_snapshot / path.relative_to(stage / 'snapshots' / new)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.is_symlink():target.symlink_to(os.path.relpath(blobs / path.resolve().name, target.parent))
            return {'job': 123}
        if args[0] == 'jobs':
            return dict(job=dict(finished=True, error=None, cancelled=False),
                **{'import':dict(finished=True, errors=[], copied_bytes=0, files=len(published['changed_files']) + bool(published['normalized_attributes']), adopted=0, skipped=0)})
        assert args[0] == 'ls'
        selected = Path(args[1]); relative = selected.relative_to('/hf-home/hub')
        local = hub / relative
        entries=[]
        for path in (local.iterdir() if local.is_dir() else [local]):
            if path.is_dir():entries.append(dict(name=path.name, kind='directory'));continue
            if path.is_symlink():
                entries.append(dict(name=path.name,kind='symlink',size=len(os.readlink(path)),sealed=False,writing_on=None,hosts=[]))
            else:
                entries.append(dict(name=path.name, kind='regular', size=path.stat().st_size,
                                    sealed=True, writing_on=None, hosts=['raptor']))
        return {'entries':entries}
    return dict(publication=receipt, root=root, hub=hub, call=call, calls=calls,
                published=published, model=model, parent=parent, delta=delta)


def run(s):
    return handoff.handoff(s['publication'], s['root'], hub=s['hub'], call=s['call'])


def test_handoff_imports_only_delta_then_aliases_and_switches_ref_last(setup, monkeypatch):
    s=setup
    original_open=Path.open
    def guarded_open(path, *args, **kwargs):
        if path.suffix == '.safetensors':raise AssertionError('weight payload opened')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    result=run(s)
    assert result['status']=='passed'
    assert result['copied_bytes']==result['fetched_bytes']==0
    assert result['imported_delta_files']==2 and result['inherited_files']==3
    assert (s['model']/'refs/main').read_text()==s['published']['revision']
    current=Path(result['snapshot'])
    for name in s['published']['parent_files']:
        assert (current/name).samefile(s['parent']/name)
    assert len([c for c,_ in s['calls'] if c[:2]==('hf','import')])==1
    assert not (s['root']/'hf-stage'/s['model'].name/'refs').exists()
    shutil.rmtree(s['delta'])
    before=len(s['calls']); assert run(s)==result
    assert all(c[0]=='ls' for c,_ in s['calls'][before:])


def test_partial_import_resumes_same_native_job(setup):
    s=setup; real=s['call']; waiting=True
    def interrupted(*args, **kwargs):
        nonlocal waiting
        if args[0]=='jobs' and waiting:
            waiting=False
            raise TimeoutError('wait interrupted')
        return real(*args,**kwargs)
    s['call']=interrupted
    with pytest.raises(TimeoutError):run(s)
    assert (s['model']/'refs/main').read_text()==s['published']['parent_revision']
    result=run(s)
    assert result['status']=='passed'
    assert len([c for c,_ in s['calls'] if c[:2]==('hf','import')])==1


def test_failed_import_retains_old_ref_and_does_not_relaunch(setup):
    s=setup;real=s['call']
    def failed(*args,**kwargs):
        r=real(*args,**kwargs)
        if args[0]=='jobs':r['import']['copied_bytes']=1
        return r
    s['call']=failed
    with pytest.raises(ValueError,match='no-copy'):run(s)
    assert (s['model']/'refs/main').read_text()==s['published']['parent_revision']
    with pytest.raises(ValueError,match='no-copy'):run(s)
    assert len([c for c,_ in s['calls'] if c[:2]==('hf','import')])==1


@pytest.mark.parametrize('defect',['unsealed','size','extra'])
def test_native_inventory_failure_never_publishes_main(setup,defect):
    s=setup;real=s['call']
    def broken(*args,**kwargs):
        r=real(*args,**kwargs)
        if args[0]=='ls':
            files=[v for v in r['entries'] if v['kind']=='regular']
            if files:
                if defect=='unsealed':files[0]['sealed']=False
                elif defect=='size':files[0]['size']+=1
                else:r['entries'].append(dict(files[0],name='unexpected.bin'))
        return r
    s['call']=broken
    with pytest.raises(ValueError,match='inventory'):run(s)
    assert (s['model']/'refs/main').read_text()==s['published']['parent_revision']


def test_changed_delta_and_parent_fail_before_native_import(setup):
    s=setup
    (s['delta']/'original-mtp-0001.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):run(s)
    assert not s['calls']


def test_unknown_ref_and_publication_fail_before_native_import(setup):
    s=setup
    (s['model']/'refs/main').write_text('c'*40)
    with pytest.raises(ValueError,match='moved outside'):run(s)
    assert not s['calls']


def test_normalized_attributes_are_delta_metadata(setup):
    s=setup
    raw=(s['parent']/'.gitattributes').read_bytes()+b'original-mtp-0001.safetensors filter=lfs diff=lfs merge=lfs -text\n'
    info=row(raw)
    s['published']['normalized_attributes']=info
    s['published']['files']['.gitattributes']=info
    dump(s['publication'],s['published'])
    (s['publication'].parent/'published.gitattributes').write_bytes(raw)
    result=run(s)
    assert result['imported_delta_files']==3 and result['inherited_files']==2
    assert (Path(result['snapshot'])/'.gitattributes').read_bytes()==raw


def test_existing_alias_wrong_inode_fails_closed(setup):
    s=setup
    # Same size and OID-shaped name is insufficient: retain exact managed inode.
    source=s['parent']/'inherited.safetensors'
    info=s['published']['files']['inherited.safetensors']
    original=source.resolve(); original.rename(original.with_name(original.name+'.old'))
    original.write_bytes(b'old weights')
    source.unlink(); source.symlink_to(os.path.relpath(original.with_name(original.name+'.old'),source.parent))
    with pytest.raises(ValueError,match='OID inode differs'):run(s)
    assert (s['model']/'refs/main').read_text()==s['published']['parent_revision']


def test_interruption_during_ref_switch_resumes_without_reimport(setup, monkeypatch):
    s=setup; replace=os.replace; interrupted=True
    def stop_once(source,target):
        nonlocal interrupted
        if Path(target)==s['model']/'refs/main' and interrupted:
            interrupted=False
            raise OSError('simulated ref publication interruption')
        return replace(source,target)
    monkeypatch.setattr(os,'replace',stop_once)
    with pytest.raises(OSError,match='ref publication interruption'):run(s)
    assert (s['root']/'snapshot-ready.json').is_file()
    assert (s['model']/'refs/main').read_text()==s['published']['parent_revision']
    assert run(s)['status']=='passed'
    assert len([c for c,_ in s['calls'] if c[:2]==('hf','import')])==1


def test_unowned_stage_and_changed_recovery_receipt_fail_closed(setup):
    s=setup
    (s['root']/'hf-stage').mkdir(parents=True)
    with pytest.raises(ValueError,match='unowned HF stage'):run(s)
    assert not s['calls']
    s['published']['identity']['source_revision']='unreviewed-change'
    dump(s['publication'],s['published'])
    with pytest.raises(ValueError,match='recovery identity differs'):run(s)


def test_native_global_content_id_can_differ_from_hub_oid(setup):
    s=setup
    # SparkNest repository blobs/HF-OID entries may themselves alias a global
    # native object. That native content ID is not a Hugging Face checksum.
    global_blobs=s['hub']/'blobs/native';global_blobs.mkdir(parents=True)
    for index,path in enumerate((s['model']/'blobs').iterdir()):
        native=global_blobs/('native-content-'+str(index))
        path.rename(native)
        path.symlink_to(os.path.relpath(native,path.parent))
        assert path.resolve().name!=path.name
    real=s['call']
    def import_global(*args,**kwargs):
        result=real(*args,**kwargs)
        if args[:2]==('hf','import'):
            for index,path in enumerate((s['model']/'blobs').iterdir()):
                if path.is_symlink():continue
                native=global_blobs/('imported-native-'+str(index))
                path.rename(native)
                path.symlink_to(os.path.relpath(native,path.parent))
        return result
    s['call']=import_global
    result=run(s)
    assert result['status']=='passed'
    for name,info in s['published']['files'].items():
        path=Path(result['snapshot'])/name
        assert path.samefile(s['model']/'blobs'/info['oid'])
        assert path.resolve().name!=info['oid']
