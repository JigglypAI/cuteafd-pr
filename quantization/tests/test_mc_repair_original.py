"""Repair tiny mixed shards without touching quantized payloads."""
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mc_repair_original as repair
import mc_package as package


def save(root,name,values):
    header,offset={},0
    raw=b''
    for tensor,(dtype,shape,data) in values.items():
        header[tensor]=dict(dtype=dtype,shape=shape,data_offsets=[offset,offset+len(data)])
        raw+=data;offset+=len(data)
    encoded=json.dumps(header).encode();encoded+=b' '*(-len(encoded)%8)
    (root/name).write_bytes(struct.pack('<Q',len(encoded))+encoded+raw)


def fixture(tmp_path,monkeypatch):
    p,s,o=[tmp_path/n for n in ['parent','source','delta']]
    p.mkdir();s.mkdir()
    g=dict(in_features=128,out_features=128,stored_out_features=128,bits=4,group='mtp_core')
    module='model.layers.2.eh_proj'
    monkeypatch.setattr(repair,'mtp_modules',lambda family:(SimpleNamespace(BASE_REPO='test/source',BASE_REVISION='source'),{module:g}))
    monkeypatch.setitem(repair.REVISIONS,'glmf',('test/model','parent'))
    packed=package.packed_specs(module,g)
    vals={n:(d,sh,b'Q'*(2*__import__('math').prod(sh) if d!='I32' else 4)) for n,(d,sh) in packed.items()}
    vals['model.layers.1.mlp.experts.0.trellis']=('I16',[2],b'EEee')
    # Shared expert and MTP occupy one shard; default stops indexing MTP but
    # neither copies nor rewrites this physical shard.
    save(p,'mixed.safetensors',vals)
    vals['head.weight']=('BF16',[1],b'HH')
    save(p,'original.safetensors',{'head.weight':vals['head.weight']})
    save(p,'compact.safetensors',{'head.trellis':('I16',[1],b'KK')})
    wm={n:'mixed.safetensors' for n in vals if n!='head.weight'}
    original=wm|{'head.weight':'original.safetensors'}
    compact=wm|{'head.trellis':'compact.safetensors'}
    q=dict(tensor_storage={module:dict(bits_per_weight=4,stored_tensors={n:dict(
               torch_dtype=package.DTYPES[d],shape=shape) for n,(d,shape) in packed.items()}),
               'expert':dict(bits_per_weight=4)},
           module_include=['.*'],mtp_bits=4,lm_head=False,kept_tensors={})
    cfg=dict(quantization_config=dict(mtp_bits=4),text_config=dict(quantization_config=dict(mtp_bits=4)))
    for view,mapping,core in [('original',original,['head.weight']),('k6-compact',compact,['head.trellis']),('k8-compact',compact,['head.trellis'])]:
        idx,cf,qf,_=package.mc_manifest.FILES[view]
        size=sum(len(vals[n][2]) if n in vals else 2 for n in mapping)
        (p/idx).write_text(json.dumps(dict(metadata=dict(total_size=size),weight_map=mapping)))
        (p/cf).write_text(json.dumps(cfg));(p/qf).write_text(json.dumps(q))
    manifest=package.mc_manifest.build(p,source_repo='test/source',source_revision='source',
        core_tensors={'original':['head.weight'],'k6-compact':['head.trellis'],'k8-compact':['head.trellis']})
    (p/'cuteafd-mc.json').write_text(json.dumps(manifest));(p/'README.md').write_text('---\nlicense: mit\n---\nOld shared MTP claim\n')
    raw=b'\x81\x3f'*16384
    save(s,'source.safetensors',{module+'.weight':('BF16',[128,128],raw)})
    (s/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={module+'.weight':'source.safetensors'})))
    return p,s,o,module,raw


def test_delta_preserves_compact_shared_and_exact_source_bytes(tmp_path,monkeypatch):
    p,s,o,module,raw=fixture(tmp_path,monkeypatch)
    before={f.name:f.read_bytes() for f in p.iterdir()}
    report=repair.prepare('glmf',p,s,o)
    assert {f.name:f.read_bytes() for f in p.iterdir()}==before
    idx=json.loads((o/'model.safetensors.index.json').read_text())
    assert not any(n.startswith(module+'.') and n!=module+'.weight' for n in idx['weight_map'])
    assert idx['weight_map']['model.layers.1.mlp.experts.0.trellis']=='mixed.safetensors'
    h,base=repair.read_header(o/'original-mtp-0001.safetensors')
    assert (o/'original-mtp-0001.safetensors').read_bytes()[base:]==raw
    assert report['new_shards']['original-mtp-0001.safetensors']['sha256']==hashlib.sha256((o/'original-mtp-0001.safetensors').read_bytes()).hexdigest()
    q=json.loads((o/'quantize_config.json').read_text())
    assert module not in q['tensor_storage'] and 'mtp_bits' not in q
    cfg=json.loads((o/'config.json').read_text())
    assert 'mtp_bits' not in cfg['quantization_config']
    assert 'mtp_bits' not in cfg['text_config']['quantization_config']
    m=json.loads((o/'cuteafd-mc.json').read_text())
    for v in m['variants']:
        assert (module+'.weight' in v['core_tensors']) == (v['name']=='original')
        assert (module+'.trellis' in v['core_tensors']) == (v['name']!='original')
        if v['name']!='original':
            for f in [v['index'],v['config'],v['quantize']]:
                assert not (o/f).exists()
                (o/f).symlink_to(p/f)
    package.mc_manifest.validate(m,o)
    assert 'license: mit' in (o/'README.md').read_text()
    assert report['new_tensor_payload_bytes']==len(raw)


@pytest.mark.parametrize('bad',['dtype','source','unknown','collision','compact','geometry'])
def test_preflight_rejects_before_any_write(tmp_path,monkeypatch,bad):
    p,s,o,module,raw=fixture(tmp_path,monkeypatch)
    if bad in ['dtype','geometry']:
        save(s,'source.safetensors',{module+'.weight':('F16' if bad=='dtype' else 'BF16',[64,256] if bad=='geometry' else [128,128],raw)})
    elif bad=='source':
        x=json.loads((p/'cuteafd-mc.json').read_text());x['source']['revision']='other';(p/'cuteafd-mc.json').write_text(json.dumps(x))
    else:
        idx=p/('model.k8.safetensors.index.json' if bad=='compact' else 'model.safetensors.index.json')
        x=json.loads(idx.read_text())
        if bad=='unknown': x['weight_map']['other.trellis']='mixed.safetensors'
        if bad=='collision': x['weight_map'][module+'.weight']='mixed.safetensors'
        if bad=='compact': x['weight_map'][module+'.trellis']='other.safetensors'
        idx.write_text(json.dumps(x))
        # Rebuild discovery to isolate the intended structural failure.
        m=json.loads((p/'cuteafd-mc.json').read_text())
        # Stale discovery itself is also a preflight rejection, before payload IO.
    with pytest.raises(ValueError):repair.prepare('glmf',p,s,o)
    assert not o.exists()


def test_hardlink_reuse_has_no_second_payload_read(tmp_path,monkeypatch):
    p,s,o,module,raw=fixture(tmp_path,monkeypatch)
    repair.prepare('glmf',p,s,o)
    monkeypatch.setattr(package,'_assemble_proved_shard',lambda *a,**k:pytest.fail('unexpected second copy'))
    second=tmp_path/'second'
    r=repair.prepare('glmf',p,s,second,reuse=o)
    assert (second/'original-mtp-0001.safetensors').stat().st_ino==(o/'original-mtp-0001.safetensors').stat().st_ino
    assert len(r['added_tensors'])==1


def refresh_manifest(parent):
    old=json.loads((parent/'cuteafd-mc.json').read_text())
    updated=package.mc_manifest.build(parent,source_repo=old['source']['repo'],
        source_revision=old['source']['revision'],
        core_tensors={v['name']:v['core_tensors'] for v in old['variants']})
    (parent/'cuteafd-mc.json').write_text(json.dumps(updated))


@pytest.mark.parametrize('shard',['../outside.safetensors','sub/weights.safetensors',
                                 'sub\\weights.safetensors','/tmp/weights.safetensors','other.bin'])
def test_rejects_source_shard_traversal_before_payload_copy(tmp_path,monkeypatch,shard):
    p,s,o,module,_=fixture(tmp_path,monkeypatch)
    (s/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={module+'.weight':shard})))
    monkeypatch.setattr(package,'_assemble_proved_shard',lambda *a,**k:pytest.fail('payload copy before preflight'))
    with pytest.raises(ValueError,match='safetensors basename'):
        repair.prepare('glmf',p,s,o)
    assert not o.exists()


def test_source_dtype_shape_extent_must_match_before_payload_copy(tmp_path,monkeypatch):
    p,s,o,module,raw=fixture(tmp_path,monkeypatch)
    save(s,'source.safetensors',{module+'.weight':('BF16',[128,128],raw[:-2])})
    monkeypatch.setattr(package,'_assemble_proved_shard',lambda *a,**k:pytest.fail('payload copy before preflight'))
    with pytest.raises(ValueError,match='shape/dtype extent differs'):
        repair.prepare('glmf',p,s,o)
    assert not o.exists()


@pytest.mark.parametrize('bad',['missing','bits','stored_tensors'])
def test_bad_quant_metadata_fails_before_output_or_payload_copy(tmp_path,monkeypatch,bad):
    p,s,o,module,_=fixture(tmp_path,monkeypatch)
    q=json.loads((p/'quantize_config.json').read_text())
    if bad=='missing':q['tensor_storage'].pop(module)
    elif bad=='bits':q['tensor_storage'][module]['bits_per_weight']=3
    else:q['tensor_storage'][module]['stored_tensors']={}
    (p/'quantize_config.json').write_text(json.dumps(q));refresh_manifest(p)
    monkeypatch.setattr(package,'_assemble_proved_shard',lambda *a,**k:pytest.fail('payload copy before preflight'))
    with pytest.raises(ValueError,match='MTP quantization metadata differs'):
        repair.prepare('glmf',p,s,o)
    assert not o.exists()


@pytest.mark.parametrize('bad',['missing','status','source','groups','sha','tensor_proof','file_changed'])
def test_bad_reuse_fails_before_any_output_link(tmp_path,monkeypatch,bad):
    p,s,donor,module,_=fixture(tmp_path,monkeypatch)
    repair.prepare('glmf',p,s,donor)
    report=donor/'original-mtp-repair.json'
    row=json.loads(report.read_text());filename='original-mtp-0001.safetensors'
    if bad=='missing':report.unlink()
    elif bad=='status':row['status']='partial'
    elif bad=='source':row['identity']['source']['revision']='another-source'
    elif bad=='groups':row['new_shards']['original-mtp-9999.safetensors']=row['new_shards'][filename]
    elif bad=='sha':row['new_shards'][filename]['sha256']='bad'
    elif bad=='tensor_proof':row['new_shards'][filename]['original_tensor_proofs'][module+'.weight']['source_name']='wrong.weight'
    elif bad=='file_changed':
        path=donor/filename
        path.write_bytes(path.read_bytes()+b'x')
    if bad!='missing':report.write_text(json.dumps(row))
    output=tmp_path/'reused'
    monkeypatch.setattr(package,'_assemble_proved_shard',lambda *a,**k:pytest.fail('unexpected source copy'))
    with pytest.raises(ValueError):
        repair.prepare('glmf',p,s,output,reuse=donor)
    assert not output.exists()
