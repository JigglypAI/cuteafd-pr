"""Hy4 direct MC preserves native bytes and isolates the alternate tables."""
import json
from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy4_flash import engram, export, recipe
from kimi_k3.uniform_k2.storage import read_header
import mc_package as package


def packed(module, geometry, bits):
    values = {}
    dtypes = {'I16': torch.int16, 'I32': torch.int32, 'F16': torch.float16}
    for name, (dtype, shape) in recipe.packed_specs(module, geometry, bits=bits).items():
        value = recipe.MCG_MULTIPLIER - 2**32 if name.endswith('.mcg') else 0
        values[name] = torch.full(shape, value, dtype=dtypes[dtype])
    return values


@pytest.mark.parametrize('drafter_bits', [6, 8])
def test_direct_hy4_engram_and_drafter_indexes(tmp_path, monkeypatch, drafter_bits):
    source = tmp_path / 'source'
    source.mkdir()
    base, draft, expert = 'lm_head', 'model.dfly_layers.0.mlp.down_proj', 'model.layers.1.mlp.experts.0.down_proj'
    table = engram.TENSOR_TEMPLATE.format(1)
    first = {'in_features': 128, 'out_features': 128, 'stored_out_features': 128, 'bits': 6}
    selection = {base: dict(first, group='core'), draft: dict(first, group='mtp')}
    experts = {expert: dict(first, bits=4, group='routed')}
    monkeypatch.setattr(recipe, 'reviewed_core_modules', lambda **_: selection)
    monkeypatch.setattr(recipe, 'reviewed_expert_modules', lambda: experts)
    monkeypatch.setattr(recipe, 'ENGRAM_TABLES', {table: 2})
    monkeypatch.setattr(engram, 'TABLE_ROWS', {1: 2})
    native = {
        base + '.weight': torch.ones((128, 128), dtype=torch.bfloat16),
        draft + '.weight': torch.ones((128, 128), dtype=torch.bfloat16),
        table: torch.ones((2, 96), dtype=torch.bfloat16),
        'model.layers.1.engram_layer.lookup_table': torch.zeros((3,), dtype=torch.int32),
    }
    weights = source / 'model.safetensors'
    save_file(native, weights)
    header, _ = read_header(weights)
    sizes = {n: native[n].numel() * native[n].element_size() for n in native}
    payload = sum(sizes.values())
    inv = dict(mapping={name: weights.name for name in native},
               locations={name: dict(shard=weights.name, source_name=name) for name in native},
               specs={name: (header[name]['dtype'], header[name]['shape']) for name in native},
               sizes=sizes, payload_bytes=payload, tensor_count=len(native),
               payload_totals=dict(declared=payload, measured=payload, reviewed=payload))
    for name in export.ASSETS:
        (source / name).write_text('asset')
    output = tmp_path / 'out'
    writer = export.DirectMCWriter(output, source, {'run': 1}, inv)

    def write_inputs(filename, tensors):
        path = tmp_path / filename
        save_file(tensors, path)
        return {name: (path, name) for name in tensors}

    k8 = write_inputs('k8.safetensors', packed(base, first, 8) | packed(draft, first, 8))
    k6_data = packed(base, first, 6)
    if drafter_bits == 6:
        k6_data.update(packed(draft, first, 6))
    k6 = write_inputs('k6.safetensors', k6_data)
    shared = write_inputs('k4.safetensors', packed(expert, first, 4))
    lookup = 'model.layers.1.engram_layer.lookup_table'
    shared[lookup] = (weights, lookup)
    writer.write_group('all', shared=shared, original={name + '.weight': (weights, name + '.weight')
        for name in selection}, k8=k8, k6=k6)
    alt = engram.export_tables(output, {table: weights}, recipe.REFERENCE_REVISION)
    errors = tmp_path / 'errors.csv'
    errors.write_text('module,bits\n')
    result = writer.finalize(core_selection=selection, engram=alt, document={'model_type': 'hy_v4_flash'},
                             error_csv=errors, card_path=tmp_path / 'card.md',
                             k6_drafter_bits=drafter_bits)
    manifest = json.loads((output / 'cuteafd-mc.json').read_bytes())
    assert manifest['engram_fp8'] == alt['manifest']
    assert result['status'] == 'validated-headers-and-byte-exactness'
    assert set(result['original_core_tensor_sha256']) == set(native)
    original, k8_map, k6_map = [json.loads((output / package.mc_manifest.FILES[view][0]).read_bytes())['weight_map']
                                for view in export.VIEWS]
    assert draft + '.weight' in original and draft + '.trellis' not in original
    assert original[table] == k8_map[table] == k6_map[table] == alt['tables'][table]
    assert (k6_map[draft + '.trellis'] == k8_map[draft + '.trellis']) == (drafter_bits == 8)
    for part in manifest['engram_fp8']['layers'][0]['parts']:
        assert all(part['file'] not in mapping.values() for mapping in (original, k8_map, k6_map))
    assert original[lookup] == k8_map[lookup] == k6_map[lookup]
