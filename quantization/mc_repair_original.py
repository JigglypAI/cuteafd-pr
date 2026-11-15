"""Append missing original MTP core without reading/repacking any existing quant.

Preparation is local and CPU-only. The output is a delta, not a full checkpoint;
compact metadata is linked only to validate discovery and is never uploaded.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import re

import mc_package as package
from core_export import packed_specs
from kimi_k3.uniform_k2.storage import atomic_json, file_identity, read_header, sha256_file

REVISIONS = {
    'glm5': ('wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC', 'c39486376dee84e5faa4904e014e07d7b6e76405'),
    'glmf': ('wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K4-MC', '91a8dbc9c12e790c65aa1a38c6aede6a3e23a77d'),
    'glmf-k325': ('wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-MC', 'b59496510b33f27d92516e8f4f27e733db4d816e'),
    'step5': ('wrldsuksgo2mars/Step-5-Preview-EXL3-K4-MC', '4483a4bd874e8c360aec486ea72bd31d1ad7a247'),
}


def mtp_modules(family):
    recipe, _ = package.family_recipe('glmf' if family == 'glmf-k325' else family)
    return recipe, {n: g for n, g in recipe.reviewed_core_modules().items() if g['group'] == 'mtp_core'}


def clear_mtp_bits(config):
    """Remove active bitrate claims only; keep historical numerical provenance."""
    if isinstance(config, dict):
        for key, value in config.items():
            if key == 'quantization_config':
                value.pop('mtp_bits', None)
            else:
                clear_mtp_bits(value)
    elif isinstance(config, list):
        for value in config:
            clear_mtp_bits(value)


def shard_path(root, mapping, tensor):
    name = mapping[tensor]
    if (not isinstance(name, str) or Path(name).name != name or '\\' in name
            or name.startswith('.') or not name.endswith('.safetensors')):
        raise ValueError('tensor shard must be a safetensors basename')
    return root / name


def reuse_receipts(donor, output, groups, specs, source, source_index, manifest, modules):
    """Check all donor receipts and headers before creating any output links."""
    donor = Path(donor).resolve()
    if output == donor or output in donor.parents or donor in output.parents:
        raise ValueError('delta must be disjoint from reused repair')
    try:
        row = package.load_json(donor / 'original-mtp-repair.json')
        if (row['status'] != 'validated-delta-headers-and-streamed-source-bytes'
                or row['identity']['source'] != manifest['source']
                or row['identity']['modules'] != sorted(modules)
                or set(row['new_shards']) != set(groups)):
            raise ValueError('reuse source/MTP selection or shard groups differs')
        receipts = deepcopy(row['new_shards'])
        for filename, names in groups.items():
            receipt = receipts[filename]
            path = donor / filename
            if file_identity(path) != receipt['file_identity']:
                raise ValueError('reused final shard changed')
            if not re.fullmatch('[0-9a-f]{64}', receipt['sha256']):
                raise ValueError('reused shard SHA proof is malformed')
            if (row['new_files'][filename]['sha256'] != receipt['sha256']
                    or row['new_files'][filename]['file_identity'] != receipt['file_identity']):
                raise ValueError('reused shard inventory differs from receipt')
            header, _ = read_header(path)
            if (set(header) - {'__metadata__'} != set(names)
                    or any((header[n]['dtype'], header[n]['shape']) != specs[n] for n in names)
                    or set(receipt['original_tensor_proofs']) != set(names)):
                raise ValueError('reused shard tensor inventory differs')
            for name in names:
                proof = receipt['original_tensor_proofs'][name]
                origin = shard_path(source, source_index, name)
                if (proof['source_name'] != name or proof['source_path'] != str(origin.resolve())
                        or proof['source_file_identity'] != file_identity(origin)
                        or not re.fullmatch('[0-9a-f]{64}', proof['sha256'])):
                    raise ValueError('reused original source proof changed')
        return donor, receipts
    except (OSError, KeyError, TypeError) as error:
        raise ValueError('reused repair is missing or malformed') from error


def prepare(family, parent, source, output, *, shard_target=2_000_000_000, reuse=None, source_identity=None):
    repo, revision = REVISIONS[family]
    recipe, modules = mtp_modules(family)
    parent, source, output = map(lambda p: Path(p).resolve(), (parent, source, output))
    if parent.name != revision or source.name != recipe.BASE_REVISION:
        raise ValueError('exact pinned parent/source snapshot paths required')
    if any(output == p or output in p.parents or p in output.parents for p in (parent, source)):
        raise ValueError('delta must be disjoint from immutable inputs')
    manifest = package.mc_manifest.validate(package.load_json(parent / 'cuteafd-mc.json'), parent)
    if manifest['source'] != dict(repo=recipe.BASE_REPO, revision=recipe.BASE_REVISION):
        raise ValueError('MC source identity differs from recipe')
    variants = {v['name']: v for v in manifest['variants']}
    index = package.load_json(parent / 'model.safetensors.index.json')
    weights = index['weight_map']
    removed = {n for m, g in modules.items() for n in packed_specs(m, dict(g, bits=4))}
    actual = {n[:-8] for n in weights if n.endswith('.trellis') and '.experts.' not in n
              and n not in variants['original']['core_tensors']}
    if actual != modules.keys() or not removed <= weights.keys():
        raise ValueError('default defect differs from exact reviewed MTP set')
    source_index = package.load_json(source / 'model.safetensors.index.json')['weight_map']
    native = {m + '.weight' for m in modules}
    if not native <= source_index.keys() or native & weights.keys():
        raise ValueError('source MTP missing or collides with default index')
    headers, sizes, specs = {}, {}, {}
    for root, mapping, names in ((parent, weights, removed), (source, source_index, native)):
        for n in sorted(names):
            p = shard_path(root, mapping, n)
            if p not in headers:
                headers[p] = read_header(p)[0]
            h = headers[p][n]
            sizes[n] = h['data_offsets'][1] - h['data_offsets'][0]
            specs[n] = (h['dtype'], h['shape'])
    expected = {n: spec for m, g in modules.items() for n, spec in packed_specs(m, dict(g,bits=4)).items()}
    if any(specs[n] != spec for n, spec in expected.items()):
        raise ValueError('existing MTP is not exact reviewed K4 geometry')
    for m, g in modules.items():
        if specs[m + '.weight'] != ('BF16', [g['out_features'], g['in_features']]):
            raise ValueError('original MTP source shape/dtype differs')
    for name, v in variants.items():
        if name == 'original':
            continue
        wm = package.load_json(parent / v['index'])['weight_map']
        if any(wm.get(n) != weights[n] for n in removed) or removed & set(v['core_tensors']):
            raise ValueError('compact view MTP sharing differs')
    source_proof = package.load_json(source_identity) if source_identity else None
    if source_proof is not None:
        if (source_proof.get('status') != 'passed'
                or source_proof.get('repo_id') != recipe.BASE_REPO
                or source_proof.get('pinned_revision') != recipe.BASE_REVISION
                or source_proof.get('pinned_index_sha256') != sha256_file(source / 'model.safetensors.index.json')
                or {n for row in source_proof['shards'] for n in row['tensors']} != native):
            raise ValueError('source alias proof differs from exact pinned MTP selection')
        for row in source_proof['shards']:
            if any(source_index[n] != row['shard'] for n in row['tensors']):
                raise ValueError('source alias tensor mapping differs')
            alias, cached = source / row['shard'], Path(row['cached_path'])
            if not alias.samefile(cached) or alias.stat().st_size != row['bytes']:
                raise ValueError('source alias no longer matches verified managed source')
    # Metadata and every donor shard must pass before any source payload copy.
    quant = package.load_json(parent / 'quantize_config.json')
    config = package.load_json(parent / 'config.json')
    storage = quant.get('tensor_storage')
    if not isinstance(storage, dict):
        raise ValueError('parent quantization storage is malformed')
    for module, geometry in modules.items():
        entry = storage.get(module)
        required = {name: dict(shape=shape, torch_dtype=package.DTYPES[dtype])
                    for name, (dtype, shape) in packed_specs(module, dict(geometry, bits=4)).items()}
        if (not isinstance(entry, dict) or entry.get('bits_per_weight') != 4
                or entry.get('stored_tensors') != required):
            raise ValueError('parent MTP quantization metadata differs from headers')
    q, c = package.view_metadata(quant, config, modules, {n:specs[n] for n in native},
                                bits=None, quantize_file='quantize_config.json')
    q.pop('mtp_bits', None)
    clear_mtp_bits(c)
    groups = {f'original-mtp-{i:04d}.safetensors': names
              for i, names in enumerate(package.shard_groups(native, sizes, shard_target), 1)}
    donor, donor_receipts = (None, {}) if reuse is None else reuse_receipts(
        reuse, output, groups, specs, source, source_index, manifest, modules)
    identity = dict(schema='cuteafd-original-mtp-repair-v1', repo=repo, parent_revision=revision,
                    source=manifest['source'], modules=sorted(modules),
                    input_metadata_sha256={f: sha256_file(parent / f) for f in
                        ('cuteafd-mc.json','model.safetensors.index.json','config.json','quantize_config.json')})
    if output.exists():
        raise FileExistsError('use a fresh delta directory; never silently reuse a partial repair')
    output.mkdir(parents=True)
    receipts, native_map = {}, {}
    for filename, names in groups.items():
        if reuse is None:
            receipt = package._assemble_proved_shard({n: (source / source_index[n], n) for n in names}, output / filename)
        else:
            receipt = donor_receipts[filename]
            if file_identity(donor / filename) != receipt['file_identity']:
                raise ValueError('reused final shard changed')
            os.link(donor / filename, output / filename)
        receipts[filename] = receipt
        h, _ = read_header(output / filename)
        if set(h) - {'__metadata__'} != set(names) or any((h[n]['dtype'],h[n]['shape']) != specs[n] for n in names):
            raise ValueError('new original MTP header differs')
        native_map.update({n: filename for n in names})
    new_index = deepcopy(index)
    new_index['weight_map'] = dict(sorted({n: s for n, s in weights.items() if n not in removed}.items() | native_map.items()))
    new_index['metadata']['total_size'] += sum(sizes[n] for n in native) - sum(sizes[n] for n in removed)
    atomic_json(output / 'model.safetensors.index.json', new_index)
    atomic_json(output / 'quantize_config.json', q)
    atomic_json(output / 'config.json', c)
    core = {v['name']: sorted(set(v['core_tensors']) | (native if v['name']=='original' else removed))
            for v in manifest['variants']}
    context_files = []
    for view, v in variants.items():
        if view != 'original':
            for f in (v['index'],v['config'],v['quantize']):
                (output / f).symlink_to(parent / f)
                context_files.append(f)
    corrected = package.mc_manifest.build(output, source_repo=recipe.BASE_REPO,
        source_revision=recipe.BASE_REVISION, core_tensors=core,
        original_core_dtype=manifest['original_core_dtype'])
    atomic_json(output / 'cuteafd-mc.json', corrected)
    for f in context_files:
        (output / f).unlink()
    # Publish an unambiguous corrected card; historical cards/diagnostics stay untouched.
    old_card = (parent / 'README.md').read_text()
    front = ''
    if old_card.startswith('---\n'):
        end = old_card.index('\n---\n',4) + 5
        front = old_card[:end] + '\n'
    card = front + '# ' + repo.split('/')[-1] + '\n\n'
    card += (f'The default index preserves the original source core, including all {len(modules)} restored MTP matrices, '
        'their source values and native dtypes. Routed expert encodes are unchanged.\n\n'
        'The stored K8/K6 compact views retain their existing target core encodes and K4 MTP core. '
        'Both views reference the same compressed MTP payload; it is stored once. '
        'Original and compact MTP weights are selected by their respective indexes.\n\n'
        f'Original source: `{recipe.BASE_REPO}@{recipe.BASE_REVISION}`. '
        f'Packaging repair parent: `{repo}@{revision}`.\n\n'
        '`original-mtp-repair.json` records the exact index delta and source-to-output streaming byte proofs. '
        'Only missing original MTP tensors and package metadata were added; no quantization was rerun. '
        'Earlier `package-validation.json` and inherited diagnostics describe the pinned parent package; '
        'the repair receipt supersedes their original-view inventory. '
        'See inherited license and usage assets for the checkpoint terms. '
        'No new full-engine quality or speed claim is made.\n')
    (output / 'README.md').write_text(card)
    new_files = sorted(list(receipts) + ['model.safetensors.index.json','quantize_config.json',
                                        'config.json','cuteafd-mc.json','README.md'])
    report = dict(identity=identity, status='validated-delta-headers-and-streamed-source-bytes',
        source_alias_proof=source_proof,
        added_tensors=sorted(native), removed_default_tensors=sorted(removed), new_shards=receipts,
        original_payload_bytes=new_index['metadata']['total_size'],
        new_tensor_payload_bytes=sum(sizes[n] for n in native),
        compact_metadata_unchanged=True, compact_mtp_core_bits=4,
        new_files={f:dict(bytes=(output/f).stat().st_size,sha256=receipts[f]['sha256'] if f in receipts else sha256_file(output/f),
                    file_identity=file_identity(output/f)) for f in new_files})
    atomic_json(output / 'original-mtp-repair.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=REVISIONS)
    parser.add_argument('--parent',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--reuse')
    parser.add_argument('--source-identity')
    a=parser.parse_args()
    r=prepare(a.family,a.parent,a.source,a.output,reuse=a.reuse,source_identity=a.source_identity)
    print(json.dumps(dict(repo=r['identity']['repo'],modules=len(r['identity']['modules']),
                         added_bytes=r['new_tensor_payload_bytes'],status=r['status'])))


if __name__ == '__main__':
    main()
