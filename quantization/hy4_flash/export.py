"""Direct three-view Hy4-flash package, with one native copy and alternate Engram."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import mc_package as package
from inkling2.export import DirectMCWriter as SharedDirectMCWriter, VIEWS
from . import recipe


ASSETS = ("LICENSE", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "processor_config.json", "preprocessor_config.json",
          "special_tokens_map.json")


def _storage_entry(module, geometry):
    stored = {name: dict(shape=shape, torch_dtype=package.DTYPES[dtype])
              for name, (dtype, shape) in recipe.packed_specs(module, geometry).items()}
    return package.tensor_storage_entry(geometry, stored, side=1,
                                       mcg_multiplier=recipe.MCG_MULTIPLIER)


def _native_specs(source_inventory, selection):
    excluded = {module + '.weight' for module in selection} | recipe.expert_source_specs().keys()
    return {name: spec for name, spec in source_inventory['specs'].items() if name not in excluded}


def view_metadata(document, inventory, selection, experts, identity, *, k6_drafter_bits):
    """Bind each view to its own index/metadata and exact packed tensor shapes."""
    native = _native_specs(inventory, selection)
    configs, quantizes = {}, {}
    for view in VIEWS:
        index, _, quantize, bits = package.mc_manifest.FILES[view]
        compact = view != 'original'
        config = deepcopy(document)
        declaration = dict(quant_method='exl3', format='mcg', per_module_metadata=f'{quantize}:tensor_storage')
        config['quantization_config'] = declaration
        core = {}
        if compact:
            for module, geometry in selection.items():
                width = k6_drafter_bits if bits == 6 and geometry['group'] == 'mtp' else bits
                core[module] = dict(geometry, bits=width)
        modules = experts | core
        # Exact name patterns for core plus a bounded routed range. The source
        # has no fused expert names in the EXL3 view.
        patterns = [r'^model\.layers\.(?:[1-9]|[1-3][0-9]|4[01])\.mlp\.experts\.'
                    r'(?:[0-9]|[1-9][0-9]|[1-4][0-9]{2}|50[0-9]|51[01])\.(?:gate_proj|up_proj|down_proj)$']
        patterns.extend('^' + re.escape(name) + '$' for name in sorted(core))
        retained = native if compact else native | {name + '.weight': inventory['specs'][name + '.weight']
                                                  for name in selection}
        meta = package.quantize_config_for_view(declaration,
            tensor_storage={module: _storage_entry(module, g) for module, g in modules.items()},
            retained_native_tensors={name: dict(dtype=dtype, shape=shape)
                                     for name, (dtype, shape) in retained.items()},
            provenance=identity, core_view=view, upstream_activations='k8-compact',
            engram_checkpoint_dtype='BF16')
        meta, config = package.gptqmodel_mc_view(config, meta, bits=4,
            core_bits=bits if compact else None, module_include=patterns)
        config['quantization_config']['safetensors_index'] = index
        meta['safetensors_index'] = index
        if compact:
            drafter_bits = bits if bits == 8 else k6_drafter_bits
            config['quantization_config']['drafter_bits'] = drafter_bits
            meta['drafter_bits'] = drafter_bits
        configs[view], quantizes[view] = config, meta
    return configs, quantizes


def verify_original_discovery(output, expected_mapping):
    """Run the pinned Transformers default discovery without loading weights."""
    from transformers import AutoConfig
    from transformers.modeling_utils import _get_resolved_checkpoint_files

    output = Path(output)
    config = AutoConfig.from_pretrained(output, local_files_only=True)
    files, metadata = _get_resolved_checkpoint_files(
        output, variant=None, gguf_file=None, use_safetensors=True,
        user_agent=None, is_remote_code=False, download_kwargs={"local_files_only": True})
    if (config.model_type != "hy_v4_flash" or not metadata
            or metadata["weight_map"] != expected_mapping
            or {Path(path) for path in files} != {output / shard for shard in expected_mapping.values()}
            or not all(Path(path).is_file() for path in files)):
        raise ValueError("Transformers default discovery differs from verified Hy4 original headers")
    return dict(status="passed", scope="default config, index and sealed header inventory",
                reference_revision=recipe.REFERENCE_REVISION,
                config_type=type(config).__name__, shards=len(files), tensors=len(expected_mapping))


def model_card(path, engram, *, k6_drafter_bits, selection):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mismatches = sum(row['mismatch_count'] for row in engram['proof'].values())
    rows = sum(row['rows'] for row in engram['proof'].values())
    if mismatches != 0:
        raise ValueError("model card cannot claim an exact Engram alternate with mismatches")
    native_gib = rows * 192 / (1 << 30)
    fp8_gib = rows * 100 / (1 << 30)
    head = 'with a compact head' if 'lm_head' in selection else 'with the original BF16 head'
    text = f"""---
library_name: cuteafd
license: tencent-hy-license
base_model: tencent/Hy4-flash
tags:
- exl3
- mixture-of-experts
---

# Hy4-flash EXL3 K4 MC

One checkpoint with K4 MCG routed experts and three indexed core views. The
`original` view is the default and keeps the source core and bundled DFly2
weights. `k8-compact` and `k6-compact` contain compact target matrices
{head}. They select K8 and K{k6_drafter_bits} DFly2 matrices respectively.
The K4 experts, both Engram representations, iHC, compressed indexers, router,
native drafter tensors, vision tower and projector are shared. A loader that
implements Hy4-flash and its EXL3 experts can discover the original view from
`config.json` and `model.safetensors.index.json` alone.

## Engram tables

All views index the four byte-identical native BF16 tables in dedicated
shards ({native_gib:.2f} GiB). The separate FP8 alternate ({fp8_gib:.2f} GiB)
packs 96 E4M3 codes and a little-endian F32 scale into each dense 100-byte row.
The export uses stock BF16 lookup arithmetic to select E4M3 codes and
stores the BF16-rounded scale as F32. It multiplies the stored code and scale
in F32 and casts the result to BF16. Every decoded row is compared bitwise
with the complete stock BF16 `x + (decoded - x)` expression, including
signed zero. Across {rows:,} source rows, the comparison found
{mismatches} mismatches.
`cuteafd-mc.json` carries the alternate layout, row ranges and proof digests.
The ordinary indexes reference the BF16 tables.

## Calibration

Source `{recipe.BASE_REPO}@{recipe.BASE_REVISION}`. Calibration uses the
source tokenizer and chat template with the standard own-text panel,
including long-sequence coverage. One causal pass feeds reconstructed K8 core
and K4 expert activations downstream. K6 encodes reuse the Hessians captured
on that path. DFly2 uses the reference target hidden-state, KV and indexer
taps at layers 25, 29, 33, 37 and 41.

Hy4-flash serving and end-to-end runtime qualification are separate work.
See [CuteAFD #2478](https://github.com/tpurtell/cuteafd/issues/2478).
"""
    path.write_text(text, encoding='utf-8')
    return path


class DirectMCWriter(SharedDirectMCWriter):
    """Reuse direct MC shard writer with a complete Hy4 native byte inventory."""

    family = 'hy4_flash'
    prove_shared_source = True

    def __init__(self, output, source_snapshot, identity, source_inventory, **kwargs):
        super().__init__(output, source_snapshot, recipe.BASE_REPO, recipe.BASE_REVISION,
                         identity, source_inventory, **kwargs)

    def finalize(self, *, core_selection, engram, document, error_csv, card_path,
                 k6_drafter_bits=6):
        """Seal and independently validate exact complete per-view inventories."""
        if set(engram['tables']) != recipe.ENGRAM_TABLES.keys() or k6_drafter_bits not in (6, 8):
            raise ValueError('Hy4-flash requires all dedicated Engram tables and a reviewed drafter width')
        reviewed = recipe.reviewed_core_modules(
            include_head='lm_head' in core_selection,
            include_kv_b='model.layers.0.self_attn.kv_b_proj' in core_selection)
        if any(module not in reviewed or any(g[key] != reviewed[module][key] for key in
               ('in_features', 'out_features', 'stored_out_features', 'group'))
               for module, g in core_selection.items()):
            raise ValueError('unreviewed selected core')
        if {n for n in core_selection if n.startswith('model.dfly_layers.')} != {
                n for n in reviewed if n.startswith('model.dfly_layers.')}:
            raise ValueError('compact views require every selected DFly2 matrix')
        native = _native_specs(self.source_inventory, core_selection)
        experts = recipe.reviewed_expert_modules()
        shared = native | {name: spec for module, g in experts.items()
                           for name, spec in recipe.packed_specs(module, g).items()}
        configs, quantizes = view_metadata(document, self.source_inventory, core_selection,
                                          experts, self.identity, k6_drafter_bits=k6_drafter_bits)
        assets = {name: self.source / name for name in ASSETS if (self.source / name).is_file()}
        if not {'LICENSE', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                'generation_config.json', 'processor_config.json'} <= assets.keys():
            raise ValueError('source license, tokenizer or processor assets are incomplete')
        assets['README.md'] = model_card(card_path, engram,
                                        k6_drafter_bits=k6_drafter_bits, selection=core_selection)
        assets['per-module-errors.csv'] = Path(error_csv)
        aliases = {module for module, geometry in core_selection.items()
                   if geometry['group'] == 'mtp'} if k6_drafter_bits == 8 else ()
        return super().finalize(core_selection=core_selection, configs=configs, quantizes=quantizes,
                               assets=assets, engram=engram, k6_use_k8=aliases,
                               expected_shared_specs=shared)
