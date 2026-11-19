"""Reviewed Hy4-flash BF16 source, K4 experts and dual compact core recipe."""
from __future__ import annotations

from collections import Counter
import hashlib
import re

from glm5.core_k6.recipe import packed_bytes
from core_export import packed_specs as core_packed_specs
from kimi_k3.uniform_k2.storage import canonical

BASE_REPO = "tencent/Hy4-flash"
BASE_REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
TARGET_REPO = "wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC"
REFERENCE_REVISION = "c2ec850510a37978ea83df226589623d5f2be776"
SOURCE_CONFIG_SHA256 = "a59ea10f98683e83c6515e340c85e24eb2e48c91d75af491ed3d6e88c3bc1b3c"
SOURCE_INDEX_SHA256 = "0cd31d2ffceb989e4d465d0c80be5731f6941dbf434ad84506e30930fa02e192"
SOURCE_SPECS_SHA256 = "3fd8338ec487da76349627376380137394fccb315b16869a5d4232d2a0be8073"
BASE_PAYLOAD_BYTES, BASE_TENSOR_COUNT, SHARD_COUNT = 440_520_833_910, 1964, 148
LAYER_COUNT, EXPERTS_PER_LAYER, MTP_LAYER_COUNT = 42, 512, 5
DFLY_TARGET_LAYERS = (25, 29, 33, 37, 41)
SPARSE_LAYERS = (0, 1, *range(5, LAYER_COUNT, 4))
SWA_LAYERS = tuple(layer for layer in range(LAYER_COUNT) if layer not in SPARSE_LAYERS)
ENGRAM_ROWS = {1: 164_994_380, 11: 164_999_008, 21: 165_004_086, 31: 165_008_942}
ENGRAM_TABLES = {f"model.layers.{layer}.engram_layer.multi_head_embedding.embedding.weight": rows
                 for layer, rows in ENGRAM_ROWS.items()}
MUL1_MULTIPLIER = 0x83DCD12D
SUFFIXES = ("trellis", "suh", "svh", "mul1")
EXPERT_MODULE_COUNT = (LAYER_COUNT - 1) * EXPERTS_PER_LAYER * 3
BASE_CORE_MODULE_COUNT, MTP_CORE_MODULE_COUNT = 461, 45
CORE_MODULE_COUNT = BASE_CORE_MODULE_COUNT + MTP_CORE_MODULE_COUNT
ENCODE_COUNT = EXPERT_MODULE_COUNT + 2 * CORE_MODULE_COUNT
_EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)")


def _geometry(inputs, outputs, bits, group):
    return dict(in_features=inputs, out_features=outputs, stored_out_features=(outputs + 127) // 128 * 128,
                bits=bits, group=group, codebook="mul1")


def packed_specs(module, geometry, *, bits=None):
    width = geometry["bits"] if bits is None else bits
    return core_packed_specs(module, dict(geometry, bits=width, codebook="mul1"))


def _attention(heads, include_kv_b):
    shapes = {
        "q_a_proj": (3072, 1024),
        "q_b_proj": (1024, heads * 256),
        "kv_a_proj_with_mqa": (3072, 576),
        "o_proj": (heads * 256, 3072),
        "linear_gate": (3072, heads * 256),
    }
    # PR 50903 expands the KV latents through an explicit kv_b_proj
    # linear, so its calibration input is available as a real boundary.
    if include_kv_b:
        shapes["kv_b_proj"] = (512, heads * (192 + 256))
    return shapes


def reviewed_core_modules(*, core_bits=6, include_head=True, include_mtp=True, include_kv_b=True):
    """Use source module names, including the bundled DFly2 core in both views."""
    if type(core_bits) is not int or core_bits not in (6, 8):
        raise ValueError("Hy4-flash core bits must be K6 or K8")
    selected = {}

    def add(prefix, shapes, group):
        selected.update({prefix + name: _geometry(inputs, outputs, core_bits, group)
                         for name, (inputs, outputs) in shapes.items()})

    for layer in range(LAYER_COUNT):
        prefix = f"model.layers.{layer}."
        add(prefix + "self_attn.", _attention(32 if layer in SPARSE_LAYERS else 64, include_kv_b), "core")
        # Layer zero is a real dense pre-MoE block with the same linear
        # capture boundaries as the shared-expert FFNs at later layers.
        intermediate = 13_824 if layer == 0 else 1536
        mlp = prefix + ("mlp." if layer == 0 else "mlp.shared_experts.")
        add(mlp, {"gate_proj": (3072, intermediate), "up_proj": (3072, intermediate),
                  "down_proj": (intermediate, 3072)}, "core")
        if layer:
            # Names are from the source, which uses F.linear [out,in].
            add(prefix + "mlp.", {"routed_expert_down_proj": (3072, 1536),
                                  "routed_expert_up_proj": (1536, 3072)}, "core")
    if include_head:
        add("", {"lm_head": (3072, 120_832)}, "core")
    if include_mtp:
        for layer in range(MTP_LAYER_COUNT):
            prefix = f"model.dfly_layers.{layer}."
            add(prefix + "self_attn.", _attention(32, include_kv_b), "mtp")
            add(prefix + "mlp.", {"gate_proj": (3072, 9216), "up_proj": (3072, 9216),
                                  "down_proj": (9216, 3072)}, "mtp")
    return selected


def reviewed_expert_modules(layer=None):
    if layer is not None and (type(layer) is not int or not 0 <= layer < LAYER_COUNT):
        raise ValueError("unreviewed Hy4-flash layer")
    if layer == 0:
        return {}
    layers = range(1, LAYER_COUNT) if layer is None else (layer,)
    return {f"model.layers.{layer}.mlp.experts.{expert}.{leaf}": _geometry(1536, 1536, 4, "routed")
            for layer in layers for expert in range(EXPERTS_PER_LAYER)
            for leaf in ("gate_proj", "up_proj", "down_proj")}


def source_tensor(module):
    """Return the source tensor name and optional expert index and output-row slice."""
    match = _EXPERT.fullmatch(module)
    if match is None:
        if ".mlp.experts." in module:
            raise ValueError("unreviewed Hy4-flash expert module")
        return module + ".weight", None, None
    layer, expert, leaf = match.groups()
    if not 1 <= int(layer) < LAYER_COUNT or not 0 <= int(expert) < EXPERTS_PER_LAYER:
        raise ValueError("unreviewed Hy4-flash expert")
    prefix = f"model.layers.{layer}.mlp.experts."
    if leaf == "down_proj":
        return prefix + "down_proj", int(expert), None
    return prefix + "gate_up_proj", int(expert), (0, 1536) if leaf == "gate_proj" else (1536, 3072)


def expert_source_specs():
    return {f"model.layers.{layer}.mlp.experts.{leaf}": ("BF16", [512, rows, 1536])
            for layer in range(1, LAYER_COUNT)
            for leaf, rows in (("gate_up_proj", 3072), ("down_proj", 1536))}


def audit_source(source):
    specs = source.specs
    if len(specs) != BASE_TENSOR_COUNT or sum(source.sizes.values()) != BASE_PAYLOAD_BYTES:
        raise ValueError("Hy4-flash source count or payload differs")
    if hashlib.sha256(canonical(specs).encode()).hexdigest() != SOURCE_SPECS_SHA256:
        raise ValueError("Hy4-flash source header dtype/shape inventory differs")
    selection = reviewed_core_modules()
    required = {name + ".weight": ("BF16", [g["out_features"], g["in_features"]])
                for name, g in selection.items()}
    required.update(expert_source_specs())
    required.update({name: ("BF16", [rows, 96]) for name, rows in ENGRAM_TABLES.items()})
    if any(specs.get(name) != shape for name, shape in required.items()):
        raise ValueError("Hy4-flash selected source dtype/shape differs")
    return dict(source_tensors=len(specs), source_bytes=sum(source.sizes.values()),
                source_dtypes=dict(Counter(dtype for dtype, _ in specs.values())),
                **size_report(source, selection))


def size_report(source, selection=None, *, k6_drafter_bits=6):
    """Scale-inclusive byte census of three standard views and physical alternate."""
    if k6_drafter_bits not in (6, 8):
        raise ValueError("K6 fallback drafter must use K8")
    selection = reviewed_core_modules() if selection is None else selection
    expert_bytes = EXPERT_MODULE_COUNT * packed_bytes(_geometry(1536, 1536, 4, "routed"))
    original_bytes = sum(source.sizes[name + ".weight"] for name in selection)
    expert_source_bytes = sum(source.sizes[name] for name in expert_source_specs())
    shared_native = source.payload_bytes - expert_source_bytes - original_bytes
    k8_bytes = sum(packed_bytes(dict(g, bits=8)) for g in selection.values())
    k6_bytes = sum(packed_bytes(dict(g, bits=(k6_drafter_bits if g["group"] == "mtp" else 6)))
                   for g in selection.values())
    # A K8 drafter fallback in K6 points at the K8 shards, once physically.
    aliased_bytes = (sum(packed_bytes(dict(g, bits=8)) for g in selection.values() if g["group"] == "mtp")
                     if k6_drafter_bits == 8 else 0)
    engram_fp8 = sum(ENGRAM_ROWS.values()) * 100
    shared = shared_native + expert_bytes
    return dict(core_modules=len(selection), expert_modules=EXPERT_MODULE_COUNT,
                encode_operations=EXPERT_MODULE_COUNT + 2 * len(selection),
                expert_source_bytes=expert_source_bytes, expert_k4_bytes=expert_bytes,
                expert_k4_bits_per_weight=8 * expert_bytes / (EXPERT_MODULE_COUNT * 1536 * 1536),
                expert_tp2_bytes=expert_bytes // 2,
                shared_native_bytes=shared_native, engram_bf16_bytes=sum(ENGRAM_ROWS.values()) * 192,
                engram_fp8_bytes=engram_fp8,
                core_bytes=dict(original=original_bytes, k8=k8_bytes, k6=k6_bytes),
                view_bytes=dict(original=shared + original_bytes, k8=shared + k8_bytes, k6=shared + k6_bytes),
                physical_payload_bytes=shared + original_bytes + k8_bytes + k6_bytes - aliased_bytes + engram_fp8)
