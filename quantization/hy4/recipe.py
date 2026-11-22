"""Full Hy4 direct MC geometry, including the five DFly2 layers."""
from __future__ import annotations

import re

from glm5.core_k6.recipe import packed_bytes
from hy4_flash.recipe import MUL1_MULTIPLIER, SUFFIXES, _geometry, packed_specs

BASE_REPO = "tencent/Hy4"
BASE_REVISION = "c10c0de5b6f850703fb0a90e65d1c32b558b9167"
REFERENCE_REVISION = "2f1bc5e321b9a736200a590d6e7743ccdeb60772"
TARGET_REPO = "wrldsuksgo2mars/Hy4-EXL3-K4-MC"
BASE_PAYLOAD_BYTES = 1_548_929_595_126
BASE_TENSOR_COUNT, SHARD_COUNT = 2710, 245
LAYER_COUNT, EXPERTS_PER_LAYER, MTP_LAYER_COUNT = 78, 256, 5
DFLY_TARGET_LAYERS = (61, 65, 69, 73, 77)
FULL_INDEX_LAYERS = (0, *range(1, LAYER_COUNT, 4))
ENGRAM_ROWS = {}
ENGRAM_TABLES = {}
EXPERT_MODULE_COUNT = (LAYER_COUNT - 1) * EXPERTS_PER_LAYER * 3
BASE_CORE_MODULE_COUNT = LAYER_COUNT * 9 + 1
MTP_CORE_MODULE_COUNT = MTP_LAYER_COUNT * 9
CORE_MODULE_COUNT = BASE_CORE_MODULE_COUNT + MTP_CORE_MODULE_COUNT
ENCODE_COUNT = EXPERT_MODULE_COUNT + 2 * CORE_MODULE_COUNT
_EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)")


def reviewed_core_modules(*, core_bits=6, include_head=True, include_mtp=True, include_kv_b=True):
    if type(core_bits) is not int or core_bits not in (6, 8):
        raise ValueError("Hy4 compact core requires K6 or K8")
    selected = {}
    for family, count, group in (("layers", LAYER_COUNT, "core"),
                                 ("dfly_layers", MTP_LAYER_COUNT if include_mtp else 0, "mtp")):
        for layer in range(count):
            prefix = f"model.{family}.{layer}."
            mlp = "mlp.shared_experts." if family == "layers" and layer else "mlp."
            intermediate = 2048 if mlp == "mlp.shared_experts." else 18432
            shapes = {"self_attn.q_a_proj": (6144, 2048), "self_attn.q_b_proj": (2048, 16384),
                      "self_attn.kv_a_proj_with_mqa": (6144, 576), "self_attn.linear_gate": (6144, 16384),
                      "self_attn.o_proj": (16384, 6144), mlp + "gate_proj": (6144, intermediate),
                      mlp + "up_proj": (6144, intermediate), mlp + "down_proj": (intermediate, 6144)}
            if include_kv_b:
                shapes["self_attn.kv_b_proj"] = (512, 28672)
            selected.update({prefix + leaf: _geometry(inputs, outputs, core_bits, group)
                             for leaf, (inputs, outputs) in shapes.items()})
    if include_head:
        selected["lm_head"] = _geometry(6144, 120832, core_bits, "core")
    return selected


def reviewed_expert_modules(layer=None):
    if layer is not None and (type(layer) is not int or not 0 <= layer < LAYER_COUNT):
        raise ValueError("unreviewed Hy4 layer")
    layers = range(1, LAYER_COUNT) if layer is None else ([layer] if layer else [])
    return {f"model.layers.{i}.mlp.experts.{e}.{leaf}": _geometry(inputs, outputs, 4, "routed")
            for i in layers for e in range(EXPERTS_PER_LAYER)
            for leaf, inputs, outputs in (("gate_proj", 6144, 2048), ("up_proj", 6144, 2048),
                                          ("down_proj", 2048, 6144))}


def source_tensor(module):
    if ".mlp.experts." not in module:
        return module + ".weight", None, None
    match = _EXPERT.fullmatch(module)
    if match is None:
        raise ValueError(f"unreviewed expert projection {module}")
    layer, expert, leaf = match.groups()
    if not (1 <= int(layer) < LAYER_COUNT and 0 <= int(expert) < EXPERTS_PER_LAYER):
        raise ValueError(f"unreviewed expert {module}")
    prefix = f"model.layers.{layer}.mlp.experts."
    if leaf == "down_proj":
        return prefix + "down_proj", int(expert), None
    return prefix + "gate_up_proj", int(expert), (0, 2048) if leaf == "gate_proj" else (2048, 4096)


def expert_source_specs():
    return {f"model.layers.{i}.mlp.experts.{leaf}": ("BF16", shape)
            for i in range(1, LAYER_COUNT)
            for leaf, shape in (("gate_up_proj", [256, 4096, 6144]), ("down_proj", [256, 6144, 2048]))}


def size_report(source, selection=None, *, k6_drafter_bits=6):
    if k6_drafter_bits not in (6, 8):
        raise ValueError("unsupported Hy4 K6 drafter selection")
    modules = reviewed_core_modules() if selection is None else selection
    expert_bytes = (LAYER_COUNT - 1) * sum(packed_bytes(g) for g in reviewed_expert_modules(1).values())
    native_experts = sum(source.sizes[name] for name in expert_source_specs())
    original = sum(source.sizes[name + ".weight"] for name in modules)
    shared_native = source.payload_bytes - native_experts - original
    k8 = sum(packed_bytes(g, bits=8) for g in modules.values())
    k6 = sum(packed_bytes(g, bits=k6_drafter_bits if g["group"] == "mtp" else 6) for g in modules.values())
    aliased = sum(packed_bytes(g, bits=8) for g in modules.values()
                  if g["group"] == "mtp" and k6_drafter_bits == 8)
    core = dict(original=original, k8=k8, k6=k6)
    shared = shared_native + expert_bytes
    return dict(core_modules=len(modules), expert_modules=EXPERT_MODULE_COUNT, encode_operations=ENCODE_COUNT,
                expert_source_bytes=native_experts, expert_k4_bytes=expert_bytes,
                expert_k4_bits_per_weight=8 * expert_bytes / (EXPERT_MODULE_COUNT * 6144 * 2048),
                expert_tp2_bytes=expert_bytes // 2, shared_native_bytes=shared_native, core_bytes=core,
                view_bytes={key: shared + count for key, count in core.items()},
                physical_payload_bytes=shared + original + k8 + k6 - aliased)
