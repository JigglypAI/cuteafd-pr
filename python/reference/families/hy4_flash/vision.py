"""Independent Hy4-flash 27-layer ViT reference over the checkpoint's own ``vit.*`` tensors.

Ports Tencent's serving tower (SGLang ``models/hunyuan_v4_vl.py`` at
``sources.SGLANG_REVISION``), not the engine. Input is the SGLang patchify
layout: raster patch rows over the (h, w) patch grid, each row ``(c, py, px)``.
The merged LM rows follow, ``h/2 * (w/2 + 1)`` of them, with the checkpoint's
own ``image_newline`` closing every merged row.

Two arithmetic modes:

- ``"serving"``: BF16 weights and activations in SGLang's order. The position
  table and its bilinear resize run in F32, then are cast to BF16 and added to the
  BF16 conv+bias. LayerNorm (eps 1e-5) is ``nn.LayerNorm`` in BF16. Attention is
  SDPA (math backend) per image, scale ``72**-0.5``. GELU is exact. The merger
  RMSNorm upcasts to F32 and rounds once to BF16 after the weight (SGLang's
  fused kernel).
- ``"f32"``: the same graph with every weight upcast and IEEE-F32 math
  (TF32 off); outputs stay F32.

``encode(...)`` returns ``Stages``. ``embed`` is ``[n, 1152]`` (patch + position);
``layers[i]`` is the ``[n, 1152]`` output of block ``i`` (27 entries);
``merger_norm`` is ``[n, 1152]``, the merger RMSNorm output with rows in raster
order; ``output`` is ``[h/2 * (w/2+1), 3072]``. ``n = h * w``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
import torch.nn.functional as F

PREFIX = "vit."
HIDDEN, HEADS, HEAD_DIM, INTER, LAYERS, PATCH, MERGE, OUT = 1152, 16, 72, 4304, 27, 16, 2, 3072
EDGE = 128  # max_image_size 2048 / patch 16
EPS = 1e-5
MODES = ("serving", "f32")


def contract():
    """Exact ``vit.*`` tensor names, shapes and BF16 storage."""
    items = {"embeddings.patch_embedding.weight": [HIDDEN, 3, PATCH, PATCH],
             "embeddings.patch_embedding.bias": [HIDDEN],
             "embeddings.position_embedding.weight": [EDGE * EDGE + 1, HIDDEN],
             "perceive.image_newline": [OUT], "perceive.norm.weight": [HIDDEN],
             "perceive.proj.0.weight": [HIDDEN * 4, HIDDEN, MERGE, MERGE], "perceive.proj.0.bias": [HIDDEN * 4],
             "perceive.proj.2.weight": [OUT, HIDDEN * 4, 1, 1], "perceive.proj.2.bias": [OUT]}
    for i in range(LAYERS):
        p = f"layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            items[p + norm + ".weight"] = items[p + norm + ".bias"] = [HIDDEN]
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            items[p + f"self_attn.{proj}.weight"] = [HIDDEN, HIDDEN]
            items[p + f"self_attn.{proj}.bias"] = [HIDDEN]
        items[p + "mlp.dense_h_to_4h.weight"] = [INTER, HIDDEN]
        items[p + "mlp.dense_h_to_4h.bias"] = [INTER]
        items[p + "mlp.dense_4h_to_h.weight"] = [HIDDEN, INTER]
        items[p + "mlp.dense_4h_to_h.bias"] = [HIDDEN]
    return {PREFIX + k: v for k, v in items.items()}


def validate_config(raw):
    v = raw["vision_config"]
    expected = dict(hidden_size=HIDDEN, intermediate_size=INTER, num_attention_heads=HEADS,
                    num_key_value_heads=HEADS, num_hidden_layers=LAYERS, patch_size=PATCH,
                    spatial_merge_size=MERGE, out_hidden_size=OUT, max_image_size=2048, rms_norm_eps=EPS,
                    hidden_act="gelu", interpolate_mode="bilinear", projector_type="rmsnorm_mlp_adapter",
                    cat_extra_token=0, temporal_patch_size=1, num_channels=3)
    bad = {k: v.get(k) for k, want in expected.items() if v.get(k) != want}
    if bad or v.get("use_2d_rope", False) or v.get("perceive_intermediate_size") not in (None, 0):
        raise ValueError(f"unsupported Hy4-flash vision_config: {bad}")
    if v.get("perceive_post_norm") is not False:
        raise ValueError("Hy4-flash vision perceive_post_norm must be false")


def load_weights(snapshot, device="cuda"):
    """BF16 tensors exactly as stored; the tower inventory must match ``contract()``."""
    from safetensors import safe_open
    snapshot = Path(snapshot)
    validate_config(json.loads((snapshot / "config.json").read_bytes()))
    index = json.loads((snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
    want = contract()
    names = {n for n in index if n.startswith(PREFIX)}
    if names != set(want):
        raise ValueError(f"vit.* inventory differs: {sorted(names ^ set(want))[:8]}")
    weights = {}
    for shard in sorted({index[n] for n in names}):
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as reader:
            for name in sorted(n for n in names if index[n] == shard):
                value = reader.get_tensor(name)
                if value.dtype != torch.bfloat16 or list(value.shape) != want[name]:
                    raise ValueError(f"{name}: {value.dtype} {list(value.shape)}; need BF16 {want[name]}")
                weights[name.removeprefix(PREFIX)] = value.to(device)
    return weights


def output_rows(grid):
    t, h, w = grid
    if t != 1 or h % MERGE or w % MERGE or h <= 0 or w <= 0 or h > EDGE or w > EDGE:
        raise ValueError("Hy4 image grid must be [1, even h<=128, even w<=128]")
    return (h // MERGE) * (w // MERGE + 1)


@dataclass
class Stages:
    embed: torch.Tensor = None
    layers: list = field(default_factory=list)
    merger_norm: torch.Tensor = None
    output: torch.Tensor = None


def _rms(x, weight, mode):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * weight.float()
    return y if mode == "f32" else y.to(torch.bfloat16)


def _positions(table, h, w, dtype):
    """F32 bilinear resize of the 128x128 grid with scale_factor (n+0.1)/128 (row 0 is the unused cls)."""
    grid = table[1:].float().reshape(1, EDGE, EDGE, HIDDEN).permute(0, 3, 1, 2)
    resized = F.interpolate(grid, scale_factor=((h + 0.1) / EDGE, (w + 0.1) / EDGE),
                            mode="bilinear", align_corners=False)
    if resized.shape[-2:] != (h, w):
        raise ValueError("position interpolation extent differs from grid")
    return resized.flatten(2).transpose(1, 2)[0].to(dtype)


def encode(weights, pixels, grid, *, mode="serving", observer=None):
    """``pixels``: ``[h*w, 3*16*16]`` normalized raster patches (F32 or BF16). One image."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    _, h, w = grid
    rows = output_rows(grid)
    n = h * w
    if tuple(pixels.shape) != (n, 3 * PATCH * PATCH):
        raise ValueError("pixel rows differ from grid")
    dtype = torch.float32 if mode == "f32" else torch.bfloat16
    W = {k: v.to(dtype) for k, v in weights.items()}
    stages = Stages()

    def keep(name, value):
        if observer is not None:
            observer(name, value)
        return value

    x = F.conv2d(pixels.to(W["embeddings.patch_embedding.weight"].device, dtype).reshape(n, 3, PATCH, PATCH),
                 W["embeddings.patch_embedding.weight"], W["embeddings.patch_embedding.bias"], stride=PATCH)
    x = x.reshape(n, HIDDEN)
    x = x + _positions(weights["embeddings.position_embedding.weight"], h, w, dtype)
    stages.embed = keep("embed", x)
    for i in range(LAYERS):
        p = f"layers.{i}."
        res = x
        y = F.layer_norm(x, (HIDDEN,), W[p + "input_layernorm.weight"], W[p + "input_layernorm.bias"], EPS)
        q, k, v = (F.linear(y, W[p + f"self_attn.{n_}_proj.weight"], W[p + f"self_attn.{n_}_proj.bias"])
                   .reshape(n, HEADS, HEAD_DIM).transpose(0, 1) for n_ in "qkv")
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            o = F.scaled_dot_product_attention(q[None], k[None], v[None], scale=HEAD_DIM ** -0.5)[0]
        o = F.linear(o.transpose(0, 1).reshape(n, HIDDEN), W[p + "self_attn.o_proj.weight"], W[p + "self_attn.o_proj.bias"])
        x = res + o
        y = F.layer_norm(x, (HIDDEN,), W[p + "post_attention_layernorm.weight"], W[p + "post_attention_layernorm.bias"], EPS)
        y = F.gelu(F.linear(y, W[p + "mlp.dense_h_to_4h.weight"], W[p + "mlp.dense_h_to_4h.bias"]))
        x = x + F.linear(y, W[p + "mlp.dense_4h_to_h.weight"], W[p + "mlp.dense_4h_to_h.bias"])
        stages.layers.append(keep(f"layer{i}", x))
    y = _rms(x, weights["perceive.norm.weight"], mode)
    stages.merger_norm = keep("merger_norm", y)
    y = y.reshape(1, h, w, HIDDEN).permute(0, 3, 1, 2)
    y = F.gelu(F.conv2d(y, W["perceive.proj.0.weight"], W["perceive.proj.0.bias"], stride=MERGE))
    y = F.conv2d(y, W["perceive.proj.2.weight"], W["perceive.proj.2.bias"])
    newline = W["perceive.image_newline"].reshape(1, OUT, 1, 1).expand(1, OUT, h // MERGE, 1)
    y = torch.cat((y, newline), dim=-1).flatten(2).transpose(1, 2)[0]
    if y.shape != (rows, OUT):
        raise ValueError("merged output rows differ from grid")
    stages.output = keep("output", y)
    return stages


def hf_tower(snapshot, weights, device="cuda"):
    """The pinned Transformers PR tower (BF16, SDPA) for the cross-check only."""
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as hf
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashConfig
    raw = json.loads((Path(snapshot) / "config.json").read_bytes())
    config = HYV4FlashConfig.from_dict(raw).vision_config
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        tower = hf.HYV4FlashVisionTransformer(config)
    renames = {"input_layernorm": "layer_norm1", "post_attention_layernorm": "layer_norm2",
               "mlp.dense_h_to_4h": "mlp.fc1", "mlp.dense_4h_to_h": "mlp.fc2"}
    state = {}
    for name, value in weights.items():
        key = name
        for old, new in renames.items():
            key = key.replace(old, new)
        key = (key.replace("perceive.norm.", "patch_merger.before_rms.").replace("perceive.proj.0.", "patch_merger.proj_conv.")
               .replace("perceive.proj.2.", "patch_merger.proj_out.").replace("perceive.", "patch_merger."))
        state[key] = value.to(device)
    result = tower.load_state_dict(state, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"HF tower state mapping differs: {result}")
    return tower.eval()
