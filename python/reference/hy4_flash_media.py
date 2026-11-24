"""Hy4-flash image processor, M-RoPE positions and bundled-tower features for media goldens.

Processor: SGLang's ``smart_resize`` arithmetic (factor 32 = patch 16 x merge 2,
4096..4194304 pixels; Python ``round``, which ties to even). The resize is
Pillow BICUBIC on uint8 RGB, as the pinned Transformers PIL processor does and
as the engine does. Normalization is ``(v/255 - 0.5)/0.5`` in F32 through a
256-entry table. Patch rows are raster over the (h, w) patch grid, each row
``(c, py, px)``, matching SGLang's ``patchify``.

Positions follow SGLang's ``get_hy_v4_vl_mrope_positions``. An image span is
``h/2 * (w/2+1)`` placeholder rows (``image_token_id``) enclosed by
``image_start``/``image_end``. Row ``r`` and column ``c`` get ``[s, s+r, s+c]``,
where ``s`` is the running text position. The next text position continues at
``s + max(rows, cols)``.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

from fidelity_media import publish_immutable, read_fixture, validate_media, write_features
from fidelity_windows import canonical

IMAGE_ID, START_ID, END_ID, VIDEO_ID = 120120, 120118, 120119, 120124
WIDTH, PATCH, MERGE = 3072, 16, 2
FACTOR = PATCH * MERGE
MIN_PIXELS, MAX_PIXELS = 4096, 4194304
MAX_ASPECT_RATIO = 200


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """SGLang hy_v4_vl_processing.smart_resize, verbatim arithmetic."""
    if min(height, width, factor, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError("Image dimensions and pixel limits must be positive and ordered")
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(f"Image aspect ratio must not exceed {MAX_ASPECT_RATIO}")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    h, w = max(h, factor), max(w, factor)
    if h * w > max_pixels:
        raise ValueError("Image aspect ratio and patch rounding exceed max_pixels")
    return h, w


def normalization_table():
    import numpy as np
    values = (np.arange(256, dtype=np.float64) * (1.0 / 255.0)).astype(np.float32)
    return (values - np.float32(0.5)) / np.float32(0.5)


def prepare_rgb(image):
    """PIL image (any mode) -> (resized uint8 HxWx3, grid [1, h, w])."""
    import numpy as np
    from PIL import Image
    rgb = image.convert("RGB")
    height, width = smart_resize(rgb.height, rgb.width)
    if (height, width) != (rgb.height, rgb.width):
        rgb = rgb.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(rgb, dtype=np.uint8).copy(), [1, height // PATCH, width // PATCH]


def patchify(rgb8):
    """uint8 HxWx3 -> F32 [h*w, 3*16*16] raster rows, (c, py, px) inside each row."""
    import numpy as np
    table = normalization_table()
    height, width, _ = rgb8.shape
    if height % FACTOR or width % FACTOR:
        raise ValueError("image dimensions must align with patch_size * merge_size")
    x = table[rgb8].transpose(2, 0, 1)  # c, H, W
    h, w = height // PATCH, width // PATCH
    return np.ascontiguousarray(x.reshape(3, h, PATCH, w, PATCH).transpose(1, 3, 0, 2, 4).reshape(h * w, 3 * PATCH * PATCH))


def span_len(grid):
    t, h, w = grid
    if t != 1 or h % MERGE or w % MERGE or h <= 0 or w <= 0:
        raise ValueError("Hy4 image grid must be [1, even h, even w]")
    return (h // MERGE) * (w // MERGE + 1)


def processor_contract():
    """validate_media's merged-extent contract: one newline column per merged row, same placeholder id."""
    return {"merge": MERGE, "row_markers": 1, "uniform_span": True, "image_token_id": IMAGE_ID}


def validate_window(window):
    """Spans are h/2*(w/2+1) image_pad rows between image_start/image_end; nothing else uses those ids."""
    tokens, spans = window["tokens"], window.get("media", [])
    covered, starts, ends = set(), [], []
    previous = 0
    for span in spans:
        if span.get("kind") != "image":
            raise ValueError("Hy4-flash media supports images only")
        start, length = span["start"], span["len"]
        end = start + length
        if length != span_len(span["grid"]):
            raise ValueError("Hy4 image span differs from its merged grid plus newline column")
        if (start < previous + 1 or end >= len(tokens) or tokens[start - 1] != START_ID or tokens[end] != END_ID
                or any(t != IMAGE_ID for t in tokens[start:end]) or end > window["score_from"]):
            raise ValueError("Hy4 image span requires native 120118/120120/120119 ids before scoring")
        covered.update(range(start, end))
        starts.append(start - 1)
        ends.append(end)
        previous = end
    if ({i for i, t in enumerate(tokens) if t == IMAGE_ID} != covered
            or [i for i, t in enumerate(tokens) if t == START_ID] != starts
            or [i for i, t in enumerate(tokens) if t == END_ID] != ends
            or VIDEO_ID in tokens):
        raise ValueError("Hy4 native image tokens differ from the pinned span list")
    return spans


def rope_positions(tokens, spans):
    """[3, T] int64 M-RoPE (T, H, W) positions. Text rows carry the same value on all axes."""
    import numpy as np
    positions = np.zeros((3, len(tokens)), dtype=np.int64)
    cursor = text = 0
    for span in sorted(spans, key=lambda s: s["start"]):
        start = span["start"]
        count = start - cursor
        positions[:, cursor:start] = np.arange(count) + text
        text += count
        _, h, w = span["grid"]
        rows, cols = h // MERGE, w // MERGE + 1
        if span["len"] != rows * cols:
            raise ValueError("Hy4 rope span extent differs from grid")
        r, c = np.divmod(np.arange(rows * cols), cols)
        positions[0, start:start + span["len"]] = text
        positions[1, start:start + span["len"]] = text + r
        positions[2, start:start + span["len"]] = text + c
        text += max(rows, cols)
        cursor = start + span["len"]
    positions[:, cursor:] = np.arange(len(tokens) - cursor) + text
    return positions


def inject_embeddings(embedding, window, features, hc_mult=4):
    """Overwrite single-width embedding rows, then replicate to the iHC streams.

    SGLang and HF scatter image features into ``inputs_embeds`` before the text model
    expands every row to ``hc_mult`` streams (``unsqueeze(2).expand``), exactly as text rows are."""
    import torch
    if embedding.ndim != 2 or embedding.shape != (len(window["tokens"]), WIDTH):
        raise ValueError("Hy4 embedding must be [T, 3072] before stream expansion")
    for span in validate_window(window):
        value = features[span["key"]]
        if tuple(value.shape) != (span["len"], WIDTH) or not torch.isfinite(value).all():
            raise ValueError("invalid Hy4 feature rows")
        embedding[span["start"]:span["start"] + span["len"]] = value.to(embedding.device, embedding.dtype)
    return embedding[:, None].expand(-1, hc_mult, -1).contiguous()


def validate_spans(manifest, snapshot):
    config = json.loads((Path(snapshot) / "config.json").read_bytes())
    if ((config.get("image_token_id"), config.get("image_start_token_id"), config.get("image_end_token_id"))
            != (IMAGE_ID, START_ID, END_ID) or config["vision_config"]["spatial_merge_size"] != MERGE
            or config["vision_config"]["patch_size"] != PATCH):
        raise ValueError("unsupported Hy4-flash image ids or geometry")
    for window in manifest["windows"]:
        validate_media(window.get("media", []), window["tokens"], window["roles"], window["score_from"],
                       processor=processor_contract())
        validate_window(window)


def _source_hashes():
    from families.hy4_flash import sources, vision
    return {"vision_sha256": hashlib.sha256(Path(vision.__file__).read_bytes()).hexdigest(),
            "media_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sglang_revision": sources.SGLANG_REVISION, "sglang_files": sources.SGLANG_VISION_FILES}


def provenance(args, manifest):
    import PIL
    snapshot = Path(args.snapshot)
    return {**_source_hashes(), "snapshot_revision": snapshot.name, "pillow": PIL.__version__,
            "tower_dtype": getattr(args, "tower_dtype", "bf16"), "resample": "PIL BICUBIC",
            **{key + "_sha256": hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
               for key, name in (("config", "config.json"), ("processor", "processor_config.json"),
                                 ("tokenizer", "tokenizer.json"), ("index", "model.safetensors.index.json"))}}


def encode_fixture(weights, root, span, mode="serving"):
    import torch
    from PIL import Image
    from families.hy4_flash import vision
    with Image.open(io.BytesIO(read_fixture(root, span))) as source:
        rgb8, grid = prepare_rgb(source)
    if grid != span["grid"]:
        raise ValueError(f"Hy4 processor grid {grid} differs from pinned span {span['grid']}")
    pixels = torch.from_numpy(patchify(rgb8))
    with torch.inference_mode():
        stages = vision.encode(weights, pixels, grid, mode=mode)
    return stages


def window_features(args, manifest):
    """Return ``{key: [len, 3072] tensor}`` from the serving-mode (BF16) tower."""
    import torch
    from families.hy4_flash import vision
    validate_spans(manifest, args.snapshot)
    identity = provenance(args, manifest)
    root = getattr(args, "media_root", None) or args.windows.parent
    mode = "serving" if getattr(args, "tower_dtype", "bf16") == "bf16" else "f32"
    weights = vision.load_weights(args.snapshot)
    features, bindings, entries = {}, {}, []
    try:
        for window in manifest["windows"]:
            for span in window.get("media", []):
                binding = (tuple(span["grid"]), span["fixture"]["sha256"])
                if span["key"] in features:
                    if bindings[span["key"]] != binding:
                        raise ValueError("one Hy4 media key binds different fixtures/grids")
                    continue
                output = encode_fixture(weights, root, span, mode).output
                features[span["key"]], bindings[span["key"]] = output.cpu(), binding
                out = getattr(args, "media_features_out", None)
                if out and not getattr(args, "_prefix_probe", False):
                    entries.append(write_features(out, span, output.bfloat16().contiguous().cpu().view(torch.uint16).numpy(),
                                                  tower_dtype="bf16" if mode == "serving" else "fp32", identity=identity))
    finally:
        del weights
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    out = getattr(args, "media_features_out", None)
    if out and not getattr(args, "_prefix_probe", False):
        index = {"schema": "cuteafd.media.features.index/1", "family": "hy4_flash",
                 "checkpoint": manifest["checkpoint"], "tokenizer_sha256": identity["tokenizer_sha256"],
                 "set_sha256": manifest["set_sha256"], "features": [entry["key"] for entry in entries]}
        publish_immutable(Path(out) / "features.json", canonical(index) + b"\n")
    return features, identity
