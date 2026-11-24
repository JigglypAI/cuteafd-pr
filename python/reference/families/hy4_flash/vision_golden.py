#!/usr/bin/env python3
"""Hy4-flash tower golden: per-stage dumps (serving BF16 and F32) and the pinned-HF cross-check.

Outputs under ``--out``. ``<fixture>/<mode>/<stage>.bf16`` or ``.f32`` hold row-major
little-endian arrays: ``embed``, ``layer00..layer26``, ``merger_norm``, ``output``.
``tower.json`` records shapes, sha256s, the serving-vs-F32 per-stage cosine and max-abs, and the
HF cross-check. The HF tower is fed SGLang raster rows. Its BF16 RMSNorm rounds before
the weight, so its output stage is expected to differ at BF16 rounding scale only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))


def metrics(actual, expected):
    a, e = actual.double().flatten(), expected.double().flatten()
    return {"cosine": float((a @ e) / (a.norm() * e.norm()).clamp_min(1e-300)),
            "max_abs": float((a - e).abs().max()), "rel_l2": float((a - e).norm() / e.norm().clamp_min(1e-300)),
            "expected_max_abs": float(e.abs().max())}


def dump(folder, name, value):
    import numpy as np
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    if value.dtype == torch.bfloat16:
        data, suffix = value.contiguous().cpu().view(torch.uint16).numpy().astype("<u2").tobytes(), "bf16"
    else:
        data, suffix = value.float().contiguous().cpu().numpy().astype("<f4").tobytes(), "f32"
    (folder / f"{name}.{suffix}").write_bytes(data)
    return {"shape": list(value.shape), "dtype": suffix, "sha256": hashlib.sha256(data).hexdigest()}


def write_raw(folder, rgb8, grid, stages):
    """Numpy-free engine fixture: rgb.u8 (H x W x 3), F32 row-major stages, meta.json."""
    import hy4_flash_media as media
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rgb.u8").write_bytes(rgb8.tobytes())
    table = media.normalization_table().astype("<f4").tobytes()
    named = [("0", stages.patch), ("pos", stages.embed)] + [(str(i + 1), x) for i, x in enumerate(stages.layers)] + [
        ("28", stages.merger_norm), ("out", stages.output)]
    files = {}
    for key, value in named:
        data = value.float().contiguous().cpu().numpy().astype("<f4").tobytes()
        (folder / f"stage_{key}.f32").write_bytes(data)
        files[key] = {"rows": int(value.shape[0]), "width": int(value.shape[1]), "sha256": hashlib.sha256(data).hexdigest()}
    meta = {"schema": "cuteafd.hy4f.vision-stages/1", "grid": [grid[1], grid[2]], "canvas": [rgb8.shape[0], rgb8.shape[1]],
            "rgb_sha256": hashlib.sha256(rgb8.tobytes()).hexdigest(), "lut_sha256": hashlib.sha256(table).hexdigest(),
            "lut": "(v/255 - 0.5)/0.5 in F32, 256 entries per channel", "mode": "serving (BF16, values widened to F32)",
            "pixel_order": "raster (h, w) patches, (c, py, px) inside each", "stages": [k for k, _ in named], "files": files}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return str(folder)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True, help="vision panel windows.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hf", action="store_true", help="cross-check against the pinned Transformers tower")
    p.add_argument("--raw-dir", type=Path, help="engine-test dumps: <name>/{rgb.u8,meta.json,stage_*.f32}, batch/")
    a = p.parse_args()
    import torch
    from PIL import Image
    import hy4_flash_media as media
    from families.hy4_flash import sources, vision
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    manifest = json.loads(a.panel.read_bytes())
    weights = vision.load_weights(a.snapshot)
    hf = vision.hf_tower(a.snapshot, weights) if a.hf else None
    if hf is not None:
        from families.hy4_flash.golden import load_pinned_transformers
        load_pinned_transformers()  # verifies the pinned modeling source digests
    report = {"schema": "cuteafd.hy4f.vision-golden/1", "set_sha256": manifest["set_sha256"],
              "snapshot_revision": a.snapshot.name, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
              "sources": {"sglang": sources.SGLANG_REVISION, "transformers": sources.TRANSFORMERS_REVISION},
              "vision_sha256": hashlib.sha256(Path(vision.__file__).read_bytes()).hexdigest(),
              "media_sha256": hashlib.sha256(Path(media.__file__).read_bytes()).hexdigest(), "fixtures": {}}
    root = a.panel.parent
    batch_inputs = []
    for window in manifest["windows"]:
        for span in window.get("media", []):
            name = window["id"]
            with Image.open(io.BytesIO(media.read_fixture(root, span))) as source:
                rgb8, grid = media.prepare_rgb(source)
                torchvision_diag = None
                if source.size != (rgb8.shape[1], rgb8.shape[0]):
                    import numpy as np
                    import torchvision.transforms.v2.functional as tvF
                    t = torch.from_numpy(np.asarray(source.convert("RGB"))).permute(2, 0, 1)
                    tv = tvF.resize(t, [rgb8.shape[0], rgb8.shape[1]], interpolation=tvF.InterpolationMode.BICUBIC,
                                    antialias=True).permute(1, 2, 0).numpy()
                    diff = np.abs(tv.astype(np.int16) - rgb8.astype(np.int16))
                    torchvision_diag = {"max_abs_u8": int(diff.max()), "mean_abs_u8": float(diff.mean()),
                                        "differing_fraction": float((diff > 0).mean())}
            pixels = torch.from_numpy(media.patchify(rgb8)).cuda()
            entry = {"grid": grid, "rows": vision.output_rows(grid), "source_size": list(source.size),
                     "resized": [rgb8.shape[1], rgb8.shape[0]], "pil_vs_torchvision_resize": torchvision_diag,
                     "modes": {}, "serving_vs_f32": {}}
            stages = {}
            for mode in vision.MODES:
                tick = time.monotonic()
                with torch.inference_mode():
                    s = vision.encode(weights, pixels, grid, mode=mode)
                torch.cuda.synchronize()
                stages[mode] = s
                folder = a.out / name / mode
                files = {"embed": dump(folder, "embed", s.embed)}
                for i, layer in enumerate(s.layers):
                    files[f"layer{i:02d}"] = dump(folder, f"layer{i:02d}", layer)
                files["merger_norm"] = dump(folder, "merger_norm", s.merger_norm)
                files["output"] = dump(folder, "output", s.output)
                entry["modes"][mode] = {"seconds": time.monotonic() - tick, "files": files}
            sv, f32 = stages["serving"], stages["f32"]
            pairs = [("embed", sv.embed, f32.embed)] + [(f"layer{i:02d}", x, y) for i, (x, y) in
                     enumerate(zip(sv.layers, f32.layers))] + [("merger_norm", sv.merger_norm, f32.merger_norm),
                     ("output", sv.output, f32.output)]
            entry["serving_vs_f32"] = {k: metrics(x, y) for k, x, y in pairs}
            if hf is not None:
                with torch.inference_mode():
                    captured = {}
                    hooks = [hf.embeddings.register_forward_hook(lambda m, i, o: captured.__setitem__("embed", o[0]))]
                    for j, layer in enumerate(hf.layers):
                        hooks.append(layer.register_forward_hook(
                            lambda m, i, o, j=j: captured.__setitem__(f"layer{j:02d}", (o[0] if isinstance(o, tuple) else o)[0])))
                    result = hf(pixels.to(torch.bfloat16), grid_thw=torch.tensor([grid], device="cuda"))
                    for hook in hooks:
                        hook.remove()
                captured["output"] = result.pooler_output[0]
                ours = {"embed": sv.embed, "output": sv.output, **{f"layer{i:02d}": x for i, x in enumerate(sv.layers)}}
                entry["serving_vs_hf"] = {k: metrics(ours[k], captured[k]) for k in ours if k in captured}
                del captured, result
            if a.raw_dir is not None:
                entry["raw"] = write_raw(a.raw_dir / name, rgb8, grid, stages["serving"])
            batch_inputs.append((name, pixels, grid, stages["serving"].output))
            report["fixtures"][name] = entry
            print(name, grid, "serving-vs-f32 output cos %.6f" % entry["serving_vs_f32"]["output"]["cosine"],
                  *(["hf layer26 maxabs %.4g output maxabs %.4g cos %.7f" % (
                      entry["serving_vs_hf"]["layer26"]["max_abs"], entry["serving_vs_hf"]["output"]["max_abs"],
                      entry["serving_vs_hf"]["output"]["cosine"])] if hf is not None else []), flush=True)
            del stages
            torch.cuda.empty_cache()
    if batch_inputs:
        with torch.inference_mode():
            batch = vision.encode_batch(weights, [b[1] for b in batch_inputs], [b[2] for b in batch_inputs])
        rows = torch.cat([s.output for s in batch])
        singles = torch.cat([b[3] for b in batch_inputs])
        report["batch_vs_singles"] = {"order": [b[0] for b in batch_inputs], **metrics(rows, singles),
                                      "bitwise_equal": bool(torch.equal(rows, singles))}
        if a.raw_dir is not None:
            folder = a.raw_dir / "batch"
            folder.mkdir(parents=True, exist_ok=True)
            data = rows.float().contiguous().cpu().numpy().astype("<f4").tobytes()
            (folder / "stage_out.f32").write_bytes(data)
            (folder / "meta.json").write_text(json.dumps({"order": [b[0] for b in batch_inputs],
                "grids": [[b[2][1], b[2][2]] for b in batch_inputs], "rows": [vision.output_rows(b[2]) for b in batch_inputs],
                "width": vision.OUT, "sha256": hashlib.sha256(data).hexdigest(),
                "bitwise_equal_to_singles": report["batch_vs_singles"]["bitwise_equal"]}, indent=1) + "\n")
        print("batch vs singles", report["batch_vs_singles"], flush=True)
    report["seconds"] = time.monotonic() - started
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tower.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{a.out / 'tower.json'}: {report['seconds']:.1f}s")


if __name__ == "__main__":
    main()
