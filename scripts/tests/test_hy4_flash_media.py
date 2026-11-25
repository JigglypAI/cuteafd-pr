"""CPU contracts for the Hy4-flash vision reference: processor, M-RoPE positions, spans, tower."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
import hy4_flash_media as media


def window(grid=(1, 4, 6), before=2, after=3):
    length = media.span_len(list(grid))
    tokens = [17] * (before - 1) + [media.START_ID] + [media.IMAGE_ID] * length + [media.END_ID] + [18] * after
    return {"tokens": tokens, "roles": ["ctx"] * (len(tokens) - 1) + ["gen"], "score_from": len(tokens) - 1,
            "media": [{"kind": "image", "start": before, "len": length, "grid": list(grid),
                       "key": "a" * 64, "fixture": {"path": "image.png", "sha256": "b" * 64}}]}


@pytest.mark.parametrize("hw,expected", [
    ((512, 512), (512, 512)), ((768, 512), (768, 512)), ((1024, 1024), (1024, 1024)),
    ((480, 640), (480, 640)), ((500, 500), (512, 512)), ((10, 10), (64, 64)),
    ((4000, 3000), (2336, 1760)), ((48, 48), (64, 64)), ((16, 2000), (32, 736)),
])
def test_smart_resize_matches_sglang_arithmetic(hw, expected):
    assert media.smart_resize(*hw) == expected
    h, w = media.smart_resize(*hw)
    assert h % 32 == 0 and w % 32 == 0 and 4096 <= h * w <= 4194304


def test_smart_resize_rounds_ties_to_even_and_refuses_extreme_aspect():
    # 48/32 = 1.5 -> 2 (even), 80/32 = 2.5 -> 2 (even): Python round, as SGLang.
    assert media.smart_resize(80, 80 * 2) == (64, 160)
    with pytest.raises(ValueError):
        media.smart_resize(1, 201)


def test_normalization_table_is_f32_half_scale():
    table = media.normalization_table()
    assert table.dtype == np.float32 and table[0] == -1.0 and table[255] == 1.0
    assert table[128] == np.float32((np.float32(128 / 255) - np.float32(0.5)) / np.float32(0.5))


def test_patchify_is_raster_c_py_px():
    rgb = np.arange(64 * 32 * 3, dtype=np.uint64).reshape(64, 32, 3) % 256
    rgb = rgb.astype(np.uint8)
    rows = media.patchify(rgb)
    table = media.normalization_table()
    assert rows.shape == (4 * 2, 3 * 16 * 16)
    # patch (r=1, c=1), channel 2, pixel (3, 5)
    assert rows[1 * 2 + 1, 2 * 256 + 3 * 16 + 5] == table[rgb[16 + 3, 16 + 5, 2]]
    with pytest.raises(ValueError):
        media.patchify(np.zeros((48, 32, 3), np.uint8))


def test_prepare_rgb_uses_pillow_bicubic_and_keeps_aligned_sizes():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (512, 768), (10, 20, 30))
    rgb, grid = media.prepare_rgb(image)
    assert rgb.shape == (768, 512, 3) and grid == [1, 48, 32]
    small = Image.new("RGB", (500, 500), (10, 20, 30))
    rgb, grid = media.prepare_rgb(small)
    assert rgb.shape == (512, 512, 3) and grid == [1, 32, 32]
    expected = np.asarray(small.resize((512, 512), Image.Resampling.BICUBIC))
    assert np.array_equal(rgb, expected)


def test_span_len_includes_one_newline_per_merged_row():
    assert media.span_len([1, 32, 32]) == 16 * 17
    assert media.span_len([1, 30, 40]) == 15 * 21
    with pytest.raises(ValueError):
        media.span_len([1, 3, 4])


def test_rope_positions_follow_hf_docstring_layout():
    # HF get_rope_index docstring: merged 2x2 (+newline column) at start 0 -> T 0, H [0,0,0,1,1,1], W [0,1,2]*2
    spans = [{"start": 0, "len": 6, "grid": [1, 4, 4]}]
    tokens = [media.IMAGE_ID] * 6 + [1, 2]
    pos = media.rope_positions(tokens, spans)
    assert pos[0, :6].tolist() == [0] * 6
    assert pos[1, :6].tolist() == [0, 0, 0, 1, 1, 1]
    assert pos[2, :6].tolist() == [0, 1, 2, 0, 1, 2]
    # Next text continues at max(rows, cols) = 3.
    assert pos[:, 6:].tolist() == [[3, 4]] * 3


def test_rope_positions_offsets_text_and_multiple_images():
    w = window(grid=(1, 4, 6), before=3, after=2)
    second = media.span_len([1, 2, 2])
    tokens = w["tokens"][:-2] + [media.START_ID] + [media.IMAGE_ID] * second + [media.END_ID, 5]
    start2 = len(w["tokens"]) - 2 + 1
    spans = w["media"] + [{"start": start2, "len": second, "grid": [1, 2, 2]}]
    pos = media.rope_positions(tokens, spans)
    # text rows 0..2 (start marker at index 2), image from 3: rows 2 cols 4, extent 4
    assert pos[:, :3].tolist() == [[0, 1, 2]] * 3
    assert pos[:, 3].tolist() == [3, 3, 3] and pos[:, 3 + 4 + 3].tolist() == [3, 4, 6]
    assert pos[:, 3 + 8].tolist() == [7, 7, 7]  # end marker resumes at 3 + max(2, 4)
    image2 = pos[:, start2:start2 + second]
    base = pos[0, start2 - 1] + 1
    assert image2.tolist() == [[base, base], [base, base], [base, base + 1]]
    assert pos[:, -2].tolist() == [base + 2] * 3 and pos[:, -1].tolist() == [base + 3] * 3  # end marker, then text
    text = list(range(10))
    assert np.array_equal(media.rope_positions(text, []), np.array([text] * 3))


@pytest.mark.parametrize("mutation", ["start", "end", "unbound", "grid", "video", "after_score", "kind"])
def test_window_validation_fails_closed(mutation):
    w = window()
    span = w["media"][0]
    if mutation == "start":
        w["tokens"][span["start"] - 1] = 17
    elif mutation == "end":
        w["tokens"][span["start"] + span["len"]] = 17
    elif mutation == "unbound":
        w["tokens"][-2] = media.IMAGE_ID
    elif mutation == "grid":
        span["grid"] = [1, 4, 4]
    elif mutation == "video":
        w["tokens"][0] = media.VIDEO_ID
    elif mutation == "after_score":
        w["score_from"] = span["start"] + 1
    else:
        span["kind"] = "audio"
    with pytest.raises(ValueError):
        media.validate_window(w)
    assert media.validate_window(window()) == window()["media"]


def test_validate_media_accepts_uniform_newline_span():
    from fidelity_media import validate_media
    w = window()
    validate_media(w["media"], w["tokens"], w["roles"], w["score_from"], processor=media.processor_contract())
    bad = window()
    bad["media"][0]["len"] -= 1
    with pytest.raises(ValueError):
        validate_media(bad["media"], bad["tokens"], bad["roles"], bad["score_from"], processor=media.processor_contract())


@pytest.mark.container_torch
def test_text_mrope_phase_is_bitwise_the_text_phase():
    torch = pytest.importorskip("torch")
    from families.hy4_common import phase
    from families.hy4_flash import model
    pos = torch.arange(300)
    for base in (10000.0, 10000000.0):
        cos, sin = phase(pos, 64, base)
        cos3, sin3 = model.mrope_phase(pos.expand(3, -1), 64, base)
        assert torch.equal(cos, cos3) and torch.equal(sin, sin3)
        cos1, sin1 = model.mrope_phase(pos, 64, base)
        assert torch.equal(cos, cos1) and torch.equal(sin, sin1)


@pytest.mark.container_torch
def test_mrope_phase_matches_hf_recomposition():
    torch = pytest.importorskip("torch")
    from families.hy4_flash import model
    pos = torch.stack([torch.full((5,), 7), torch.arange(5) + 7, torch.arange(5) * 2 + 7])
    cos, _ = model.mrope_phase(pos, 64, 10000.0)
    inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = pos[..., None].float() * inv  # [3, T, 32]
    thw = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        idx = slice(offset, 10 * 3, 3)
        thw[..., idx] = freqs[dim][..., idx]
    expected = torch.cat((thw, thw), dim=-1).cos()  # HF layout: halves; ours is adjacent-pair interleaved
    assert torch.equal(cos[:, 0, 0::2], expected[:, :32]) and torch.equal(cos[:, 0, 1::2], expected[:, :32])


@pytest.mark.container_torch
def test_text_only_oracle_attention_is_bitwise_unchanged_by_slot_rope():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from families.hy4_flash import model
    torch.manual_seed(0)
    c = SimpleNamespace(attention=lambda _: (2, 2, 4, 2), layer_types=[model.SLIDING],
                        rope_parameters={model.SLIDING: {"rope_theta": 10000, "mrope_section": [1, 1, 0]}},
                        rms_norm_eps=.01, kv_lora_rank=2, sliding_window=3)
    shapes = {"q_a_proj": (3, 4), "q_a_layernorm": (3,), "q_b_proj": (12, 3), "kv_a_proj_with_mqa": (6, 4),
              "kv_a_layernorm": (2,), "kv_b_proj": (8, 2), "linear_gate": (4, 4), "o_proj": (4, 4),
              "post_proj_norm": (4,)}
    weights = {f"self_attn.{k}.weight": torch.randn(v) for k, v in shapes.items()}
    weights["self_attn.learnable_sink_param"] = torch.zeros(2)
    x = torch.randn(7, 4)
    plain = model.attention(x, weights, c, 0)
    slots = torch.arange(7).expand(3, -1)
    assert torch.equal(model.attention(x, weights, c, 0, slots), plain)
    moved = slots.clone()
    moved[1, 3:] += 2
    assert not torch.equal(model.attention(x, weights, c, 0, moved), plain)


@pytest.mark.container_torch
def test_tiny_tower_matches_shapes_and_newline_column(monkeypatch):
    torch = pytest.importorskip("torch")
    from families.hy4_flash import vision
    monkeypatch.setattr(vision, "LAYERS", 2)
    torch.manual_seed(0)
    weights = {k.removeprefix(vision.PREFIX): (torch.randn(v) * 0.02).bfloat16()
               for k, v in vision.contract().items() if not k.startswith("vit.layers.") or
               int(k.split(".")[2]) < 2}
    grid = [1, 4, 6]
    pixels = torch.randn(24, 768)
    for mode in vision.MODES:
        s = vision.encode(weights, pixels, grid, mode=mode)
        assert s.embed.shape == (24, 1152) and len(s.layers) == 2 and s.merger_norm.shape == (24, 1152)
        assert s.output.shape == (2 * 4, 3072)
        newline = weights["perceive.image_newline"].to(s.output.dtype)
        assert torch.equal(s.output[3], newline) and torch.equal(s.output[7], newline)
    serving = vision.encode(weights, pixels, grid, mode="serving").output.float()
    f32 = vision.encode(weights, pixels, grid, mode="f32").output
    assert torch.nn.functional.cosine_similarity(serving.flatten(), f32.flatten(), dim=0) > 0.99


@pytest.mark.container_torch
def test_inject_replaces_rows_before_stream_expansion():
    torch = pytest.importorskip("torch")
    w = window()
    span = w["media"][0]
    flat = torch.randn(len(w["tokens"]), media.WIDTH)
    original = flat.clone()
    feature = torch.randn(span["len"], media.WIDTH)
    out = media.inject_embeddings(flat, w, {span["key"]: feature})
    assert out.shape == (len(w["tokens"]), 4, media.WIDTH)
    rows = slice(span["start"], span["start"] + span["len"])
    assert torch.equal(out[rows], feature[:, None].expand(-1, 4, -1))
    assert torch.equal(out[:span["start"]], original[:span["start"], None].expand(-1, 4, -1))
    with pytest.raises(ValueError):
        media.inject_embeddings(torch.randn(len(w["tokens"]), media.WIDTH), w, {span["key"]: feature[:-1]})


def test_engine_identity_matches_probe_contract(tmp_path):
    import hashlib
    snapshot = tmp_path / "cc72179113573d751ed3245e3e58d3640694378d"
    snapshot.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        (snapshot / name).write_text(name)
    identity = media.engine_identity(snapshot)
    assert identity == {"snapshot_revision": snapshot.name, "family": "Hy4Flash", "source": "sglang@ccb8d9cc",
                        **{f"{n}_sha256": hashlib.sha256(n.encode()).hexdigest()
                           for n in ("config.json", "tokenizer.json", "model.safetensors.index.json")}}


@pytest.mark.container_torch
def test_varlen_batch_matches_singles_and_keeps_images_independent(monkeypatch):
    torch = pytest.importorskip("torch")
    from families.hy4_flash import vision
    monkeypatch.setattr(vision, "LAYERS", 1)
    torch.manual_seed(1)
    weights = {k.removeprefix(vision.PREFIX): (torch.randn(v) * 0.02).bfloat16()
               for k, v in vision.contract().items() if not k.startswith("vit.layers.") or k.split(".")[2] == "0"}
    grids = [[1, 4, 6], [1, 2, 2]]
    pixels = [torch.randn(24, 768), torch.randn(4, 768)]
    batch = vision.encode_batch(weights, pixels, grids, mode="f32")
    for value, grid, got in zip(pixels, grids, batch):
        single = vision.encode(weights, value, grid, mode="f32")
        assert got.patch.shape == single.patch.shape and len(got.layers) == 1
        torch.testing.assert_close(got.output, single.output, rtol=1e-5, atol=1e-5)


def test_vision_panel_spans_take_the_serving_engines_prepared_identity():
    import importlib.util
    spec = importlib.util.spec_from_file_location("hy4_vision_panel", ROOT / "python/reference/families/hy4_flash/vision_panel.py")
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    identity = {"encoder": "a" * 64, "processor": "b" * 64}
    echo = {"start": 18, "len": 272, "kind": "image", "key": "c" * 64, "grid": [1, 32, 32], "identity": identity}
    seen = []
    probe = lambda body: seen.append(body) or {"probe": {"media": [echo]}}
    span = panel.served_span(probe, panel.CHECKPOINT, b"png", [1, 32, 32], 272)
    assert span["key"] == "c" * 64 and span["identity"] == identity
    assert seen[0]["model"] == panel.CHECKPOINT and seen[0]["messages"][0]["content"][0]["type"] == "image_url"
    for bad in ({**echo, "len": 256}, {**echo, "grid": [1, 30, 40]}, {k: v for k, v in echo.items() if k != "identity"}):
        with pytest.raises(ValueError, match="prepared span differs"):
            panel.served_span(lambda body, bad=bad: {"probe": {"media": [bad]}}, panel.CHECKPOINT, b"png", [1, 32, 32], 272)
