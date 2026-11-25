#!/usr/bin/env python3
"""Build the fixed Hy4-flash vision golden panel: 4 image+text windows and 1 text control.

Images are first-party synthetic renders from ``scripts/bench/generate-media-fixtures.py``,
re-rendered at the target canvas sizes with nearest-neighbour upscaling (deterministic,
no external assets). Prompts render through the checkpoint chat template (reasoning
effort ``no_think``), then a fixed, about 64-token teacher-forced answer. Only the answer rows
are scored. Media spans take the serving engine's prepared-image key and its sealed
{encoder, processor} identity from one cold probe per fixture (as fidelity-set.py does),
so later servers recompute the same key from pixels; the span also records the fixture
sha. Usage: vision_panel.py --snapshot DIR --out DIR --server URL --bench-token-file FILE
"""
from __future__ import annotations

import argparse
import hashlib
import io
import base64
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
from fidelity_windows import SET_SCHEMA, bucket, canonical, set_hash, validate_set  # noqa: E402
from fidelity_media import prepared_span  # noqa: E402
import hy4_flash_media as media  # noqa: E402

CHECKPOINT, REVISION = "tencent/Hy4-flash", "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
# (window id, fixture index in generate-media-fixtures, canvas (w, h), question, answer)
WINDOWS = [
    ("img512", 2, (512, 512), "What does the terminal screenshot report? Answer briefly.",
     "The screenshot shows a synthetic terminal log for the command cargo test media. It collected the tests "
     "resize_exact, hash_binding and cold_restore, and each one is marked ok. The summary line reports 17 passed "
     "and 0 failed, and the final exit status is 0, so the run succeeded."),
    ("img512x768", 4, (512, 768), "Read the chart and summarize the batch counts.",
     "The chart is titled Synthetic batch counts and shows four bars. Alpha is 12, Beta is 28, Gamma is 19 and "
     "Delta is 35. Delta is the largest batch and Alpha is the smallest. Adding them gives a total of 94 across "
     "the four batches shown in the figure."),
    ("img1024", 6, (1024, 1024), "List the stages of the pipeline diagram in order.",
     "The diagram is labelled Synthetic pipeline and has four numbered boxes stacked vertically. The stages are "
     "1. INPUT, 2. DECODE, 3. ENCODE and 4. CACHE. Arrows point downward from each box to the next one, so data "
     "flows from input through decode and encode into the cache."),
    ("img630x470", 0, (630, 470), "Describe the code screenshot in one or two sentences.",
     "The screenshot is titled REPOSITORY CODE and shows the first lines of a CUDA source file from this "
     "repository. It contains include directives and the start of a normalization kernel, rendered in a "
     "monospace font on a light background, with each line truncated to fit the image width."),
]
CONTROL = ("text", "Explain in two sentences why a prefix cache must restore exact state.",
           "A prefix cache must restore exactly the state that prefilling the same prefix from scratch would have "
           "produced, otherwise later tokens are computed from a different context. Approximate restores can "
           "change the model's outputs silently, which breaks reproducibility and makes quality results "
           "impossible to compare across runs.")


def render_fixtures(out):
    """Re-render the eight 512x512 fixtures in a scratch dir, then resample the chosen ones."""
    import importlib.util
    import tempfile
    from PIL import Image
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts/bench/generate-media-fixtures.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    with tempfile.TemporaryDirectory() as scratch:
        base = gen.generate(Path(scratch))
        records = {}
        for wid, index, (width, height), *_ in WINDOWS:
            record = base["fixtures"][index]
            with Image.open(Path(scratch) / record["path"]) as source:
                canvas = source.convert("RGB").resize((width, height), Image.Resampling.NEAREST)
            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
            data = buffer.getvalue()
            path = out / "fixtures" / f"{wid}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != data:
                raise ValueError("panel fixtures are immutable; choose a new output directory")
            path.write_bytes(data)
            records[wid] = {"path": f"fixtures/{wid}.png", "sha256": hashlib.sha256(data).hexdigest(),
                            "source": record["path"], "source_sha256": record["sha256"], "canvas": [width, height]}
    return records


def template(snapshot):
    import jinja2
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True,
                             extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
    return env.from_string((snapshot / "chat_template.jinja").read_text())


def probe_client(url, token, timeout=600):
    """Cold, draft-free media probe against a live server: POST /v1/bench/probe."""
    def call(body):
        headers = {"Content-Type": "application/json", "x-cuteafd-bench": token, "Authorization": "Bearer " + token}
        request = urllib.request.Request(url.rstrip("/") + "/v1/bench/probe",
                                         canonical({"body": body, "spec": {"cold": True, "no_speculation": True}}), headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    return call


def served_span(probe, model, data, grid, length):
    """The serving engine's prepared span for one fixture; its grid and rows must match the processor."""
    chat = probe({"model": model, "max_tokens": 1, "temperature": 0, "stream": False, "messages": [{"role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
                    {"type": "text", "text": "media identity probe"}]}]})
    spans = chat.get("probe", {}).get("media", [])
    if len(spans) != 1 or spans[0].get("grid") != grid or spans[0].get("len") != length or "identity" not in spans[0]:
        raise ValueError(f"server prepared span differs from the Hy4 processor: {spans}")
    return prepared_span(spans[0])


def build(snapshot, out, probe):
    from PIL import Image
    from tokenizers import Tokenizer
    snapshot, out = Path(snapshot), Path(out)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    chat = template(snapshot)
    fixtures = render_fixtures(out)
    windows = []

    def encode_turn(content, answer):
        """Generation prompt exactly as served (template output), then the answer and hy_end."""
        prompt = chat.render(messages=[{"role": "user", "content": content}], add_generation_prompt=True,
                             reasoning_effort="no_think")
        if prompt.endswith("\n"):
            prompt = prompt[:-1]  # jinja keep_trailing_newline artifact; HF templating strips it too
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        end = tokenizer.token_to_id("<\uff5chy_end:opensource\uff5c>")
        answer_ids = tokenizer.encode(answer, add_special_tokens=False).ids
        if end is None or not 48 <= len(answer_ids) <= 96:
            raise ValueError(f"answer must be about 64 tokens and hy_end must exist: {len(answer_ids)}")
        return prompt_ids, prompt_ids + answer_ids + [end]

    for wid, _, (width, height), question, answer in WINDOWS:
        content = [{"type": "image"}, {"type": "text", "text": question}]
        prompt_ids, full_ids = encode_turn(content, answer)
        with Image.open(out / fixtures[wid]["path"]) as image:
            _, grid = media.prepare_rgb(image)
        length = media.span_len(grid)
        marker = full_ids.index(media.IMAGE_ID)
        if full_ids.count(media.IMAGE_ID) != 1 or full_ids[marker - 1] != media.START_ID or full_ids[marker + 1] != media.END_ID:
            raise ValueError("chat template image placeholder differs from start/pad/end")
        tokens = full_ids[:marker] + [media.IMAGE_ID] * length + full_ids[marker + 1:]
        score_from = len(prompt_ids) + length - 1
        served = served_span(probe, CHECKPOINT, (out / fixtures[wid]["path"]).read_bytes(), grid, length)
        span = {"kind": "image", "start": marker, "len": length, "grid": grid, "key": served["key"],
                "identity": served["identity"],
                "fixture": {"path": fixtures[wid]["path"], "sha256": fixtures[wid]["sha256"]}}
        windows.append({"id": wid, "block": "V", "bucket": bucket(score_from), "tokens": tokens,
                        "roles": ["ctx"] * score_from + ["gen"] * (len(tokens) - score_from),
                        "score_from": score_from, "media": [span],
                        "provenance": {"question": question, "answer": answer, "canvas": [width, height],
                                       "fixture_source": fixtures[wid]["source"]}})
    wid, question, answer = CONTROL
    prompt_ids, tokens = encode_turn(question, answer)
    windows.append({"id": wid, "block": "V", "bucket": bucket(len(prompt_ids)), "tokens": tokens,
                    "roles": ["ctx"] * len(prompt_ids) + ["gen"] * (len(tokens) - len(prompt_ids)),
                    "score_from": len(prompt_ids), "provenance": {"question": question, "answer": answer}})
    sha = lambda name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
    manifest = {"schema": SET_SCHEMA, "family": "hy4_flash", "checkpoint": CHECKPOINT, "model": CHECKPOINT,
                "version": "vision-panel-v1", "image_processor": media.processor_contract(),
                "tokenizer_sha256": sha("tokenizer.json"),
                "reference_snapshot": {"checkpoint": CHECKPOINT, "snapshot_revision": REVISION,
                                       "config_sha256": sha("config.json"), "tokenizer_sha256": sha("tokenizer.json")},
                "generation_arm": {"text_source": "fixed teacher-forced answers", "chat_template_sha256":
                                   sha("chat_template.jinja"), "reasoning_effort": "no_think"},
                "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "fixtures": fixtures, "quick_windows": [w["id"] for w in windows], "windows": windows}
    manifest["set_sha256"] = set_hash(manifest)
    validate_set(manifest)
    media.validate_spans(manifest, snapshot)
    path = out / "windows.json"
    content = canonical(manifest) + b"\n"
    if path.exists() and path.read_bytes() != content:
        raise ValueError("panel manifest changed; choose a new output directory")
    path.write_bytes(content)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--server", required=True, help="live Hy4 server (served as tencent/Hy4-flash) for prepared keys")
    p.add_argument("--bench-token-file", type=Path, required=True)
    a = p.parse_args()
    manifest = build(a.snapshot, a.out, probe_client(a.server, a.bench_token_file.read_text().strip()))
    for w in manifest["windows"]:
        spans = w.get("media", [])
        print(w["id"], len(w["tokens"]), "score_from", w["score_from"],
              [(s["grid"], s["len"]) for s in spans])
    print("set_sha256", manifest["set_sha256"])


if __name__ == "__main__":
    main()
