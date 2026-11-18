"""Render the standard own-text panel with the pinned native Hy4 tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from kimi_k3 import prepare_corpus as panel
from kimi_k3.uniform_k2.storage import atomic_json, checkpoint_io, fsync_directory

SOURCE_REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
SCHEMA = "cuteafd-hy4-flash-corpus-v1"
ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
CHAT = dict(tokenize=True, add_generation_prompt=True, return_dict=False,
            padding=False, truncation=False, date_string="2026-11-18")
CONTRACT = dict(renderer="native-chat-template", reasoning_effort="default-high",
                batch_size=1, packing=False, order="dataset", **CHAT)


def long_witness(tokenizer):
    for length in (48, 72, 96, 128):
        prompt = "\n".join(
            f"Observation {i}: The airship records the position and colour of every star. "
            "The archivist compares the last recorded coordinate with the first."
            for i in range(length)
        )
        ids = tokenizer.apply_chat_template([dict(role="user", content=prompt)], **CHAT)
        if len(ids) > 2048:
            return dict(input_ids=ids, tokens=len(ids), hessian_eligible=False,
                        input_ids_sha256=panel.ids_sha256(ids))
    raise ValueError("long context witness never crossed the sparse selector cutoff")


def load_token_batches(root, split="calibration"):
    yield from panel.load_token_batches(root, split, schema=SCHEMA, contract=CONTRACT)


def prepare(source, corpus, output):
    import torch
    from transformers import PreTrainedTokenizerFast

    source, output = Path(source).absolute(), Path(output).absolute()
    if source.name != SOURCE_REVISION or output.exists():
        raise ValueError("require the pinned source and a fresh corpus directory")
    identity = {name: panel.sha256_file(source / name) for name in ASSETS}
    tokenizer = PreTrainedTokenizerFast.from_pretrained(source, local_files_only=True)
    if (not tokenizer.is_fast or not isinstance(tokenizer.chat_template, str)
            or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != identity["chat_template.jinja"]):
        raise ValueError("native tokenizer did not load the pinned chat template")
    rows, provenance = panel.read_panel(panel.download_corpus(corpus))
    manifest = dict(schema=SCHEMA, status="passed", contract=CONTRACT,
        dataset=dict(repo_id=panel.REPO_ID, revision=panel.REVISION, files=panel.PAYLOAD_SHA256),
        historical_selection=provenance["selection"],
        source=dict(repo_id="tencent/Hy4-flash", revision=SOURCE_REVISION, files=identity),
        software=panel.software_identity(), tokenizer=dict(vocab_size=len(tokenizer)), splits={})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        for split in panel.SPLITS:
            ids, offsets, report = panel.tokenize_rows(rows[split], tokenizer, chat_kwargs=CHAT)
            path = stage / (split + ".safetensors")
            receipt = checkpoint_io().write_safetensors(path,
                dict(input_ids=torch.frombuffer(ids, dtype=torch.int64).clone(),
                     offsets=torch.tensor(offsets, dtype=torch.int64)),
                metadata=dict(schema=SCHEMA, split=split), hash_algorithm="sha256")
            manifest["splits"][split] = dict(report, file=path.name, sha256=receipt["sha256"])
        witness = long_witness(tokenizer)
        atomic_json(stage / "long-context-smoke.json", witness)
        manifest["long_context_smoke"] = dict(file="long-context-smoke.json", tokens=witness["tokens"],
            sha256=panel.sha256_file(stage / "long-context-smoke.json"), hessian_eligible=False)
        if identity != {name: panel.sha256_file(source / name) for name in ASSETS}:
            raise ValueError("source tokenizer changed during preparation")
        atomic_json(stage / "manifest.json", manifest)
        for split in panel.SPLITS:
            panel.load_token_records(stage, split, schema=SCHEMA, contract=CONTRACT)
        stage.rename(output)
        fsync_directory(output.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "corpus", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source, args.corpus, args.output)
    print(json.dumps({"splits": {name: {key: value[key] for key in ("count", "tokens", "longest")}
                                for name, value in manifest["splits"].items()},
                      "long_context_smoke": manifest["long_context_smoke"]}), flush=True)


if __name__ == "__main__":
    main()
