"""Local-only checkpoint lookup for optional real-checkpoint test witnesses."""
import json
import os
from pathlib import Path

import pytest

V41_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
V41_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"


def local_snapshot(repo_id=V41_REPO, revision=V41_REVISION, *, files=(), indexed=False):
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = os.environ.get("HF_HUB_CACHE")
    if cache is None:
        home = os.environ.get("HF_HOME")
        cache = Path(home) / "hub" if home else HF_HUB_CACHE
    snapshot = Path(cache).expanduser() / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    required = list(files)
    if indexed:
        required += ["config.json", "model.safetensors.index.json"]
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        pytest.skip(f"local checkpoint {repo_id}@{revision} unavailable at {snapshot}: missing {', '.join(missing)} (no downloads)")
    if indexed:
        index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        missing = sorted({name for name in index["weight_map"].values() if not (snapshot / name).is_file()})
        if missing:
            pytest.skip(f"local checkpoint {repo_id}@{revision} incomplete at {snapshot}: missing shards {', '.join(missing)} (no downloads)")
    return snapshot
