"""Portable optional checkpoint witnesses must never download missing assets."""
import json

import pytest

from checkpoint_cache import V41_REPO, V41_REVISION, local_snapshot


def snapshot(cache):
    root = cache / ("models--" + V41_REPO.replace("/", "--")) / "snapshots" / V41_REVISION
    root.mkdir(parents=True)
    return root


def test_hf_home_and_pinned_revision(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    root = snapshot(tmp_path / "hub")
    (root / "config.json").write_text("{}")
    assert local_snapshot(files=("config.json",)) == root
    (root / "config.json").unlink()
    with pytest.raises(pytest.skip.Exception, match="local checkpoint.*missing config.json.*no downloads"):
        local_snapshot(files=("config.json",))


def test_cache_override_and_complete_index(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "unused"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    root = snapshot(tmp_path / "cache")
    (root / "config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"w": "model.safetensors"})))
    with pytest.raises(pytest.skip.Exception, match="incomplete.*missing shards model.safetensors"):
        local_snapshot(indexed=True)
    (root / "model.safetensors").write_bytes(b"header fixture")
    assert local_snapshot(indexed=True) == root


def test_sdk_cache_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    root = snapshot(tmp_path)
    assert local_snapshot() == root
