"""Exercise the qualified SDK itself, mocking only remote/transport boundaries."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi
from huggingface_hub import _commit_api
from huggingface_hub.hf_api import RepoFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hub_requirements
from upload_model import _restored_operation


@pytest.mark.parametrize("version,xet", [("1.26.1", True), ("2.2.0", True), ("1.33.0", False)])
def test_unqualified_sdk_or_disabled_xet_is_rejected(monkeypatch, version, xet):
    monkeypatch.setattr(hub_requirements.importlib.metadata, "version", lambda _: version)
    monkeypatch.setattr("huggingface_hub.utils._runtime.is_xet_available", lambda: xet)
    with pytest.raises(ValueError, match="qualified huggingface_hub 1.33.0 with Xet"):
        hub_requirements.require_qualified_hub()


def test_real_preupload_uses_xet_and_restored_receipt_avoids_reupload(tmp_path, monkeypatch):
    hub_requirements.require_qualified_hub()
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"synthetic weights")
    digest = hashlib.sha256(path.read_bytes()).digest()
    op = CommitOperationAdd(path.name, path)
    assert not op.upload_info.is_hashed

    def classify(**kwargs):
        assert kwargs["gitignore_content"] == ""
        for addition in kwargs["additions"]:
            addition._upload_mode = "lfs"
            addition._should_ignore = False

    def xet_upload(**kwargs):
        assert kwargs["additions"] == [op]
        op.upload_info.sha256 = digest

    classify_mock = Mock(side_effect=classify)
    xet_mock = Mock(side_effect=xet_upload)
    monkeypatch.setattr(_commit_api, "_fetch_upload_modes", classify_mock)
    monkeypatch.setattr(_commit_api, "_upload_xet_files", xet_mock)
    monkeypatch.setattr(_commit_api, "post_lfs_batch_info", Mock(side_effect=AssertionError("legacy LFS")))
    api = HfApi(token=False)
    api.preupload_lfs_files("test/model", [op], repo_type="model", revision="main",
                           num_threads=1, free_memory=False, gitignore_content="")
    assert op._is_uploaded and op.upload_info.sha256 == digest
    assert op.path_or_fileobj == str(path)
    restored = _restored_operation(path.name, path, dict(mode="lfs", blob=digest.hex()))
    api.preupload_lfs_files("test/model", [restored], free_memory=False, gitignore_content="")
    assert restored._is_uploaded and restored.upload_info.sha256 == digest
    assert classify_mock.call_count == xet_mock.call_count == 1


def test_real_commit_copies_lfs_server_side_without_reading_source(monkeypatch):
    hub_requirements.require_qualified_hub()
    revision = "a" * 40
    op = CommitOperationCopy("source.safetensors", "model.safetensors", src_revision=revision,
                             src_repo_id="test/source", src_repo_type="model")
    source = RepoFile(path="source.safetensors", size=123, oid="b" * 40,
                      lfs=dict(oid="c" * 64, size=123, pointerSize=130), xetHash="d" * 64)
    files = {_commit_api._CopySource("test/source", "model", "source.safetensors", revision): source}
    fetch = Mock(return_value=files)
    monkeypatch.setattr(_commit_api, "_fetch_files_to_copy", fetch)
    response = Mock(status_code=200)
    response.json.return_value = {"failed": []}
    session = Mock()
    session.post.return_value = response
    monkeypatch.setattr("huggingface_hub.hf_api.get_session", lambda: session)
    monkeypatch.setattr("huggingface_hub.hf_api.hf_raise_for_status", lambda _: None)
    send = Mock(return_value=SimpleNamespace(oid="e" * 40))
    monkeypatch.setattr(_commit_api, "_send_commit", send)
    result = HfApi(token=False).create_commit("test/dest", [op], repo_type="model", revision="main",
                                           parent_commit=revision, commit_message="Mock copy")
    assert result.oid == "e" * 40
    assert op._is_duplicated
    assert fetch.call_args.kwargs["copies"] == [op]
    assert session.post.call_args.args == ("https://huggingface.co/api/models/test/source/lfs-files/duplicate",)
    assert session.post.call_args.kwargs["json"] == dict(
        target=dict(type="model", name="test/dest"),
        files=[dict(xetHash="d" * 64, sha256="c" * 64, filename="source.safetensors")])
    assert send.call_args.kwargs["operations"] == [op]
    assert send.call_args.kwargs["parent_commit"] == revision
