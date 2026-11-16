"""Exercise qualified publication adapters without network/GPU."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.uniform_k2.publish import install_upload_compat, main


def test_modern_api_preserves_upload_selection_and_uses_streaming_folder_api():
    class Api:
        def upload_folder(self, **kwargs):
            self.received = kwargs
            return "committed"

    with patch("huggingface_hub.HfApi", Api), patch("huggingface_hub.utils._runtime.is_xet_available", return_value=True):
        install_upload_compat()
    api = Api()
    selection = dict(repo_id="owner/model", repo_type="model", folder_path="/export",
                     ignore_patterns=["*-receipt.json", ".cache/**"])
    assert api.upload_large_folder(**selection, num_workers=4, print_report=True,
                                   print_report_every=1800) == "committed"
    assert api.received == selection


def test_existing_adapter_is_not_replaced():
    class Api:
        def upload_large_folder(self):
            return "legacy"

    original = Api.upload_large_folder
    with patch("huggingface_hub.HfApi", Api), patch("huggingface_hub.utils._runtime.is_xet_available", return_value=True):
        install_upload_compat()
    assert Api.upload_large_folder is original


def test_modern_api_refuses_nonresumable_fallback_without_xet():
    class Api:
        pass

    with patch("huggingface_hub.HfApi", Api), patch("huggingface_hub.utils._runtime.is_xet_available", return_value=False):
        with pytest.raises(ValueError, match="with Xet enabled"):
            install_upload_compat()
    assert not hasattr(Api, "upload_large_folder")


def test_uniform_producer_publication_qualifies_before_remote_mutation(tmp_path, monkeypatch):
    from kimi_k3.uniform_k2 import publish as adapter, run
    import hub_requirements

    events = []

    class UploadReached(Exception):
        pass

    class Api:
        def __init__(self):
            events.append("api")

        def create_repo(self, *args, **kwargs):
            events.append("create")

        def upload_folder(self, **kwargs):
            assert kwargs["folder_path"] == str(tmp_path / "export")
            events.append("upload")
            raise UploadReached

    real_check = adapter.require_qualified_hub

    def check():
        events.append("qualify")
        real_check()

    monkeypatch.setattr(adapter, "require_qualified_hub", check)
    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    with pytest.raises(UploadReached):
        run.publish(tmp_path / "export", tmp_path / "state", "test/model")
    assert events == ["qualify", "api", "create", "upload"]

    events.clear()
    monkeypatch.setattr(hub_requirements.importlib.metadata, "version", lambda _: "2.2.0")
    with pytest.raises(ValueError, match="qualified huggingface_hub 1.33.0"):
        run.publish(tmp_path / "export", tmp_path / "state", "test/model")
    assert events == ["qualify"]


def test_publication_entry_point_cannot_start_quantization():
    with patch.object(sys, "argv", ["publish.py"]):
        with pytest.raises(SystemExit, match="requires --publish-only"):
            main()
