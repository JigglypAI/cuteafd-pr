"""Validate full Hy4's publication identity with the same proof checked before Hub upload."""
import pytest

from hy4 import publish, source
from inkling2 import publish as hub
from kimi_k3.uniform_k2.storage import atomic_json


def test_full_hy4_requires_its_own_sealed_source_and_manifest(tmp_path):
    codebook = dict(name="mul1", multiplier=0x83DCD12D)
    state = dict(identity=dict(source_repo=source.BASE_REPO, source_revision=source.BASE_REVISION,
                               codebook="mul1", codebook_multiplier=codebook["multiplier"]))
    manifest = dict(source=dict(repo=source.BASE_REPO, revision=source.BASE_REVISION),
                    original_core_dtype="BF16", exl3_codebook=codebook,
                    variants=[dict(name=n) for n in ("original", "k8-compact", "k6-compact")])
    path = tmp_path / "cuteafd-mc.json"
    atomic_json(path, manifest)
    validation = dict(new_file_sha256={path.name: hub.metadata_identity(path)["sha256"]})
    publish.TARGET.validate(tmp_path, state, validation)
    state["identity"]["source_repo"] = "tencent/Hy4-flash"
    with pytest.raises(ValueError, match="pinned source"):
        publish.TARGET.validate(tmp_path, state, validation)
    state["identity"]["source_repo"] = source.BASE_REPO
    atomic_json(path, dict(manifest, engram_fp8={}))
    validation["new_file_sha256"][path.name] = hub.metadata_identity(path)["sha256"]
    with pytest.raises(ValueError, match="unexpected Engram"):
        publish.TARGET.validate(tmp_path, state, validation)
