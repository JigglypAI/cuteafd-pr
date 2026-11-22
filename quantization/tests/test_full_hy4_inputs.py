"""A checkpoint or token stream for Flash cannot be substituted for full Hy4."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy4 import prepare_corpus, source
from hy4_flash import prepare_corpus as flash


def test_full_hy4_rejects_flash_snapshot(tmp_path):
    snapshot = tmp_path / flash.SOURCE_REVISION
    snapshot.mkdir()
    with pytest.raises(ValueError, match="pinned revision"):
        source.HeaderSource(snapshot)
    with pytest.raises(ValueError, match="pinned source"):
        prepare_corpus.prepare(snapshot, tmp_path / "panel", tmp_path / "out")


@pytest.mark.parametrize("flash_field", ["schema", "contract"])
def test_full_hy4_rejects_flash_panel(tmp_path, flash_field):
    manifest = dict(status="passed", schema=prepare_corpus.SCHEMA, contract=prepare_corpus.CONTRACT,
                    dataset=prepare_corpus.DATASET)
    manifest[flash_field] = flash.SCHEMA if flash_field == "schema" else flash.CONTRACT
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prepared corpus contract differs"):
        list(prepare_corpus.load_token_batches(tmp_path, "heldout"))
