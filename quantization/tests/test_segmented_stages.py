"""Byte preservation and crash barriers for future activation scratch files."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kimi_k3.core_k6 import storage as stages
from kimi_k3.uniform_k2 import storage


def batches(count):
    for index in range(count):
        # Raw BF16 payloads, including signed zero/NaN bits: storage never
        # interprets or canonicalizes them (the caller owns finite checks).
        yield index, {"hidden": torch.tensor([0x8000, 0x7FC1, index], dtype=torch.uint16).view(torch.bfloat16)}


def test_groups_reduce_payload_barriers_and_never_read_back_on_write(tmp_path, monkeypatch):
    store = stages.StageStore(tmp_path, {"run": "one"})
    original_open, original_sync = Path.open, os.fsync
    synced = []

    def guarded_open(path, mode="r", *args, **kwargs):
        if "r" in mode and path.suffix in (".bin", ".partial"):
            raise AssertionError("write-time payload reread")
        return original_open(path, mode, *args, **kwargs)

    def tracked_sync(fd):
        synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        return original_sync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded_open)
        patch.setattr(os, "fsync", tracked_sync)
        manifest = store.write_batches("a", batches(12), segment_batches=4)
    payload_syncs = [name for name in synced if "segment-" in name]
    assert len(payload_syncs) == 3
    assert all(name.endswith(".bin.partial") for name in payload_syncs)
    assert len(synced) == 7  # 3 payloads + phase dir + parent + manifest + phase dir.
    assert synced[-3] == str(tmp_path)
    assert len({entry["filename"] for entry in manifest["entries"]}) == 3
    assert store.load("a", checksum=True) == manifest
    for index, expected in batches(12):
        actual = store.read_batch("a", index)["hidden"]
        assert torch.equal(actual.view(torch.uint16), expected["hidden"].view(torch.uint16))


@pytest.mark.parametrize("after", [False, True])
def test_manifest_is_the_only_numerical_commit_barrier(tmp_path, monkeypatch, after):
    store = stages.StageStore(tmp_path, {"run": "one"})
    publish = stages.atomic_json

    def interrupted(path, value):
        if after:
            publish(path, value)
        raise RuntimeError("power loss")

    with monkeypatch.context() as patch:
        patch.setattr(stages, "atomic_json", interrupted)
        with pytest.raises(RuntimeError, match="power loss"):
            store.write_batches("a", batches(5), segment_batches=2)
    restarted = stages.StageStore(tmp_path, store.identity)
    assert (restarted.load("a") is not None) == after
    if after:
        def must_not_run():
            raise AssertionError("recomputed committed phase")
            yield
        result = restarted.write_batches("a", must_not_run())
    else:
        # Orphan sealed groups have no numerical standing. Replay cleans all
        # of this phase's old groups, including surplus leaves from the crash.
        result = restarted.write_batches("a", batches(2), segment_batches=2)
        assert len(list((tmp_path / "a").glob("segment-*.bin"))) == 1
    assert restarted.load("a", checksum=True) == result


def test_unfinished_segment_is_replayed_and_unknown_leaves_are_preserved(tmp_path):
    store = stages.StageStore(tmp_path, {})

    def interrupted():
        yield from batches(3)
        raise RuntimeError("lost producer")

    with pytest.raises(RuntimeError):
        store.write_batches("a", interrupted(), segment_batches=2)
    assert (tmp_path / "a/segment-00001.bin.partial").exists()
    unknown = tmp_path / "a/source-checkpoint.bin"
    unknown.write_bytes(b"preserve")
    leaves = set((tmp_path / "a").iterdir())
    with pytest.raises(ValueError, match="unexpected"):
        store.write_batches("a", batches(1))
    assert set((tmp_path / "a").iterdir()) == leaves
    unknown.unlink()
    store.write_batches("a", batches(1))
    assert not list((tmp_path / "a").glob("*.partial"))


@pytest.mark.parametrize("filename", [".manifest.json.abc123_4.tmp", ".json-abc123_4"])
def test_manifest_temp_from_process_death_does_not_block_replay(tmp_path, filename):
    store = stages.StageStore(tmp_path, {})
    directory = store.directory("a")
    directory.mkdir()
    # SIGKILL/power loss bypasses atomic_json's finally unlink; neither the
    # old helper's temporary name nor the new path-specific one is a commit.
    (directory / filename).write_text('{"incomplete":')
    store.write_batches("a", batches(1))
    assert not (directory / filename).exists()
    assert store.load("a", checksum=True)


def test_byte_target_seals_between_records_and_preserves_batch_bound(tmp_path):
    store = stages.StageStore(tmp_path, {})
    result = store.write_batches("a", batches(3), segment_bytes=1)
    assert len({entry["filename"] for entry in result["entries"]}) == 3
    assert all(entry["offset"] == 0 for entry in result["entries"])
    assert store.load("a", checksum=True)


@pytest.mark.parametrize("truncate", [False, True])
def test_corrupt_or_truncated_group_fails_recovery(tmp_path, truncate):
    store = stages.StageStore(tmp_path, {})
    manifest = store.write_batches("a", batches(4))
    path = tmp_path / "a" / manifest["entries"][0]["filename"]
    stat = path.stat()
    with path.open("r+b") as stream:
        if truncate:
            stream.truncate(stat.st_size - 1)
        else:
            stream.seek(manifest["entries"][1]["offset"] + manifest["entries"][1]["bytes"] - 1)
            stream.write(b"\xff")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed|checksum"):
        stages.StageStore(tmp_path, {}).load("a", checksum=True)


def test_legacy_sha_phase_and_new_xxh3_receipts_are_explicit(tmp_path):
    store = stages.StageStore(tmp_path, {})
    path = store.directory("old") / "00000.safetensors"
    path.parent.mkdir()
    expected = torch.arange(6).bfloat16()
    save_file({"hidden": expected}, str(path))
    entry = dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 file_identity=storage.file_identity(path))
    store.commit("old", [entry])
    assert store.load("old", checksum=True)
    assert torch.equal(store.read_batch("old", 0)["hidden"], expected)
    receipt = storage.commit_safetensors(tmp_path / "new.safetensors", {"hidden": expected})
    assert receipt["hash_algorithm"] == "xxh3-128" and "sha256" not in receipt
    storage.verify_receipt(tmp_path / "new.safetensors", receipt, checksum=True)
    # Existing capture receipts use offset for corpus rows, not file bytes.
    storage.verify_receipt(tmp_path / "new.safetensors", dict(receipt, offset=10000), checksum=True)


def test_segment_extent_overlap_and_unknown_algorithm_fail_closed(tmp_path):
    store = stages.StageStore(tmp_path, {})
    manifest = store.write_batches("a", batches(2))
    path = tmp_path / "a/manifest.json"
    manifest["entries"][1]["offset"] -= 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="extent"):
        stages.StageStore(tmp_path, {}).load("a")

    manifest["entries"][1]["offset"] += 1
    manifest["entries"][0]["hash_algorithm"] = "md5"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="extent"):
        stages.StageStore(tmp_path, {}).load("a")


def test_migration_inventory_reuses_new_checksum_without_sha_reread(tmp_path, monkeypatch):
    from step5.core_k6 import migrate
    path = tmp_path / "packed.safetensors"
    receipt = storage.commit_safetensors(path, {"packed": torch.arange(4)})
    monkeypatch.setattr(migrate, "sha256_file", lambda path: pytest.fail("scratch SHA reread"))
    inventory = migrate.Objects()
    saved = inventory.add(path, receipt)
    assert saved["hash_algorithm"] == "xxh3-128"
    assert saved["xxh3_128"] == receipt["xxh3_128"]
    assert inventory.add(path, receipt) == saved
    inventory.unchanged()
