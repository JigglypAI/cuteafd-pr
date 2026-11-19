"""Committed phase files between two rolling Kimi layer boundaries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil

from kimi_k3.uniform_k2.storage import (atomic_json, checkpoint_io, file_identity,
    fsync_directory, verify_receipt)


class StageStore:
    """Trust a phase only after its complete manifest has been published.

    Callers can discard earlier phases once their successor contains everything
    needed to resume. Packed projection receipts live outside these directories
    until the layer's export and frontier have both committed.
    """
    def __init__(self, root, identity):
        self.root = Path(root)
        missing, ancestor = [], self.root
        while not ancestor.exists():
            missing.append(ancestor)
            ancestor = ancestor.parent
        if self.root.is_symlink():
            raise ValueError("refuse symlink phase root")
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in reversed(missing):
            fsync_directory(directory.parent)
        self.identity = identity
        self._manifests = {}

    def directory(self, name):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError("invalid phase name")
        path = self.root / name
        if path.is_symlink():
            raise ValueError("refuse symlink phase directory")
        return path

    @staticmethod
    def _validate_entries(entries):
        if any(entry.get("storage_format") not in (None, "safetensors-segment-v1") for entry in entries):
            raise ValueError("unknown phase storage format")
        segmented = any(entry.get("storage_format") == "safetensors-segment-v1" for entry in entries)
        if segmented and any(entry.get("storage_format") != "safetensors-segment-v1" or
                             type(entry.get("index")) is not int or entry["index"] != index
                             for index, entry in enumerate(entries)):
            raise ValueError("invalid phase segment batch inventory")
        names = [entry["filename"] for entry in entries]
        keys = [(entry["filename"], entry.get("offset")
                 if entry.get("storage_format") == "safetensors-segment-v1" else None) for entry in entries]
        if not names or len(set(keys)) != len(keys):
            raise ValueError("empty or duplicate phase inventory")
        if any(not isinstance(name, str) or Path(name).name != name
               or name in (".", "..", "manifest.json") for name in names):
            raise ValueError("phase filename must be a local basename")
        groups = {}
        for entry in entries:
            groups.setdefault(entry["filename"], []).append(entry)
        for group in groups.values():
            if any(entry.get("storage_format") == "safetensors-segment-v1" for entry in group):
                cursor = 0
                for entry in group:
                    if (entry.get("storage_format") != "safetensors-segment-v1" or
                            entry.get("hash_algorithm") != "xxh3-128" or
                            type(entry.get("offset")) is not int or entry["offset"] != cursor or
                            type(entry.get("bytes")) is not int or entry["bytes"] <= 0):
                        raise ValueError("invalid or noncontiguous phase segment extent")
                    cursor += entry["bytes"]
                if any(entry["file_identity"] != group[0]["file_identity"] for entry in group):
                    raise ValueError("phase segment identities differ")
                if cursor != group[0]["file_identity"]["bytes"]:
                    raise ValueError("phase segment size differs")

    def load(self, name, *, checksum=False):
        directory = self.directory(name)
        path = directory / "manifest.json"
        if not path.exists():
            return None
        manifest = json.loads(path.read_text())
        if manifest.get("identity") != self.identity or manifest.get("stage") != name:
            raise ValueError("phase identity differs")
        self._validate_entries(manifest["entries"])
        for entry in manifest["entries"]:
            target = directory / entry["filename"]
            if target.is_symlink():
                raise ValueError("refuse symlink phase payload")
            verify_receipt(target, entry, checksum=checksum)
        self._manifests[name] = manifest
        return manifest

    def commit(self, name, entries, metadata=None):
        self._validate_entries(entries)
        directory = self.directory(name)
        for entry in entries:
            target = directory / entry["filename"]
            if target.is_symlink():
                raise ValueError("refuse symlink phase payload")
            verify_receipt(target, entry)
        record = dict(identity=self.identity, stage=name, entries=entries,
                      metadata={} if metadata is None else metadata)
        prior = self.load(name)
        if prior is not None:
            if prior != record:
                raise ValueError("refuse to replace a committed phase")
            return prior
        # A durable manifest inside a newly created directory also needs the
        # parent entry persisted. This is one barrier per phase, never per batch.
        fsync_directory(self.root)
        atomic_json(directory / "manifest.json", record)
        self._manifests[name] = record
        return record

    def write_batches(self, name, batches, metadata=None, *, segment_bytes=512 * 1024 * 1024,
                      segment_batches=32, record_bytes=None, resume=None):
        """Seal bounded groups, then publish the existing complete-phase barrier.

        By default the historical format replays the entire incomplete phase.
        A caller can supply a serialized record size for pre-append rollover.
        A resumable activation producer supplies a journal and accepts a starting
        record ordinal. Packed fit groups retain their phase-level metric commit.
        """
        if (type(segment_bytes) is not int or segment_bytes <= 0 or
                type(segment_batches) is not int or segment_batches <= 0):
            raise ValueError("invalid phase segment bounds")
        prior = self.load(name)
        if prior is not None:
            return prior
        directory = self.directory(name)
        directory.mkdir(parents=True, exist_ok=True)
        if resume is None:
            entries, ordinal = [], 0
        else:
            entries, ordinal = resume.recover(directory)
        # Validate every leaf before removing any. Never delete unknown source,
        # symlink, or committed manifest data while cleaning an aborted phase.
        leaves = list(directory.iterdir()) if resume is None else []
        owned = re.compile(r"(?:[0-9]{5}\.safetensors(?:\.partial)?|segment-[0-9]{5}\.bin(?:\.partial)?|"
                           r"\.manifest\.json\.[a-z0-9_]{8}\.tmp|\.json-[a-z0-9_]{8})")
        if any(path.is_symlink() or not path.is_file() or not owned.fullmatch(path.name) for path in leaves):
            raise ValueError("unexpected uncommitted phase leaf")
        for path in leaves:
            path.unlink()
        pending, stream = [], None

        def seal():
            nonlocal stream, pending, ordinal
            if stream is None:
                return
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            stream = None
            path = directory / f"segment-{ordinal:05d}.bin"
            os.replace(path.with_suffix(".bin.partial"), path)
            identity = file_identity(path)
            sealed = [dict(entry, file_identity=identity) for entry in pending]
            if resume is not None:
                resume.seal(directory, ordinal, sealed)
            entries.extend(sealed)
            pending = []
            ordinal += 1

        try:
            source = batches if resume is None else (() if len(entries) == resume.count else batches(len(entries)))
            for index, values in source:
                if type(index) is not int or index != len(entries) + len(pending):
                    raise ValueError("phase batch indices must be contiguous from zero")
                if record_bytes is not None:
                    size = record_bytes(values)
                    if resume is not None and (index >= resume.count or size > segment_bytes):
                        raise ValueError("activation record exceeds the declared count or segment limit")
                    if stream is not None and stream.tell() + size > segment_bytes:
                        seal()
                if stream is None:
                    filename = f"segment-{ordinal:05d}.bin"
                    stream = (directory / (filename + ".partial")).open("xb")
                receipt = checkpoint_io().append_safetensors(stream,
                    {key: value.detach().cpu().contiguous() for key, value in values.items()})
                if record_bytes is not None and receipt["bytes"] != size:
                    raise ValueError("serialized checkpoint size differs from its planned size")
                pending.append(dict(filename=filename, index=index,
                                    storage_format="safetensors-segment-v1", **receipt))
                if stream.tell() >= segment_bytes or len(pending) >= segment_batches:
                    seal()
            if resume is not None and len(entries) + len(pending) != resume.count:
                raise ValueError("incomplete activation record inventory")
            seal()
            # Persist every segment rename together before the manifest refers
            # to them. atomic_json supplies the later manifest/file+dir barrier.
            fsync_directory(directory)
            return self.commit(name, entries, metadata)
        finally:
            if stream is not None:
                stream.close()

    def read_batch(self, name, index, *, device="cpu"):
        manifest = self._manifests.get(name)
        if manifest is None:
            manifest = self.load(name)
        if manifest is None:
            raise ValueError("phase is not committed")
        entries = manifest["entries"]
        if type(index) is not int or not 0 <= index < len(entries):
            raise IndexError(index)
        entry = entries[index]
        path = self.directory(name) / entry["filename"]
        verify_receipt(path, entry)
        if "index" in entry and entry["index"] != index:
            raise ValueError("phase batch index differs")
        if entry.get("storage_format") == "safetensors-segment-v1":
            values = checkpoint_io().read_safetensors(path, offset=entry["offset"], length=entry["bytes"])
        else:
            from safetensors.torch import load_file
            values = {key: value.clone() for key, value in load_file(str(path)).items()}
        return {key: value.to(device) for key, value in values.items()}

    def retire(self, *names):
        for name in names:
            directory = self.directory(name)
            if directory.exists():
                shutil.rmtree(directory)
            self._manifests.pop(name, None)
        fsync_directory(self.root)

    def clear(self):
        self.retire(*(path.name for path in self.root.iterdir() if path.is_dir()))
