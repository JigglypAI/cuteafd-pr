"""Sequential Hy4 scratch stages and ordered expert dispatch."""
from __future__ import annotations

import torch

from kimi_k3.core_k6.storage import StageStore
from kimi_k3.uniform_k2.storage import read_header, verify_receipt


_FIELDS = {"F32": torch.float32, "I64": torch.int64, "I32": torch.int32}


class Stages(StageStore):
    """Expose committed record extents directly to the MC writer."""

    def records(self, name, *, device="cpu", keys=None):
        manifest = self.load(name)
        if manifest is None:
            raise ValueError(f"Hy4 stage is missing: {name}")
        for index in range(len(manifest["entries"])):
            if keys is None:
                values = self.read_batch(name, index, device=device)
            else:
                values = self.fields(name, index, keys, device=device)
            yield index, values

    def fields(self, stage, index, keys, *, device="cpu"):
        """Read just the used tensors of a sealed record without rereading large peers."""
        path, _, offset, length = self.reference(stage, index, "")
        verify_receipt(path, self._manifests[stage]["entries"][index])
        header, base = read_header(path, offset=offset, length=length)
        values = {}
        with path.open("rb") as stream:
            for key in keys:
                spec = header[key]
                start, stop = spec["data_offsets"]
                dtype = _FIELDS[spec["dtype"]]
                if start == stop:
                    value = torch.empty(spec["shape"], dtype=dtype)
                else:
                    raw = bytearray(stop - start)
                    stream.seek(base + start)
                    if stream.readinto(raw) != len(raw):
                        raise ValueError("truncated Hy4 stage field")
                    value = torch.frombuffer(raw, dtype=dtype).reshape(spec["shape"])
                values[key] = value.to(device)
        return values

    def reference(self, stage, index, tensor):
        manifest = self._manifests.get(stage) or self.load(stage)
        if manifest is None:
            raise ValueError(f"Hy4 stage is missing: {stage}")
        record = manifest["entries"][index]
        return (self.directory(stage) / record["filename"], tensor,
                record["offset"], record["bytes"])


class Routes:
    """Stable expert-major view of immutable token-major route records."""

    def __init__(self, batches, experts, tokens):
        self.latent = None
        count = 0
        for _, batch in batches:
            latent, ids, gamma = (batch[k] for k in ("latent", "ids", "gamma"))
            if (latent.dtype != torch.float32 or ids.ndim != 2 or ids.shape != gamma.shape
                    or len(latent) != len(ids) or ids.dtype != torch.int64
                    or gamma.dtype != torch.float32
                    or bool((ids < 0).any()) or bool((ids >= experts).any())
                    or not bool(torch.isfinite(gamma).all())):
                raise ValueError("invalid Hy4 natural-route record")
            if self.latent is None:
                self.latent = torch.empty((tokens, latent.shape[-1]), dtype=latent.dtype)
                choices = torch.empty((tokens, ids.shape[-1]), dtype=ids.dtype)
                weights = torch.empty_like(choices, dtype=gamma.dtype)
            stop = count + len(latent)
            self.latent[count:stop].copy_(latent)
            choices[count:stop].copy_(ids)
            weights[count:stop].copy_(gamma)
            count = stop
        if count != tokens or self.latent is None:
            raise ValueError("Hy4 route records disagree with the corpus token count")
        width = choices.shape[-1]
        ids = choices.flatten()
        self.bounds = [0] + torch.bincount(ids, minlength=experts).cumsum(0).tolist()
        order = torch.argsort(ids, stable=True)
        self.rows = order.div(width, rounding_mode="floor")
        self.gamma = weights.flatten()[order]
        sorted_ids = ids[order]
        same_expert = sorted_ids[1:] == sorted_ids[:-1]
        if bool(((self.rows[1:] == self.rows[:-1]) & same_expert).any()):
            raise ValueError("Hy4 router assigned the same expert twice for a token")

    def expert(self, index):
        start, stop = self.bounds[index:index + 2]
        return self.rows[start:stop], self.gamma[start:stop]

    def combined(self, outputs):
        """Accumulate weighted outputs in the reference's ascending expert order."""
        combined = torch.zeros_like(self.latent)
        expected = len(self.bounds) - 1
        count = 0
        for expert, weighted in outputs:
            if count >= expected or expert != count:
                raise ValueError("Hy4 expert outputs are out of order")
            rows, _ = self.expert(expert)
            if weighted.shape != (len(rows), self.latent.shape[-1]) or weighted.dtype != combined.dtype:
                raise ValueError("Hy4 weighted output differs from its natural route geometry")
            combined.index_add_(0, rows, weighted)
            count += 1
        if count != expected:
            raise ValueError("Hy4 expert output inventory is incomplete")
        return combined
