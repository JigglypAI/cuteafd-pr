"""Validate row ordering, bounded workspaces and reuse in full Hy4 calibration."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hy4_flash.storage import HostExpertStaging


@pytest.mark.parametrize("capacity", [4, 8])
def test_expert_passes_preserve_natural_prefix_and_workspace_bound(capacity, monkeypatch):
    bank = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    selected = torch.tensor([1, 3, 7, 8, 6, 2])  # natural rows, then augmented rows
    row_bytes = bank.shape[1] * bank.element_size()
    staging = HostExpertStaging(bank, "cpu", chunk_rows=2, max_bytes=capacity * row_bytes)
    assert staging.staged.shape == (capacity, 3)
    assert staging.host_chunk.shape == (2, 3)
    stage_ptr = staging.staged.data_ptr()
    copies = []
    copy_rows = staging.copy_rows

    def record(part, target):
        copies.append(part.tolist())
        copy_rows(part, target)

    monkeypatch.setattr(staging, "copy_rows", record)
    reader = staging.select(selected)
    for count in (6, 6, 3):
        # A streamed slice belongs to this iteration only.
        observed = [part.clone() for part in reader(count)]
        assert [len(part) for part in observed] == [len(part) for part in selected[:count].split(2)]
        assert torch.equal(torch.cat(observed), bank[selected[:count]])
    expected = [[1, 3], [7, 8], [6, 2]]
    assert copies == (expected if capacity >= len(selected) else expected * 2 + [[1, 3], [7]])
    next_rows = selected[[2, 4]]
    assert torch.equal(torch.cat(staging.select(next_rows)(2)), bank[next_rows])
    assert staging.staged.data_ptr() == stage_ptr
