from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy4_flash.progress import layer_progress


def test_just_completed_layer_does_not_count_twice_against_forecast():
    prior = dict(phase="target_layer_complete", current_layer=1, layer_seconds=100)
    status = dict(phase="target_layer_complete", current_layer=1, completed_target_layers=2,
                  target_layers=42, expert_projections=1536, total_expert_projections=1536)
    report, count = layer_progress(datetime.now(timezone.utc), status, [prior])
    assert count == 1 and report["remaining_seconds"] == 4000
    status.update(phase="expert_group_complete", current_layer=2, expert_projections=768)
    report, _ = layer_progress(datetime.now(timezone.utc), status, [prior])
    assert report["remaining_seconds"] == 3950
