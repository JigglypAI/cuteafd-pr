"""Adopt full Hy4 through an offline no-copy import and a single eight-store placement."""
from hy4_flash.handoff import Placement, run_cli
from .publish import TARGET

LAYOUT = Placement(TARGET, 'hy4-k4-mc')


if __name__ == '__main__':
    run_cli(LAYOUT)
