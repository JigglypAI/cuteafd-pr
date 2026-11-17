"""ext-launch.sh must pin Antigravity to a Carbon model, never a bare --effort.

Antigravity's model-list reset made its own default Argon, so the single-turn
launcher follows ext-agent.py (#2388): an explicit model wins, otherwise the
effort selects gemini-4-carbon-{high,medium,low}, and neither gets
gemini-4-carbon-high. Grok keeps its bare --reasoning-effort.
"""
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'agents' / 'ext-launch.sh'

# Records its argv NUL-separated so the prompt (which contains newlines) stays
# exactly one argument.
AGENT_STUB = '''#!/usr/bin/env python3
import os, sys
with open(os.environ['FAKE_ARGV'], 'w') as f:
    f.write('\\0'.join(sys.argv[1:]))
'''

# Runs the session command in the foreground, then reports no session, so the
# launcher finishes without a real tmux server or a 30 s poll.
TMUX_STUB = '''#!/bin/sh
case "$1" in
  has-session) exit 1 ;;
  new-session) n=$#; shift $((n - 1)); exec bash -c "$1" ;;
esac
exit 0
'''


def launch(tmp_path, tool, *extra):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stub = bin_dir / 'agent-stub'
    stub.write_text(AGENT_STUB)
    stub.chmod(0o755)
    for name in ('agy', 'grok'):
        (bin_dir / name).symlink_to(stub)
    tmux = bin_dir / 'tmux'
    tmux.write_text(TMUX_STUB)
    tmux.chmod(0o755)
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'task.md').write_text('brief\n')
    argv_file = tmp_path / 'argv'
    env = {
        **os.environ,
        'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
        'AGENT_REPO': str(SCRIPT.parents[2]),
        'AGENT_RUNS': str(runs),
        'FAKE_ARGV': str(argv_file),
    }
    result = subprocess.run(['bash', str(SCRIPT), tool, 'task', *extra],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return (argv_file.read_text()).split('\0')


def value(argv, flag):
    assert flag in argv, argv
    return argv[argv.index(flag) + 1]


def test_agy_pins_carbon_high_by_default(tmp_path):
    argv = launch(tmp_path, 'agy')
    assert value(argv, '--model') == 'gemini-4-carbon-high'
    assert '--effort' not in argv


@pytest.mark.parametrize('effort', ['high', 'medium', 'low'])
def test_agy_effort_maps_to_the_matching_carbon_model(tmp_path, effort):
    argv = launch(tmp_path, 'agy', '', effort)
    assert value(argv, '--model') == f'gemini-4-carbon-{effort}'
    assert '--effort' not in argv


def test_agy_explicit_model_wins(tmp_path):
    argv = launch(tmp_path, 'agy', 'gemini-4-argon-low', 'high')
    assert value(argv, '--model') == 'gemini-4-argon-low'
    assert '--effort' not in argv


def test_agy_unrecognized_effort_falls_back_to_carbon_high(tmp_path):
    argv = launch(tmp_path, 'agy', '', 'turbo')
    assert value(argv, '--model') == 'gemini-4-carbon-high'
    assert '--effort' not in argv


def test_grok_keeps_its_bare_reasoning_effort(tmp_path):
    argv = launch(tmp_path, 'grok')
    assert value(argv, '--reasoning-effort') == 'high'
    assert '--model' not in argv and '-m' not in argv


def test_grok_explicit_model_is_unchanged(tmp_path):
    argv = launch(tmp_path, 'grok', 'grok-4.7-build-fast')
    assert value(argv, '-m') == 'grok-4.7-build-fast'
    assert value(argv, '--reasoning-effort') == 'high'
