"""Shared history summaries separate content and count only reached positions."""
import runpy
from pathlib import Path

summarize = runpy.run_path(str(Path(__file__).parents[1] /
                             'qualify/deepseek_v41/collect-content-acceptance.py'))['summarize_history']


def request(identity=1, **changes):
    value = dict(id=identity, room=7, proposed=3, accepted_drafts=2,
                 explore_skipped=None, positions=[
                     dict(track='neural', position=0, outcome=True, explore=False),
                     dict(track='neural', position=1, outcome=True, explore=False),
                     dict(track='neural', position=2, outcome=False, explore=True),
                     dict(track='copy', position=0, outcome=None, explore=True)])
    value.update(changes)
    return value


def test_content_counts_and_censored_copy_positions():
    records = [dict(type='cycle'), dict(type='draft_round', policy='history', requests=[
        request(), request(2, proposed=1, accepted_drafts=0, explore_skipped=0.1),
        request(3, room=0, proposed=0, accepted_drafts=0, positions=[])])]
    result = summarize(records, {'1': 'code', '2': 'prose', '3': 'code'})
    assert result['all']['proposed'] == 4
    assert result['all']['accepted'] == 2
    assert result['all']['acceptance'] == 0.5
    assert result['all']['short_share'] == 0.5
    assert result['all']['explore_skipped'] == 1
    assert result['code']['requests'] == 2
    assert result['code']['copy_reached'] == 0
    assert result['code']['copy_acceptance'] is None
    assert result['prose']['short_share'] == 1


def test_copy_acceptance_and_empty_trace():
    r = request(positions=[dict(track='copy', position=0, outcome=True, explore=True),
                           dict(track='copy', position=1, outcome=False, explore=True)])
    result = summarize([dict(type='draft_round', policy='history', requests=[r])])
    assert result['all']['copy_reached'] == 2
    assert result['all']['copy_acceptance'] == 0.5
    assert summarize([]) == {}


def test_fixed_width_is_not_reported_as_history():
    import pytest
    with pytest.raises(ValueError, match='expected history'):
        summarize([dict(type='draft_round', policy='fixed 7', requests=[request()])])
