"""Shared history summaries separate content and count only reached positions."""
import runpy
from pathlib import Path

summarize = runpy.run_path(str(Path(__file__).parents[1] /
                             'qualify/deepseek_v41/collect-content-acceptance.py'))['summarize_history']


def request(identity=1, **changes):
    value = dict(id=identity, room=7, proposed=3, accepted_drafts=2, positions=[
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
    assert result['all']['short_share'] == 1 / 3
    assert result['all']['explore_skipped'] == 1
    explore = next(p for p in result['all']['positions'] if p['track'] == 'neural' and p['position'] == 2)
    assert (explore['reached'], explore['accepted'], explore['explored'], explore['explore_skipped']) == (1, 0, 1, 1)
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


def test_report_joins_exact_http_ids_and_keeps_c1_content_label():
    load = runpy.run_path(str(Path(__file__).parents[1] /
                            'qualify/deepseek_v41/collect-content-acceptance.py'))['report_content']
    timing = dict(speculation=dict(request_id='http-code', proposed=7, accepted=5, cycles=2))
    report = dict(baseline=dict(card=dict(
        decode=[dict(content='code', runs=[timing]), dict(content='prose', runs=[
            dict(speculation=dict(request_id='http-prose'))])],
        concurrency=[dict(width=1, runs=[dict(timing=timing)]), dict(width=16, runs=[
            dict(timing=dict(speculation=dict(request_id='http-wide')))])])))
    content = load(report)
    assert content == {'http-code': 'code', 'http-prose': 'prose', 'http-wide': 'concurrency/16'}
    r = request(request_id='http-code')
    result = summarize([dict(type='draft_round', policy='history', requests=[r])], content)
    assert 'code' in result and 'unclassified' not in result


def test_skipped_explore_is_not_a_verified_conditional_label():
    r = request(proposed=1, accepted_drafts=1, explore_skipped=0.1,
                positions=[dict(track='neural', position=0, outcome=True, explore=False),
                           dict(track='neural', position=1, outcome=None, explore=True)])
    result = summarize([dict(type='draft_round', policy='history', requests=[r])])['all']
    assert result['positions'][1]['reached'] == 0
    assert result['positions'][1]['explore_skipped'] == 1
