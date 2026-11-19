#!/usr/bin/env python3
"""Summarize shared history draft_round JSONL, optionally grouped by request content.

This reads an existing SPECULATION_TRACE from any family; it never launches
traffic. --report joins request-local speculation usage to trace HTTP request ids;
--request-content instead accepts an explicit id-to-content JSON mapping.
Terminal rounds retain their verified counts; per-position acceptance uses only
positions the verifier reached, excluding censored suffixes and any synthetic
skips found in historical traces.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize_history(records, content=None):
    groups = defaultdict(list)
    for record in records:
        if record.get('type') != 'draft_round':
            continue
        if record.get('policy') != 'history':
            raise ValueError('expected history policy, not a fixed-width or off trace')
        for request in record['requests']:
            request_id = request.get('request_id')
            label = (content or {}).get(request_id, (content or {}).get(str(request['id']), 'unclassified'))
            for group in {'all', label}:
                groups[group].append({**request, 'block': record.get('block', 2)})
    result = {}
    for name, requests in sorted(groups.items()):
        proposed = sum(r['proposed'] for r in requests)
        accepted = sum(r['accepted_drafts'] for r in requests)
        eligible = [r for r in requests if min(r['room'], r.get('block', 2)) > 1]
        copies = [p for r in requests for p in r['positions']
                  if p['track'] == 'copy' and p['outcome'] is not None]
        copy_accepted = sum(p['outcome'] for p in copies)
        positions = defaultdict(lambda: [0, 0, 0, 0])
        for request in requests:
            skipped = request.get('explore_skipped') is not None
            for position in request['positions']:
                counts = positions[(position['track'], position['position'])]
                if skipped and position['track'] == 'neural' and position['explore']:
                    counts[3] += 1
                elif position['outcome'] is not None:
                    counts[0] += 1
                    counts[1] += int(position['outcome'])
                    counts[2] += int(position['explore'])
        result[name] = dict(
            request_rounds=len(requests), requests=len({r['id'] for r in requests}),
            proposed=proposed, accepted=accepted,
            acceptance=accepted / proposed if proposed else None,
            rows_per_round=1 + proposed / len(requests),
            short_rounds=sum(r['proposed'] <= 1 for r in eligible),
            short_share=sum(r['proposed'] <= 1 for r in eligible) / len(requests),
            explore_skipped=sum(r.get('explore_skipped') is not None for r in requests),
            copy_reached=len(copies), copy_accepted=copy_accepted,
            copy_acceptance=copy_accepted / len(copies) if copies else None,
            positions=[dict(track=track, position=position, reached=n, accepted=ok,
                            explored=explore, explore_skipped=skip,
                            acceptance=ok / n if n else None)
                       for (track, position), (n, ok, explore, skip) in sorted(positions.items())])
    return result


def report_content(report):
    """Use request-local usage identities, never concurrent global deltas."""
    card = report['baseline']['card']
    content = {}
    for point in card['decode']:
        for run in point.get('runs', []):
            if run.get('speculation'):
                content[run['speculation']['request_id']] = point['content']
    for point in card.get('concurrency', []):
        for run in point.get('runs', []):
            spec = run['timing'].get('speculation')
            if spec:
                content.setdefault(spec['request_id'], f"concurrency/{point['width']}")
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument('--request-content', type=Path)
    mapping.add_argument('--report', type=Path, help='report.json: join content via usage request_id')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    content = json.loads(args.request_content.read_text()) if args.request_content else None
    if args.report:
        content = report_content(json.loads(args.report.read_text()))
        if not content:
            parser.error('report has no request-local speculation usage identities')
    report = dict(trace_sha256=hashlib.sha256(raw).hexdigest(),
                  content=summarize_history((json.loads(line) for line in raw.splitlines() if line.strip()), content))
    if not report['content']:
        parser.error('no shared history draft_round records')
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
