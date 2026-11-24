#!/usr/bin/env python3
"""Score DFlash2 draft-count policies offline from a serve-glm trace.

Read the shared per-round CUTEAFD_SPECULATION_TRACE schema. At temperature 0,
the tokens emitted after each verify anchor are the target's greedy continuation,
so counterfactual acceptance from the full neural proposal can be priced
with a verify-step table:

    tokens/s = sum(1 + min(k, accepted)) / sum(step_ms(1 + k) + draft_ms)

The actual policy is scored the same way (its own k per cycle, same table),
so the comparison isolates the choice of k from step-time noise. Cycles that
used a copy window, and C>1 steps, are priced as single-sequence steps.
"""
import argparse
import json
from collections import defaultdict

# Historical rows -> verify ms (GLM 5.3 K4, 1 RTX PRO 6000 at 325 W + 4 Sparks TP4).
K4_TP4 = [(1, 35.0), (2, 50.5), (3, 60.2), (4, 67.2), (5, 77.6), (6, 85.3), (7, 92.8), (8, 102.6)]


def interpolate(table, rows):
    for (r0, m0), (r1, m1) in zip(table, table[1:]):
        if rows <= r1:
            return m0 + (m1 - m0) * (rows - r0) / (r1 - r0)
    return table[-1][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trace')
    parser.add_argument('--table', help='rows:ms,... verify-step table (default: historical K4_TP4 measurements)')
    parser.add_argument('--draft-ms', type=float, default=4.7, help='drafter step ms added to every cycle')
    args = parser.parse_args()
    table = K4_TP4 if not args.table else [tuple(map(float, p.split(':'))) for p in args.table.split(',')]
    rounds, generated, done = defaultdict(list), defaultdict(dict), set()
    with open(args.trace) as source:
        for line in source:
            record = json.loads(line)
            if 'neural_tokens' not in record:
                continue  # shared draft-policy observations are separate records
            for rid, start, draft, verify, emitted, accepted, finished in zip(
                    record['request_ids'], record['starts'], record['neural_tokens'],
                    record['verify_tokens'], record['committed_tokens'], record['accepted'], record['finished'],
                    strict=True):
                if not emitted:
                    continue
                generated[rid][start] = verify[0]
                generated[rid].update(enumerate(emitted, start + 1))
                if draft:
                    rounds[rid].append((start, draft, verify[1:], accepted))
                if finished:
                    done.add(rid)
    rows = []
    for rid in done:
        for start, draft, verified, accepted in rounds[rid]:
            truth = generated[rid]
            matched = 0
            for i, token in enumerate(draft, start + 1):
                if truth.get(i) != token:
                    break
                matched += 1
            rows.append(dict(accepted=matched, actual=len(verified),
                copy=verified != draft[:len(verified)], committed=1 + accepted))
    if not rows:
        raise SystemExit('no drafted cycles with a finished request in the trace')
    cost = lambda k: interpolate(table, 1 + k) + args.draft_ms

    def score(choose):
        tokens = sum(1 + min(choose(r), r['accepted']) for r in rows)
        ms = sum(cost(choose(r)) for r in rows)
        return tokens / len(rows), ms / len(rows), 1e3 * tokens / ms

    plain = [r for r in rows if not r['copy']]
    agree = sum(r['committed'] == 1 + min(r['actual'], r['accepted']) for r in plain)
    print(f'consistency: {agree}/{len(plain)} DFlash2 cycles committed 1 + min(verified, counterfactual) rows')
    print(f'{len(rows)} drafted cycles from {len(done)} requests; mean counterfactual acceptance '
          f"{sum(r['accepted'] for r in rows) / len(rows):.2f} of 7; histogram "
          f"{[sum(r['accepted'] == a for r in rows) for a in range(8)]}")
    print(f"{'policy':>10s} {'tok/cycle':>9s} {'ms/cycle':>9s} {'tok/s':>7s}")
    for name, choose in [('actual', lambda r: r['actual'])] + [(f'fixed {k}', lambda r, k=k: k) for k in range(8)] \
            + [('oracle', lambda r: r['accepted'])]:
        t, m, rate = score(choose)
        print(f'{name:>10s} {t:9.2f} {m:9.1f} {rate:7.2f}')


if __name__ == '__main__':
    main()
