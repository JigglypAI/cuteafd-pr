#!/usr/bin/env python3
"""Offline speculation replay over fixed-length draft traces (#2460).

Reads wip-cards job directories whose server ran `DRAFT_LENGTH=<block>` with
`SPECULATION_TRACE`, and writes one tidy row per (round, request) plus the
analysis the brief asks for. Analysis only: it never launches anything.

  speculation-replay.py tidy JOB_DIR... --out DIR      # tidy.jsonl + manifests
  speculation-replay.py analyze DIR --out DIR          # cycle fit, acceptance, replay

Each tidy row carries the family, card, content and phase, the request's
context at the round, the batch's rows and requests, every drafted position's
score / history estimate / outcome, and the round's host-clock times. Times come
from the round record itself (`round_us`/`verify_us`/`draft_us`, or the shared
scheduler's `cycle_ms`/`verify_ms`/`draft_ms`, or GLM 5.3's per-request
`verify_ms`).

Replay. A fixed-L trace reveals the first rejection, so for any rule that
verifies a prefix of the block, the accepted drafts are min(rule length,
accepted run) and the round commits that plus one target token. The round's
time is the family's fitted cycle time at the replayed row count, context
and concurrency. A rule never sees outcomes past the first rejection, and
only a prefix rule is replayable, which every candidate here is.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

CONTENT_KEYS = ('code', 'prose', 'json')


# ---------------------------------------------------------------- loading

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # one unterminated final record when the writer was cut


def request_index(report):
    """HTTP request id -> (panel, content, phase, c, prompt_tokens) from a smoke report.json."""
    index = {}

    def visit(node, panel, content, c, phase):
        if isinstance(node, dict):
            content = node.get('content', content) if isinstance(node.get('content'), str) else content
            c = node.get('c', c) if isinstance(node.get('c'), int) else c
            ctx = node.get('context') if isinstance(node.get('context'), int) else None
            spec = node.get('speculation')
            if isinstance(spec, dict) and spec.get('request_id'):
                index[spec['request_id']] = dict(panel=panel, content=content, c=c, phase=phase,
                    prompt_tokens=node.get('prompt_tokens'), context=ctx)
            timing = node.get('timing')
            if isinstance(timing, dict) and isinstance(timing.get('speculation'), dict):
                rid = timing['speculation'].get('request_id')
                if rid:
                    index[rid] = dict(panel=panel, content=content, c=c, phase=phase,
                        prompt_tokens=timing.get('prompt_tokens'), context=ctx)
            for key, value in node.items():
                visit(value, panel, content, c, key if key in ('warm', 'timed', 'runs', 'requests') else phase)
        elif isinstance(node, list):
            for value in node:
                visit(value, panel, content, c, phase)

    for panel in report.get('panels', []):
        visit(panel, panel.get('id'), None, None, None)
    if isinstance(report.get('baseline'), dict):  # smoke profile: the basic card's decode rows
        visit(report['baseline'], 'baseline', None, None, None)
    # Context points carry the context on the point; propagate to their requests.
    for panel in report.get('panels', []):
        if panel.get('id') != 'context_load':
            continue
        for p in panel.get('passes', []):
            for point in (p.get('points') or []):
                waves = [('timed', [r.get('timing') or {} for r in point.get('requests') or []]),
                         ('warm', point.get('warm_requests') or [])]
                for phase, timings in waves:
                    for timing in timings:
                        rid = (timing.get('speculation') or {}).get('request_id')
                        if rid:
                            index[rid] = dict(panel='context_load', content='context', c=point.get('c'),
                                phase=phase, prompt_tokens=timing.get('prompt_tokens'), context=point.get('context'))
    return index


def round_times(record):
    """(round_us, verify_us, draft_us) from whichever fields the record has."""
    if 'round_us' in record:
        return record.get('round_us'), record.get('verify_us'), record.get('draft_us')
    if 'cycle_ms' in record:
        ms = lambda k: None if record.get(k) is None else int(1000 * record[k])
        return ms('cycle_ms'), ms('verify_ms'), ms('draft_ms')
    return None, None, None


def tidy_job(job_dir):
    """Rows of one job directory: one per (round, request)."""
    job_dir = Path(job_dir)
    job = json.loads((job_dir / 'job.json').read_text())
    entry = job['entry']
    reports = sorted(job_dir.glob('reports/**/report.json'))
    index = request_index(json.loads(reports[0].read_text())) if reports else {}
    family = entry['family']
    rounds, cycles, glm_cycles = [], [], collections.defaultdict(list)
    for record in read_jsonl(job_dir / 'speculation-trace.jsonl'):
        kind = record.get('type') or record.get('kind')
        if kind == 'draft_round':
            rounds.append(record)
        elif kind == 'cycle':  # GLM 5.3: one line per verified sequence
            glm_cycles[record['id']].append(record)
        elif 'cycle_ms' in record or 'verify_ms' in record:
            cycles.append(record)
    # The shared scheduler writes its cycle record after observe: same order, same members.
    paired = len(cycles) == len(rounds)
    seen = collections.Counter()  # per-request round counter, for GLM 5.3 cycle lines
    generated = collections.Counter()
    rows = []
    for k, record in enumerate(rounds):
        cycle = cycles[k] if paired else None
        times = round_times(record) if 'round_us' in record else (round_times(cycle) if cycle else (None, None, None))
        requests = record['requests']
        contexts = record.get('context')
        if contexts is None and cycle and cycle.get('contexts'):
            contexts = [c[0] - 1 if c else None for c in cycle['contexts']]
        batch_rows = sum(q['proposed'] + 1 for q in requests)
        for i, q in enumerate(requests):
            meta = index.get(q.get('request_id') or '', {})
            context = contexts[i] if contexts and i < len(contexts) else None
            verify_ms = None
            features = []
            if glm_cycles.get(q['id']):
                n = seen[q['id']]
                if n < len(glm_cycles[q['id']]):
                    line = glm_cycles[q['id']][n]
                    context = line.get('position', context)
                    verify_ms = line.get('verify_ms')
                    features = line.get('features') or []
                seen[q['id']] += 1
            if context is None and meta.get('prompt_tokens') is not None:
                context = meta['prompt_tokens'] + generated[q['id']]
            positions = [p for p in q['positions'] if p.get('track', 'neural') == 'neural']
            panel, content, c = meta.get('panel'), meta.get('content'), meta.get('c')
            if not meta:
                # Concurrency waves carry no per-request ids: label short unmatched rounds by batch width.
                panel = 'concurrency' if context is not None and context < 4096 else 'unlabeled'
                content, c = ('distinct-v1', len(requests)) if panel == 'concurrency' else (None, None)
            rows.append(dict(
                family=family, card=job['card'], round=record['round'], lane=record.get('lane'),
                request=q['id'], request_id=q.get('request_id'),
                panel=panel, content=content, c=c, phase=meta.get('phase'),
                context=context, batch_requests=len(requests), batch_rows=batch_rows,
                block=record['block'], policy=record['policy'], room=q['room'],
                proposed=q['proposed'], accepted=q['accepted_drafts'], censored=q['censored'],
                score=[p.get('score') if p.get('score') is not None else p.get('calibrated') if p.get('calibrated') is not None
                       else (features[k][1] if k < len(features) and len(features[k]) > 1 else None)
                       for k, p in enumerate(positions)],
                history=[p.get('p') for p in positions],
                outcome=[p.get('outcome') for p in positions],
                round_us=times[0], verify_us=times[1] if times[1] is not None else (
                    None if verify_ms is None else int(1000 * verify_ms)), draft_us=times[2]))
            generated[q['id']] += q['accepted_drafts'] + 1
    return rows


def manifest(job_dir):
    job_dir = Path(job_dir)
    job = json.loads((job_dir / 'job.json').read_text())
    result = json.loads((job_dir / 'result.json').read_text()) if (job_dir / 'result.json').is_file() else {}
    keep = ('MODEL_ID', 'MODEL_REVISION', 'SPECULATOR', 'SPECULATOR_MODEL_ID', 'CORE_VIEW', 'DRAFT_LENGTH',
            'RTX_GPUS', 'SPARK_COUNT', 'GLM5_FLASH_DECODE_ROWS', 'COORDINATOR_DOCKER_INFERENCE')
    thermal = []
    for report in sorted(job_dir.glob('reports/**/report.json')):
        for panel in json.loads(report.read_text()).get('panels', []):
            for p in panel.get('passes', []):
                if isinstance(p, dict) and p.get('thermal'):
                    thermal.append(dict(panel=panel.get('id'), thermal=summarize_thermal(p['thermal'])))
    return dict(card=job['card'], family=job['entry']['family'], status=result.get('status'),
        launcher=result.get('launcher'), artifacts=result.get('artifacts'),
        settings={k: v for k, v in job['entry']['set'].items() if k in keep}, thermal=thermal)


def summarize_thermal(window):
    """Per-GPU power-cap and thermal seconds, peak temperature and min T.Limit headroom."""
    out = []
    for gpu in (window.get('gpus') or []) if isinstance(window, dict) else []:
        out.append({k: gpu.get(k) for k in ('index', 'power_cap_s', 'sw_thermal_s', 'hw_thermal_s',
            'power_brake_s', 'max_temperature_c', 'min_tlimit_margin_c', 'busy_sm_clock_mean_mhz',
            'min_power_limit_w') if k in gpu})
    return out


# ---------------------------------------------------------------- analysis

def accepted_run(row):
    """Leading accepted drafts the fixed trace proves (outcomes stop at the first rejection)."""
    n = 0
    for outcome in row['outcome']:
        if outcome is True:
            n += 1
        else:
            break
    return n


def full_block(row):
    """Whether the round verified the whole block (C16 can be cut by the row cap)."""
    return row['proposed'] >= min(row['block'], row['room'])


def context_band(context):
    if context is None:
        return 'unknown'
    for edge, name in ((2048, '0'), (12288, '8K'), (49152, '32K'), (1 << 30, '128K')):
        if context < edge:
            return name
    return '128K'


def concurrency_band(n):
    return 'C1' if n <= 1 else 'C2-4' if n <= 4 else 'C5-8' if n <= 8 else 'C9-16'


def fit_cycle(rows):
    """Linear model plus the measured table (see `fit_table`); `predict` prefers the table."""
    fit = fit_linear(rows)
    if fit is not None:
        fit['table'] = fit_table(rows)
    return fit


def fit_linear(rows):
    """Least squares round_us ~ a + b*rows + c*rows*ctx_k + d*requests*ctx_k + e*requests per family.

    ctx_k is the batch's mean context in thousands of tokens. Returns
    coefficients and the R^2 over rounds (one sample per round, not per request).
    """
    samples = [(t, rws, n, BAND_K[band]) for t, rws, n, band in round_samples(rows)]
    if len(samples) < 8:
        return None
    X = [[1., rws, rws * ck, n * ck, n] for (_, rws, n, ck) in samples]
    y = [t for (t, *_rest) in samples]
    coef = lstsq(X, y)
    pred = [sum(c * x for c, x in zip(coef, xs)) for xs in X]
    mean = statistics.fmean(y)
    ss_tot = sum((v - mean) ** 2 for v in y) or 1.
    ss_res = sum((v - p) ** 2 for v, p in zip(y, pred))
    return dict(coef=coef, r2=1 - ss_res / ss_tot, rounds=len(samples),
                terms=['1', 'rows', 'rows*ctx_k', 'requests*ctx_k', 'requests'])


def lstsq(X, y, ridge=1e-6):
    """Normal equations with a tiny ridge (pure Python; five columns)."""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (ridge if i == j else 0.) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    for col in range(k):  # Gauss-Jordan with partial pivoting
        pivot = max(range(col, k), key=lambda r: abs(A[r][col]))
        A[col], A[pivot], b[col], b[pivot] = A[pivot], A[col], b[pivot], b[col]
        d = A[col][col] or 1e-12
        for j in range(col, k):
            A[col][j] /= d
        b[col] /= d
        for r in range(k):
            if r != col and A[r][col]:
                f = A[r][col]
                for j in range(col, k):
                    A[r][j] -= f * A[col][j]
                b[r] -= f * b[col]
    return b


def round_samples(rows, timed_only=True):
    """One (round_us, rows, requests, context band) per batch round."""
    rounds = {}
    for r in rows:
        if timed_only and r.get('phase') == 'warm':
            continue
        t = r['round_us'] if r['round_us'] is not None else r['verify_us']
        if t is None:
            continue
        key = (r['card'], r['round'], r['lane'])
        e = rounds.setdefault(key, dict(t=t, rows=r['batch_rows'], n=r['batch_requests'], ctx=[]))
        if r['context'] is not None:
            e['ctx'].append(r['context'])
    return [(e['t'], e['rows'], e['n'], context_band(statistics.fmean(e['ctx']) if e['ctx'] else None))
            for e in rounds.values()]


def fit_table(rows):
    """Measured median round time by (context band, rows): the curve the replay prices.

    Rounds at every concurrency contribute to their row count, so a C8 round
    of 64 rows and a C16 round of 64 rows share one point; the request count is
    kept as a column for review but the replay prices rows within a band.
    """
    cells = collections.defaultdict(list)
    for t, rws, n, band in round_samples(rows):
        cells[(band, rws)].append(t)
    table = {}
    for (band, rws), ts in cells.items():
        if len(ts) >= 3:
            table.setdefault(band, {})[rws] = statistics.median(ts)
    return table


def predict(fit, rows, requests, ctx_k):
    """Round microseconds at `rows`: interpolate the measured table of the
    nearest context band when present, else the linear model."""
    table = fit.get('table') if isinstance(fit, dict) else None
    if table:
        band = context_band(ctx_k * 1000)
        curve = table.get(band) or table.get(min(table, key=lambda b: abs(BAND_K[b] - ctx_k)))
        xs = sorted(curve)
        if rows <= xs[0]:
            lo, hi = xs[0], xs[1] if len(xs) > 1 else xs[0]
        elif rows >= xs[-1]:
            lo, hi = xs[-2] if len(xs) > 1 else xs[-1], xs[-1]
        else:
            lo = max(x for x in xs if x <= rows)
            hi = min(x for x in xs if x >= rows)
        if hi == lo:
            return curve[lo]
        return max(1., curve[lo] + (curve[hi] - curve[lo]) * (rows - lo) / (hi - lo))
    a, b, c, d, e = fit['coef']
    return max(1., a + b * rows + c * rows * ctx_k + d * requests * ctx_k + e * requests)


BAND_K = {'0': 0., '8K': 8., '32K': 32., '128K': 128., 'unknown': 0.}


# -- rules: each maps a row to the number of drafts it would verify (a prefix length).

def rule_fixed(n):
    return lambda row: min(n, row['proposed'])


def rule_history(cut=0.5, trained_cut=0.3):
    """Dev `history`: keep while the per-token score clears its cut, plus one explore.

    Token-only drafters (no score) use the history estimate with the 0.3 cut.
    The keep prefix is followed by one explored position.
    """
    def length(row):
        scores = row['score']
        if any(s is not None for s in scores):
            c = cut
            vals = scores
        else:
            c = trained_cut
            vals = row['history']
        keep = 0
        for v in vals:
            if v is not None and v > c:
                keep += 1
            else:
                break
        return min(keep + 1, row['proposed'])
    return length


def rule_product(threshold):
    """Keep while the running product of per-token scores (or history) stays above threshold, plus one explore."""
    def length(row):
        vals = row['score'] if any(s is not None for s in row['score']) else row['history']
        product, keep = 1., 0
        for v in vals:
            product *= (v if v is not None else 0.)
            if product > threshold:
                keep += 1
            else:
                break
        return min(keep + 1, row['proposed'])
    return length


def platt(score, a=0.59, b=-0.84):
    """#2460 part 2 pooled calibration of raw candidate probability."""
    if score is None:
        return 0.
    s = min(max(score, 1e-6), 1 - 1e-6)
    z = a * math.log(s / (1 - s)) + b
    return 1 / (1 + math.exp(-z))


def calibrated(row):
    if any(s is not None for s in row['score']):
        return [platt(s) for s in row['score']]
    return [h if h is not None else 0. for h in row['history']]


def settle(lengths, members, max_rows=64):
    """Dev `settle`: fit the round into `max_rows` (anchors included), granting
    positions across requests by highest next score, each keeping a prefix."""
    budget = max_rows - len(members)
    if sum(lengths) <= budget:
        return lengths
    scores = [m['score'] if any(v is not None for v in m['score']) else m['history'] for m in members]
    granted = [0] * len(members)
    for _ in range(max(budget, 0)):
        best = None
        for i, m in enumerate(members):
            if granted[i] >= lengths[i]:
                continue
            v = scores[i][granted[i]] if granted[i] < len(scores[i]) and scores[i][granted[i]] is not None else 0.
            key = (v, -granted[i], -m['request'])
            if best is None or key > best[0]:
                best = (key, i)
        if best is None:
            break
        granted[best[1]] += 1
    return granted


def replay(rows, fit, rule, batch_rule=None, max_rows=64):
    """Expected emitted tok/s of `rule` over the rows' batches, with time from `fit`.

    A batch round is the set of rows sharing (card, round, lane). Its replayed
    rows are sum(1 + L_i); committed tokens are sum(1 + min(L_i, accepted_i)).
    `batch_rule`, when given, maps a round's rows to their lengths jointly.
    """
    rounds = collections.defaultdict(list)
    for r in rows:
        rounds[(r['card'], r['round'], r['lane'])].append(r)
    tokens = seconds = verify_rows = 0.
    for members in rounds.values():
        lengths = batch_rule(members, fit) if batch_rule else settle([rule(m) for m in members], members, max_rows)
        committed = sum(1 + min(L, accepted_run(m)) for L, m in zip(lengths, members))
        nrows = sum(1 + L for L in lengths)
        ctx = [m['context'] for m in members if m['context'] is not None]
        t = predict(fit, nrows, len(members), statistics.fmean(ctx) / 1000 if ctx else 0.)
        tokens += committed
        seconds += t / 1e6
        verify_rows += nrows
    return dict(tok_s=tokens / seconds if seconds else 0., tokens=tokens, rounds=len(rounds),
                rows_per_round=verify_rows / max(len(rounds), 1), tokens_per_round=tokens / max(len(rounds), 1))


def rule_marginal(members, fit):
    """Greedy over the whole batch: add the next row with the highest marginal expected
    tokens per marginal microsecond while that ratio beats the batch's running tok/us.

    Expected acceptance of position k is the calibrated product along the prefix.
    """
    probs = [calibrated(m) for m in members]
    lengths = [0] * len(members)
    cum = [1.] * len(members)
    ctx = [m['context'] for m in members if m['context'] is not None]
    ck = statistics.fmean(ctx) / 1000 if ctx else 0.
    n = len(members)
    rows = n
    expected = float(n)
    t = predict(fit, rows, n, ck)
    while True:
        best, gain = None, 0.
        for i, m in enumerate(members):
            if lengths[i] >= m['proposed']:
                continue
            g = cum[i] * probs[i][lengths[i]]
            if g > gain:
                best, gain = i, g
        if best is None:
            break
        t_next = predict(fit, rows + 1, n, ck)
        if (expected + gain) / t_next <= expected / t:
            break
        cum[best] *= probs[best][lengths[best]]
        lengths[best] += 1
        rows += 1
        expected += gain
        t = t_next
    return lengths


def acceptance_table(rows):
    """Per position: conditional and cumulative acceptance, by family x context band x content."""
    table = collections.defaultdict(lambda: [[0, 0, 0] for _ in range(16)])  # reached, accepted, rounds
    for r in rows:
        key = (r['family'], context_band(r['context']), r['content'] or 'unknown')
        run = accepted_run(r)
        for k, outcome in enumerate(r['outcome'][:16]):
            cell = table[key][k]
            cell[2] += 1
            if outcome is None:
                continue
            cell[0] += 1
            cell[1] += outcome is True
    out = []
    for (family, band, content), cells in sorted(table.items()):
        cumulative = 1.
        for k, (reached, accepted, total) in enumerate(cells):
            if not total:
                continue
            conditional = accepted / reached if reached else None
            cumulative = cumulative * conditional if conditional is not None else None
            out.append(dict(family=family, context=band, content=content, position=k + 1, reached=reached,
                accepted=accepted, conditional=conditional, cumulative=cumulative))
            if cumulative is None:
                break
    return out


def reliability(rows, field='score', bins=10):
    """Observed acceptance per score bin over reached positions, by family."""
    table = collections.defaultdict(lambda: [[0, 0] for _ in range(bins)])
    for r in rows:
        for value, outcome in zip(r[field], r['outcome']):
            if value is None or outcome is None:
                continue
            b = min(int(value * bins), bins - 1)
            table[r['family']][b][0] += 1
            table[r['family']][b][1] += outcome is True
    return {f: [dict(bin=f'{i/bins:.1f}-{(i+1)/bins:.1f}', n=n, accepted=a / n if n else None)
                for i, (n, a) in enumerate(cells)] for f, cells in table.items()}


RULES = {
    **{f'fixed-{n}': rule_fixed(n) for n in range(0, 8)},
    'history': rule_history(),
    **{f'product>{t}': rule_product(t) for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)},
}


def analyze(rows, weights=None):
    """Everything the report needs, from tidy rows only."""
    fits = {}
    by_family = collections.defaultdict(list)
    for r in rows:
        by_family[r['family']].append(r)
    for family, fam_rows in by_family.items():
        fits[family] = fit_cycle(fam_rows)
    cells = []
    for family, fam_rows in by_family.items():
        fit = fits[family]
        if fit is None:
            continue
        groups = collections.defaultdict(list)
        for r in fam_rows:
            groups[(context_band(r['context']), r['c'] or r['batch_requests'])].append(r)
        for (band, c), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            rounds = {(g['card'], g['round'], g['lane']) for g in group}
            # Replay only on batches where every member is in this cell: a round mixing cells is split.
            row = dict(family=family, context=band, c=c, rounds=len(rounds),
                       truncated=sum(not full_block(g) for g in group) / len(group))
            for name, rule in RULES.items():
                row[name] = replay(group, fit, rule)['tok_s']
            row['marginal'] = replay(group, fit, None, rule_marginal)['tok_s']
            best = max((k for k in row if k.startswith(('fixed', 'history', 'product', 'marginal'))), key=lambda k: row[k])
            row['best'] = best
            cells.append(row)
    return dict(fits=fits, cells=cells, acceptance=acceptance_table(rows),
                reliability=reliability(rows), mix=traffic_mix(cells, weights))


DEFAULT_MIX = {'0': 0.25, '8K': 0.25, '32K': 0.3, '128K': 0.2}


def traffic_mix(cells, weights=None):
    """Context-weighted mean tok/s per family and rule (half or more of the weight at 32K+)."""
    weights = weights or DEFAULT_MIX
    assert sum(w for k, w in weights.items() if k in ('32K', '128K')) >= 0.5 - 1e-9, 'brief: >= half the weight at 32K+'
    out = {}
    for family in {c['family'] for c in cells}:
        rules = [k for k in cells[0] if k.startswith(('fixed', 'history', 'product', 'marginal'))]
        row = {}
        for rule in rules:
            total = weight = 0.
            for band, w in weights.items():
                values = [c[rule] for c in cells if c['family'] == family and c['context'] == band]
                if values:
                    total += w * statistics.fmean(values)
                    weight += w
            row[rule] = total / weight if weight else None
        out[family] = row
    return out


def render(result):
    """Markdown tables: cycle curves, replay cells and the traffic mix."""
    lines = ['## Cycle time (median ms per batch round, by verify rows)', '']
    for family, fit in sorted(result['fits'].items()):
        if not fit:
            continue
        for band, curve in sorted(fit['table'].items(), key=lambda kv: BAND_K.get(kv[0], 0)):
            pts = ', '.join(f'{r}:{t / 1000:.1f}' for r, t in sorted(curve.items()) if r in (1, 2, 4, 8, 16, 24, 32, 48, 56, 64, 96, 128))
            lines.append(f'- {family} {band}: {pts}')
    rules = ['history', 'fixed-1', 'fixed-2', 'fixed-3', 'fixed-4', 'fixed-7', 'product>0.3', 'product>0.5', 'marginal']
    lines += ['', '## Replay tok/s by family x context x concurrency', '',
              '| Family | Context | C | rounds | trunc | ' + ' | '.join(rules) + ' | best |',
              '|' + '---|' * (len(rules) + 6)]
    for c in result['cells']:
        if c['rounds'] < 20:
            continue
        lines.append(f"| {c['family']} | {c['context']} | {c['c']} | {c['rounds']} | {c['truncated']:.2f} | "
                     + ' | '.join(f"{c[r]:.1f}" for r in rules) + f" | {c['best']} |")
    lines += ['', '## Traffic mix (25% 0, 25% 8K, 30% 32K, 20% 128K)', '', '| Family | ' + ' | '.join(rules) + ' |',
              '|' + '---|' * (len(rules) + 1)]
    for family, row in sorted(result['mix'].items()):
        lines.append(f'| {family} | ' + ' | '.join('' if row.get(r) is None else f'{row[r]:.1f}' for r in rules) + ' |')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tidy')
    t.add_argument('jobs', nargs='+')
    t.add_argument('--out', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('dataset')
    a.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == 'tidy':
        with open(out / 'tidy.jsonl', 'a') as f:
            for job in args.jobs:
                rows = tidy_job(job)
                for row in rows:
                    f.write(json.dumps(row) + '\n')
                m = manifest(job)
                (out / 'manifests').mkdir(exist_ok=True)
                (out / 'manifests' / f"{m['card']}.json").write_text(json.dumps(m, indent=1))
                timed = sum(r['round_us'] is not None for r in rows)
                print(f"{job}: {len(rows)} request-rounds, {timed} timed, "
                      f"{sum(r['context'] is not None for r in rows)} with context")
    else:
        rows = list(read_jsonl(Path(args.dataset) / 'tidy.jsonl'))
        result = analyze(rows)
        (out / 'analysis.json').write_text(json.dumps(result, indent=1, default=str))
        (out / 'report.md').write_text(render(result))
        print(json.dumps({'families': list(result['fits']), 'cells': len(result['cells'])}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
