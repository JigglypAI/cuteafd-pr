#!/usr/bin/env python3
"""Sparse host-cache study (#2576): tidy tables from selection dumps.

Reads one family's `CUTEAFD_DIAG_SELECTION_DUMP` directory (SSEL0001 layer
files, see rust/crates/cuteafd-daemon/src/shared/selection_dump.rs) and writes
tidy CSV tables, a provenance JSON and one summary figure:

  scripts/bench/sparse_selection_study.py tidy --family glm5 --dump DIR \
      --out OUT [--session SESSION.json] [--index-bytes-per-token B ...]

Every number is computed per selecting layer and per context band from the
decode records in step order (one record per decode step at C1). Definitions
(also in the dataset README):

* unit: what the selector picks (token, 4-token pool, compressed row of 1 or 2
  tokens, 128-token block per KV head). `ids` are logical, ascending.
* band: the record's engine position: 0-16K, 16-32K, 32-64K, 64-128K, 128K+.
* segment: a maximal run of records whose positions grow by exactly 1. A gap
  is a turn boundary (a new prompt was prefilled: tool result, user turn).
* working set W(N): |union of the selected units over N consecutive decode
  steps| / units in the causal frontier at the window's last step, averaged
  over non-overlapping windows inside one segment (N = 1, 16, 128) and over a
  whole segment ("session"; segments shorter than N are skipped).
* churn: Jaccard(S_t, S_{t-1}) and |S_t - S_{t-1}| per consecutive step pair;
  pairs that straddle a segment boundary are reported separately ("boundary").
* clustering: distinct pages of P tokens touched per step (P = 1, 4, 64, 128;
  MiniMax pages are per KV head), and amplification = P * pages / selected
  tokens (bytes read if paged at P over bytes needed).
* recency: distance = position - last token of the unit; shares within 1K,
  4K and 16K; quantiles; the farthest reach per step.
* shared index: layer pairs whose selections are recorded identical at every
  common step (GLM 5.3 records only full-indexer layers; shared layers reuse
  them by construction: listed from the config, verified absent in the dump).
* cache simulation (per layer, one cache per request; positions restarting
  mark a new request): LRU over units at step granularity with resident
  capacity R units (2K..32K), computed exactly from stack distances (a unit
  hits iff fewer than R distinct other units were referenced since its last
  reference; a step's units count as referenced together). `lru_pin1k` pins
  the last 1K tokens' units outside R. Misses per step = selected units not
  resident. A previous-step prefetch is the same as LRU here (the previous
  set is resident), so the prefetch figure is `source_covers`: the share of a
  layer's misses at R that the nearest earlier selecting layer (same unit)
  selected at the same step, i.e. what a one-layer-ahead prefetch could hide.
* resident bytes: page tables at 4 bytes per page per virtual token / P, plus
  the index-key bytes per token given per family, plus the smallest LRU R
  (in tokens of KV) whose miss rate per step is below 1%, 0.1% and 0.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"SSEL0001"
HEADER = 40
BANDS = [(0, 16384, "0-16K"), (16384, 32768, "16-32K"), (32768, 65536, "32-64K"),
         (65536, 131072, "64-128K"), (131072, 1 << 40, "128K+")]
WINDOWS = [1, 16, 128]
PAGES = [1, 4, 64, 128]
RESIDENT = [2048, 4096, 8192, 16384, 32768]
PIN_TOKENS = 1024


def band_of(position: int) -> str:
    return next(name for lo, hi, name in BANDS if lo <= position < hi)


def read_layer(path: Path):
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: not an SSEL0001 dump")
    layer, width, unit_tokens, source, heads, _ = struct.unpack_from("<6I", data, 8)
    rows = struct.unpack_from("<Q", data, 32)[0]
    size = 24 + 4 * width
    if len(data) != HEADER + rows * size:
        raise ValueError(f"{path}: payload length differs from {rows} rows")
    raw = np.frombuffer(data, dtype=np.uint8, offset=HEADER).reshape(rows, size)
    head = raw[:, :24].copy().view(np.dtype([("step", "<u8"), ("position", "<u8"), ("frontier", "<i4"), ("count", "<i4")]))
    ids = raw[:, 24:].copy().view("<i4").reshape(rows, width)
    return dict(layer=layer, width=width, unit_tokens=unit_tokens, source=source, heads=heads,
                step=head["step"].ravel(), position=head["position"].ravel().astype(np.int64),
                frontier=head["frontier"].ravel(), count=head["count"].ravel(), ids=ids)


def sets_of(dump) -> list[np.ndarray]:
    return [row[:count] for row, count in zip(dump["ids"], dump["count"])]


def segments(positions: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) record ranges whose positions grow by exactly one."""
    cuts = np.flatnonzero(np.diff(positions) != 1) + 1
    bounds = np.concatenate([[0], cuts, [len(positions)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def unit_tokens_of(ids: np.ndarray, dump) -> tuple[np.ndarray, np.ndarray]:
    """(first, last) token of each unit (MiniMax ids fold the KV head)."""
    unit = ids // dump["heads"]
    first = unit * dump["unit_tokens"]
    return first, first + dump["unit_tokens"] - 1


def pages_of(ids: np.ndarray, dump, page: int, position: int) -> np.ndarray:
    """Distinct (page, head) keys a set touches when paged at `page` tokens
    (a unit's tokens past `position`, the open block's tail, are not read)."""
    heads, ut = dump["heads"], dump["unit_tokens"]
    unit, head = ids // heads, ids % heads
    lo, hi = unit * ut // page, np.minimum(unit * ut + ut - 1, position) // page
    if np.all(lo == hi):
        keys = lo * heads + head
    else:
        keys = np.concatenate([np.arange(a, b + 1) * heads + h for a, b, h in zip(lo, hi, head)])
    return np.unique(keys)


def frontier_units(position: int, dump) -> int:
    """Units in the causal frontier (all heads)."""
    return max(1, (position + 1) // dump["unit_tokens"] + ((position + 1) % dump["unit_tokens"] > 0)) * dump["heads"]


class Fenwick:
    """Counts per decode step (index = step + 1), vectorized point add and prefix sum."""

    def __init__(self, n: int):
        self.n, self.tree = n + 1, np.zeros(n + 2, dtype=np.int64)

    def add(self, index: np.ndarray, value: np.ndarray):
        index = np.asarray(index, dtype=np.int64) + 1
        value = np.asarray(value, dtype=np.int64)
        while index.size:
            np.add.at(self.tree, index, value)
            index = index + (index & -index)
            keep = index <= self.n
            index, value = index[keep], value[keep]

    def prefix(self, index: np.ndarray) -> np.ndarray:
        """Sum of counts at steps < index."""
        index = np.asarray(index, dtype=np.int64)
        out = np.zeros(index.shape, dtype=np.int64)
        while np.any(index > 0):
            live = index > 0
            out[live] += self.tree[index[live]]
            index = np.where(live, index - (index & -index), 0)
        return out


def stack_distances(sets: list[np.ndarray], resets: np.ndarray, pinned: list[np.ndarray] | None, key_space: int):
    """Per step, the LRU stack distance of each selected (unpinned) unit at step
    granularity: the number of distinct other units referenced since its last
    reference (inf for a first reference in this request). Requests reset the
    cache where `resets` is set. A unit hits at capacity R iff distance < R."""
    last = np.full(key_space, -1, dtype=np.int64)
    tree = Fenwick(len(sets))
    out = []
    for t, ids in enumerate(sets):
        if resets[t]:
            live = last >= 0
            tree.add(last[live], -np.ones(int(live.sum()), dtype=np.int64))
            last[:] = -1
        keys = ids if pinned is None else np.setdiff1d(ids, pinned[t], assume_unique=True)
        prev = last[keys]
        seen = prev >= 0
        dist = np.full(len(keys), np.inf)
        if seen.any():
            # distinct units whose latest reference is at or after the unit's last one,
            # minus the unit itself
            total = tree.prefix(np.array([t]))[0]
            dist[seen] = total - tree.prefix(prev[seen]) - 1
        if seen.any():
            tree.add(prev[seen], -np.ones(int(seen.sum()), dtype=np.int64))
        tree.add(np.array([t]), np.array([len(keys)]))
        last[keys] = t
        out.append(dist)
    return out


def pinned_units(position: int, dump) -> np.ndarray:
    lo = max(0, position + 1 - PIN_TOKENS)
    units = np.arange(lo // dump["unit_tokens"], position // dump["unit_tokens"] + 1)
    return (units[:, None] * dump["heads"] + np.arange(dump["heads"])[None, :]).ravel().astype(np.int32)


def analyze_layer(dump, source_sets=None) -> dict:
    sets = sets_of(dump)
    pos = dump["position"]
    segs = segments(pos)
    out = collections.defaultdict(list)
    seg_of = np.empty(len(pos), dtype=np.int64)
    for i, (a, b) in enumerate(segs):
        seg_of[a:b] = i
    # 1. working set
    for a, b in segs:
        for n in WINDOWS + ["session"]:
            span = (b - a) if n == "session" else n
            if span < 1 or (n != "session" and b - a < n):
                continue
            for w in range(a, b - span + 1, span):
                union = np.unique(np.concatenate(sets[w:w + span])) if span > 1 else sets[w]
                last = w + span - 1
                out["working"].append(dict(window=str(n), band=band_of(int(pos[last])), position=int(pos[last]),
                                           selected=len(union), frontier=frontier_units(int(pos[last]), dump),
                                           fraction=len(union) / frontier_units(int(pos[last]), dump)))
    # 2. churn
    for t in range(1, len(sets)):
        prev, cur = sets[t - 1], sets[t]
        inter = len(np.intersect1d(prev, cur, assume_unique=True))
        union = len(prev) + len(cur) - inter
        out["churn"].append(dict(band=band_of(int(pos[t])), position=int(pos[t]),
                                 kind="boundary" if seg_of[t] != seg_of[t - 1] else "step",
                                 jaccard=inter / union if union else 1.0, entered=len(cur) - inter,
                                 size=len(cur), gap=int(pos[t] - pos[t - 1])))
    # 3. clustering and 4. recency
    for t, ids in enumerate(sets):
        if not len(ids):
            continue
        first, last = unit_tokens_of(ids, dump)
        tokens = int(np.sum(np.minimum(last, pos[t]) - first + 1))
        row = dict(band=band_of(int(pos[t])), position=int(pos[t]), units=len(ids), tokens=tokens)
        for p in PAGES:
            n = len(pages_of(ids, dump, p, int(pos[t])))
            row[f"pages_{p}"] = n
            row[f"amplification_{p}"] = p * n / tokens if tokens else math.nan
        dist = pos[t] - np.minimum(last, pos[t])
        row.update({f"share_{k}": float(np.mean(dist < k * 1024)) for k in (1, 4, 16)})
        row.update(dist_p50=float(np.percentile(dist, 50)), dist_p90=float(np.percentile(dist, 90)),
                   dist_p99=float(np.percentile(dist, 99)), dist_max=int(dist.max()),
                   reach_fraction=float(dist.max() / max(1, pos[t])))
        out["cluster"].append(row)
    # 6. cache simulation (step-granular LRU stack distances, every capacity at once)
    resets = np.zeros(len(sets), dtype=bool)
    resets[0] = True
    resets[1:] = pos[1:] < pos[:-1]  # positions restart: a new request, its own cache
    key_space = int(max((s.max() for s in sets if len(s)), default=0)) + 1
    pins = [pinned_units(int(p), dump) for p in pos]
    for policy, pinned in (("lru", None), ("lru_pin1k", pins)):
        dist = stack_distances(sets, resets, pinned, key_space)
        for t, d in enumerate(dist):
            row = dict(band=band_of(int(pos[t])), position=int(pos[t]), policy=policy, selected=len(sets[t]),
                       unpinned=len(d), first_reference=int(np.isinf(d).sum()))
            for capacity in RESIDENT:
                misses = int(np.sum(d >= capacity))
                row[f"misses_{capacity}"] = misses
                # one layer ahead: misses of this layer that an earlier selecting layer picked at this step
                if source_sets is not None and policy == "lru_pin1k":
                    keys = sets[t] if pinned is None else np.setdiff1d(sets[t], pinned[t], assume_unique=True)
                    missed = keys[d >= capacity]
                    src = source_sets.get(int(pos[t]))
                    row[f"source_covers_{capacity}"] = int(np.isin(missed, src).sum()) if src is not None else None
            out["cache"].append(row)
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def summarize(family: str, dumps: dict, per_layer: dict, index_bytes: float) -> dict:
    """Per band (and per layer), the headline numbers the README tables print."""
    table = collections.defaultdict(dict)
    for layer, out in per_layer.items():
        for band in [b for _, _, b in BANDS]:
            key = (layer, band)
            ws = [r for r in out["working"] if r["band"] == band]
            if not ws:
                continue
            for n in WINDOWS + ["session"]:
                vals = [r["fraction"] for r in ws if r["window"] == str(n)]
                table[key][f"working_{n}"] = float(np.mean(vals)) if vals else None
            ch = [r for r in out["churn"] if r["band"] == band and r["kind"] == "step"]
            bd = [r for r in out["churn"] if r["band"] == band and r["kind"] == "boundary"]
            table[key].update(jaccard_step=float(np.mean([r["jaccard"] for r in ch])) if ch else None,
                              entered_step=float(np.mean([r["entered"] for r in ch])) if ch else None,
                              entered_step_p99=float(np.percentile([r["entered"] for r in ch], 99)) if ch else None,
                              jaccard_boundary=float(np.mean([r["jaccard"] for r in bd])) if bd else None,
                              entered_boundary=float(np.mean([r["entered"] for r in bd])) if bd else None,
                              boundaries=len(bd), steps=len(ch) + 1)
            cl = [r for r in out["cluster"] if r["band"] == band]
            for p in PAGES:
                table[key][f"pages_{p}"] = float(np.mean([r[f"pages_{p}"] for r in cl]))
                table[key][f"amp_{p}"] = float(np.mean([r[f"amplification_{p}"] for r in cl]))
            for k in ("share_1", "share_4", "share_16", "dist_p50", "dist_p90", "dist_p99", "reach_fraction"):
                table[key][k] = float(np.mean([r[k] for r in cl]))
            table[key]["dist_max"] = int(max(r["dist_max"] for r in cl))
            for policy in ("lru", "lru_pin1k"):
                rows = [r for r in out["cache"] if r["band"] == band and r["policy"] == policy]
                if not rows:
                    continue
                sel = sum(r["selected"] for r in rows)
                table[key][f"{policy}_first_reference_rate"] = sum(r["first_reference"] for r in rows) / max(1, sel)
                table[key][f"{policy}_first_reference_per_step"] = float(np.mean([r["first_reference"] for r in rows]))
                for capacity in RESIDENT:
                    misses = [r[f"misses_{capacity}"] for r in rows]
                    table[key][f"{policy}_{capacity}_hit"] = 1 - sum(misses) / max(1, sel)
                    table[key][f"{policy}_{capacity}_miss_per_step"] = float(np.mean(misses))
                    table[key][f"{policy}_{capacity}_zero_miss_steps"] = float(np.mean([m == 0 for m in misses]))
                    covers = [r.get(f"source_covers_{capacity}") for r in rows]
                    if any(c is not None for c in covers):
                        table[key][f"{policy}_{capacity}_source_covers_misses"] = (
                            sum(c or 0 for c in covers) / max(1, sum(misses)))
    return {f"{layer}|{band}": v for (layer, band), v in table.items()}


def resident_bytes(dumps: dict, per_layer: dict, index_bytes: float, kv_bytes: float) -> list[dict]:
    """Bytes resident per virtual token (page tables at 4 B per page entry, index keys as
    given) and, per layer and band, the smallest simulated LRU (1K pinned window) that keeps
    the per-step miss rate within each bar, as KV tokens of that layer per request."""
    rows = [dict(item=f"page_table_P{p}", bytes_per_virtual_token=4 / p) for p in PAGES]
    rows.append(dict(item="index_keys", bytes_per_virtual_token=index_bytes))
    for layer, out in per_layer.items():
        d = dumps[layer]
        for band in [b for _, _, b in BANDS]:
            cache = [r for r in out["cache"] if r["band"] == band and r["policy"] == "lru_pin1k"]
            if not cache:
                continue
            sel = sum(r["selected"] for r in cache)
            for bar_name, bar in (("1%", 0.01), ("0.1%", 0.001), ("0", 0.0)):
                need = next((c for c in RESIDENT if sum(r[f"misses_{c}"] for r in cache) <= bar * sel
                             and (bar > 0 or all(r[f"misses_{c}"] == 0 for r in cache))), None)
                tokens = None if need is None else need * d["unit_tokens"] // d["heads"]
                rows.append(dict(item=f"resident_lru_pin1k_miss_{bar_name}", layer=layer, band=band,
                                 resident_units=need if need is not None else f">{RESIDENT[-1]}",
                                 resident_tokens=tokens, pinned_tokens=PIN_TOKENS,
                                 kv_bytes=None if tokens is None else (tokens + PIN_TOKENS) * kv_bytes))
    return rows


def figure(family: str, per_layer: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for layer, data in sorted(per_layer.items()):
        cl = data["cluster"]
        if not cl:
            continue
        x = [r["position"] for r in cl]
        axes[0, 0].plot(x, [r["share_4"] for r in cl], lw=0.5, alpha=0.6)
        ch = [r for r in data["churn"] if r["kind"] == "step"]
        axes[0, 1].plot([r["position"] for r in ch], [r["jaccard"] for r in ch], lw=0.5, alpha=0.6)
        axes[1, 0].plot(x, [r["amplification_64"] for r in cl], lw=0.5, alpha=0.6)
        cache = [r for r in data["cache"] if r["policy"] == "lru_pin1k"]
        sel = max(1, sum(r["selected"] for r in cache))
        axes[1, 1].plot(RESIDENT, [1 - sum(r[f"misses_{c}"] for r in cache) / sel for c in RESIDENT],
                        marker="o", lw=1, label=f"L{layer}")
    axes[0, 0].set(title="share of selected units within 4K tokens", xlabel="position")
    axes[0, 1].set(title="step-to-step Jaccard (within a turn)", xlabel="position")
    axes[1, 0].set(title="read amplification at 64-token pages", xlabel="position")
    axes[1, 1].set(title="LRU + pinned 1K: hit rate vs resident units", xlabel="resident units", xscale="log")
    if len(per_layer) <= 12:
        axes[1, 1].legend(fontsize=6)
    fig.suptitle(f"{family}: per-layer selection on an agentic session")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tidy(args) -> int:
    files = sorted(args.dump.glob("layer-*.bin"))
    if not files:
        raise SystemExit(f"no layer-*.bin under {args.dump}")
    dumps = {d["layer"]: d for d in (read_layer(f) for f in files)}
    by_pos = {layer: {int(p): s for p, s in zip(d["position"], sets_of(d))} for layer, d in dumps.items()}
    per_layer = {}
    for layer, d in sorted(dumps.items()):
        # Prefetch source: the nearest earlier selecting layer of the same unit
        # geometry (a model's later layers often share an index source).
        earlier = [l for l in dumps if l < layer and dumps[l]["unit_tokens"] == d["unit_tokens"] and dumps[l]["heads"] == d["heads"]]
        source = by_pos[max(earlier)] if earlier else None
        per_layer[layer] = analyze_layer(d, source)
        print(f"{args.family} layer {layer}: {len(d['position'])} records", file=sys.stderr, flush=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("working", "churn", "cluster", "cache"):
        write_csv(out / f"{kind}.csv", [dict(layer=layer, **row) for layer, data in sorted(per_layer.items()) for row in data[kind]])
    summary = summarize(args.family, dumps, per_layer, args.index_bytes_per_token)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_csv(out / "resident.csv", resident_bytes(dumps, per_layer, args.index_bytes_per_token, args.kv_bytes_per_token))
    # Cross-layer identity (shared index check).
    pairs = []
    layers = sorted(dumps)
    for i, a in enumerate(layers):
        for b in layers[i + 1:]:
            common = set(by_pos[a]) & set(by_pos[b])
            if not common:
                continue
            same = sum(np.array_equal(by_pos[a][p], by_pos[b][p]) for p in common)
            jac = [len(np.intersect1d(by_pos[a][p], by_pos[b][p])) / max(1, len(np.union1d(by_pos[a][p], by_pos[b][p]))) for p in sorted(common)]
            pairs.append(dict(layer_a=a, layer_b=b, common_positions=len(common), identical=same,
                              jaccard_mean=float(np.mean(jac)) if dumps[a]["heads"] == dumps[b]["heads"] and dumps[a]["unit_tokens"] == dumps[b]["unit_tokens"] else None))
    write_csv(out / "cross_layer.csv", pairs)
    provenance = dict(schema="cuteafd.sparse-selection-study/1", family=args.family, script=str(Path(__file__).resolve()),
                      script_sha256=sha256(Path(__file__)), dump_dir=str(args.dump.resolve()),
                      inputs={f.name: dict(sha256=sha256(f), bytes=f.stat().st_size) for f in files},
                      layers={l: dict(width=d["width"], unit_tokens=d["unit_tokens"], heads=d["heads"], source=d["source"],
                                      records=int(len(d["position"])), positions=[int(d["position"].min()), int(d["position"].max())],
                                      segments=len(segments(d["position"]))) for l, d in dumps.items()},
                      session=str(args.session) if args.session else None,
                      session_sha256=sha256(args.session) if args.session else None,
                      index_bytes_per_token=args.index_bytes_per_token, kv_bytes_per_token=args.kv_bytes_per_token,
                      pin_tokens=PIN_TOKENS, resident_units=RESIDENT, windows=WINDOWS, pages=PAGES,
                      bands=[b for _, _, b in BANDS])
    if (args.dump / "manifest.jsonl").exists():
        provenance["manifest_sha256"] = sha256(args.dump / "manifest.jsonl")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    figure(args.family, per_layer, out / "summary.png")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tidy")
    t.add_argument("--family", required=True)
    t.add_argument("--dump", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--session", type=Path)
    t.add_argument("--index-bytes-per-token", type=float, required=True,
                   help="resident indexer-key bytes per virtual token, all selecting layers")
    t.add_argument("--kv-bytes-per-token", type=float, required=True,
                   help="KV bytes per token of one selecting layer (to convert resident tokens to bytes)")
    args = parser.parse_args(argv)
    return tidy(args)


if __name__ == "__main__":
    sys.exit(main())
