"""Post new sealed production gate receipts to the Hy4 tracking issue."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import fcntl
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

import xxhash

from kimi_k3.uniform_k2.storage import atomic_json

ISSUE = "repos/tpurtell/cuteafd/issues/2478/comments"
MARK = "<!-- cuteafd-hy4-production-gate-v2 "
TAIPEI = ZoneInfo("Asia/Taipei")
LIMIT = 56_000  # Allow room for a repeated heading, column key and receipt marker.


def gh(*args, data=None):
    return subprocess.run(["gh", "api", *args], input=data, capture_output=True,
                          check=True, text=True, timeout=90).stdout


def posted_parts():
    user = gh("user", "--jq", ".login").strip()
    # The API paginates in creation order; only fetch the matching comment's
    # marker and ID into memory. Ignore text copied into someone else's reply.
    lines = gh(ISSUE + "?per_page=100", "--paginate", "--jq",
               '.[] | select(.user.login == ' + json.dumps(user) +
               ' and (.body | startswith(' + json.dumps(MARK) + '))) | '
               '[ (.body | split("\\n")[0]), (.id | tostring) ] | @tsv')
    return {key: int(value) for line in lines.splitlines() for key, value in [line.split("\t", 1)]}


def forecast(events, stage, as_of):
    if not stage.startswith("target-"):
        return "Target layers complete (0 remaining)."
    layer = int(stage.split("-")[1])
    remaining = 41 - layer
    if not remaining:
        return "Target layers complete (0 remaining)."
    starts = Counter(r["current_layer"] for r in events if r["phase"] == "target_layer")
    interrupted = {r["current_layer"] for r in events if r["phase"] == "expert_group_replayed"}
    seconds = [r["layer_seconds"] for r in events if r["phase"] == "target_layer_complete"
               and 0 < r["current_layer"] <= layer and starts[r["current_layer"]] == 1
               and r["current_layer"] not in interrupted and r["layer_seconds"] > 0][-6:]
    if not seconds:
        return "Target-layer time remaining and finish time unavailable until a complete uninterrupted MoE layer is timed."
    values = [remaining * value for value in seconds]
    mid = sum(values) / len(values)
    end = as_of + timedelta(seconds=mid)
    estimate = (f"Target layers: {remaining} left, estimated {mid/3600:.2f} h, "
                f"finishing {end:%Y-%m-%d %H:%M} Asia/Taipei, based on {len(values)} uninterrupted MoE layer(s).")
    if len(values) >= 2:
        return estimate + f" Observed timing range gives {min(values)/3600:.2f}–{max(values)/3600:.2f} h."
    return estimate + " Uncertainty unavailable from one sample."


def comment_parts(receipt, status_line, as_of):
    label = receipt["stage"]
    rows = receipt["projections"]
    intro = [f"**Hy4 production gate `{label}` — {receipt['status']}**",
             f"Gate took {receipt['seconds']/60:.2f} min through held-out checks. Measured {as_of:%Y-%m-%d %H:%M} Asia/Taipei.",
             status_line,
             "Head, DFly2, Engram export, upload and placement are separate from the target-layer estimate."]
    if receipt.get("recovery"):
        r = receipt["recovery"]
        intro.append(f"Of {r['experts']} experts, {r['zero_natural']} had no natural rows, "
                     f"{r['augmented']} used ranked candidates and {r['identity']} used identity padding. "
                     f"Minimum natural/effective observations: {r['min_natural']}/{r['min_effective']}. "
                     f"Total augmented/identity observations: {r['augmented_rows']}/{r['identity_rows']}.")
    if receipt.get("engram"):
        g = receipt["engram"]
        intro.append(f"Engram calibration addressed {g['unique_rows']:,} distinct rows in {g['lookups']:,} lookups; "
                     f"{g['repeated_lookups']:,} repeats and {g['untouched_rows']:,} untouched table rows.")
    sizes = receipt["size_estimate"]["view_bytes"]
    tp = receipt["tp2_even_weight_bytes"]
    intro.append("Projected weight GiB (view / even TP2 per device, excluding cache and workspace): " +
                 ", ".join(f"{name} {n/2**30:.2f} / {tp[name]/2**30:.2f}" for name, n in sizes.items()) + ".")
    if "dfly_top1_agreement" in receipt:
        a = receipt["dfly_top1_agreement"]
        intro.append(f"DFly2 top-1 on {receipt['compared_slots']} held-out slots: K8 {a['8']:.5f}, K6 {a['6']:.5f}. "
                     + receipt["comparison"] + ".")
    if "published_k6_drafter_bits" in receipt:
        intro.append(f"K6 view uses K{receipt['published_k6_drafter_bits']} DFly2.")
    if "hidden_correction" in receipt:
        intro.append(f"DFly2 hidden correction: {receipt['hidden_correction']}, "
                     f"{receipt['hidden_correction_parameters']:,} parameters.")
    intro.append("Activation-weighted held-out NMSE limits: K8 0.0001, K6 0.001, K4 0.01.")
    yield "\n\n".join(intro)

    core = []
    experts = {}
    for row in rows:
        module = row["module"]
        match = re.fullmatch(r"model\.layers\.[0-9]+\.mlp\.experts\.([0-9]+)\.(gate|up|down)_proj", module)
        if not match:
            core.append(f"| `{module}` | {row['bits']} | {row['heldout_nmse']:.9g} | "
                        f"{row['heldout_rows']} | {row['effective_bits_per_weight']:.7g} | "
                        f"{'native fallback' if row['kept_native'] else 'pass' if row['heldout_passed'] else 'miss'} |")
        else:
            experts.setdefault(int(match[1]), {})[match[2]] = row
    core_title = ("Core gates (last 16 rows per held-out prompt, all output columns)\n\n"
                  "| Module | Bits | NMSE | Rows | Scale-inclusive bpw | Decision |\n"
                  "|---|---:|---:|---:|---:|---|")
    expert_title = ("Routed gates. NMSE and scale-inclusive bpw are ordered gate/up/down. "
                    "A zero natural count marks a held-out latent-tail probe excluded from the model output.\n\n"
                    "| Expert | Gate NMSE | Up NMSE | Down NMSE | Bpw (g/u/d) | Evaluated rows | Natural rows |\n"
                    "|---:|---:|---:|---:|---|---:|---:|")
    expert_lines = []
    for i, modules in sorted(experts.items()):
        ordered = [modules[k] for k in ("gate", "up", "down")]
        numeric = " | ".join(f"{r['heldout_nmse']:.9g}" for r in ordered)
        bpw = "/".join(f"{r['effective_bits_per_weight']:.7g}" for r in ordered)
        first = ordered[0]
        expert_lines.append(f"| {i} | {numeric} | {bpw} | {first['heldout_rows']} | {first['heldout_natural_rows']} |")
    for heading, lines in ((core_title, core), (expert_title, expert_lines)):
        page = []
        size = 0
        for line in lines:
            size += len(line.encode("utf-8")) + 1
            if size > LIMIT and page:
                yield "\n".join([heading, *page])
                page = []
                size = len(line.encode("utf-8")) + 1
            page.append(line)
        if page:
            yield "\n".join([heading, *page])


def read_events(path):
    rows = []
    if path.exists():
        with path.open() as stream:
            for line in stream:
                # The producer may still be appending its last event.
                if line.endswith("\n"):
                    rows.append(json.loads(line))
    return rows


def seal_event(row, label):
    phase = row["phase"]
    if phase == "group_frontier_recovered":
        return row.get("group") == label
    if phase == "head_complete":
        return label == "head"
    if phase == "target_layer_complete":
        return label == f"target-{row['current_layer']:03d}"
    if phase == "drafter_layer_complete":
        return label == f"draft-{row['current_drafter_layer']:03d}"
    return False


def report(root):
    receipts = sorted((root / "gates").glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    status_path = root / "status.json"
    status = json.loads(status_path.read_bytes()) if status_path.exists() else {}
    error = status.get("error", "")
    failed = (status.get("status") == "blocked" and error.startswith("ValueError:")
              and any(key in error for key in (" K4 held-out NMSE ", " K6 training error ",
                                               "nonfinite or zero-energy held-out projection")))
    if not receipts and not failed:
        return
    identity_bytes = (root / "identity.json").read_bytes()
    identity = json.loads(identity_bytes)
    if identity.get("schema") != "cuteafd-hy4-causal-mc-v2" or identity.get("codebook") != "mul1":
        raise ValueError("refusing to report a different production identity")
    run_id = xxhash.xxh3_128_hexdigest(identity_bytes)
    directory = root / "gate-posts"
    directory.mkdir(exist_ok=True)
    with (directory / "post.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        remote = None
        events = read_events(root / "events.jsonl")
        if failed:
            # A hard projection gate stops before its group can seal. Report
            # the observed alarm as such, never advertise that stage as passed.
            digest = xxhash.xxh3_128_hexdigest(error.encode())
            local = directory / f"failure-{digest}.json"
            if not local.exists():
                remote = posted_parts()
                marker = f"{MARK}{run_id}/failed/{digest}/1 -->"
                if marker not in remote:
                    body = (marker + "\n\n**Hy4 production stopped at a projection gate**\n\n" +
                            error + f"\n\nLayer {status.get('current_layer', 'unknown')}, "
                            f"{status.get('expert_projections', 0)} sealed expert projections. "
                            "Time remaining and Asia/Taipei completion time unavailable while blocked.")
                    remote[marker] = int(gh(ISSUE, "--method", "POST", "--input", "-", "--jq", ".id",
                                            data=json.dumps({"body": body})))
                atomic_json(local, dict(run=run_id, error=error, comment_ids=[remote[marker]]))
        for path in receipts:
            payload = path.read_bytes()
            receipt = json.loads(payload)
            label = path.stem
            if (receipt.get("schema") != "hy4-production-gate-v1" or receipt["stage"] != label
                    or not re.fullmatch(r"(?:target|draft)-[0-9]{3}|head", label)):
                raise ValueError("invalid Hy4 production gate receipt")
            digest = xxhash.xxh3_128_hexdigest(payload)
            local = directory / path.name
            if local.exists():
                proof = json.loads(local.read_bytes())
                if (proof["run"], proof["receipt"]) != (run_id, digest):
                    raise ValueError(f"already posted Hy4 gate changed: {label}")
                continue
            complete = any(seal_event(row, label) for row in events)
            if not complete and not receipt["status"].startswith("failed"):
                continue
            # Small durable event/metric receipts are independent of the report generator.
            metrics = json.loads((root / "metrics" / path.name).read_bytes())
            verified = {(m["module"], m["bits"]): m for m in metrics}
            projections = receipt["projections"]
            report_keys = {(row["module"], row["bits"]) for row in projections}
            if (len(verified) != len(metrics) or len(projections) != len(metrics)
                    or verified.keys() != report_keys or any(any(verified[(r["module"], r["bits"])].get(k) != v for k, v in r.items())
                           for r in receipt["projections"])):
                raise ValueError(f"{label} gate and committed projection evidence disagree")
            as_of = datetime.fromtimestamp(path.stat().st_mtime, TAIPEI)
            sections = list(comment_parts(receipt, forecast(events, label, as_of), as_of))
            if remote is None:
                remote = posted_parts()
            ids = []
            for part, body in enumerate(sections, 1):
                marker = f"{MARK}{run_id}/{label}/{digest}/{part} -->"
                if marker not in remote:
                    header = f"`{label}` · {part}/{len(sections)}\n\n" if part > 1 else ""
                    data = json.dumps({"body": marker + "\n\n" + header + body})
                    remote[marker] = int(gh(ISSUE, "--method", "POST", "--input", "-", "--jq", ".id", data=data))
                ids.append(remote[marker])
            atomic_json(local, dict(run=run_id, receipt=digest, issue=2478, comment_ids=ids))
            print(f"{label}: {len(ids)} comments verified on #2478", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Hy4 numerical production directory")
    report(parser.parse_args().root)


if __name__ == "__main__":
    main()
