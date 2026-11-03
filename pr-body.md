Re-submits #966 (closed before review). Rebased onto `dev` @ `ae207cb95`.

## What this is

The §2 loop-abort detector as **shared code**, wired into the shared serving path of every family — not a V4.1-only patch.

- `shared/repeat_guard.rs` (new): rules R (32-token shingle, 8th-or-later verbatim copy, run 512) and F (short unit repeated for a span), 4 knobs, 25 tests.
- `shared/loop_abort.rs` (new): the per-request guard, built at admission from the live knobs; `push(token)` returns `true` on a loop.
- Wired into `Active::emit_one` (V4.1), and the equivalent emit path in GLM 5, GLM 5.3 Flash, MiMo v2, Qwen 3.5, Qwen 4, Step 5, Kimi K3, DeepSeek V4 — every family's serve loop.
- An abort reuses the `max_tokens` finish path: `finish_reason: "length"`, additive `usage.finish_detail: "repetition"`, `totals.loop_aborts`, one `cuteafd::loop_abort` tracing line.
- **Retention fix (TJ):** `cacheable = !looped && !cold` — a looped turn is never retained in the prefix cache.
- **Generic env names (TJ):** `LOOP_ABORT_RUN`, `LOOP_ABORT_COPIES`, `LOOP_FAST_SPAN`, `LOOP_FAST_COPIES` (no `V41_` prefix), forwarded by `run-family.sh` and accepted by `release-common.sh`.
- Emission truncation in `single_lane_round`: a loop abort cuts the round's emission short, so the console counts only what was actually emitted, not the accepted-but-unemitted tail.

## Changes since #966

- Rebased onto `dev` @ `ae207cb95`; conflicts were the #1026 lane-cap/`prefill_slots` refactor and the #954 KV-reserve keys. Resolved by keeping upstream's new signatures and adding only the loop-abort wiring (the old `.min(DECODE_ROWS)` workaround is gone — upstream now validates and refuses >64).
- The Rust `#[ignore]` fidelity test is replaced by a self-contained Python harness + evidence JSON (below), so the gate runs without a GPU.

## Rework (2026-11-03) — per TJ's review of #1138

### V1 (BLOCKING) — the daemon test target now compiles
- `families/glm5/ml4_media.rs:188` now sets the required `finish_detail` field.
- Two more test-only literals were also stale against `dev` and are fixed: `v41_native_serve/console.rs:240` (`Proposal.drafted` is `Vec<u8>`, not an array) and `shared/qwen/scheduler.rs:1906` (`Active` gained a `lease` field).
- **Proof:** `cargo test -p cuteafd-daemon --no-run` -> **rc 0** (the merge blocker).

### V2 — Rule R vs Rule F are now disjoint by period
- **Chosen resolution: gate Rule R on a minimum period.** Rule R only fires for `period > FAST_WINDOW` (256); Rule F owns `period 1..=256`. The two rules no longer overlap, so the "Rule R fires first for every P <= 80" case is gone.
- **Why this one:** the shipped design is "8 copies / 1,024-token patience". Without the gate a period-1 run aborted at 550 tokens (Rule R at `7P + 543`), which silently undercut that patience for every short period. With the gate the 1,024-token span protects every short-period repeat, and Rule R keeps its job for long blocks (period > 256), which Rule F cannot see (its window is 256).
- **Behaviour change:** a period-1 loop now aborts at 1,025 tokens (Rule F) instead of 550; still far below the 32k cap. Rule F is **not** dead code — it is the only detector for `period <= 256` and fires first for `P = 81..256` (unchanged).
- **Tests at/past 550 tokens** (the old "legitimate" tests stopped at 400): `identical_run_below_rule_f_span_does_not_abort` (1,024 identical -> no abort), `identical_run_fires_rule_f_at_span` (1,025 -> Fast, period 1), `short_period_repeat_waits_for_rule_f` (period 8 -> Fast at 1,032), `long_block_still_fires_rule_r` (period 300 -> Repeat), `rule_r_is_gated_to_periods_above_the_fast_window` (256 -> Fast, 257 -> Repeat).

### V3 — fidelity gate: gap stated honestly, DEFERRED
- The local dataset has **1 of the 13** published configs (`deepseek_v41-v2_20261005`); the other 12 need a dataset download. The sweep over all 13 is **DEFERRED** (a data fetch, not a code change; the harness already takes `--config`).
- **Rule F cannot fire in a gen window:** the dataset recipe sets `max_tokens: 1024`, and Rule F needs >= 1,025 identical tokens. Rule R can fire there only for `period <= 68` (`7P + 543 <= 1,024`) and did not. The gen-row metrics also show `max_shingle_copies` 2 (`gen_rows`) / 6 (`all_roles`) — never the 7 copies Rule R needs — so the 0-fire result does not exercise Rule R's shipped threshold. The long-generation control (real 32k-token generations) is what reaches both rules.
- The evidence JSON now says this explicitly.

### V4 — PR body matches the commit
- Removed the non-existent GPU test `aborted_row_is_retired_in_the_round` and the non-existent "offline source guard".
- Test count corrected: `repeat_guard.rs` has **25** `#[test]` functions (was 19 at `bf878c5b7`; +6 for V2/N3), `loop_abort` has **12**.

### Nits
- **N1 (V4.1 round log uses pre-truncation counts):** stats only; the `cuteafd::timing` line still logs the raw `accepted`/`emitted` from `commit_lane` while the console tally is capped. Not changed (behaviour is correct; only the log/tally disagree). Noted for a follow-up.
- **N2 (usage ledger cannot distinguish a loop abort):** `finish_detail` is injected into the HTTP/SSE usage object only; `usage.jsonl` still records `"length"`. Not changed (additive schema change; noted for a follow-up).
- **N3 (`Shingle.last` initialized to `pushed - WINDOW`):** **fixed** — `first`/`last` are now both the `pushed` index of the sighting, so `pushed - last` is exactly the period. New test `second_sighting_reports_the_true_period` (copies=2, period 600 -> 600, not 632).
- **N4 (`--loop-abort-run 0` alone is not a kill switch):** documented and tested. Each rule has its own off switch; the detector is fully off only when **both** `--loop-abort-run 0` and `--loop-fast-span 0`. The misleading test is renamed `loop_abort_run_zero_disables_rule_r_only`; new `loop_abort_both_rules_zero_is_the_kill_switch` proves the full switch. `cuteafd.config` and the clap help say so.

## Tests

`cargo test -p cuteafd-daemon loop_abort` -> **12 passed / 0 failed** (V4.1 + Qwen 4 emit paths, CLI defaults/kill switch, console totals, tally truncation).
`cargo test -p cuteafd-daemon repeat_guard` -> **25 passed / 0 failed** (fixture pin, disabled, 2/7 copies, 8th-copy fire point, rule F span/copies, the V2 threshold tests, legitimate 400-token runs, map bound, N3 period).
`cargo test -p cuteafd-daemon --no-run` -> **rc 0** (V1).
Full daemon suite: see the rework comment for the before -> after counts.

**Fidelity gate (TJ):** `python3 rust/crates/cuteafd-daemon/tests/loop-abort-fidelity-harness.py --fidelity <dataset>` -> **53 transcripts / 28,820 generated tokens / 0 fires**; the crate fixture's known loops fire (`looped` -> Repeat, `fast_loop` -> Fast) and the `draft_then_write` control never fires. Evidence: `tests/loop-abort-fidelity-evidence.json`.

## Deferred

- **Fidelity over every published set:** 1 of 13 configs is local; the other 12 need a dataset download. The harness takes `--config`, so this is a data fetch, not a code change.
- **Long-generation control** (`tests/loop-abort-corpus-replay.py`) needs the opencode DB and an external `loop-replay.py` (`LOOP_REPLAY`); not runnable in CI.
- **GPU test** of same-round retirement needs `exercise_distributed_decode`; not run here (no GPU in this environment).

## Open question for TJ

Default is ON (run 512) per Jonathan's direction. For default-off, set **both** `--loop-abort-run 0` and `--loop-fast-span 0` (setting only the first leaves Rule F on).
