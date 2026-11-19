# MoE quantization brief template: direct MC, MUL1, rolling recovery

**Every quant brief starts from this file.** To write one, copy it whole, keep every section, and append a **Model addendum** (§14) with only what's specific to the checkpoint. The addendum never restates or weakens a template rule. If a model truly needs an exception, the addendum names the rule and gives the reason.

These rules come from past runs: Inkling 2, Kimi K3/K3.1, GLM 5.3/Flash, ML4, Muse, Step-5, MiniMax, V4.1 and Hy4. Each **Why** names the run that paid for it. Most were found mid-run, and cost a stopped quant, a restart or a thrown-away WIP. **They're the required design, not later optimizations:** build them in from the first line of code. Survey evidence and conflicts: the #2511-era quant-learnings sweep (2026-11-19).

**What not to copy.** `quantization/inkling2/output_io.py`, `io_resume.py` and the expert-output parts of `inkling2/run.py` (≈ 311–366) carry an "ASCII dragon" warning from TJ: *wrong design, do not copy* (`4a5cdcaf17`, on `quant/hy4-flash-k4-mc`). They're a compatibility shim for a frozen run.
- **Safe to copy:** Inkling's route spool and recovery (`storage.py`, `run.py` `router()`), its activation segments with XXH3 (`2a95c7c8af`), the MC exporter (`export.py`), and the metadata builders.
- **Not safe to copy:** its expert-output spool.

## 0. Operating context (fill in per run)
- **Who runs it:** TJ, in Codex on maxq. Run from the engine checkout `~/Developer/cuteafd` at the latest **origin/dev**, with recursive submodules, or a task worktree of it. Local `dev` checkouts go stale, so always fetch. Quant code lives under `quantization/<family>/`.
- **Quantizer:** GPTQModel is the submodule `third_party/gptqmodel` (remote `tpurtell/GPTQModel`).
  - Make quantizer changes there, merge them to **fork main** and push, then bump the engine's submodule pin and `third_party/gptqmodel.lock.json`, and encode from that.
  - Production refuses a dirty fork, or a fork that differs from the pin.
  - Check fork main with `git ls-remote`, not a local clone.
- **Upload:** approved in the brief, with the full lifecycle (§10).
- **Decisions go to TJ, not to main.** Report anything that changes numerics before acting on it. When TJ says stop testing, run only the production arm, with no extra A/Bs.
- **Re-verify every architecture fact** in the brief against the complete download before starting. **Why:** briefs written mid-download have been wrong: the Hy4 drafter KV layers, its Engram and LatentMoE naming, and Inkling's 2× expert-size estimate.

## 1. Recipe: direct MC, one causal pass
- **Package:** one MC package, schema `cuteafd-mc-v1`. The authoritative spec is `docs/mc-checkpoints.md`; read all of it first. It holds shared routed experts plus three indexed views:
  - `original`: the source core byte for byte, in its native dtype;
  - `k8-compact`;
  - `k6-compact`.
- **Every MC family must serve original, k8, k6 and fp8-compact uniformly.** A view the engine refuses is a release gap.
- **Routed experts:** EXL3 at the addendum's bit width (K4 by default), encoded once and shared by every view.
- **Core: two encodes from one Hessian.** Each core module is quantized to **both K8 and K6 from the same Hessian**.
  - Downstream layers, experts and the drafter see the **K8 reconstruction**. K6 is a second encode with no separate pass.
  - Record `upstream_activations: k8-compact` in `quantize_config.k6.json`.
  - Core widths are exactly {4, 6, 8}.
- **Bundled drafter** (MTP, DFlash, dSpark or DFly): it rides the core views, calibrated in the same pass from the K8-core target's hidden states, exactly as the reference runs it.
  - `original` restores the native drafter **byte for byte**. **Why:** published original views once carried a K4 MTP and needed repair (`c5b1e4aeab`).
- **Write the MC package directly into one output directory.** No intermediate plain packages, staging copies or server-side parent copies.
- **FP8-origin sources aren't MC** and never get a quantized core. Start from the BF16 original whenever one exists.
- **Keep native, byte for byte, in every view:** embeddings, router weights and biases, all norms, sinks, the DSA/compressed indexer, absorbed `kv_b_proj`, recurrent/KDA pieces, vision towers, the drafter's special heads, and any lookup tables (see the addendum).

## 2. EXL3 contract: MUL1 codebook
- **Codebook (TJ 2026-11-19):** **MUL1** (multiplier `0x83DCD12D`) for every EXL3 tensor, experts and core K8/K6 alike.
  - Write `.trellis`/`.suh`/`.svh`/`.mul1`, with `out_scales: auto`.
  - Whole projections record side 1.
  - A padded head records logical vs stored N, padded to 128, with zero padding scales.
- **Quantizer support:** GPTQModel already handles MUL1 (`gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py`, `codebook_mul1_mult`, `mul1` in `quant_args`).
- **Record the codebook** in `quantize_config*.json`, `cuteafd-mc.json` and the model card's recipe table.
- **Engine status:** the core path loads MUL1. **The routed-expert kernels require MCG today**: `deepseek_v41/v41_exl3.rs:489` ("routed expert kernels require MCG"), and the fork's `w4a16/prepare.py` refuses the MUL1 runtime marker. Expert MUL1 support is separate engine work, so a MUL1-expert checkpoint won't serve until it lands. Produce the correct MUL1 checkpoint anyway.
- **Don't use an E4M3-rounded codebook at K6.** **Why:** it failed all 20 pilot tensors at 2.28–2.39× NMSE.

## 3. Calibration data
- **Corpus:** the pinned panel `wrldsuksgo2mars/agentic-quant-panel-v1@875ff1d3`, hash-attested. Its calibration, screening and heldout splits are disjoint by group and prompt hash. It's 1,426 prompts, about 1.08M tokens, records of 512–1,024 tokens covering code/agentic, multilingual wiki, math, reasoning-termination and structured/function-call traffic. Every family reuses `quantization/kimi_k3/prepare_corpus.py`.
  - Optional, TJ's call: add the ExLlamaV3 standard set (`exllamav3/conversion/standard_cal_data`, 250 × 2,048 tokens by default; wiki, C4, code, multilingual, technical, plus 15% random-token rows) as a separate, labelled, hashed component. It adds about 47% tokens.
  - Its rows are **raw text, not chat-rendered**, and the random rows are calibration-only.
  - Gate it with a pilot-layer A/B: held-out NMSE plus the §4 recovery distribution, with and without it.
- **Rendering:** use **this checkpoint's own** tokenizer and chat template. One unpadded prompt per batch, in dataset order. No truncation, padding, sorting or packing. Freeze the template date.
  - **Never re-encode rendered text.** Re-encoding erases special-token permissions (`kimi_k3/prepare_corpus.py:143`).
  - Never reuse another model's token IDs.
- **Thinking/effort mode:** state it explicitly in the addendum. Kimi, GLM and Inkling used effort `max`.
- **Long-context witnesses** (past the SWA window or the sparse top-k cutoff) are `hessian_eligible=False`, and used only for parity checks.
  - The panel's longest prompt is about 1.2K tokens, so long attention paths aren't exercised in the Hessians. If the addendum needs them in the Hessians, it must add them explicitly. Otherwise say that they aren't.
- **Pilot slices** use real prompt **tails**, not prefixes, because prefixes are mostly the shared template preamble (`f48ad65c5b`).
- **Transformers:** pin the family-matched version, or the reference PR head, and record it in provenance.
- **Preflight:** the calibration forward matches the reference implementation on a few prompts, covering every special path, before any Hessian is captured.

## 4. Routed-expert coverage: low-row recovery (required)
**Every routed expert gets at least 1024 Hessian observations.**

**Why:** at 384–512 experts per layer, some experts get few or **zero** natural rows. Hy4's run **stopped at layer 26, on an expert with no natural calibration rows** (`"Hy4 layer 26 expert 456 has no natural calibration rows"`), and its first brief left this out. Inkling handled it: of 18,432 experts, 327 needed candidates and 134 needed the identity fallback.

**Never raise on, or silently skip, a low-row expert.** Fill it in this order:
1. **Natural rows:** the reference router's native top-k routes, in corpus order.
2. **Near-expert candidates:** if it's still short of 1024, add rows where this expert ranks **k+1..2k** by the reference router's own ranking score. The bias and correction terms affect *ranking* exactly as the reference applies them.
   - Fill rank-major, then corpus order, until the expert reaches 1024. A row never appears at two recovery ranks.
   - **Weight each candidate with renormalized hypothetical gamma:** substitute it for the last native route (rank k), recompute the reference's routing normalization over that hypothetical set, then apply the routed scaling factor.
3. **Identity fallback:** if it's still short, add `missing × I` to the **raw XᵀX, before count normalization**. That's contract `raw-XTX-missing-times-I-equivalent-to-normalized-2I` (`inkling2/storage.py:240`). It keeps count above 0, so GPTQModel's uncalibrated `q_fallback` path (`exl3_lib/quantize.py`, `count==0` or `diag_mean<1e-20`) can never trigger for a routed expert.

- **Reference implementation (copy and generalize):**
  - `quantization/inkling2/run.py` `router()` (≈ 230–246);
  - `quantization/inkling2/storage.py` `RouteSpool` (≈ 47–112, 213–247), with `minimum=1024` and `CANDIDATE_GAMMA_CONTRACT = native-last-top8-substitution-renormalized-v1`.
  - **The Inkling code hardcodes top-8:** `ids[:, :7]`, the 8-wide sum, and `_routes_valid` shape `[:,8]`. Generalize it to the model's top-k and routing function (softmax or sigmoid, biases, scaling).
- **Recovery ranks by top-k, from past recipes:** k=8 uses ranks 9–16 (GLM, GLM Flash, Step-5, Muse), k=16 uses 17–32 (Kimi), and k=4 uses 5–8 (ML4).
- **Provenance per expert:** natural rows, augmented rows with their ranks, identity rows, effective rows, and a selection hash. Record it in the receipts and the per-module CSV. The pilot reports the distribution before the full run.
- **Routed experts only.** Dense FFNs and shared experts use `core_identity_rows=0`.
- **Known disagreement:** for down-projection Hessians, V4.1 used unit route weight, and Kimi used none. Use the renormalized candidate gamma above unless the addendum justifies otherwise.

## 5. Numerics
- **Covariance:** FP32 products bounded to 128 rows, a persistent **FP64** outer sum, and FP32 storage (`fp32-products-k128-fp64-sum-fp32-storage-v1`). **Why:** Muse's FP32 outer sum gave 7.97e-6 matrix error and an indefinite Hessian; fixed to 2.85e-8 by `4ac29e1e2c`.
- **Factorization:** checked. **On failure, fix the input. Never retry with more damping.** `sigma_reg=0.025` (× `diag_mean`) is the only regularizer, with `seed=787`. The Hadamard congruence runs in FP64.
- **TF32 off,** including NGC's environment overrides: `NVIDIA_TF32_OVERRIDE=0`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` (`3977fc5671`).
- **Propagate the serialized reconstruction:** the packed trellis/suh/svh decoded to BF16, **never the search's raw `weight_q`**, which differs by 6.2e-4. Down Hessians see the serialized gate and up.
- **Fail closed** on non-finite values at every stage boundary. Stop any K6 encode whose training error exceeds 1e-3 before installing it.
- **Finiteness checks stay on the GPU, batched per activation chunk,** with one device-to-host scalar per chunk. **Never** a GPU→CPU boolean per product, module or element. **Why:** Hy4 synced per product until the fork's `d606eaccc` (TJ: "we don't keep the check on device, we download it to check over and over?"). The fix was 1.08–1.16× faster and bitwise-identical.

## 6. Activation and expert-output dataflow (required design)
- **Accumulate routed-expert outputs on the GPU.** The weighted top-k combine is computed on the device as an ordered FP32 reduction. Only the **summed** per-layer output feeds the next layer or the causal checkpoint.
  - **Never persist unsummed per-expert outputs.**
  - **Why:** Hy4 copied Inkling's unsummed-output spool. That was **100.5 GiB written plus 100.5 GiB read back per routed layer**, and it hung maxq. `495873eaa` replaced it with about 12.6 GiB of resident buffers.
  - On resume, replay the sealed packed experts in ascending expert order. Don't reload stored outputs.
- **Keep routed inputs resident** for the layer when they fit, otherwise stream bounded chunks. Never bounce through host memory per expert.
- **Read each source weight once,** and rely on the page cache. Source reads go through SparkNest at about 1 GB/s on maxq's 10G link, so don't be wasteful.
- **Caching:** cache only the native non-expert weights for the current phase, and clear the cache at phase boundaries. Retire stages and stores after the downstream commit.
- **Huge lookup tables** (Engram, PLE): mmapped and reclaimable, gathering only the rows the calibration tokens hash to. **Never materialize them.** Watch anonymous memory, not just RSS.

## 7. Recovery and I/O (required)
- **Log-style recovery:** append records to **bounded sequential segments**. The fork uses 32 batches per segment, capped at 256 MiB (`9820e963a`). **Fsync each segment once when it closes,** then atomically publish a small per-phase or per-layer manifest. On resume, replay the incomplete tail.
  - **Never scatter writes** into a big preallocated or mmapped file. **Why:** Inkling's 90 GiB token×rank scatter memmap was flushed and fsynced after every expert, under an 80 GiB container cap, at **107 s per expert**. Sequential staging made it 4.8 s (`0220f41abc`). That shim is compatibility-only, so don't copy it.
  - **Never fsync per record, expert or tensor.** **Why:** `2a95c7c8af` cut payload fsyncs from 12 to 3 on a 12-record fixture, with grouped activation segments.
- **Cheap hashing:** XXH3-128, **streamed in the write pass,** for every scratch and recovery receipt. **Never read a file back just to hash it.** **Why:** XXH3 runs at 21.75 GiB/s against SHA256's 2.11 GiB/s (`cb9ed4c2ee`, `8eb5a68a5b`).
- **SHA256 only where a format requires it:** the `cuteafd-mc.json` digests, the Hub/LFS identity and the source identity. Compute those in the write pass too.
- **Frozen identity:** cover the source, corpus, numerical-file hashes and fork pin. The numerical files are committed, and the fork is clean and equal to the pin.
  - Refuse resume on any mismatch.
  - Require a diagnosis before resuming a failed producer.
  - Never restart a completed job.
- **Mid-run changes that don't touch numerics** (I/O, scheduling) go in as reviewed recovery receipts bound to the frozen identity, never as silent numeric changes. Hy4's `resident-expert-recovery.json`, `batched-hessian-recovery.json` and `paced-trellis-recovery.json` are examples.
  - Build §5–§8 in from the start, so you don't need a chain of them.
- **Infra faults:** diagnose, then resume. Never restart from scratch. **Why:** SparkNest EIO hit Inkling at layers 20 and 24, and both resumed.

## 8. GPU, host and throughput on maxq
- **maxq's GPU is also the WSL display GPU.**
  - Don't flood the PCIe/driver queue with huge host↔device ping-pong, or the host hangs: screen, keyboard, ssh and gpu-powerd. `dmesg` shows `dxgkio_escape` floods.
  - Structure work like Kimi's: bounded spans, resident buffers, few syncs.
  - Pace LDLQ spans with the fork's `GPTQMODEL_EXL3_SPAN_PAUSE_MS=1` (TJ-requested, `43180de6c`): "you are fucking the pci bus… structure it more like kimi".
- **Memory and GPU access:**
  - **Memory:** 94 GiB RAM with no swap, so cap containers at about 80G. Page cache from mmapped scratch counts against the cap.
  - **GPU access:** request GPUs via CDI, not `--gpus`, because `daemon-reload` kills `--gpus` containers. Keep timeouts in the supervisor's process group.
- **Hub upload throughput:**
  - Xet's adaptive concurrency oscillated between 1 and 64 streams. `HF_XET_FIXED_UPLOAD_CONCURRENCY=32` gave about 299 Mbps, and xet 1.8 (direct to S3) about 500 Mbps.
  - The code pins Hub SDK 1.33.0 (`quantization/hub_requirements.py`), but Inkling actually published with hub 2.2.0 and xet 1.8. **Confirm the SDK the run uses supports xet 1.8 before uploading.**
- **Don't repeat known dead ends:** for example, page-cache hint trials (released 65 GiB, no speedup).
- **Progress reporting:** an update every 30 min via a durable timer, with remaining time and an Asia/Taipei ETA. Exclude the interrupted layer from averages, and allow separately for head/MTP, upload and placement.

## 9. Output and validation
- **MC contract:** follow `docs/mc-checkpoints.md` exactly. Earlier packages needed post-publication repairs (`quantization/mc_repair_*.py`), because of:
  - incomplete `tensor_storage`: use the shared builder, with quant format, bpw, codebook multiplier, in/out/stored features and stored tensors (`85d080f37f`);
  - view declarations not mirrored into the metadata (`05d890fbcf`);
  - nested text-block metadata not rewritten per view (`0fd3199183`, Kimi K3.1 #2098);
  - integer base bits lost in mixed experts (`e0f0901430`);
  - mixed core and expert shards blocking clean copies (`5a4a6e0e4b`);
  - per-view config not referencing its own index;
  - unsorted or non-unique `core_tensors`;
  - mismatched shared-inventory digests;
  - K6 names leaking into the K8 index;
  - the original index naming trellis tensors.
- **Readers and writers:** tolerant readers ignore unknown fields, and writers and validators stay strict. Anything that changes how weights are interpreted goes in `required_features`, or bumps the schema. Advisory sections (like Hy4's `engram_fp8`) don't.
- **Derived alternates** (precomputed tables, for example): put them in dedicated shards that no standard view indexes, and verify every row bit-exact against the vendor's own expression, including ±0.
- **Reuse:**
  - `quantization/mc_package.py` (writer and validator);
  - `core_export.py`, `core_launch.py`;
  - `mc_publish.py`, `mc_postpublish.py`;
  - `quantization/tests/test_mc_*.py`;
  - the Inkling MC-direct exporter `quantization/inkling2/export.py`, with `9e71bb5320` and `05d890fbcf`.
- **Model card (`README.md`):** generated by the exporter.
  - **Template:** the **GLM 5.3 MC card** (`wrldsuksgo2mars/GLM-5.3-EXL3-K4-MC`), not Inkling's, which still has boilerplate TJ rejected.
  - **Front matter:** `library_name: cuteafd`, the source's real license with the LICENSE file kept, `base_model`, and tags `exl3`, `mixture-of-experts`.
  - **Sections:**
    - "What multi-core means", with a view table of sizes and when to use each view;
    - "How it was made";
    - a Recipe table: codebook and multiplier, bits, seed, sigma, Hessian accumulation, ordering, replay, calibration count, low-row recovery policy;
    - "Running it";
    - "License".
  - **Describe the panel accurately:** "token-budgeted real source code, worked mathematics, function-call schemas, multilingual text".
  - **Leave out** repair narratives and "no quality claim" boilerplate.
- **Validate before upload** with the loader's own checks: discovery, every view's index, metadata hashes, fingerprints, shared-digest equality, and original-view core bytes equal to the source. Also run a plain-loader check of `original`.

## 10. Publishing lifecycle
- **Before publishing:** release the GPU. Run publication as a separate, CPU-only unit (`af7528e1b7`).
- **Sealing:** predeclare `.gitattributes` in the sealed package as a regular file, because the Hub appends LFS rules. **Why:** Inkling's publication omitted it, and its staging used a symlink.
- **The lifecycle:**
  1. **Upload** as `wrldsuksgo2mars/<Model>-EXL3-K<bits>-MC` (standard upload).
  2. **Verify** by size plus the LFS/Xet sha256 the Hub already reports. **No extra hash passes,** no hashing over 10G, and no re-downloading your own quant; reuse the writer's hashes. (Older briefs said otherwise, and this supersedes them.)
  3. **Stage** a hard-link HF cache with the Hub OIDs as blob names, then run `nest hf import --offline --no-copy`. Update `refs/main`. Pinned revisions get `refs/keep/<name>`, written with `printf %s`, with no trailing newline.
  4. **Place it, before deleting anything.** Use SparkNest only: `nest plan --place hf:<repo> --host raptor:d2 --host raptor:d3 --host ostrich --host dodo --host emu --host kiwi --host rhea --host moa --factor 1/8`, then `nest plan show` (check for removals), then one `nest plan apply --wait`.
     - Random-read tables get a pin rule to raptor first.
     - Never `nest replicate --hosts`.
     - Check coverage with `nest ls --json`.
  5. **Clean up:** delete the source export, staging and intermediates. Keep receipts, logs and the per-module CSV.
- **Hub folder limits:** bundle large diagnostic sets to stay under them (`5b15c6d5d3`).

## 11. Preflight and pilot gates (report before the full run)
- **Preflight:** a full header and source audit over the complete download, with a fail-closed inventory and byte totals. Then the reference-forward match on a few prompts, covering every special path.
- **Pilot:** representative layers (first dense, first MoE, a mid layer of each attention type, the last layer, and the drafter target). Arms per core module are K8, K6 and keep-native; experts run at the addendum's bits. Keep it under about 15–30 minutes of GPU time. Report:
  - held-out NMSE per arm;
  - scale-inclusive bpw;
  - view sizes;
  - a TP2 projection;
  - the §4 recovery distribution;
  - anything odd, with a keep-native option.
- **Bars:** held-out NMSE K8 ≤ 1e-4, K6 ≤ 1e-3, expert ≤ 1e-2, all finite (`inkling2/pilot.py:21`). For reference, Inkling's worst values were K8 1.41e-5, K6 1.86e-4 and K4 4.55e-3.

## 12. Hardware and run hygiene (TJ's runs)
- **No cluster locks or leases:** no `hwlock.py`, `build-slot.sh`, `maxq.lock`, `gpu*.lock` or seed leases. (Inkling ran under a maxq lease; this rule supersedes that.)
- **Durability:** `systemd-run --user --expand-environment=no --setenv=PATH --setenv=HOME`, detached. `nohup setsid` dies with the session's cgroup. Chained units need a failure path that wakes the owner, and resume-after-reboot units are gated on `boot_id`.
- **venv:** use a task venv under the run root. **Never use or modify `~/Developer/cuteafd/.venv` on maxq,** not even by symlinking it into the worktree.
- **Storage:** work storage on `~/Developer`, never `/mnt/scratch`. Stream the source from the HF cache.
- **Secrets:** the HF token goes over stdin only.
- **Process hygiene:** no `pkill -f`, no `git stash`, and never print full process argv.
- **Off limits:** don't touch power settings, raptor's GPUs or the serving Sparks.

## 13. Code delivery and report
- **Engine code:** `quantization/<family>/`, `mc_package.py` changes, docs and tests go on a branch, with a PR against `dev` (`scripts/dev/pr-sync init`). Keep it pushed during the run; main reviews and merges.
- **GPTQModel changes:** merged to fork main and pushed before encoding. The engine PR bumps the pin and lock, and main merges it after the quant.
- **Report** on a GitHub issue:
  - per-layer time and total;
  - sizes per view;
  - the per-module error table (K8/K6);
  - the §4 recovery distribution;
  - repo@sha.

---

## 14. Model addendum
*(Append: source and pinned revision; reference code in authority order and the pinned Transformers version; architecture; size estimate; top-k and routing function (for §4); thinking mode (for §3); core module list; keep-native list; special tensors and their formats; model-specific calibration traffic; pilot layers; extra bars; the published name; and the engine follow-ups this checkpoint needs.)*
