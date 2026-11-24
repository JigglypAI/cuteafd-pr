# Hy4-flash

The v7 Release-smoke checkpoint is `wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC`
with EXL3 K4 MUL1 routed experts and host-resident FP8 Engram tables. The
cards use automatic KV sizing and `CORE_VIEW=k8-compact`. With bundled vision
support, `VISION=auto` admits the tower whenever the KV pool still meets
`KV_MIN_TOKENS`.

On two RTX GPUs, `ATTENTION_PLACEMENT=auto` (default) resolves to mixed
placement through the family resolver: ContextSplit (`context`) on the 12
MLA/DSA layers (parity-owned KV/index pages with no DSA KV replication, ~2.0×
KV pool capacity) and head-split on the 30 SWA-512 layers. Explicit
`ATTENTION_PLACEMENT=heads` opts into replicated-KV head-split across all
layers; single-GPU layouts keep whole layers.

| Layout | Starting config |
|---|---|
| 1 RTX PRO 6000, no Sparks | [Min](../../examples/configs/hy4-flash-exl3-k4-mc-min.config) |
| 2 RTX PRO 6000, no Sparks | [Max](../../examples/configs/hy4-flash-exl3-k4-mc-k8-max.config) |
| 1 RTX PRO 6000 + 1 Spark | [One Spark](../../examples/configs/hy4-flash-exl3-k4-mc-1rtx-1spark.config) |
| 1 RTX PRO 6000 + 4 Sparks | [Four Sparks](../../examples/configs/hy4-flash-exl3-k4-mc-1rtx-4spark.config) |

The [MC comparison](../../scripts/bench/release-smoke-v7-mc-compact.json)
runs `original`, `k6-compact` and `fp8-compact` on the same two-RTX
layout as the K8 base. Cards and quality comparisons are pending measurement.

## Changelog

| Release | Date | Changes | Basic eval |
| --- | --- | --- | --- |
