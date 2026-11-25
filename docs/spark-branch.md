# The official `spark` branch as the base

The recipe was built on the third-party port `andreasknopke/basalt-arm-sm121` because upstream had no Arm64/sm_121
support. Upstream now has an official branch for it — [`jesdga95/basalt`, branch `spark`](https://github.com/jesdga95/basalt/tree/spark)
— and this is the base to move to and to send changes to.

## What the branch is

Head at the time of writing: `f3bd352e` (2026-11-24), **Basalt 0.5.7**. The Arm64 work arrived as PR #450
("arm64: building for Arm64 and the integrated 12.1 GPU") and follow-ups, and the branch also carries the
0.5.7 release.

What it already has, checked in the source and the docs:

- **Arm64/GB10 build**: `CMAKE_CUDA_ARCHITECTURES` defaults to 121 on Arm64, NEON CPU kernels, the
  memory-ordering and synchronous-copy rules of the platform layer, `docs/arm64.md`.
- **The page-cache trap fixed upstream**: `gpu_mem_info` reads the host's `MemAvailable`, not the driver's
  `MemFree`, so a download's page cache no longer makes the planners refuse to load. This supersedes the
  `posix_fadvise` workaround we carried for 0.5.0 (the start scripts can keep it harmlessly).
- **GB10 benchmark support**: an integrated GPU row (no PCIe link or NVML power), Arm CPU naming from
  `/proc/cpuinfo`, CI on `ubuntu-24.04-arm`.
- **Hy4 and Qwen model code** and the Engram FP32 epsilon (#556).
- **0.5.7 features**: batched decoding, the `gpu` / `gpu-mode` / `gpu-share` settings, and the model fixes of
  0.5.6/0.5.7. The `.basalt` and config formats stay version 2.

## What it does not have, and what of ours is still valid

Checked against the branch's files:

| Our change | In `spark`? | Action |
|---|---|---|
| NVFP4 prompt rows tiled on sm_121 (`mmq_config.cuh`) | **No** — all NVFP4 rows are `stream_k = true` | Port if the prompt fault still reproduces; 0.5.7 still lists the Hy4 illegal access on prompts around 600K and more |
| Expert ring off (Hy4, sm_121) | **No** — the ring code is present, no `BASALT_RING` guard | Test the long-prompt path on 0.5.7; the 0.5.0 ring fault may or may not reproduce |
| Qwen batch windows with FP16 K/V | **No** — `kv_batch_refusal` is int8-only | Port the 0.5.6 version and test 2/4 slots; a self-contained PR |
| Hy4 prompt-chunk policy (32k up to 128k, 8k above 262k per slot) | **No** — `kMaxChunk = 16384`, `kAttRows = 2048` | Port and re-measure 33k/80k and the >262k case |
| Page-cache fadvise workaround | Superseded by `MemAvailable` | Keep in the start scripts (harmless), drop the claim from the docs on migration |
| `BASALT_DBG_Q` diagnostics / product tags | Not needed as a patch on 0.5.7 | Re-evaluate only if debugging a new fault |

Our three patches **do not apply** to `spark`: the 0.5.7 files changed (ring, MMQ table, QSA, chunk code).
They are ports, not cherry-picks.

## PR plan for the official branch

Fork `jesdga95/basalt`, branch from `spark`, one topic per PR, each with a GB10 test record:

1. **`qwen-fp16-batch`** — **opened as [PR #691](https://github.com/jesdga95/basalt/pull/691)**, rebased
   onto 0.5.7 with the Q4_0 K/V path preserved. Tested on 2 × 262,144 and 4 × 131,072 with `--kv fp16`
   (all slots answer), the int8 4-slot regression, and the resident-FP16 refusal. The fork PR #1 is the
   0.5.6 reference and is superseded by this one.
2. **`hy4-sm121-nvfp4`** — tile the NVFP4 rows and test the large-prompt fault at 0.5.7 (the 0.5.7 notes
   still report it around 600K); include the ring question only if the 0.5.7 path reproduces it.
3. **`hy4-chunk-policy`** — the 32k/8k prompt-chunk policy, with the 33k/80k prefill numbers and the
   >262k stability test.

The official branch was built and run on the GB10 before the PR: Hy4 4 × 262,144 loads and answers
(a 33k prompt reads in 41.1 s at 0.5.7, against ~32–36 s on the 0.5.0 tree with the chunk policy);
Qwen int8 4 slots answers, and `--kv fp16` with 2 slots was refused exactly as above before the patch.

## Recipe migration

The validated 0.5.0 tree stays the `spark-2026-11-24` tag. A migration branch would pin the official
`spark` commit, point the installer at `jesdga95/basalt`, carry only the ported patches, rebuild for 121,
and re-measure the presets, the four-slot behaviour and the quality gate against 0.5.7. Nothing of this
changes the current recipe until that gate passes.
