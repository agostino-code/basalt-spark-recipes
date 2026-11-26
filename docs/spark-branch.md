# The official main as the base

The recipe was built on the third-party port `andreasknopke/basalt-arm-sm121` because upstream had no
Arm64/sm_121 support. Upstream now has it on `main`: the temporary `spark` branch was removed on
2026-11-25 and its DGX Spark work landed in `jesdga95/basalt` main (the same commits rebased; the head
is `c70e8011`, Basalt 0.5.7).

The base is now **`agostino-code/basalt` main**, a direct fork of `jesdga95/basalt`, pinned at
`c70e80118ead375dbcb37948d0cccad8937ad03a` in the installer. Nothing in this repository depends on the
old port fork.

## What main already has

- **Arm64/GB10**: `CMAKE_CUDA_ARCHITECTURES` defaults to 121 on Arm64, NEON CPU kernels, the platform
  memory-ordering and synchronous-copy rules, `docs/arm64.md`, the `ubuntu-24.04-arm` CI.
- **The page-cache trap fixed upstream**: `gpu_mem_info` reads the host's `MemAvailable`, so a
  download's page cache no longer makes the planners refuse to load. The `posix_fadvise` workaround
  in the start scripts is now harmless rather than necessary.
- **GB10 benchmark support** and the Engram FP32 epsilon (#556).
- **0.5.7**: batched decoding, the GPU placement settings and the model fixes of 0.5.6/0.5.7.

## What remains ours

These are the patches the installer applies, with their upstream state (the chunk policy is merged into the pinned main):

| Patch | What it does | Upstream |
|---|---|---|
| `0001-qwen-fp16-batch.patch` | FP16 K/V for Qwen's batch windows, with the 0.5.7 Q4_0 path preserved | [PR #694](https://github.com/jesdga95/basalt/pull/694) (open; the `qwen4exp_kv_batch_test` wording fix is pushed) |
| `0003-hy4-fp4-prompt.patch` | experimental Hy4 FP4 prompt experts (one-term W4A4), 1536×1536 prefill path | [PR #793](https://github.com/jesdga95/basalt/pull/793) (draft, experimental) |
| `0004-qsa-seg-parity-int8.patch` | the `qsa_seg_parity` CI fix (the `int8` argument); lets the full build pass | part of [PR #694](https://github.com/jesdga95/basalt/pull/694) |

The old `0001-spark-sm121-fixes` patch is **gone**: its platform part is in main, and the prompt
chunk policy is the separate patch above. The 0.5.0 tree and its measurements remain the historical
`spark-2026-11-24` tag; the installer and the appendices now label which numbers were taken on which
base.

## PR history

- #691 and #692 (Qwen FP16 from the nested/clean fork, base `spark`) were closed when `spark` was
  removed; #694 is the same change rebased onto `main`.
- #693 (chunk policy, base `spark`) was closed the same way; #695 is the rebased one.
- The head fork is `agostino-code/basalt`, a direct fork of `jesdga95/basalt`.
- #695 (**merged**) carried the chunk policy, now in the pinned main (no 0002 patch); #694 (open)
  carries 0001 and the `qsa_seg_parity` fix (0004); #793 (draft) carries 0003, the experimental
  one-term FP4 prompt path.
