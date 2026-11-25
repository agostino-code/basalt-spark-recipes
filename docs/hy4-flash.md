# Hy4-Flash on Spark

> **This is the original Hy4-Flash recipe**, kept as the per-model appendix of the unified repository
> that merges it with the Qwen3.8-Flash-Next one. Install and start commands now live in the
> [README](../README.md) and [install-basalt.sh](../install-basalt.sh) — one engine tree,
> `./install-basalt.sh --model hy4`, `basalt-port/start-hy4.sh` (or `./run.sh hy4`).

Running **Tencent Hy4-Flash** (224B total / 12B active, natively multimodal) on a **DGX Spark**
(NVIDIA GB10, compute capability 12.1) with the [Basalt](https://github.com/jesdga95/basalt)
inference engine.

This repository is the **DGX Spark validation that the Arm64/sm_121 port lists as missing**: the
port's own README says it was measured on an RTX Spark N1X and that *"the DGX Spark is the same GPU
generation but has not been tested"*. We tested it, found two faults that only show up on sm_121,
fixed them far enough to run agent workloads for hours, and wrote the numbers down.

**What is ours** is the fork [`andreasknopke/basalt-arm-sm121`](https://github.com/andreasknopke/basalt-arm-sm121)
(branch `hy4-port`, pinned in the installer): it merges the upstream **v0.5.0** into yanmindi's
Arm64/sm_121 port — that merge is where the Hy4-flash support enters the port — adds the MMQ
read-past fix for the dense tensors, the two sm_121 prompt-fault fixes, and (one commit past the tree
the Qwen numbers were measured on) upstream **#556**, the Engram epsilon fix, plus the crash
reproduction, the lane configuration and the measurements below. The engine, the port and the
model underneath stay other people's work — see [Credits](#credits). The sm_121 fixes are **workarounds, not root causes**: we know which code path
faults and that the other one does not, not which instruction does it (see
[Open questions](#open-questions)).

---

## What we measured

| | |
|---|---|
| Prefill | **~790–830 tok/s, essentially flat from 16k to 262k tokens** — 822 at 16.5k, 828 at 165k, 808 at 220k, 796 at 250k, 791 at 262k. It is the per-token cost of this model (top-16 of 512 experts over 41 layers, plus the sparse-layer indexer), not a long-context effect |
| Decode | **20–23 tok/s**, essentially independent of context: 20.3 at 165k, 20.0 at 220k, 24.9 at 250k and 262k with one active slot. |
| Context per request | **262 144 tokens, prompt + answer together** — the hard limit is the slot window; the server refuses more instead of truncating (`requests are never truncated`). Measured: 165 022, 220 022, 249 997 and 261 987 tokens pass, 264 022 is refused — so the limit is the window itself, and the largest prompt is 262 144 minus the answer budget. The expert tier does not bind here: 100 % decode expert-cache hits at 220k |
| KV ceiling (4 slots) | `elastic KV: about 262144 cells fit in the 4 sessions (1048576 at full context)` |
| Stability | **0 engine faults** across both benchmark runs (≈500 requests, peak request 128k tokens) |

Patch 0003 raises the prompt chunk to 32,768 rows while the prompt is at most 131,072 tokens: the
33k prefill reads 32.4 s against 36.1 s with 16,384-row chunks (~10%), and 80k reads in 78.8 s; the
16k depth is unchanged, above 128k the chunk stays 16,384 (the long-context numbers above) and above
262,144 per slot it stays 8,192 (the large-slot cap above). The workspace grows by ~2.5 GiB; every expert stays resident.

Benchmark results, both on the public leaderboards:

* **opencode-bench-plusplus, suite `locker-v5`, variant `default`** — quality **0.9881**, wall
  **42:36**, 102 steps, model time 2500 s, peak request 104 014 tokens, eligible, gate pass →
  [issue #34](https://github.com/knopkem/opencode-bench-plusplus/issues/34) (measured on `8172978`,
  before #556)
* **codec-reverse-bench, variant `high`** — **57.0** points, roundtrip solved in **stage 3**
  (35 of 57 points), 11/15 structure questions right, compression ratio 61.2 →
  [issue #13](https://github.com/andreasknopke/codec-reverse-bench-leaderboard/issues/13)

For reference: on the codec leaderboard the cap without a verified roundtrip is 25, the top of the
board is around 127, strong local lanes around 85 — 57 is a real codec, mid-field, about on par with Qwen3.8-Flash-Next NVFP4

### Where the prefill goes

`nsys` kernel summary of one 33k and one 80k prompt through `hy4-agent` (the engine under a
`--engine-bin` wrapper that carries the flag list the server reads; `nsys stats --report
cuda_gpu_kern_sum`; the sum includes the load and decode tails, so the shares are indicative):

| Kernel group | 33k | 80k |
|---|---:|---:|
| Dense Q8_0 MMQ (`mul_mat_q`, type 8) | 20 % | 20 % |
| NVFP4 expert MMQ (`mul_mat_q`, type 40) | 19 % | 19 % |
| Attention (`attend_mma`) | 14 % | 14 % |
| Activation quantize (`quantize_mmq_q8_1`) | 11 % | 10 % |
| Router + SwiGLU + weighted sum | 11 % | 10 % |
| Hyper-connections (`hc_*` + `plain_partial` + `mul_sigmoid`) | 12 % | 10 % |
| K/V prepare (`prepare_kernel`) | 4 % | 4 % |
| Indexer scores + select | 2 % | 2 % |
| Engram and other | ~1 % | ~1 % |

The prefill is **MMQ-bound**: dense Q8_0 and NVFP4 experts together are ~39–40 %, and the activation
quantizer another 10–11 %. Attention is 14 % plus 4 % of K/V prepare; a 30 % faster attention would
be about 5 % of the prefill. The hyper-connection kernels (~11 %) and the MoE routing/activation
(~10 %) are the next Hy4-specific blocks. At longer contexts attention grows faster than the rest, so
a 262k profile is the one to take before deciding to rewrite it; at 33k/80k it is not the first
target. The full kernel tables are in [evidence/hy4-prefill-kernels.txt](../evidence/hy4-prefill-kernels.txt).

Two cheap dense-path knobs were tried and rejected: 4,096-row attention steps (the doubled buffers
ran the lane into RAM/checkpoint failures) and tiling the Q8_0 MMQ rows (about 20% slower than the
stream-k rows at 33k). The dense path's next real steps are fusing the activation quantizer into the
MMQ (the quantizer is ~10% on its own) or moving the prompt experts to FP4 tensor cores (19% plus a
share of the quantizer).

### Quality against the BF16 reference

The Engram gate normed the stream and the key with bf16's machine epsilon (`0.0078125`) where the
model uses FP32's. That epsilon swamps small keys: after a line end sampled as `},` `\n` ` ` instead
of `},\n` ` `, the next row put 77–94 % of its mass on `...\n`, with byte tokens, `\x7f` and U+FEFF
behind it. Upstream **#556**, which the pinned commit carries, uses FP32's epsilon; the same prefix
then gives ` {"` at p 1.000. The stray JSON/code answers in #556's harness went from 3–4 of 48 to
0 of 48.

The gate (`basalt-bench --quality 100 --kld-base hy4flash-bf16-base.kld --quality-t 1`: 100 chunks,
25,500 rows, T=1) on this box against the rebuilt BF16 base (base PPL 18.4162, sha256
`8502b5d2…` from [basalt-kld-bases](https://huggingface.co/datasets/jesdga/basalt-kld-bases)):

| | This lane, GB10 (`f9bf55aa`) | Upstream 0.5.6, RTX 5090 | Pre-#556 engine |
|---|---:|---:|---:|
| PPL | 17.5207 ± 0.4442 | 17.484 | 17.565 |
| KLD mean | 0.20300 | 0.209 | 0.782 |
| same top-1 | 86.32 % | 86.14 % | 76.77 % |

The GB10 lane reproduces the fixed engine's quality: KLD and top-1 are where 0.5.6 puts them, not
where the pre-fix engine does. Our run is the engine's auto placement (512 of 512 experts on the
card); upstream's 5090 number pins `--expert-cache 4800`. The 0.9881 opencode-bench row above is
**not** a post-fix number — it was measured on `8172978` — and the fix changes Hy4 output, so the
gate above is the figure to compare across the fix.

With `--kv fp8` (the `hy4-fp8` preset) the same gate reads PPL 17.6522 ± 0.4489, KLD 0.20744,
top-1 86.30 %: +0.0044 KLD for roughly 6 GiB of KV back. The decode fixtures are level on struct
and prose (39.3 against 39.4, 19.1 against 19.6 tok/s); `conv` read 22.4 against 24.8 in one run
each, so we do not claim a speed gain — the trade is memory. The `hy4-2` preset (two slots) and
`--spec` 4/8 were also measured; `--spec` 4 is worse on `struct` and 8 is level with the default 6.

---

## Install

One script builds the whole lane with the versions this recipe pins — port commit
`f9bf55aa`, CUDA 13, `CMAKE_CUDA_ARCHITECTURES=121`, the NVFP4 pack `Hy4-flash-NVFP4` (147.8 GiB):

```bash
git clone https://github.com/agostino-code/basalt-spark-recipes
cd basalt-spark-recipes

./install-basalt.sh --model hy4              # text only: 4 slots x 262144 — the agent configuration
./install-basalt.sh --model hy4 --vision     # image input: 1 slot, 262144 total
```

**Pick one — the two configurations are mutually exclusive on this box**, and that is measured, not
assumed (the numbers are in [Vision is optional](#vision-is-optional-and-it-costs-concurrency)):

| | `./install-basalt.sh --model hy4` | `./install-basalt.sh --model hy4 --vision` |
|---|---|---|
| Slots | 4 | 1 |
| Context | 1048576 cells, 262144 per slot | 262144 in the single slot |
| Image input | no — `images=off` | yes — `images=on`, `--vision dynamic` |
| For agent workloads | yes; a long request does not block the others | no; one request at a time |
| Start line it writes | `--parallel 4 --max-context 1048576 --fit-max-tokens` | `--parallel 1 --max-context 262144 --fit-max-tokens --vision dynamic` |

Both write the same `start-hy4.sh` and, with `--install-unit`, the same systemd user unit — only the
engine flags differ. Nothing else changes, so switching later is one flag and a rerun.

What it does, in order — every step is idempotent, so rerunning it after a failure is cheap:

| Step | What happens | Time |
|---|---|---|
| 1 | machine checks: aarch64, CUDA 13, cmake >= 3.24, gcc >= 13, python >= 3.11, 160 GiB free for the pack | seconds |
| 2 | clones `andreasknopke/basalt-arm-sm121` (branch `hy4-port`) at **`f9bf55aa`**, the commit we validated, into `basalt-port/` | ~1 min |
| 3 | applies `patches/*.patch` in order when the tree does not already carry each one — the pinned commit carries 0001 (NVFP4 on the tiled MMQ path, expert ring off, product names in the MMQ launch errors); 0002 is Qwen-only, 0003 is the prompt-chunk policy (8k above 262,144 per slot, 32k up to 128k-token prompts, 16k above) | seconds |
| 4 | builds it when `bin/basalt` is missing or was last built at another commit, `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121` | ~10 min |
| 5 | creates `./.venv` and installs the pinned `requirements.txt` | ~1 min |
| 6 | downloads the pack from [jesdga/Hy4-flash-Basalt](https://huggingface.co/jesdga/Hy4-flash-Basalt) into `models/`, resumable and size-checked | 1–3 h |
| 7 | writes `start-hy4.sh`, optionally the systemd user unit, starts the lane once and smoke-tests it | ~1 min |

Flags that matter: `--vision` (the single-slot image configuration), `--dir <path>`, `--pack <file>`
(use a pack you already have), `--skip-pack` (build only), `--port <n>`, `--jobs <n>`, `--install-unit`,
`--no-smoke`, `--kv int8`.

Nothing needs root. Afterwards:

```bash
basalt-port/start-hy4.sh                   # foreground (start-lane.sh is the old name; ./run.sh hy4 also works)
systemctl --user start hy4-basalt          # if installed with --install-unit
```

### Quick command

`./run.sh hy4` in this checkout starts the lane from the engine tree it finds — `$BASALT_DIR`, else
`./basalt-port` in this checkout — and passes any extra argument to the server. The Qwen3.8-Flash-Next
lane has its own recipe and launcher (`./run.sh qwen`, `start-qwen.sh`) in the same tree.

The unit is deliberately left **disabled**: port 30000 is shared with other lanes on our box, so an
automatic start would race them. The start script refuses a busy port instead of starting into
someone else's server.

### Vision is optional, and it costs concurrency

The lane as installed runs with `images=off` — text only, four slots, 262144 per slot. Image input is
one flag away (`--vision dynamic`), but on this box it only starts with a **single slot**:

| Configuration | Result |
|---|---|
| `--parallel 1 --max-context 262144 --vision dynamic` | loads in 30 s, `ctx=262144 images=on`, reads a test image correctly |
| `--parallel 2 / 3 / 4` with `--vision dynamic` | dies after 10 s in the planning step: `RAM ran out loading` — at every context size from 65536 to 524288 |
| `--vision cpu` | refused by the model: `this model's images are encoded in the engine: --vision dynamic, not cpu` |

The encoder's reservation is taken per slot, and two of them already exceed what is left next to the
86.2 GiB of weights. So vision and the four-slot agent configuration are mutually exclusive here:
either image input with one slot (a long request blocks the others), or four slots without vision.
A measured image answer: a 900x360 PNG reading "SPARK-42 / Basalt NVFP4 / 22 Aug 2026" came back
verbatim, `finish_reason: stop`, 363 prompt tokens.

### Requirements

| | |
|---|---|
| Machine | DGX Spark (GB10) or another Arm64 box with the compute-capability 12.1 GPU |
| CUDA | 13.x toolkit (`nvcc`); measured with 13.0.88 |
| Toolchain | GCC 13+, CMake 3.24+, Python 3.11+ |
| Disk | ~150 GiB for the pack, ~5 GiB for the build |
| Memory | 121 GiB unified; the lane holds 86.2 GiB of weights plus the KV |

The acceptance test for the installed lane is the one in
[Acceptance test before every run](#acceptance-test-before-every-run) — `tools/basalt-repro.py` now also takes `PORT` and `LANE_LOG` from the environment, so it can
point at a lane on a non-default port.

---|---|
| Machine | DGX Spark, NVIDIA GB10, compute capability 12.1, 121 GiB unified memory |
| Driver / CUDA | 580.159.03 / CUDA 13.0.88 |
| OS | Ubuntu 24.04.4 LTS (aarch64), kernel 6.17.0-1021-nvidia |
| Toolchain | GCC 13.3, CMake 3.28 |
| Model | [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) — 224B total (incl. 64B Engram), 12B active, 1M context |
| Pack | `Hy4-flash-NVFP4.basalt`, 147.8 GiB, from [jesdga/Hy4-flash-Basalt](https://huggingface.co/jesdga/Hy4-flash-Basalt) |
| Engine | [andreasknopke/basalt-arm-sm121](https://github.com/andreasknopke/basalt-arm-sm121) — our fork (`hy4-port`) of [yanmindi/basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121), which is the Arm64/sm_121 port of [jesdga95/basalt](https://github.com/jesdga95/basalt); the fork merges upstream v0.5.0 and carries the fixes |

`./install-basalt.sh --model hy4` does the following; manually it is:

```bash
git clone https://github.com/andreasknopke/basalt-arm-sm121 basalt-port
cd basalt-port && git checkout f9bf55aab0b507e2130db34c2463b209c319d8e2
# the pin already carries the sm_121 patch and #556; the installer applies
# patches/0001-spark-sm121-fixes.patch only to a tree that does not carry it
export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build -j 4          # ~10 min for the full build
```

### Lane configuration

```bash
PORT=30000 PARALLEL=4 CTX=1048576 LOG=/home/owc/basalt-serve.log \
  ./start-basalt-hy4.sh          # tools/start-basalt-hy4.{sh,py}
```

`--parallel 4 --max-context 1048576` gives **four slots of 262 144 tokens each** and leaves the
engine's full KV budget intact. That is not decoration: the agent runs we measured peak at
**104k–128k tokens in a single request**. With the context divided too small the lane does not fail
at the start, it fails in the middle of the job — the most expensive misconfiguration here.

The scripts in `tools/` carry our Spark's absolute paths (`/home/owc/...`); adjust them for your
machine. `start-basalt-hy4.py` is the wrapper our supervisor calls, `SLOT_FULL=1` switches it to
`--slot-full-context`.

Measured, four slots × 262 144 cells: KV ≈ 15 GiB (≈14.6 KB per cell for this model), experts
resident in VRAM (`ram-experts=0.0GB`), no throttling (2463 of 3003 MHz, 66 °C).

### Acceptance test before every run

`tools/basalt-repro.py` is the crash reproduction (below). A lane is only good if it passes it and
the log stays clean:

```bash
python3 tools/basalt-repro.py crash      # R1 16522 / R2 33022 / R3 33390 tokens, ~20/40/40 s
grep -c "illegal memory" /home/owc/basalt-serve.log    # must print 0
ctest --test-dir build -j 2 -R "prefill_mmq|hy4flash"  # 10/10 with the lane stopped
```

---

## The two sm_121 faults

Both produce the same error class — `basalt: error: prefill mmq: <product>: an illegal memory
access was encountered` — and both are **memory-layout dependent**: the identical request succeeds
or dies depending on how the KV and the buffers happen to be allocated at that moment.

### 1. The MMQ stream-k path for NVFP4

**Symptom.** After a large prompt prefix is reused, the engine dies mid-prefill:

```
basalt: error: prefill mmq: moe-down: an illegal memory access was encountered
```

**Reproduction (deterministic, ~70 s).** Two requests that reuse a big prefix, then a third that
appends a small chunk:

```
R1: 16 522 tokens          -> OK
R2: 33 022 tokens          -> OK
R3: 33 390 tokens (33 037 reused + ~350 new) -> crash
```

The trigger is not the prompt length alone and not the small chunk alone — it needs both. A second
form (`basalt-repro.py ladder …`, seven fresh prompts before the long one) crashes the same way.

**Attribution.** With `CUDA_LAUNCH_BLOCKING=1` the error lands in the MMQ product itself; the last
product before the fault is always a *small* one (`moe-down`, NVFP4, weights [1536,1536], `n = 1`,
834 rows, with `ids_dst` + `expert_bounds`). Small products are exactly the ones that take the
**stream-k** path (`fixup_needed = true`). Synchronised checkpoints around the staging copies, the
`put_i32` writes and the SwiGLU ruled those out; upstream llama.cpp's fixup kernel is identical to
the port's.

**Fix.** In `src/prefill/mmq/mmq_config.cuh` all 16 NVFP4 rows run the **tiled** path
(`stream_k = false`) instead of stream-k. The tiled path writes straight into `dst`, so there is no
fixup buffer involved at all.

**Verification.** `ctest -R "prefill_mmq|hy4flash"` → 10/10. The reproduction passes in both forms.
Prefill time is unchanged (16.5k: 20.8 s vs 20.9 s; 33k: 39.9 s vs 40.0 s). Both benchmark runs
then went through ~500 requests with 0 faults.

### 2. The expert ring

**Symptom.** Same error class, but it survives fix 1 as soon as the lane runs with *large*
per-slot contexts (4 × 262 144 instead of 4 × 65 536) — and then it dies within two seconds of
starting the chunk.

The ring (`ring_open`/`ring_copy` in `src/models/hy4flash/session.cpp`) is a ~4 GB staging area:
both slots hold all 512 experts of one layer, copied ahead while the previous layer computes. It is
the only buffer in the prompt path that ends **exactly on the 2 MiB allocation boundary**
(`2 × 512 × stride_` = 1944 × 2 MiB), so the tile loaders reading a little past the last expert's
blob hit an unmapped page there — while the same overread lands harmlessly inside the staging
buffers. What we ruled out, by measurement:

* **Not the copy.** With `BASALT_DBG_Q=1` a synchronised check after the copy loop reports
  `err=no error` for every layer, including the last one before the crash — so the fault is in the
  product reading from the ring, not in filling it.
* **Not a small tail overread.** 8 MiB of padding behind the ring did *not* help.
* **Decode is unaffected.** The ring is prompt-only (chunks ≥ `kRingRows` = 1024 rows, created in
  `open_prompt`, freed in `release_prompt`); decode steps (≤ `kMaxWindow` = 8 rows) read the experts
  straight from the tier. Measured with the ring off: decode 20.7–24.3 tok/s, prefill 833–859 tok/s,
  and the prompt times are *identical* to the ring-on runs.

**Fix.** The ring is disabled by default (`BASALT_RING=1` turns it back on for re-measuring). The
per-layer staging costs nothing measurable here, so this is a workaround with no price tag.

### Smaller traps worth knowing

* **Page cache can look like an out-of-memory error.** The lane refuses to start with
  `--max-context 65536: the first 32768 positions need 456 MiB, more than half of the 601 MiB free`
  — the kernel compares `cudaMemGetInfo` against `MemFree`, and the page cache of the 148 GiB pack
  eats the machine. No root needed to fix:
  `os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)` on the pack returns ~100 GiB instantly and
  leaves running downloads untouched. The same pressure has a quieter form: in our first lane start
  after downloading the 6.2 GB KLD base (still cached, the pack dropped), the plan put 26 of 512
  experts per layer in the host RAM tier; the first request passed and a second slot died with
  `expert 376 of layer 2 is not in RAM`. Dropping the base's cache too brought the plan back to
  512 of 512 experts on the card, where three sequential requests ran. The lane's start script drops
  only the pack, not a base file downloaded for measurement.
* **More than 262,144 tokens per slot needs a smaller prompt chunk.** The engine accepts
  `--parallel 2 --max-context 1048576` (2 × 524,288) and a single 1,048,576 slot, and a 16k prompt
  runs; an 80,031-token prompt then died in `prefill mmq: quantize(gemm)` (an illegal memory access)
  on both, while the same prompt on 4 × 262,144 passed. With `CUDA_LAUNCH_BLOCKING=1` the exact
  launch is `moe-down` (NVFP4) — the product class of the original sm_121 fault; the unblocked tag
  is the sticky one. Neither tiling the Q8_0 MMQ rows nor an 8 MiB guard after the staging buffers
  changed it. **Patch 0003** caps the prompt chunk at 8,192 rows when the per-slot context is above
  262,144: 80k and 200k prompts then pass on 2 × 524,288 (140 s and 267 s), and 80k on one 1,048,576
  slot (139 s); the 4 × 262,144 lane does not fault, and the same patch raises its chunk to 32k up to 128k. The root cause
  (the tiled NVFP4 `moe-down` launch at ~15k rows) is still open, and a full 524k/1M prompt is not
  measured.
* **compute-sanitizer is not usable at this model size.** `memcheck` needs shadow memory; with an
  86 GiB model resident, the engine aborts loading (`RAM ran out loading (839 MiB free)`) and the
  expert tier shrinks to 11 of 512 experts — which changes the state and hides the bug. Use targeted
  synchronised checkpoints plus `CUDA_LAUNCH_BLOCKING=1` instead.
* **Driving the engine under a wrapper.** If you do want the sanitizer, start the server with
  `--engine-bin <wrapper>` (`tools/basalt-san-wrapper.sh`). Two details: the server reads the
  accepted engine flags out of the **bytes of the binary**, so a shell wrapper needs the flag list as
  a comment line, otherwise every flag is rejected; and the filter syntax is
  `--kernel-name kns=<substring>` (not `regex:<pattern>`).
* **Error messages now name the product.** The patch adds the call-site tag to the MMQ launch check
  (`ck(cudaGetLastError(), p.what)` with `p.what` = `gemm` / `heads` / `moe-gu` / `moe-down`). That
  is how we learned the fault is always on a `moe-down` product; with `CUDA_LAUNCH_BLOCKING=1` the
  attribution is exact. The diagnostics are behind `BASALT_DBG_Q=1` and cost nothing when unset.

---

## Open questions

* **Which instruction faults** inside the ring path is not found. Upstream's Blackwell MMQ table uses
  `GGML_CUDA_MMQ_SRAM_LAYOUT_FP4` + `MMQ_ITER_K_FP4` for NVFP4 where the port uses
  `Q3_K` + `MMQ_ITER_K`; that is consistent with the port's own loader, but we could not turn it into
  a fault in isolation.
* The stream-k fault was pinned to the MMQ product with blocking launches, but not to a line. A
  standalone MMQ test at the failing geometry is the obvious next step.
* Both fixes are **not upstreamed** — this repository only offers the patch. If the port's maintainer
  wants it, a PR is the natural place; we did not open an issue yet.

---

## Credits

Everything that makes this run is other people's work. Our thanks to:

* **[jesdga95](https://github.com/jesdga95)** and the Basalt contributors — the
  [Basalt](https://github.com/jesdga95/basalt) inference engine itself (MIT): expert tiering across
  VRAM/RAM/peer GPUs, MTP/chunk-draft speculative decoding, OpenAI- and Anthropic-compatible APIs,
  and the Hy4-flash support. Also for the NVFP4 pack
  [jesdga/Hy4-flash-Basalt](https://huggingface.co/jesdga/Hy4-flash-Basalt).
* **[yanmindi](https://github.com/yanmindi)** — the
  [basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121) port that makes Basalt build and
  run on Arm64 CPUs with the integrated 12.1 GPU, including the NEON kernels, the NVFP4/ARM dot
  products and the Arm64 build system. This work is a working copy of that port plus the fixes in
  `patches/`.
* **[Tencent Hunyuan](https://huggingface.co/tencent/Hy4-flash)** — the Hy4-flash model (224B total,
  12B active, 1M context, natively multimodal) released as open weights.
* **[AngelSlim](https://huggingface.co/AngelSlim/Hy4-flash-GGUF)** — the SPARK-4Bit GGUF and its spark-patch,
  which powered the *llama.cpp* comparison lane on the same box. **It is not what this recipe runs**: the
  lane described here serves the NVFP4 pack from the setup table above through Basalt. The two are separate
  lanes, and a comparison across quantisations is not a model comparison.
* **The [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) authors** — the MMQ kernels,
  tile configurations and quantisation layouts that Basalt's prefill path is built on, and the Arm
  kernels the port reuses.
* **[Niko1221](https://github.com/Niko1221/Strata) and the Strata contributors** — listed in Basalt's
  third-party licences.
* **[NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** — the DGX Spark / GB10
  platform and CUDA.
* **[opencode](https://github.com/sst/opencode)** and the benchmark harnesses —
  [opencode-bench-plusplus](https://github.com/knopkem/opencode-bench-plusplus) and
  [codec-reverse-bench](https://github.com/andreasknopke/codec-reverse-bench-leaderboard) — that
  produced the two results above.

## How this was produced

The bring-up, the fault hunting and the measurements were done on the Spark itself, driven by
[Hermes Agent](https://hermes-agent.nousresearch.com/) under human direction: the agent ran the
reproductions, bisected the configuration space, read the kernel sources, wrote the patches, and
ran both benchmarks. Numbers in this README come from tool output, not from summaries; where we
only have a hypothesis it says so.

## Licence

The patch and the tooling in this repository are offered under the MIT licence, the same terms as
Basalt upstream. Basalt, the Arm64/sm_121 port, llama.cpp/ggml, Strata, the model and the packs keep
their own licences — see the respective repositories.
