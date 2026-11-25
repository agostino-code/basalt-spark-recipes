# Basalt on Spark: Hy4-Flash and Qwen3.8-Flash-Next in NVFP4

Running **Tencent Hy4-Flash** (224B total / 12B active, natively multimodal) and
**Qwen3.8-Flash-Next** (`Qwen4ExpForCausalLM` — NVFP4 experts, a 262,144-token native window,
natively multimodal) on a **DGX Spark** (NVIDIA GB10, Arm64, compute capability 12.1) with the
[Basalt](https://github.com/jesdga95/basalt) inference engine.

This is one engine tree with two NVFP4 packs on one box. The two lanes share the Arm64/sm_121 port,
the pinned commit, the patch, the venv and the machine; only the pack, the alias and the KV default
differ — 30000 and 30001 are just the two launchers' ports. One build of `bin/basalt` serves both;
the packs sit side by side in `models/`; one lane fits in the box's 121 GiB of unified memory at a
time.

**This repository merges two former recipe repositories**, `hy4-flash-on-spark` and
`qwen38-flash-next-on-spark`. Their full measured records are kept as the two per-model appendices —
[docs/hy4-flash.md](docs/hy4-flash.md) and [docs/qwen38-flash-next.md](docs/qwen38-flash-next.md) —
with paths and cross-repo links updated, and the Hy4 appendix's stale port pin (`df6a10cf`) corrected
to the tree this unified installer builds. This README is the shared install and the unified view.

> **Base change, 2026-11-25.** The installer now builds the maintained fork
> [`agostino-code/basalt`](https://github.com/agostino-code/basalt), a direct fork of the official
> `jesdga95/basalt`, at main `c70e8011` (0.5.7 + the DGX Spark / Arm64 support that used to live in
> the third-party port). The 0.5.0-specific sm_121 patch is gone: main has the platform support. Only
> two patches remain — 0001 Qwen FP16 batch windows, 0002 Hy4 prompt-chunk policy. The 0.5.7 base
> was validated on the GB10: 33k **31,4 s**, 71k **77,5 s**, 222k **270 s** (0.5.0: 32,4 / 78,8 /
> 292), the 100-chunk gate **17,6184 PPL / 0,20460 KLD / 86,22 % top-1**, all 12 measured presets (and the experimental YaRN one) and both
> installer smokes pass, and a **624.830-token prompt** completes in 1325 s with no illegal access
> (the 8k cap). The 0.5.0 tables in the appendices remain the historical record. Upstream:
> [#694](https://github.com/jesdga95/basalt/pull/694) and
> [#695](https://github.com/jesdga95/basalt/pull/695).

What is ours is small: the fork, the two patches, the lane configuration and the measurements. The
engine, the port, the packs and the models are other people's work — see [Credits](#credits). The
port's own README measures the RTX Spark N1X and lists the DGX Spark as *the same GPU generation but
not tested*; both appendices are that validation for their pack.

---

## What is here

* [README.md](README.md) — this document: the unified install, the two lanes, the numbers and the traps.
* [LICENSE](LICENSE) — MIT for the text and scripts in this repository.
* [install-basalt.sh](install-basalt.sh) — one installer for both lanes: `--model hy4|qwen|all`.
* [run.sh](run.sh) — `./run.sh [--preset NAME|hy4|qwen] [server args]`, starts a lane from `basalt-port/`.
* [presets/](presets/) — the named launch configurations: agent, fp8, two-slot, vision, single
  window; `./run.sh --list-presets` lists them.
* [patches/0001-qwen-fp16-batch.patch](patches/0001-qwen-fp16-batch.patch) — Qwen's batch windows
  with FP16 K/V: the segment K/V append stores the resident FP16 pools, so `--parallel` 2–4 works
  with `--kv fp16` (streamed FP16 is refused). Rebases the 0.5.6 port onto main; PR #694.
* [patches/0002-hy4-chunk-policy.patch](patches/0002-hy4-chunk-policy.patch) — the prompt-chunk policy:
  32k rows up to 128k tokens, 16k up to 262k, 8k above 262,144 per slot (the NVFP4 `moe-down`
  workaround). On 0.5.7, 33k reads in 37.6 s against 41.1 s and 71k in 89.5 s against 99.3 s (engine);
  PR #695.
* [tools/](tools/) — the crash reproduction, the sanitizer wrapper, the two Hy4 launchers, the
  concurrency ladder and the Qwen acceptance check:
  * [tools/basalt-repro.py](tools/basalt-repro.py) — deterministic Hy4 crash reproduction (`crash`, `ladder`, `pair`).
  * [tools/basalt-san-wrapper.sh](tools/basalt-san-wrapper.sh) — runs the engine under `compute-sanitizer`, with the engine flag list as a comment line.
  * [tools/start-basalt-hy4.py](tools/start-basalt-hy4.py) and [tools/start-basalt-hy4.sh](tools/start-basalt-hy4.sh) — the supervisor wrapper for the Hy4 lane; they carry our Spark's absolute paths.
  * [tools/concurrency.py](tools/concurrency.py) — the concurrency ladder used for the Qwen four-window numbers.
  * [tools/qwen-lane-check.py](tools/qwen-lane-check.py) — the Qwen acceptance check (R1/R2/R3, then a fault scan of the log).
* [evidence/](evidence/) — the raw logs: the Qwen lane (`lane.log`, `smoke.log`, `basalt-bench.log`,
  `quality.log`, `bench-random-body.log`, `lane-quiet.log`), and the Hy4 lane after the Engram fix
  (`hy4-quality-f9bf55aa.log`, `hy4-quality-fp8.log`, `hy4-smoke-f9bf55aa.log`), the preset smoke
  (`presets-smoke.log`), the two-slot Qwen run (`qwen-two-slots-89k.log`), the FP16 batch windows
  (`qwen-fp16-batch-single.log`, `-pair.log`, `-quad.log`, `-full4.log`), the int8 regression and the
  resident refusal (`qwen-batch-int8-regression.log`, `qwen-fp16-resident-refused.log`), the batch
  logprob comparison (`qwen-batch-logprobs.log`), the pre-patch fp16 refusal (`qwen-fp16-batch-refused.log`),
  the long Hy4 windows after 0003 (`hy4-long-2x524.log`, `hy4-long-1m.log`), the 33k/80k prefill
  after 0003 (`hy4-prefill-33k.log`, `hy4-prefill-80k.log`), the nsys kernel profile
  (`hy4-prefill-kernels.txt`), plus the one RAM-tier fault seen before
  dropping the measurement base's page cache (`hy4-ram-tier-fault.log`).
* [docs/hy4-flash.md](docs/hy4-flash.md) — the full Hy4-Flash recipe: the two sm_121 faults, the fixes,
  the benchmark tables and the credits.
* [docs/spark-branch.md](docs/spark-branch.md) — the official main as the base: what it already has,
  the two remaining patches and the upstream PRs.
* [docs/qwen38-flash-next.md](docs/qwen38-flash-next.md) — the full Qwen3.8-Flash-Next recipe: the
  measured tables, the four-slot trap, the Infernix comparison and the credits.
* `basalt-port/` — the engine tree: a gitignored working checkout of the fork, with `build/`, `.venv/`,
  `models/` and `bin/basalt`. It is not committed; the installer clones it if missing.

## The two lanes

| | Hy4-Flash | Qwen3.8-Flash-Next |
|---|---|---|
| Model | Tencent Hy4-flash — 224B total / 12B active, 1M context, natively multimodal | Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`) — NVFP4 experts, 262,144-token native window, natively multimodal |
| Pack | `Hy4-flash-NVFP4.basalt`, 147.8 GiB | `Qwen3.8-FN-NVFP4-DQ6_K.basalt`, 122.6 GiB |
| Port | 30000 | 30001 |
| Slots / context | 4 × 262,144 (1,048,576 cells) | 4 × 262,144 (1,048,576 cells) |
| KV | engine default: fp16, ≈15 GiB for four windows | int8, 15.2 GiB for four windows |
| Alias | `hy4-flash` | `qwen3.8-flash-next` |
| Start script | `basalt-port/start-hy4.sh` | `basalt-port/start-qwen.sh` |

Only one lane is up at a time anyway. The rest of the lane shape is shared: four slots at 262,144
tokens each, prompt and answer inside the window, `--fit-max-tokens`.

## Install

On the box: a DGX Spark (GB10) or another Arm64 machine with a compute-capability 12.1 GPU, CUDA 13.x
(`nvcc`; measured with 13.0.88), GCC 13+, CMake 3.24+, Python 3.11+, 121 GiB of unified memory, and
disk for the build (~5 GiB) plus the packs you install (147.8 GiB for Hy4, 122.6 GiB for Qwen).
Nothing needs root.

```bash
git clone https://github.com/agostino-code/basalt-spark-recipes
cd basalt-spark-recipes

./install-basalt.sh --model hy4     # the Hy4-flash lane
./install-basalt.sh --model qwen    # the Qwen3.8-Flash-Next lane, in the same engine tree
./install-basalt.sh --model all     # both packs, both start scripts, one build
```

What it does, in order — every step is idempotent, so rerunning it after a failure is cheap:

| Step | What happens | Time |
|---|---|---|
| 1 | machine checks: aarch64, CUDA 13, cmake >= 3.24, gcc >= 13, python >= 3.11, free disk (160 GiB for the Hy4 pack, 130 GiB for Qwen) | seconds |
| 2 | engine tree: clones `agostino-code/basalt` (main) at **`c70e80118ead375dbcb37948d0cccad8937ad03a`** into `basalt-port/` (or `--dir`), or reuses the checkout already there | ~1 min |
| 3 | applies `patches/*.patch` in order when the tree does not already carry each one — 0001 Qwen FP16, then 0002 Hy4 chunk policy | seconds |
| 4 | builds it when `bin/basalt` is missing or was last built at another commit, `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121` | ~10 min |
| 5 | creates `basalt-port/.venv` and installs the pinned `requirements.txt` when the venv is missing | ~1 min |
| 6 | downloads each selected pack into `basalt-port/models/` when it is missing or incomplete, resumable and size-checked | 1–3 h per pack |
| 7 | writes `basalt-port/start-hy4.sh` / `start-qwen.sh`, and with `--install-unit` the systemd user unit(s) | seconds |
| 8 | drops the pack's page cache, starts and smoke-tests each selected lane — one after the other, never together | ~1 min (Hy4) / ~2 min (Qwen) |

The flags that matter:

| Flag | Meaning |
|---|---|
| `--model hy4\|qwen\|all` | which lane to install (default `hy4`); `all` builds one tree and fetches both packs |
| `--dir <path>` | engine tree; defaults to `$BASALT_DIR` if set, else `basalt-port/` in this repo |
| `--port <n>` | the lane's port (defaults 30000 for Hy4, 30001 for Qwen); single lane only |
| `--jobs <n>` | build parallelism, defaults to `nproc` |
| `--pack <file>` | use a pack you already have instead of downloading one; single lane only |
| `--skip-pack` | build only — no pack, no start script, no smoke test |
| `--no-smoke` | skip the smoke test |
| `--vision` | Hy4 only: the single-slot image configuration (`--parallel 1 --max-context 262144 --fit-max-tokens --vision dynamic`); refused with `--model qwen`, applied to Hy4 only with `--model all` |
| `--install-unit` | write the systemd `--user` unit(s) (`hy4-basalt` / `qwen-basalt`); they are left disabled on purpose |
| `--kv <fmt>` | KV format; Hy4 defaults to fp16 (`fp8` halves it), Qwen defaults to int8 (fp16 works with patch 0001); single lane only |
| `-h`, `--help` | usage |

`--pack`, `--port` and `--kv` describe one lane, so the installer refuses them with `--model all`.

Install both (`--model all`) is fine; running both is not. The weights alone — 86.2 GiB for Hy4 and
68.4 GiB for Qwen — exceed the 121 GiB of unified memory before any KV. Stop one lane before starting
the other. The start scripts refuse a port that is already serving; they cannot stop a busy box.

Afterwards:

```bash
basalt-port/start-hy4.sh                  # four slots x 262,144, port 30000
basalt-port/start-qwen.sh                 # four slots x 262,144, int8 KV, port 30001
./run.sh qwen                             # the same lane through the repo-level launcher
./run.sh hy4 --port 30100                 # extra arguments go to the server

systemctl --user start hy4-basalt         # if installed with --install-unit
systemctl --user start qwen-basalt
```

Presets are named flag sets for those same start scripts ([presets/](presets/), format in
[presets/README.md](presets/README.md)):

```bash
./run.sh --list-presets
./run.sh --preset hy4-fp8                 # Hy4 with fp8 KV
./run.sh --preset qwen-2 --port 30100     # Qwen, two full windows, another port
```

| Preset | Lane | Shape |
|---|---|---|
| `hy4-agent` | Hy4 | 4 slots × 262,144, fp16 KV — the measured baseline |
| `hy4-fp8` | Hy4 | 4 slots × 262,144, fp8 KV — half the KV, the difference in expert slots |
| `hy4-2` | Hy4 | 2 slots × 262,144, fp16 KV — lower memory |
| `hy4-2x524` | Hy4 | 2 slots × 524,288, fp16 KV — needs patch 0002 |
| `hy4-1m` | Hy4 | 1 slot × 1,048,576, fp16 KV — needs patch 0002 |
| `hy4-vision` | Hy4 | 1 slot × 262,144, `--vision dynamic` |
| `hy4-fp4` | Hy4 | 4 slots × 262,144, FP4 prompt experts — experimental; needs patch 0003 |
| `qwen-agent` | Qwen | 4 slots × 262,144, int8 KV — the measured baseline |
| `qwen-2` | Qwen | 2 slots × 262,144, int8 KV — the largest clean configuration |
| `qwen-single` | Qwen | 1 slot × 262,144, int8 KV |
| `qwen-fp16` | Qwen | 1 slot × 262,144, fp16 KV — one full window |
| `qwen-fp16-2` | Qwen | 2 slots × 262,144, fp16 KV — needs patch 0001 |
| `qwen-fp16-4` | Qwen | 4 slots × 131,072, fp16 KV — needs patch 0001; 15.2 GiB of KV |
| `qwen-yarn-524` | Qwen | 1 slot × 524,288, fp16 KV, YaRN ×2 — experimental, twice the native 262,144 window |

`hy4-fp8` halves the KV for +0.004 KLD and no top-1 loss; every preset's tested numbers are in
[presets/README.md#measured](presets/README.md#measured).

`basalt-port/` is a working checkout, not a vendored copy: it is gitignored and never committed. Its
`build/`, `.venv/` and `models/` are exactly what the two lanes share — one engine tree serves both
packs, and the disk holds each pack once. The start scripts drop the pack's page cache, refuse a busy
port, and pass every extra argument to `serve.server`. The units are left **disabled** on purpose:
the ports are shared with other lanes on our box, so an automatic start would race them.

## What we measured

Both lanes were measured on the same box: DGX Spark, GB10, 121 GiB of unified memory, Ubuntu 24.04,
CUDA 13.0.88, Basalt 0.5.0 plus the pinned fork's sm_121 fixes and (for the Hy4 quality gate) #556. The
installer now builds 0.5.7 main instead; the re-characterisation is in progress.
The full per-depth tables, the conditions and the raw logs are in the appendices; this is the
condensed record, with the measured figures kept exact.

### Hy4-Flash

| | |
|---|---|
| Prefill | **~790–830 tok/s, essentially flat from 16k to 262k tokens** — 822 at 16.5k, 828 at 165k, 808 at 220k, 796 at 250k, 791 at 262k |
| Decode | **20–23 tok/s**, essentially independent of context: 20.3 at 165k, 20.0 at 220k, 24.9 at 250k and 262k with one active slot |
| Context per request | **262,144 tokens, prompt + answer together**; 165,022, 220,022, 249,997 and 261,987 tokens pass, 264,022 is refused, so the window itself is the limit |
| Slots / KV | four slots × 262,144 = 1,048,576 cells; KV ≈ 15 GiB, experts resident in VRAM (`ram-experts=0.0GB`) |
| Quality vs BF16 | **PPL 17.5207 ± 0.4442** (base 18.4162), **KLD mean 0.20300**, **same top-1 86.32 %** — 100 chunks, 25,500 rows, T=1, after the Engram epsilon fix (#556). The pre-fix engine reads 0.782 KLD / 76.77 % on the same base |
| Prefill profile | **dense Q8_0 20 %, NVFP4 experts 19 %, attention 14 %, activation quantize 10–11 %, router + SwiGLU ~9 %, hyper-connections ~11 %** — nsys at 33k/80k, [details](docs/hy4-flash.md#where-the-prefill-goes) |
| Benchmarks | opencode-bench-plusplus, suite `locker-v5`, variant `default`: quality **0.9881**, wall **42:36**, 102 steps, peak request 104,014 tokens — measured on `8172978`, before #556. codec-reverse-bench, variant `high`: **57.0** points, roundtrip solved in stage 3 (35 of 57), 11/15 structure questions, compression ratio 61.2 |
| Stability | **0 engine faults** across both benchmark runs (≈500 requests, peak request 128k tokens) |

Full tables: [docs/hy4-flash.md#what-we-measured](docs/hy4-flash.md#what-we-measured). The two
benchmark results are on the public leaderboards: [issue #34](https://github.com/knopkem/opencode-bench-plusplus/issues/34)
and [issue #13](https://github.com/andreasknopke/codec-reverse-bench-leaderboard/issues/13).

### Qwen3.8-Flash-Next

| | |
|---|---|
| Load | **68.4 GiB in 54.1 s** (2,006–2,913 MB/s from the pack), server ready in ~100 s |
| Prefill | **1,797–2,083 tok/s from 64k to 252k** — 2,083 at 63.6k, 1,944 at 185.6k, 1,797 at 252.4k; 1,432 at 4k |
| Decode | **56.7–57.7 tok/s at every depth measured** (63.6k, 185.6k, 252.4k); 68.9 at 4k |
| Context per request | **262,144-token window** (prompt + answer); a 252,425-token prompt answers correctly and the server reports `ctx=262144 images=off` |
| KV | **int8, 15.2 GiB for four windows** (elastic, `1.5/15.2GiB` at start, ~15.5 KB per token) |
| KV (fp16) | patch 0002: 15.2 GiB for two 262,144-token windows, or four 131,072-token ones; four full 262,144 windows need 30.4 GiB |
| Quality | **PPL 3.3032 ± 0.0436** (base 3.2733), **KLD mean 0.06246**, **same top-1 91.68 %** — the pack card's 3.304 / 0.0624 / 91.65 % reproduced |
| Four full windows | **does not hold on this box**: four 252,010-token sessions took **621.7 s** wall, TTFT 187–620 s, decode 0.4–34.9 tok/s, almost 8 GB of swap in use |
| Stability | 0 engine faults in ~25 requests, including the acceptance ladder and the full-window runs |

Full tables: [docs/qwen38-flash-next.md#what-we-measured](docs/qwen38-flash-next.md#what-we-measured)
and [the four-slot trap](docs/qwen38-flash-next.md#the-four-slot-trap-this-box-cannot-hold-four-full-windows).
The Infernix comparison, same model and same box, is in
[the appendix](docs/qwen38-flash-next.md#basalt-vs-infernix-same-model-same-box).

## The shared engine

Both lanes run the maintained fork [`agostino-code/basalt`](https://github.com/agostino-code/basalt),
a direct fork of the official [`jesdga95/basalt`](https://github.com/jesdga95/basalt), pinned in the
installer at main `c70e80118ead375dbcb37948d0cccad8937ad03a` (Basalt 0.5.7). Main carries the
Arm64/sm_121 and DGX Spark support (the temporary `spark` branch was removed on 2026-11-25), the
Engram FP32 epsilon (#556), the `MemAvailable` memory planning for the integrated GPU and the 0.5.7
engine. One tree, one `bin/basalt` for both models.

The 0.5.0 port tree and its two sm_121 fault fixes are the **historical validation** (the
`spark-2026-11-24` tag and the appendices): the installer no longer builds them, because main has the
platform support and the fault workaround is now the chunk policy below.

The two patches still applied, in order:

[patches/0001-qwen-fp16-batch.patch](patches/0001-qwen-fp16-batch.patch) — Qwen's batch windows with
FP16 K/V. The segment K/V append (`qsa_dec_kv_post_seg`) stores into the resident FP16 pools, the
segment attention is told the slot's format, the fused pre path runs for a resident FP16 window, and
the batch init admits resident FP16 while refusing a streamed one. The Q4_0 path added in 0.5.7 is
preserved. Tested on 2 × 262,144 and 4 × 131,072 (all slots answer) and on the int8 4-slot
regression. Upstream: [PR #694](https://github.com/jesdga95/basalt/pull/694).

[patches/0002-hy4-chunk-policy.patch](patches/0002-hy4-chunk-policy.patch) — the Hy4 prompt-chunk
policy: 32,768 rows up to 128k tokens, 16,384 up to 262k, 8,192 above 262,144 per slot (the NVFP4
`moe-down` workaround). On 0.5.7 a 33k prompt reads in 37.6 s against 41.1 s and a 71,142-token prompt
in 89.5 s against 99.3 s (engine); two 524,288 slots read the same prompt in 104.7 s without a fault.
0.5.7 is slower than 0.5.0 overall, so this recovers part of the regression, not all of it. Upstream:
[PR #695](https://github.com/jesdga95/basalt/pull/695).

Both PRs are open; the patch files remain in this repository until they merge.

## The traps

* **Page cache looked like an out-of-memory error on 0.5.0.** The 0.5.0 engine compared
  `cudaMemGetInfo` against `MemFree`, and the page cache of a freshly downloaded 122.6–147.8 GiB pack
  counted as used. Main fixes the planning with `MemAvailable`; the installer still drops the cache
  with `os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)` before the smoke test, which is now
  harmless.
* **The device pool is smaller than the machine.** The GPU has no memory of its own: the firmware
  gives it a carve-out of the machine's unified memory, and the engine plans against that device
  figure, not against `free -g`. Another lane holding the GPU leaves a new engine with no memory at
  all — `nvidia-smi` and `cudaGetDeviceCount` both fail until it stops.
* **The four-slot trap (Qwen).** The server accepts `--parallel 4 --max-context 1048576` and plans
  four full windows, and one full window works exactly as planned. Four at once do not fail, but they
  swap. On 0.5.7 main+patches four **246,455-token** sessions all answered (HTTP 200): TTFT
  111 / 227 / 343 / 462 s, decode 15.4 / 20.3 / 4.7 / 36.3 tok/s, 120 GiB in use and 2 GiB of swap.
  That is better than 0.5.0 (621.7 s wall, TTFT 187–620 s, decode 0.4–34.9 tok/s) but still ~7.7 min
  to the last TTFT and one stream at 4.7 tok/s; two full windows remain the clean configuration.
* **Vision costs concurrency (Hy4).** The image encoder's reservation is taken per slot: with
  `--vision dynamic` only a single slot starts; `--parallel 2/3/4` dies in the planning step with
  `RAM ran out loading` at every context size from 65536 to 524288. Vision and the four-slot agent
  configuration are mutually exclusive on this box.
* **`compute-sanitizer` is not usable at this model size.** `memcheck` needs shadow memory; with an
  86 GiB model resident the engine aborts loading and the expert tier shrinks, which changes the
  state and hides the bug. Use targeted synchronised checkpoints plus `CUDA_LAUNCH_BLOCKING=1`
  instead — or the wrapper in `tools/`, with the two caveats in the Hy4 appendix.
* **`--fit-max-tokens` is not a memory guarantee.** It sizes the plan, not the working set of four
  concurrent prefills; it happily brought up four full windows on a box that then swapped. Treat it
  as a planner flag, not as admission control.

## Credits

Everything that makes this run is other people's work. Our thanks to:

* **[jesdga95](https://github.com/jesdga95)** and the Basalt contributors — the
  [Basalt](https://github.com/jesdga95/basalt) inference engine itself (MIT), the `.basalt` format,
  and the NVFP4 packs [jesdga/Hy4-flash-Basalt](https://huggingface.co/jesdga/Hy4-flash-Basalt) and
  [jesdga/Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt).
* **[yanmindi](https://github.com/yanmindi)** — the
  [basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121) port that makes Basalt build and
  run on Arm64 CPUs with the integrated 12.1 GPU, including the NEON kernels, the NVFP4/ARM dot
  products and the Arm64 build system. This work is a working copy of that port plus the fixes in
  `patches/`.
* **[Tencent Hunyuan](https://huggingface.co/tencent/Hy4-flash)** — the Hy4-flash model (224B total,
  12B active, 1M context, natively multimodal), released as open weights.
* **[Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** — the Qwen3.8-Flash-Next model,
  released as open weights.
* **[NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** — the DGX Spark / GB10
  platform and CUDA; and the NVFP4 checkpoint the Qwen pack's experts come from
  ([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)).
* **[AngelSlim](https://huggingface.co/AngelSlim/Hy4-flash-GGUF)** — the SPARK-4Bit GGUF and its
  spark-patch, which powered the *llama.cpp* comparison lane on the same box. **It is not what this
  recipe runs**: both lanes here serve the NVFP4 packs above through Basalt. A comparison across
  quantisations is not a model comparison.
* **The [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) authors** — the MMQ kernels, tile
  configurations and quantisation layouts that Basalt's prefill path is built on, and the Arm kernels
  the port reuses.
* **[Niko1221](https://github.com/Niko1221/Strata) and the Strata contributors** — listed in Basalt's
  third-party licences.
* **[eugene-kamenev](https://github.com/eugene-kamenev/Infernix-dgx-spark-gb10)** — Infernix and its
  DGX Spark retune, the Qwen comparison lane; its numbers are quoted from that repository and were
  measured on this box.
* **[opencode](https://github.com/sst/opencode)** and the benchmark harnesses —
  [opencode-bench-plusplus](https://github.com/knopkem/opencode-bench-plusplus) and
  [codec-reverse-bench](https://github.com/andreasknopke/codec-reverse-bench-leaderboard) — that
  produced the two Hy4 results above.

## How this was produced

The bring-up, the fault hunting and the measurements were done on the Spark itself, by agents under
human direction: the Hy4 lane was driven by [Hermes Agent](https://hermes-agent.nousresearch.com/),
which ran the reproductions, bisected the configuration space, read the kernel sources and wrote the
patch; the Qwen lane was brought up by an agent under human direction as well, with the pack
downloaded from Hugging Face (resumable, sha256-checked) and the lane run as a systemd user service.
The numbers come from tool output — `basalt-bench`, the engine's `serve_speed.py`,
`tools/qwen-lane-check.py` and the lane's own per-slot log lines — not from summaries. The raw Qwen
logs are in [evidence/](evidence/); where we only have a hypothesis, the appendices say so.

## Licence

The text and scripts in this repository — the README, the two appendices, the installer, `run.sh`,
`tools/` and the patch — are offered under the MIT licence, the same terms as Basalt upstream; see
[LICENSE](LICENSE). Basalt, the Arm64/sm_121 port, llama.cpp/ggml, Strata, the models and the packs
keep their own licences — see the respective repositories.
