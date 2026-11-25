# Qwen3.8-Flash-Next on Spark

> **This is the original Qwen3.8-Flash-Next recipe**, kept as the per-model appendix of the unified
> repository that merges it with the Hy4-Flash one. Install and start commands now live in the
> [README](../README.md) and [install-basalt.sh](../install-basalt.sh) — one engine tree,
> `./install-basalt.sh --model qwen`, `basalt-port/start-qwen.sh` (or `./run.sh qwen`).

Running **Qwen3.8-Flash-Next** (`Qwen4ExpForCausalLM` — NVFP4 experts, a 262,144-token native window,
natively multimodal) on a **DGX Spark** (NVIDIA GB10, compute capability 12.1) with
[Basalt](https://github.com/jesdga95/basalt), through the same Arm64/sm_121 fork, the same pinned commit
and the same sm_121 patch as the [Hy4-flash recipe](hy4-flash.md). Only the pack, the alias and
the KV default differ.

This repository fills the same gap the Hy4 recipe filled for Hy4-flash: the port's own README measures
Qwen3.8-Flash-Next on an **RTX Spark N1X** (Windows, 96 GB of GPU memory) and lists the DGX Spark as *the
same GPU generation but not tested*. We ran it on the Spark, measured what fits in the 121 GiB of unified
memory, found where the four-slot agent configuration breaks, and put the same model next to Infernix on
the same box.

**What is ours** is the pinned build, the measurements, the lane configuration, the traps and the
comparison. The engine ([Basalt](https://github.com/jesdga95/basalt)), the Arm64/sm_121 port
([yanmindi](https://github.com/yanmindi/basalt-arm-sm121)), the pack
([jesdga](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt)) and the model
([Qwen](https://huggingface.co/Qwen)) are other people's work — see [Credits](#credits).

---

## What we measured

Basalt `0.5.0`, pack `Qwen3.8-FN-NVFP4-DQ6_K` (122.6 GiB, sha256 verified), int8 KV, one DGX Spark GB10,
Ubuntu 24.04 (kernel 7.0.0), CUDA 13.0. The lane was started as `--parallel 4 --max-context 1048576
--kv int8`, i.e. four slots × 262,144 tokens.

| | |
|---|---|
| Load | **68.4 GiB in 54.1 s** (2,006–2,913 MB/s from the pack), server ready in ~100 s; "every expert is resident" with the device-planned MoE tail off |
| Prefill | **1,797–2,083 tok/s from 64k to 252k** — 2,083 at 63.6k, 1,944 at 185.6k, 1,797 at 252.4k; 1,432 at 4k (short prompts pay the fixed cost) |
| Decode | **56.7–57.7 tok/s at every depth measured** (63.6k, 185.6k, 252.4k); 68.9 at 4k. Expert-cache hit 99.5–99.9 %, draft acceptance 85–88 % |
| Context per request | **262,144-token window** (prompt + answer); a **252,425-token** prompt answers correctly and the server reports `ctx=262144 images=off` |
| KV | **int8, 15.2 GiB for four windows** (elastic, `1.5/15.2GiB` at start, ~15.5 KB per token); the engine reports `elastic_kv_ceiling: 1048576` |
| One full window | 252,425-token prompt: read 140.5 s (1,797 tok/s), decode 56.8 tok/s, 64 tokens — the shape of the single-session preset |
| Two full windows | two 252,010-token sessions at once: prefills 152.0 s and 167.7 s, decode 26.0 and 35.9 tok/s, 323 s wall together |
| **Four full windows** | **does not hold up on this box**: four 252,010-token sessions took 621.7 s wall, TTFT 187–620 s, decode 0.4–34.9 tok/s, and almost 8 GB of swap ended up in use — see [the four-slot trap](#the-four-slot-trap-this-box-cannot-hold-four-full-windows) |
| Stability | 0 engine faults in ~25 requests, including the acceptance ladder and the full-window runs; `illegal memory` appears 0 times in the lane log |

Single-stream, thinking off, output 128 tokens unless stated (the engine's own `tools/serve_speed.py`,
client-side read = first token, decode = last-first):

| Prompt | Read | Decode |
|---|---|---|
| 3,988 | 1,432 tok/s (2.8 s) | 68.9 tok/s |
| 63,625 | 2,083 tok/s (30.5 s) | 56.7 tok/s |
| 185,581 | 1,944 tok/s (95.5 s) | 57.7 tok/s (server: 52.2, acc 88 %) |
| 252,425 | 1,797 tok/s (140.5 s) | 56.8 tok/s (server: 54.5, acc 85 %, hit 99.5 %) |

A prompt that asks for a long answer (700 max tokens, thinking off) decodes at **44.3 tok/s** in one
stream; the 56–57 tok/s above are short 60–128-token answers.

## bench and quality

`basalt-bench` runs its own engine (fresh process, int8 KV, drafting at the pack's default window 8;
one run, so no standard deviation) over the bench fixtures and four prefill depths:

| Decode fixture | tok/s | draft acceptance | tokens per window |
|---|---:|---:|---:|
| conv (1,056 tokens) | 53.7 | 68.9 % | 2.41 |
| struct (7,230 tokens) | 95.0 | 98.1 % | 6.91 |
| prose (1,083 tokens) | 49.0 | 64.5 % | 1.98 |

| Prefill depth | prompt tokens | engine | wall | time |
|---|---:|---:|---:|---:|
| 16,384 | 16,676 | 2,027 tok/s | 2,013 tok/s | 8.2 s |
| 65,536 | 66,617 | 2,068 tok/s | 2,065 tok/s | 32.3 s |
| 131,072 | 133,205 | 1,989 tok/s | 1,987 tok/s | 67.0 s |
| 262,144 | 266,381 | 1,798 tok/s | 1,797 tok/s | 148.2 s |

**Treat those as fixture numbers, not as this engine's prefill.** Two later runs of the same command measured
2,460/2,521/2,451/2,229 tok/s at the four depths and 70.5/121.5/62.0 on the decode fixtures — a
session-to-session swing of ~20 % on the prefill with the same build, clock and pack — and the fixture's body
is one repeated filler text: with a body of random tokens at the same depth the 64k prefill reads **2,015
tok/s** instead of 2,536. The served lane's real-text numbers in [What we measured](#what-we-measured) moved
far less (2,040–2,090 at 64k, 1,830–2,010 at 250k) and are the ones we quote. The pack's card has a table
measured on RTX 5090 + RTX 5060 Ti, and the port's N1X table has 2,558/2,638 tok/s at 16k/64k; with this
fixture's own ~25 % spread we do not read a platform gap out of the comparison.

The engine reports all **63.3 GiB of experts resident** (24,576 slots, 0 GiB in RAM or on the SSD),
2,470–2,509 MHz during the run and 29.4 GiB of VRAM free after the load.

The quality gate (`basalt-bench --quality 100`) scores 100 chunks of wikitext-2 through the decode path
against the published BF16 reference base (`flashnext-bf16-base.kld`, sha256 checked), 25,500 scored rows,
T=3, int8 KV:

| | This lane, DGX Spark | Pack card (RTX 5090 + 5060 Ti) |
|---|---:|---:|
| PPL | 3.3032 ± 0.0436 (base 3.2733) | 3.304 |
| KLD mean | 0.06246 (median 0.01366, p99 0.7042) | 0.0624 |
| same top-1 | 91.68 % | 91.65 % |

The pack reproduces its published quality on the Spark to the third decimal — what one expects from a
numerical quantity, and worth checking anyway: a wrong KV format or a placement bug moves KLD, not the
model's own PPL. The gate took 583 s (5.8 s per chunk).

## Install

> **Base change, 2026-11-25.** The installer now builds `agostino-code/basalt` main `c70e8011`
> (0.5.7 + DGX Spark); the numbers below were measured on the 0.5.0 pin `f9bf55aa`. The FP16 batch
> patch is now 0001 and targets main (PR [#694](https://github.com/jesdga95/basalt/pull/694)).

One script builds the lane with the versions this recipe pins — main commit `c70e8011`, CUDA 13,
`CMAKE_CUDA_ARCHITECTURES=121`, the `Qwen3.8-FN-NVFP4-DQ6_K` pack (122.6 GiB):

```bash
git clone https://github.com/agostino-code/basalt-spark-recipes
cd basalt-spark-recipes

./install-basalt.sh --model qwen              # basalt-port/ (the shared tree), port 30001, int8 KV
./install-basalt.sh --model qwen --pack /mnt/Qwen3.8-FN-NVFP4-DQ6_K.basalt   # pack you already have
```

What it does, in order — every step is idempotent:

| Step | What happens | Time |
|---|---|---|
| 1 | machine checks: aarch64, CUDA 13, cmake >= 3.24, gcc >= 13, python >= 3.11, 130 GiB free | seconds |
| 2 | clones `agostino-code/basalt` (main) at **`c70e8011`** | ~1 min |
| 3 | applies `patches/*.patch` in order when the tree does not already carry each one (0001 Qwen FP16, 0002 Hy4 chunk policy) | seconds |
| 4 | builds it when `bin/basalt` is missing or was last built at another commit, `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121` | ~10 min |
| 5 | creates `./.venv` and installs the pinned `requirements.txt` | ~1 min |
| 6 | downloads the pack from [jesdga/Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt), resumable, size- and sha256-checked | 1–3 h |
| 7 | writes `start-qwen.sh`, optionally the systemd user unit, starts the lane once and smoke-tests it | ~2 min |

The sm_121 patch is shared with the Hy4 recipe: one half (the NVFP4 MMQ tiled path, `src/prefill/mmq/`)
applies to every model whose prompt path runs NVFP4 expert products, Qwen included; the other half (the
expert ring in `src/models/hy4flash/session.cpp`) is Hy4-specific and does not touch this model.

Nothing needs root. Afterwards:

```bash
basalt-port/start-qwen.sh                   # foreground (./run.sh qwen also works)
systemctl --user start qwen-basalt          # if installed with --install-unit
```

The unit is deliberately left **disabled**, and the script refuses a busy port instead of starting into
someone else's server.

### One engine tree, several packs

A pack is a file, not a build: this recipe and the
[Hy4-flash one](hy4-flash.md) use the same `bin/basalt`. The one installer puts both models in
**the same tree** (`$BASALT_DIR`, else the in-repo `basalt-port/`) and drops their packs in
`models/`, so one build serves both models and the disk holds the packs once. The tree carries one
launcher per model — `start-hy4.sh` (four slots × 262,144, `:30000`) and `start-qwen.sh` (four slots ×
262,144, int8 KV, `:30001`); `start-lane.sh` is kept as the old name and points at the Hy4 one.

### Quick command

`./run.sh qwen` in this checkout starts this lane — it finds the engine tree as `$BASALT_DIR`, else
`./basalt-port` in this checkout, and passes any extra argument to the server:

```bash
./run.sh qwen               # four slots x 262144 on :30001
./run.sh qwen --port 30102  # another port, same model
./run.sh --preset qwen-2    # two full windows - the largest clean configuration
./run.sh --preset qwen-single
./run.sh --preset qwen-fp16-2   # two full windows in fp16 (patch 0002)
./run.sh --preset qwen-fp16-4   # four 131,072-token windows in fp16
```

The named presets are in [presets/](../presets/); `qwen-agent` is the four-slot baseline above and
`qwen-2` the shape the two-window measurement used.

### Requirements

| | |
|---|---|
| Machine | DGX Spark (GB10) or another Arm64 box with a compute-capability 12.1 GPU |
| CUDA | 13.x toolkit (`nvcc`); measured with 13.0.88 |
| Toolchain | GCC 13+, CMake 3.24+, Python 3.11+ |
| Disk | 122.6 GiB for the pack, ~5 GiB for the build |
| Memory | 121 GiB unified: 68.4 GiB of weights, plus the KV (15.2 GiB for four windows) |
| GPU | **free**: another lane holding the GPU leaves a new engine with no memory — `nvidia-smi` and `cudaGetDeviceCount` both fail until it stops |

### Lane configuration

```bash
# what the installer writes
--parallel 4 --max-context 1048576 --fit-max-tokens --kv int8
```

That is four slots of 262,144 tokens (prompt + answer inside each window), KV int8, 15.2 GiB of KV in
total. The server divides `--max-context` among the slots and reports `ctx=262144 (1048576/4 slots)
kv=int8 spec=8`; `/v1/models` exposes `max_model_len: 262144`.

Measured, at start: `loaded 68.4GiB 54.1s`, `kv 1.5/15.2GiB cuda0 elastic slots=4`,
`par=4 ctx/slot=262144, others queue`, `ctx=262144 images=off`.

`--kv fp16` is the model's default. At the pin alone the batch windows refused it — the server
started, then the first request failed with `--batch, stage 1: verify: batch windows need INT8 K/V
without rotation, resident or streamed (--kv int8)` (upstream 0.5.6 refuses that combination at the
start). **Patch 0002 removes that limit**: the segment K/V append now stores resident FP16 pools and
the segment attention is told the slot's format, so `--parallel` 2–4 works with FP16.

| Shape | KV | Preset | Tested |
|---|---|---|---|
| 1 slot × 262,144 | 15.2 GiB | `qwen-fp16` | yes |
| 2 slots × 262,144 | 15.2 GiB | `qwen-fp16-2` | starts, both slots answer, no faults |
| 4 slots × 131,072 | 15.2 GiB | `qwen-fp16-4` | starts, all four answer, no faults |
| 4 slots × 262,144 | 30.4 GiB | — | starts and answers; experts spill to RAM |

The FP16 batch path is **as reproducible as int8's, which is not bitwise against a one-slot window**
at this pin. The same greedy prompt through one slot and through a batch moves a row's logits by
~0.1 nats and can flip a near-tie (`>` against `>=` in a `max2` answer). The unchanged int8 batch
does the same, so this is the 0.5.0 batch path's arithmetic, not the FP16 store: for exact
reproduction run one slot. Upstream 0.5.6 adds the batch conformance check that pins this down.

FP16 K/V must be **resident**: `--kv-resident` with FP16 is refused at the batch init (the pool
resolve copies INT8 pools). The int8 lane stays the default because it gives the same four windows
at half the KV; FP16 is for a two-window lane or a 4 × 131,072 one.

### Acceptance test before every run

`tools/qwen-lane-check.py` exercises the shape that faulted the Hy4-flash lane (a large prompt prefix
reused, then a small chunk appended) and fails if any request fails or the log records a fault:

```bash
python3 tools/qwen-lane-check.py --base http://127.0.0.1:30001 --log <the lane's log>
```

Measured on this box: R1 16,406 tokens (9.2 s, 1,887 tok/s), R2 32,828 (16.7 s, 2,015 tok/s),
R3 33,304 (10.3 s, 1,695 tok/s with the prefix reused) — **PASS**, 0 fault-like lines.

## The four-slot trap: this box cannot hold four full windows

The server accepts `--parallel 4 --max-context 1048576` and plans it: four full 262,144-token windows,
15.2 GiB of int8 KV next to the 68.4 GiB of weights. One full window works exactly as planned. Four
windows at once do not, and the failure is not a refusal — it is swapping:

| Sessions in flight | TTFT | Decode | Wall |
|---|---|---|---|
| 1 × 252,425 | 140.5 s (1,797 tok/s) | 56.8 tok/s | ~145 s |
| 2 × 252,010 | 152 s / 168 s | 26.0 / 35.9 tok/s | 323 s |
| 4 × 252,010 | 187 / 335 / 480 / 620 s | 8.6 / 23.4 / 34.9 / 0.4 tok/s | **621.7 s** |

The engine's own log shows the prefills serialising (each 140–145 s of engine time) while the box has
`MemAvailable` under 2 GiB and almost 8 GB of swap in use. The engine still answered every request — no fault,
no refusal — but the lane spends minutes queueing and some streams decode at under 1 tok/s. For agent
workloads this is the expensive failure the Hy4 README warns about, in a new form: not a crash, a
throughput collapse.

Batching itself is fine when the prompts are small: with 81-token prompts asking for 700-token answers,
one stream decodes at 44.3 tok/s, two at 28.0 each (53 tok/s together) and four at 18.6 each (67 tok/s
together) — the same shape the port measured on the N1X (63.7 / 78.4 / 80.1). The collapse above is
memory, not scheduling.

Two full windows are the largest configuration that measured cleanly (323 s wall, 26–36 tok/s each). If
all four windows must stay resident, the pack has to be smaller or the KV quantised further
(`--kv k8v4` is 816 B/cell against int8's 1,056; `--kv q4_0` halves int8) — not measured here.

## Basalt vs Infernix, same model, same box

Infernix ([eugene-kamenev/Infernix-dgx-spark-gb10](https://github.com/eugene-kamenev/Infernix-dgx-spark-gb10))
runs the same model on the same Spark with its own `HC4-W4Dense` NVFP4 artifact (73.96 GB) and its
measured presets. Its numbers, from that repository, measured on this box on 2026-11-23:

| | Infernix (measured here) | Basalt (measured here) |
|---|---|---|
| Single stream, 4k prompt | 1,473 tok/s read, 63.6 tok/s decode | 1,432 tok/s read, 68.9 tok/s decode |
| Single stream, 191k prompt | 2,387 tok/s read (80 s), 55.1 tok/s decode | 1,944 tok/s read (95.5 s), 57.7 tok/s decode |
| Single stream, 260k prompt | 2,224 tok/s read (117 s), 54.3 tok/s decode | 1,797 tok/s read (140.5 s), 56.8 tok/s decode |
| One full window (262,144) | 42.0 tok/s decode, prefill 80.9 s (3,238 tok/s) | 56.8 tok/s decode, prefill 140.5 s (1,797 tok/s) |
| Four full windows | 352 s wall, 31.0 tok/s per session, KV 13.13 GiB | 621.7 s wall, 0.4–34.9 tok/s |
| Prose, one lane | 71.8 tok/s (5-run mean, MTP + n-gram) | 49.0 tok/s (bench prose fixture, drafting on) |
| Quality | tool-eval-bench 96/100 (65/65 passed) | Basalt quality gate, see above |

Same model, so this is an engine comparison, not a model comparison — but not an equal-quantisation one
either: Infernix's artifact is 73.96 GB of `HC4-W4Dense`, Basalt's pack 122.6 GiB with NVFP4 experts and
Q6_K dense projections. What the table says, holding the box and the model fixed: **Infernix reads prompts faster** — about
20–25 % at depth on the server-side numbers (its own bench, not the server, reports 3,238 tok/s for a
full 262k window) — and its four-window preset fits where Basalt's does not; **Basalt decodes faster per
stream** at long context (57 vs 54 at 190k–260k, and 57 vs 42 on a full window). Neither wins everywhere.

## Smaller traps worth knowing

* **A second lane cannot even enumerate the GPU.** With Infernix holding its 110 GiB, `nvidia-smi` fails
  with `Failed to initialize NVML: Unknown Error` and a fresh CUDA process gets `cudaGetDeviceCount → 0`
  with `OS call failed or operation not supported on this OS`. Stop the other lane first; nothing else
  needs fixing.
* **Page cache looks like an out-of-memory error.** Right after the 122.6 GiB download the box reports a
  handful of GiB `MemFree` and the engine's planner refuses to load (`RAM ran out loading`). No root
  needed: `os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)` on the pack returns the cache instantly.
  The installer does it before the smoke test.
* **The pack is checked, not trusted.** 131,681,979,197 bytes, sha256
  `177f9b70c63ecd7efd17fc705ad85a8c8253d59e4f04dd10a001b556d15fad33`, format v2 (`BASALTMF`), sections
  `tokenizer/*`, `dense` (1.43 GiB), `experts` (63.28 GiB), `gguf` (55.60 GiB), `mtp/*` (0.77 GiB),
  `vision/*` (0.86 GiB), `hc8` (0.66 GiB). The installer verifies size and sha256.
* **`--fit-max-tokens` is not a memory guarantee.** It made the server bring up four full windows on a
  box that then swapped (above): the flag sizes the plan, not the working set of four concurrent
  prefills.
* **`basalt-bench`'s prefill fixture inflates prefill by ~25 %.** The bench builds its 16k/64k depths by
  repeating one filler body, so the experts route to a narrow set: 2,536 tok/s at a 64k fixture prompt
  against **2,015 tok/s** with a body of random tokens at the same depth (and 2,040–2,090 on the served
  lane's real text). Compare prefill across engines only on the same text; every prefill number in this
  README outside the bench table is real-text.
* **The SM clock is ~2,500 MHz, not the 3,003 MHz the driver reports as maximum.** Under a prefill the GPU
  sits at 2,450–2,550 MHz (62–77 W, 73–81 °C, no throttle reason active). `sudo nvidia-smi -lgc 3003` is
  accepted and then ignored: with the lock, 16k/64k prefill measured 2,015/2,063 against 2,027/2,068
  without. Reset it with `nvidia-smi -rgc`.
* **`--prefill 65536` is a fixture-sized win.** On the bench fixture it reads 262,144 tokens at 2,267 tok/s
  against 2,229 with `auto` (+1.7 %), and 16,384 at 1,989 against 2,027 (−3 %); 131,072-token chunks make
  the engine exit during the request. Not verified on real text, so the lane keeps `auto` (32,768).
* **The lane's log is worth keeping.** `serve.server` prints the per-slot lines
  (`s1 done pp=... tg=... acc=... hit=...`) that give server-side decode rates and expert-cache hits —
  the numbers in the tables above are from those lines and from the client measurements, not from
  estimates.

## Open questions

* Whether `--kv k8v4` (816 B/cell) makes four full windows fit and what it costs in decode: not measured.
* Whether a smaller expert format (the pack's `UD-IQ1_S`, 65.6 GiB with 31.1 GiB of experts) changes the
  four-window verdict, and at what quality cost (KLD 0.2676 against 0.0624 for this pack, published).
* The port's N1X table was measured with Windows and a 96 GB GPU partition; the per-depth shape here
  (prefill flat 1.8–2.1k, decode flat 57) may differ on other sm_121 boxes.
* The sm_121 patch's NVFP4 half is generic, but whether this model hits the stream-k fault without it was
  not reproduced here: we ran the fixed build only.
* Whether the served lane can reach the bench fixture's prefill rate on real text. Nothing we changed moved
  it: grammar on/off (the engine flag and a server patch to honor `--no-grammar`), 1 vs 4 slots, a 66k vs
  262k window, 14 vs 19 pool workers, cpu-share 0/0.3/0.6/0.8 and 16k/32k/64k chunks all land at
  2,010–2,090 tok/s on real text. The remainder is the prompt content, not a knob.

## Credits

* **[jesdga95](https://github.com/jesdga95)** and the Basalt contributors — the engine (MIT), the
  `.basalt` format and the `Qwen3.8-Flash-Next-Basalt` packs.
* **[yanmindi](https://github.com/yanmindi)** — the [basalt-arm-sm121](https://github.com/yanmindi/basalt-arm-sm121)
  port that makes Basalt build and run on Arm64 + compute capability 12.1, including the NEON kernels.
* **[Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** — the model, released as open weights.
* **[NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)** — the NVFP4 checkpoint the pack's
  experts come from.
* **[eugene-kamenev](https://github.com/eugene-kamenev/Infernix-dgx-spark-gb10)** — Infernix and its
  DGX Spark retune, the comparison lane; its numbers here are quoted from that repository and were
  measured on this box.
* **[NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** — the DGX Spark / GB10
  platform and CUDA.

## How this was produced

The bring-up and the measurements were done on the Spark itself by an agent under human direction: the
pack was downloaded from Hugging Face (resumable, sha256-checked), the lane started as a systemd user
service, and the numbers come from `basalt-bench`, `tools/serve_speed.py`, `tools/qwen-lane-check.py`
and the lane's own log lines. Where we only have a hypothesis it says so. The raw logs of the runs are
in [`evidence/`](../evidence/): the lane's per-slot log, the `basalt-bench` detail and the quality gate; the
concurrency ladder is [`tools/concurrency.py`](../tools/concurrency.py).

## Licence

The installer and tooling in this repository are offered under the MIT licence, the same terms as Basalt
upstream. Basalt, the Arm64/sm_121 port, the pack and the model keep their own licences — see the
respective repositories.
