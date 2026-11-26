# Presets

Named launch configurations for the two lanes. A preset is a `.conf` file read by
[../run.sh](../run.sh); each one names its model and the flags appended to that lane's
`start-<model>.sh`:

```sh
./run.sh --list-presets
./run.sh --preset hy4-fp8
./run.sh --preset qwen-2 --port 30100   # extra arguments still go to the server
```

The start script always passes its own `--parallel 4 --max-context 1048576 --fit-max-tokens`
first; a preset's `--parallel`/`--max-context` come after it, and the server keeps the last
value. That is how `hy4-2` and `qwen-2` get two slots without a second start script.

| Preset | Lane | Shape | When |
|---|---|---|---|
| `hy4-agent` | Hy4 | 4 slots × 262,144, fp16 KV | the measured agent baseline; port 30000 |
| `hy4-fp8` | Hy4 | 4 slots × 262,144, fp8 KV | half the KV (~15 → ~8.7 GiB), the difference in expert slots |
| `hy4-2` | Hy4 | 2 slots × 262,144, fp16 KV | lower memory, shorter queue |
| `hy4-2x524` | Hy4 | 2 slots × 524,288, fp16 KV | two long windows; chunk policy in main |
| `hy4-1m` | Hy4 | 1 slot × 1,048,576, fp16 KV | the model's full 1M window; chunk policy in main |
| `hy4-vision` | Hy4 | 1 slot × 262,144, `--vision dynamic` | image input; mutually exclusive with the four-slot shape |
| `hy4-fp4` | Hy4 | 4 slots × 262,144, FP4 prompt experts | experimental one-term W4A4; needs patch 0003; quality trade, see below |
| `hy4-fp4-1m` | Hy4 | 1 slot × 1,048,576, FP4 prompt experts | experimental one-term W4A4; needs patch 0003; the full 1M window |
| `qwen-agent` | Qwen | 4 slots × 262,144, int8 KV | the measured baseline; port 30001 |
| `qwen-2` | Qwen | 2 slots × 262,144, int8 KV | the largest configuration that measured cleanly |
| `qwen-single` | Qwen | 1 slot × 262,144, int8 KV | one full window at a time |
| `qwen-fp16` | Qwen | 1 slot × 262,144, fp16 KV | one full window in fp16 |
| `qwen-fp16-2` | Qwen | 2 slots × 262,144, fp16 KV | two full windows in fp16; needs patch 0001 |
| `qwen-fp16-4` | Qwen | 4 slots × 131,072, fp16 KV | four windows in fp16 at int8's KV size; needs patch 0001 |
| `qwen-yarn-524` | Qwen | 1 slot × 524,288, fp16 KV, YaRN ×2 | experimental: twice the native 262,144 window; check long-context quality before trusting it |

## Measured

The twelve stock presets were started on the Spark (pin `f9bf55aa` + patches 0002/0003, 2026-11-24) and answered a completion
(HTTP 200); the banners confirmed the shape (`kv=fp8`, two slots, `vision=gpu0`). What the variants
cost:

| Test | Result |
|---|---|
| `hy4-fp8` quality, 100 chunks T=1 | PPL 17.6522, KLD 0.20744, top-1 86.30 % — against fp16's 17.5207 / 0.20300 / 86.32 %. |
| `hy4-fp8` decode fixtures | fp16 24.8 / 39.4 / 19.6 tok/s, fp8 22.4 / 39.3 / 19.1 (one run each): struct and prose level, `conv` 10 % lower, so the gain is memory, not speed. |
| `qwen-2` under load | two ~89k-token prompts at once: both completed, 90.2 s wall, TTFT 42.9 / 51.9 s. |
| `--spec` on Hy4 | 4 is worse (struct 33.0 vs 39.4 tok/s); 8 is level with 6 (39.3 / 19.3 vs 39.4 / 19.6). Keep 6. |
| `qwen --kv k8v4` / `q4_0` | accepted at one slot and answered; not quality-measured, and not tried at four slots. |
| `qwen` fp16 batch (patch 0001) | two slots and four slots × 131,072 start, answer and show no faults; four full 262,144-token fp16 windows also start and answer, but take 30.4 GiB of KV. |
| `qwen` batch numerics | the slots are not bitwise against a one-slot window: the same greedy prompt moves a row's logits by ~0.1 nats and a near-tie token can flip. The unchanged int8 batch does the same, so it is the 0.5.0 batch path, not the FP16 store; one slot for exact reproduction. |
| `qwen` int8 regression | the unchanged int8 batch path with two slots still answers. |
| `hy4-2x524` / `hy4-1m` (in main) | an 80k-token prompt passes on both (2 × 524,288: 200, 140 s; 1 × 1,048,576: 200, 139 s); 2 × 524,288 also passes 200k tokens (200, 267 s). A full 524k/1M prompt is not measured. |
| Hy4 prefill (in main) | 33k tokens in 32.4 s against 36.1 s with 16k chunks (~10%), 80k in 78.8 s; the 16k depth is unchanged and prompts above 128k keep 16,384-row chunks. |
| `qwen` fp16 + `--kv-resident` | refused at the batch init: `a batch window's FP16 K/V must be resident (drop --kv-resident)`. |
| `--reasoning-budget-tokens 64` with `max_tokens 96` | the answer is kept (110 chars) instead of spending the cap on thinking. |

## Experimental FP4

`hy4-fp4` enables the one-term FP4 prompt experts (W4A4) through `BASALT_FP4_PROMPT=1` and
**patch 0003** (the FP4 prompt path; `install-basalt.sh` applies it; [PR #793](https://github.com/jesdga95/basalt/pull/793), draft). On a 0.5.7 tree without the
patch the variable is ignored and the normal MMQ experts run.

```bash
./run.sh --preset hy4-fp4        # 4 slots x 262,144
./run.sh --preset hy4-fp4-1m     # 1 slot x 1,048,576
```

Measured with the prototype: 33k 24.8 s vs MMQ 31.4 s, 222k 237.5 s vs 270 s, 620k one slot
1222.7 s vs 1325.4 s. The full 1M prompt is not measured; 620k is the longest FP4 run so far. The
quality trade is real: KLD 0.22542 / top-1 85.40 % against MMQ 0.20460 / 86.22 %. It is
experimental, not the default; the two-term 1536 path that would close the gap is still broken
(KLD 1.67-1.78 in the attempts so far).

Not tested: `--route-tail-skip` / `--route-prior` (they need experts off the card; this box keeps all
512 resident) and `--pool-workers` (only used when the CPU pool runs experts).

## Add one

Create `presets/<name>.conf` with the three required lines and optional `ENV=` lines
(comments and blank lines are ignored):

```sh
MODEL=hy4
DESC=what this preset is for, shown by --list-presets
ARGS=--kv fp8 --parallel 4
ENV=BASALT_FP4_PROMPT=1    # optional, NAME=VALUE exported before the start; repeatable
```

`MODEL` is `hy4` or `qwen`; `ARGS` is a flag list, not a shell command — values must not contain
spaces. It is appended to the lane's start script, so use the same names the server and engine
take (`--kv`, `--parallel`, `--max-context`, `--vision`, `--spec`, ...).
