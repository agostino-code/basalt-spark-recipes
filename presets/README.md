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
| `hy4-vision` | Hy4 | 1 slot × 262,144, `--vision dynamic` | image input; mutually exclusive with the four-slot shape |
| `qwen-agent` | Qwen | 4 slots × 262,144, int8 KV | the measured baseline; port 30001 |
| `qwen-2` | Qwen | 2 slots × 262,144, int8 KV | the largest configuration that measured cleanly |
| `qwen-single` | Qwen | 1 slot × 262,144, int8 KV | one full window at a time |

## Measured

All seven presets were started on the Spark (pin `f9bf55aa`, 2026-11-24) and answered a completion
(HTTP 200); the banners confirmed the shape (`kv=fp8`, two slots, `vision=gpu0`). What the variants
cost:

| Test | Result |
|---|---|
| `hy4-fp8` quality, 100 chunks T=1 | PPL 17.6522, KLD 0.20744, top-1 86.30 % — against fp16's 17.5207 / 0.20300 / 86.32 %. |
| `hy4-fp8` decode fixtures | fp16 24.8 / 39.4 / 19.6 tok/s, fp8 22.4 / 39.3 / 19.1 (one run each): struct and prose level, `conv` 10 % lower, so the gain is memory, not speed. |
| `qwen-2` under load | two ~89k-token prompts at once: both completed, 90.2 s wall, TTFT 42.9 / 51.9 s. |
| `--spec` on Hy4 | 4 is worse (struct 33.0 vs 39.4 tok/s); 8 is level with 6 (39.3 / 19.3 vs 39.4 / 19.6). Keep 6. |
| `qwen --kv k8v4` / `q4_0` | accepted at one slot and answered; not quality-measured, and not tried at four slots. |
| `--reasoning-budget-tokens 64` with `max_tokens 96` | the answer is kept (110 chars) instead of spending the cap on thinking. |

Not tested: `--route-tail-skip` / `--route-prior` (they need experts off the card; this box keeps all
512 resident) and `--pool-workers` (only used when the CPU pool runs experts).

## Add one

Create `presets/<name>.conf` with three lines (comments and blank lines are ignored):

```sh
MODEL=hy4
DESC=what this preset is for, shown by --list-presets
ARGS=--kv fp8 --parallel 4
```

`MODEL` is `hy4` or `qwen`; `ARGS` is a flag list, not a shell command — values must not contain
spaces. It is appended to the lane's start script, so use the same names the server and engine
take (`--kv`, `--parallel`, `--max-context`, `--vision`, `--spec`, ...).
