# Hy4 prompt experts on FP4 tensor cores — feasibility

**Verdict: feasible, and without a new tensor-core kernel.** The pinned tree already contains the
shared prompt-FP4 module (`include/basalt/prefill/moe_fp4.hpp`, `src/prefill/moe_fp4.cu`), Qwen3.8-Flash-Next's
prompt path uses it in the same checkout (`src/models/qwen4exp/prefill/expert_fp4.cpp`), and the Hy4
expert blobs are the same NVFP4 format the module reads: E2M1 codes, a UE4M3 scale per 16 values, a
matrix scale, and the 16-byte tail `{s_gate, s_up, s_down, 0}` (the comment on `kTypeNvfp4` in
`src/models/hy4flash/session.cpp`). The FP4 tensor cores are therefore already exercised on this GB10
by the Qwen lane; what is missing is the Hy4 plumbing, not a kernel.

## What the module gives

- `fp4::quantize_tokens(x, tokens, cols, ld, a4, gs, stream, two)` — one FP4 rounding per token row;
  with `two`, a second FP4 term of the residual (about 8 bits of activation precision).
- `fp4::quantize_f16(...)` — the same for the fp16 H rows between the gate/up and down products.
- `fp4::Product` — n experts' NVFP4 matrices in place in their blobs (`w_off`, `w_rows`, `w_cols`,
  `tail_off`), with `swiglu = true` giving `silu(g s_gate) (u s_up) s_down` in the gate/up product and
  an optional fp16 output for H; `bounds`, `tile_off`, `ids` and `dst` map rows through the routing.
- `fp4::Group` / `run_group` — a run of consecutive experts, H planes, pieces and runs (`cut_pieces`),
  up to `kMaxExperts = 16`; this is the API Qwen's `expert_fp4.cpp` uses.

## Hy4's shape against it

| Hy4 prompt path | FP4 path |
|---|---|
| 16 staged experts per launch (`kStageExperts`) | `kMaxExperts = 16`; `Product.blob[16]` |
| Gate/up `[3072, 1536]`, down `[1536, 1536]` | any multiple of 128; supported |
| Input `xlat_` (latent 1536), H after SwiGLU | `quantize_tokens(two=true)` → gate/up `swiglu` → `quantize_f16` → down |
| `bounds_tab_`, `pair_ident_`, `pair_rows_` | `bounds`/`tile_off`/`ids`/`dst`; `plan`, `plan_blocks`, `rebase` |
| `mmq` int8 products + `swiglu_split` | two FP4 products, SwiGLU fused |
| Expert blobs with the 16-byte tail | same tail at `tail_off` |

## Work needed

1. **Layout mapping** (2–3 days): find `w_off` of gate/up/down and `tail_off` inside Hy4's expert stride
   (`m_.expert_stride()`), and map the 16-expert staging buffer (`stage_[i].blobs`) to
   `Product.blob[]`/`base`. A one-layer prototype compares the FP4 products with the MMQ products
   (RMS / max row error) before any integration.
2. **MoE integration** (3–5 days): replace the prompt path's two `mmq` calls and `swiglu_split` with
   the token-plane quantize, the gate/up `Product` (`swiglu`, fp16 H), the H quantize and the down
   `Product`; feed the routing's bounds/ids and write `lat_out_`; keep the MMQ path behind a flag and
   as fallback. Only prompt chunks (`R > 8`); decode stays as it is.
3. **Quality** (2–3 days): KLD and top-1 over 100 chunks against the current int8-prompt engine, and
   greedy token identity with the tiers pinned; the two-term activation is the same choice Inkling
   makes for ~8 bits, but Hy4's experts need their own gate.
4. **Performance** (1–2 days): the same 33k/80k prompts and the nsys kernel summary; the FP4 product
   must beat the 19% expert MMQ share plus its share of the quantizer to be worth the complexity.

Total: the plumbing alone is ~1.5–2 weeks, but the 1536 kernel/tile support is the real work (see
“The module is dimension-specialized” below); the main risk is the tensor-core code, not the routing.

## Risks

- **Blob offsets**: Hy4's expert stride and the pack's gate/up/down/tail offsets must be exact; a
  prototype at one layer catches this cheaply.
- **Routing/bounds**: Hy4 groups pairs and stages 16 experts; Qwen's `Group` uses a run of experts and
  H planes. The `Product` path (16 blobs) is closer to Hy4's staging; `Group` is the fallback if the
  run/piece logic is needed.
- **Memory**: token planes (`a4`, `gs`) and H plans add a few hundred MiB at 16k rows; the prompt
  workspace already has ~2.5 GiB of headroom after patch 0003.
- **Quality**: two FP4 activation terms are ~8 bits; the gate is mandatory.
- **Gain uncertain**: the expert MMQ is 19% of the prefill; an FP4 product twice as fast is ~9–10%
  before the quantizer/SwiGLU fusion, and the token-plane quantize takes some of it back.

## Go / no-go

Go to the **one-layer prototype**: map the offsets, run gate/up/down on one 16-expert group at 33k
rows, compare with the MMQ products (error and time). If the products match within the int8 tolerance
and the group is at least ~30% faster, continue to the full MoE integration; otherwise stop — the
kernel is there, but the plumbing and the quality gate are not free.

No code has been changed for this study.

## Verified on the GB10 (0.5.7 main)

The shared module is not just theory: the Qwen prefill FP4 parity test (`tests/models/qwen4exp/prefill/prefill_fp4_test.cpp`,
built with `-DBUILD_TESTING=ON -DBASALT_MODELS=qwen4exp`) passes on this box — `parity: PASS`, bitwise against the
reference for every routing/run shape, edge tokens finite.

Its benchmark, 16,384 tokens × top 10 over 512 experts:

| Path | Time | vs MMQ |
|---|---:|---:|
| NVFP4 MMQ (gathers + q8_1 + gate/up + SwiGLU + down) | 73.7 ms | 1.00× |
| FP4 layer (plain gate/up + SwiGLU + down) | 47.1 ms | **1.56×** |
| FP4 token planes + fused SwiGLU | 27.9 ms | **2.64×** |

The FP4-vs-MMQ relative RMS is 0.095 (gate/up) and 0.111 (down), the two-term activation rounding; the
quality gate is still the real acceptance test. The log is `evidence/fp4-prefill-parity.log`.

## The module is dimension-specialized — the Hy4 blocker

The verified numbers above are for the **Qwen/Inkling dimensions**, not for Hy4. `moe_fp4.cu` dispatches
`quantize_tokens(..., two=true)` only for `cols == 2304` (one-term: 2560 or 640), and the high-level
`run_experts` hardcodes `kN = 2560, kFF = 640`. Hy4's latent and FF are both **1536**: the first
prototype call exits with

```
basalt: error: prefill fp4: 1536 token columns (two terms)
```

(`evidence/fp4-hy4-1536-error.log`). I then added the 1536 quantize dispatches (`quantize_kernel<3>`
for tokens/planes/f16); they compile, but the next step, the FP4 **gemm**, faults with
`prefill fp4: gemm: misaligned address` (`evidence/fp4-hy4-1536-gemm-error.log`), also with one-term
activations and one-term H (`evidence/fp4-hy4-1536-gemm-oneterm.log`). **But the 1536 product itself
works**: the micro-test `tools/fp4-1536-test.cpp` (one and two experts at the exact 1536×1536 geometry,
one-term token planes and H, real `fp4::Product`/`quantize` sequence) prints

```
fp4 1536: OK, finite=1, out[0]=1.04887
```

(`evidence/fp4-1536-microtest.log`). The kernel/tile geometry is therefore **not** the blocker; the
Hy4 prototype's misalignment is in the **real-mode plumbing**. The real geometry was logged
(`evidence/fp4-1536-real-geometry.log`): `lay.bytes` 3,981,328, `up_off` 1,327,104, `down_off`
2,654,208, `gu_row`/`d_row` 864, `stride_` 3,981,456 (`bytes` + 128); all the activation/H/blob pointers
are 16- or 128-byte aligned. Those match the synthetic layout. The micro-test now runs at the real
stage shape — **16 experts, 17,232 rows, max 11,771 per expert, R = 29,363 tokens, the real
`bytes + 128` stride, and a blob base at 16 mod 128** — and still prints
`fp4 1536: OK` (`evidence/fp4-1536-microtest-real-shape.log`). So the kernel, the multi-segment H
planes, the token/row split and the real stride/alignment are all correct. Forcing **one expert per launch**
in the full path still dies at `prefill fp4: gemm: misaligned address`
(`evidence/fp4-hy4-one-expert-fault.log`), and replacing its H buffer with a dedicated device
scratch does not help (`evidence/fp4-hy4-device-h-fault.log`). Zeroing every staged blob before the
FP4 product still faults (`evidence/fp4-hy4-zero-blob-fault.log`), so the **real blob data is not the
cause** either. With the synthetic real-shape micro passing, the remaining difference is the engine's
own stage plumbing (`B.tok`/`B.bounds`, the stage copy, or an overrun from one of the FP4 buffers),
not the product or the blob. The next test isolates the gate/up product from the down product and
dumps the stage tables; still local, no PR.

**Qwen already has this**: its prompt path uses `expert_fp4.cpp`/`run_group` at 2304/2560/640, and the
`prefill_fp4_test` above is its parity test. Nothing has to be implemented for Qwen; it is the reference
and the performance proof for the FP4 tensor cores on this box.

## Mapping to Hy4's MoE

Hy4's `NativeExpertLayout` (n_embd = latent 1536, n_ff = 1536) maps directly:

| Hy4 | `fp4::Product` |
|---|---|
| staged expert blob (`stage_[i].blobs`) | `blob[16]`, `base`/`stride` if all experts |
| gate/up at offset 0, `2 * n_ff` rows | `w_off = 0`, `w_rows = 2 * n_ff`, `w_cols = n_latent` |
| down at `lay.down_off`, `n_latent` rows | `w_off = lay.down_off`, `w_rows = n_latent`, `w_cols = n_ff` |
| 16-byte tail `{s_gate, s_up, s_down, 0}` | `tail_off = lay.bytes - 16` |
| `xlat_` rows, token id per pair row | `quantize_tokens(xlat_, R, L, two=true)`, `ids = pair_ident_`, `tokens = R` |
| `B.bounds` per stage | `bounds`, `max_rows` |
| H after the gate/up | `dst_f16` (or `dst`), then `quantize`/`quantize_f16` with `plan` + `tile_off` |
| down output pair rows | `dst = pair_rows_ + pos`, `ld_dst = n_latent` |
| 16 experts per launch | `n = nb <= 16` |

The token-plane path (`swiglu = true`) is the fast one; it reads each token's FP4 row once for all its
pair rows, which is exactly Hy4's pair expansion.

## The one semantic gap: the SwiGLU limit

Hy4 clamps the experts' SwiGLU (`swiglu_split_kernel`): `g = min(g, limit)`, `u = clamp(u, -limit, limit)`,
with the default limit **10.0** (`clamp_exp`), after the tail scales. `moe_fp4`'s fused SwiGLU has the tail
scales and an optional **softcap** (`tanh`), but not this min-clamp. Two ways out:

1. **Add `swiglu_limit` to the fused FP4 path** (a `Product` field plus a few instructions in
   `moe_fp4.cu` before the silu). Small, keeps the 2.64× path, but changes the shared module.
2. **Run the gate/up product plain (`swiglu = false`) and call Hy4's `swiglu_split_clamped`**, then
   quantize H and run the down product. No shared-kernel change and exact Hy4 semantics; it loses the
   token-plane fusion, staying near the 1.56× layer number.

The prototype should start with (2) for exactness, then try (1) for the fused gain.

## Prototype plan

1. A `BASALT_FP4_PROMPT=1` switch in `moe_prompt`'s `launch`: allocate the token planes once per layer
   (`quantize_tokens`), replace the gate/up and down `mmq_.run` with `fp4::Product` calls, keep the Hy4
   `swiglu_split_clamped` (option 2).
2. One layer, 33k prompt: compare the FP4 `pair_rows_`/`lat_out_` against the MMQ path (rel RMS) and the
   next-token logits; then all layers, greedy identity check.
3. 100-chunk quality gate (same `hy4flash-bf16-base.kld`): the FP4 two-term rounding must stay within the
   int8-prompt variance; otherwise stop.
4. Performance: 33k/71k/222k; the expert MMQ is 21% of the 222k prefill, so 1.56–2.64× there is ~7–13%
   end-to-end; measure the actual.

Estimate: prototype 3–5 days, full path and quality 1–2 weeks. The main risk is the H plane/tile layout
and the SwiGLU limit, not the FP4 kernel.


## Working prototype (one-term) and the quality trade

The full-path fault was the **down product's destination row offset**: the stage code passed
`pair_rows_ + pos` where `pos` is a pair-row index, not a row byte/float offset. With
`pair_rows_ + pos * n_latent` the FP4 path runs:

| Prompt | MMQ | FP4 one-term |
|---|---:|---:|
| 33k | 31.4 s | **24.8 s** |
| 222k | 270 s | **237.5 s** (943 tok/s) |
| MoE stage, 33k | 11.3 s | **4.5 s** |
| MoE stage, 222k | 85.3 s | **36.2 s** |

The quality gate, though, pays for the 4-bit activations: **PPL 17.2129, KLD 0.22542, top-1
85.40 %** against MMQ's 17.6184 / 0.20460 / 86.22 % (`evidence/hy4-quality-fp4-1term.log`). It is a
real but modest degradation, not parity.

The obvious fix would be the two-term (~8-bit) activations, but at 1536 the two-term path is
**wrong**: two-term gate/up and two-term H/down give KLD **1.775 / 1.745** and top-1 **62 %**
(`evidence/hy4-quality-fp4-2term.log`, `hy4-quality-fp4-hybrid.log`). The two-term 1536 quantize
dispatch compiles, but its plane/tile layout is not the one the product reads; that is a separate
kernel task. The correct, working prototype is therefore the one-term one; the two-term quality path
is still open.
