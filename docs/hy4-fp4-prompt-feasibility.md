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

Total: roughly **1.5–2 weeks**, no new CUDA kernel; the main risk is the layout/routing plumbing, not
the tensor-core code.

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
