# Deep optimization plan — Hy4 & Qwen (branch `deep-opt`)

Baseline misurata su 0.5.7 + patch 0001/0003/0004 (FP4 one-term, 8k cap): Hy4 33k 24,8 s, 222k 237,5 s,
620k 1222,7 s; qualità KLD 0,22542 / top-1 85,40 % vs MMQ 0,20460 / 86,22 %.

## Dove va il tempo (profilo)

| Contesto | attention+dense | MoE | Engram | ring copies |
|---|---:|---:|---:|---:|
| 222k | 177,9 s (75%) | 38,2 s (16%) | 7,6 s | — |
| 620k | **1005,6 s (82%)** | 143,6 s (12%) | 65,7 s (5%) | 341,7 s |

Profilo 222k FP4: dense Q8_0 22%, `attend_mma` 16%, FP4 10%, quantize 7%, router 6%, HC 7%, `scores_mma` 5%, prepare 4%.

## Hy4 — in ordine di leva

### 1. Chunk 16k/32k a >262k (lo stallo) — il più concreto
- Repro: il 16k a 620k stalla a ~393k (`stage "reading the prompt" 807 s; 0 layers`); l'8k passa.
- Escluso: ring (total-based, `kRingRows=1024`), buffer (`pair_rows_`/`gu_`/`h_`/`a4h` dimensionati giusti), expert pool (idle).
- Il commento del codice dice: "a ~15k-row chunk faults in the NVFP4 moe-down launch".
- Bug latente trovato: il loop degli stage aggiunge il primo esperto **senza cap** `kLaunchRows` → con `kLaunchRows=8192` e un esperto >8192 righe, `xq1_`/`xq2_`/`gu_`/`h_`/`a4h` sforano (lo slice fatto, corretto ma non risolve lo stallo 620k).
- **Next**: instrumentare il prompt stage (contatori per layer/chunk) e il moe-down per trovare l'hang; poi fix + chunk 16k.

### 2. Attention (82% a 620k) — il più grande
- `attend_mma` (16%) + `scores_mma` (5%), MMA BF16; `kMmaCells=32` (64 rompe: smem/stride accoppiati a 32).
- Opzioni: redesign tile/smem (bank-conflict), FP8 attention (half-rate su sm_121, rischio qualità), riduzione chiavi QSA.
- **Next**: redesign `kMmaCells`/`kLd`/`kPLd` con analisi bank-conflict, o variante FP8; quality gate.

### 3. Ring / expert streaming (341 s a 620k)
- Seconda card 0 righe, CPU pool 0, PCIe 0%; esperti streamati.
- Opzioni: resident/mmap, pool CPU, seconda card, layout pack, prefetch/overlap delle copie.

### 4. Dense / router / HC (35% a 222k)
- Dense FP8: **già testato, nessun guadagno** (half-rate e4m3, errore 5×).
- Opzioni: fusione router/HC, tuning.

### 5. Engram (5%)
- `--ple-io ram`, row cache/inflight.

## Qwen

### 1. Contesti >262k (RAM guard)
- 524k fp16 KV ~15 GB + modello 123 GB su HBM 121 + altri programmi → `RAM critical 808 MiB`.
- Fix: `--kv int8`/`nvfp4` (preset), liberare RAM; il preset `qwen-yarn-524` è stato corretto a `--kv int8` (~7.6 GiB di KV invece di ~15 GiB).

### 2. FP16 batch (PR #694)
- Il segment FP16 append; parity bitwise; manca il benchmark con power label (bot).

### 3. Decode
- `--spec 6` misurato; route-tail/prior non testati (esperti residenti); MTP.

## Regole di lavoro
- Tutto su `deep-opt`; `main` resta lo stato validato.
- Ogni modifica: misura 33k/222k/620k + quality gate + PR.
- Niente costanti kernel cieche senza il redesign smem/bank-conflict.

## Diagnosi 16k — round 1 (in corso)

- Il 16k a 620k è uno **hang non deterministico** nel prompt stage: tre run si sono fermati al chunk
  24 (393 216), 30 (491 520) e 31 (507 904) — sempre a un confine di chunk (multipli di 16 384),
  sempre a **layer 0** del chunk in corso.
- Strumentazione `[DEBUG-stall]` aggiunta nel layer loop (`ring / attn / ffn`); `kAttRows` riportato a
  2048 (era rimasto 4096 da un test precedente, senza effetto sullo stall).
- La sequenza di layer 0 è `ring_ahead → engram → attention → ffn`; il prossimo run con la
  strumentazione fine individua il kernel. Il loop resta lento (~10 min, posizione variabile).

## Diagnosi 16k — round 2

- Strumentazione fine nel `moe_prompt` (`moe_start`/`moe_router`/`moe_sync`): lo hang è **dentro
  `ffn(l)` → `moe_prompt(l)`, prima della `moe_sync` finale** (run reale: `ffn(3)` a pos 442 368, nessun
  `moe_sync l3`).
- `moe_prompt` ha due sync: il primo `cudaStreamSynchronize(st)` (attende lo stream dell'attention del
  layer) e la `moe_sync` finale. Il prossimo run pulito (in corso) distingue il primo sync (→ attention)
  dal launch/down.
- Il commento del codice sul "moe-down" riguarda il fault con `kLaunchRows=8192`, non questo stall: lo
  stall 620k è nel percorso `ffn`/`moe_prompt`/attention, non il down.

## Diagnosi 16k — round 3 (concluso: è l'attention)

- Il run pulito `debug4` conferma: lo hang è tra `moe_start l6` e `moe_router l6` = il **primo
  `cudaStreamSynchronize(st)`** di `moe_prompt`, che attende lo stream dell'**attention** del layer.
- Quindi lo stall 16k/620k è un **hang GPU nei kernel dell'attention** (`attention_step` →
  `mla_prepare` / `indexer` / `mla_attend`), non il moe-down, non il ring, non i buffer, non il host.
- La posizione è non deterministica (chunk 1, 24, 26, 30) ma sempre dentro l'attention; coerente con il
  `kMmaCells=64` che rompeva l'engine (il kernel `attend_mma` è fragile).
- La strumentazione host è esaurita (tutti i kernel sono accodati prima della sync): il prossimo passo
  è `compute-sanitizer`/`nsys` o un contatore di progresso dentro i kernel per individuare quale
  (`mla_prepare`/`indexer`/`mla_attend`) pende.

## Round 4 — Qwen >262k: preset int8 (validazione in corso)

- `qwen-yarn-524` usava `--kv fp16` (~15 GiB di KV a 524k) e scavalcava l'`--kv int8` dello start:
  il RAM guard fermava le richieste >262k ("other programs took ~12.8 GiB").
- Corretto a **`--kv int8`** (~7.6 GiB): commit `390d703`. Validazione in corso: server 524k int8
  avviato (ram-experts 16.2 GB, pool 14), prompt 524k; atteso il completamento senza RAM critical.
- Hy4 attention: il kernel `attend_mma` ha `n`/`__syncthreads` uniformi (nessuna divergenza); il 16k
  (8 step di attention vs 4 dell'8k) alza la probabilità di una race. Prossimo: `nsys`/sanitizer o un
  contatore di progresso kernel per isolare `mla_prepare`/`indexer`/`mla_attend`.

## Round 5 — Qwen 524k int8: validato

- `run.sh --preset qwen-yarn-524` (int8): `ctx=524288 kv=int8`, KV `0.7/7.8 GiB` (vs ~15 GiB fp16).
- Prompt **480,052 token**: **HTTP 200**, prefill **270,7 s (1774 t/s)**, TTFT 270,8 s, 1 token
  (`end=length`); RAM available 17 GiB dopo, **nessun** RAM critical.
- Il limite RAM >262k è superato: il fix `--kv int8` è verificato end-to-end.

## Round 6 — l'hang è l'indexer della QSA

- Sonde con sync dopo ogni fase dell'attention (`mla_prepare` / `indexer` / `mla_attend`): lo stall è
  ora a **`attn_prepare l0 pos 497664` senza `attn_indexer`** → il blocco è nell'**indexer** (i kernel
  della QSA indexer: `gemm` / `idx_pool` / `idx_scores_rows` / `select`), non in `mla_prepare` né in
  `mla_attend`.
- Le sonde (sync) spostano il punto di hang ma non lo eliminano: la posizione resta non deterministica
  (chunk 24/30/31), sempre dentro l'indexer.
- Prossimo: sonde più fini dentro `indexer()` (sync+log dopo `idx_pool`, `idx_scores_rows`, `select`)
  per isolare il singolo kernel.

## Round 7/8 — indexer: il blocco è scores/select (fix difensivo in prova)

- Sonde `idx_pool` / `idx_sel`: lo stall arriva **dopo `idx_pool`** e prima di `idx_sel` → il blocco è
  la seconda parte dell'indexer (`idx_scores_rows` / `idx_select_rows`), non il compressore.
- Fix difensivo nel `select_kernel`: inizializzazione di `sh_prefix`/`sh_rem` prima dei pass (evita
  shared stale se un pass non trova bin) e guardia `b > 0` nel loop del prefisso (evita l'indice -1).
- Sonda `idx_scores` (sync dopo gli scores) aggiunta: il run con il fix dirà se il blocco è
  `scores_mma_kernel` o `select_kernel`.
