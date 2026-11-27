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

## Round 10 — ring: 341,7 s di streaming esperti

- `t_ring_ms_` **341,7 s** a 620k (8k, 1 slot); ~6,4 TB di esperti (512 × 41 layer × 78 chunk × 4 MB)
  → **~18,7 GB/s** effettivi. Il `ring_copy` usa già `cudaMemcpy2DAsync` per i run contigui (`gap` =
  blob), quindi il collo è il bus/DMA, non l'issue delle copie.
- Seconda card **0 righe**, pool CPU **0**: tutto passa dalla card e il ring è sul percorso critico
  (la `st` attende `ring_copied_[slot]`).
- Leve: esperti residenti (slot HBM), seconda card, layout pack, pool CPU; l'8k non cambia il plateau
  di streaming.

## Round 13 — guardia cells: negativa

- Guardia `if (slot < k)` sull'output del `select_kernel` compilata e provata: lo stall **persiste**
  (`idx_scores l41 pos 487424`, nessun `idx_sel`). La scrittura OOB di `cells` **non** è la causa.
- Prossimo: `cuda-gdb` attach sul processo bloccato (`info cuda kernels`) o
  `compute-sanitizer --tool synccheck` per il deadlock del `select_kernel`.

## Round 14 — il cap a 16k è la regressione: lo stall è il piano di memoria

Il run 620k con `nsys` **completa** (600.022 token, ~348 s contro 1.222,7 s a 8k): il premio è reale,
ma il percorso 16k si blocca solo fuori dal profiler — quindi è **dipendente dal timing**, non un
evento mai registrato.

**Cosa è stato escluso, misurato:**

- **Doppio buffer nello staging del `put_i32`** (eventi per metà buffer): lo stall resta
  (`pp=442368/600022`, 676 s) e senza sonde (`pp=442368`). Non è la race host/GPU dello staging.
- **Ring degli esperti disattivato** (gate temporaneo `BASALT_NO_RING=1`, rimosso): lo stall resta e
  cade di nuovo su **chunk 26 (`pp=425984`)**, dopo aver superato i chunk 16–25. Il ring **non** è la
  causa: `ring_ready` fa join del thread prima di attendere `ring_copied_`, quindi la catena di eventi
  è ordinata.
- `cuda-gdb` attach (ptrace_scope=0): host fermo in `cudaStreamSynchronize`, `info cuda kernels` =
  **No CUDA kernels** → GPU inattiva, quindi sulla `stream_` c'è un'operazione non-kernel pendente.
- `compute-sanitizer --tool synccheck`: 0 errori (non è un deadlock di `__syncthreads`).

**Causa individuata (diff contro il baseline):** il cap sperimentale

```cpp
const int cap = m_.context() > 262144 ? 16384 : (total > 131072 ? 16384 : kMaxChunk);
```

abbassa `want` a **16384 prima della sonda `partial_lend`**. La sonda chiede meno slot, il lend
parziale **riesce**, e i chunk da 16k girano con la cache esperti elastica già piena. Gli stall
cadono tutti dove quella cache si esaurisce: **393.216 = 24 × 16.384** (poi 26/29/30/31, sempre
multipli di 16.384, sempre a inizio layer / `moe_start` del primo layer del chunk). Con il baseline
`want = 32768`, la sonda chiede di più, il lend fallisce, si passa a `lend_all()` e il chunk scende a
8k: è la configurazione con cui i 620k sono stati validati.

**Ripristino (working tree `basalt-port/`):** rimossi cap, doppio buffer di staging ed eventi, gate
`BASALT_NO_RING`; `kMaxChunk`/`kLaunchRows` = 32768. Resta solo il percorso FP4 prompt dietro
`BASALT_FP4_PROMPT` (patch 0003).

**Numero strano da segnalare:** il 33k su questo build fa **43,8 s (690 t/s, chunk 16.384)**; il
24,8 s registrato prima non è riproducibile qui, quindi quel dato veniva da un albero/binario diverso
(probabilmente pre-patch 0004 o con chunk 32k disponibili). Da rimisurare sull'albero ripristinato
insieme a 222k/620k prima di aggiornare la tabella delle prestazioni.

## Round 15 — 32k chunk: RAM critical a ~164k, e un errore di checkpoint separato

Con `kMaxChunk/kLaunchRows = 32768` (il resto ripristinato) il 620k **non** arriva in fondo, per due
motivi distinti dal cap:

1. **RAM critical a metà prefill**: a `pp=163840` il motore logga
   `RAM critical: 1017 MiB free, hard floor 1024 MiB` (poi 963 MiB). Il chunk da 32k chiede un
   footprint KV/esperti che su questa macchina (85,1 GiB di esperti in VRAM, 113 GiB di RAM) sfonda il
   floor: è coerente con la scelta validata degli **8192 righe**, che è quella che arriva a 600k.
2. **`a checkpoint past the session's position`**, non deterministico: un run si è fermato con
   HTTP 500 a `pp=65536` al confine di una parte (`part_checkpoint(to)` vero per
   `to == turn_at|root_at|media_at`), un altro ha superato 65536 e 163840 senza errore. Il
   salvataggio chiede `pos` > `hist_.size()` in `running_state()`: va strumentato `checkpoint_at(to)`
   con `to` e `hist_.size()` e un timeout di richiesta più lungo, perché serve arrivare al confine di
   parte che lo scatena.

**Stato del deliverable:** la configurazione validata (8192 righe) resta l'unica che porta i 620k in
fondo; il percorso 16k/32k è una regressione nota e revertita. Il PR Qwen #1 (kv int8, 480.052 token,
270,7 s) resta il risultato validato.

## Round 16 — correzione: lo stall NON era il cap; due bug del resume trovati e uno corretto

**Correzione al round 14.** Ripristinato *tutto* il baseline (cap, `put_i32` a singolo buffer, gate ring
rimosso, `kMaxChunk/kLaunchRows` = 32768), il 620k **si blocca ancora**: `pp=360448/600022` dopo 643 s a
618 t/s. Quindi il cap a 16k cambiava *dove* si blocca (prima), non *se*: lo stall è intermittente anche
sul percorso validato. Il numero "620k = 1222,7 s" non è riproducibile su questo albero e va considerato
un run fortunato, non una baseline stabile.

**Copia pageable trovata e rimossa.** `place_rope()` caricava la tabella rotary da
`std::vector<int32_t> tab` — memoria **pageable** — con `cudaMemcpyAsync` sulla stream del chunk, subito
seguita da `cudaStreamSynchronize`: è esattamente "host fermo in una stream sync, nessun kernel in
esecuzione" e gira a ogni confine di chunk, prima del layer 0. Ora la tabella va in un buffer **mappato**
(`cudaHostAlloc(..., cudaHostAllocMapped)`, 3 × kMaxChunk interi) e la copia è il kernel
`copy_i32x3_from_mapped` già usato da `put_i32`: nessun copy engine, nessuno staging del driver.
Da validare sul 620k (i run successivi si sono fermati prima, per il bug qui sotto).

**Bug del resume #1 (corretto).** `checkpoint_mount()` ripristinava lo stato di running ma **non la
storia dei token**: `hist_` restava vuoto, così il primo checkpoint salvato dopo un mount falliva con
"a checkpoint past the session's position" e un decode successivo sarebbe stato rifiutato con "position
P does not follow the state's 0". Ora chiama `set_history(c.ids)`, come già faceva il percorso delle
conversazioni parcheggiate (`restore_image`).

**Bug del resume #2 (localizzato, non corretto).** Dopo un mount a fine prompt il lettore parte con
`at` = posizione del checkpoint mentre la sessione ha un'altra storia: misurato
`hy4flash: position 600021 does not follow the state's 65536` (HTTP 500 in 97 s, dopo 2 chunk letti a
682 t/s). È la stessa classe del #1 ma sul bookkeeping `at`/`resume` invece che su `hist_`: da
allineare (la posizione di ripresa deve essere quella che la sessione ha davvero, o la sessione va
riportata a quella posizione).

**Guardia aggiunta.** Un read interrotto (cancel) non salva più un checkpoint oltre le righe lette:
`rows_read` registra l'ultima riga riportata dal progress e `read_to` esce con `Flow::quit` se
`rows_read < to`. Il 33k resta HTTP 200 in 43,7 s (nessuna regressione sul percorso normale).

**Stato:** il 620k non è ancora validato; il 16k/32k non è raggiungibile su questa macchina (RAM floor a
~164k con chunk 32k). Restano da fare: allineare `at`/`resume` (#2), poi rimisurare 33k/222k/620k e il
quality gate.

## Round 17 — lo stall è memoria, non una race: 117/121 GiB occupati a 360k

Riprodotto due volte **esattamente** a `pp=360448/600022`, una con KV fp16 (643 s) e una con
`--kv fp8` (644 s, 622 t/s): stesso punto, quindi non dipende dal tipo di KV. Al momento dello stall:

- **RAM host: 117 GiB usati su 121, 3 GiB liberi** (`free -g`), 0 in swap;
- host fermo in `cudaStreamSynchronize`, `info cuda kernels` = **No CUDA kernels**;
- il RAM guard del motore non è ancora scattato (floor 1024 MiB), quindi non è un `halted()`.

Il quadro corretto è quindi: il prefill da 600k token **esaurisce la memoria host** attorno ai 360k
token; con la memoria al limite il percorso host (page cache, compaction, copie zero-copy) si blocca e
la `cudaStreamSynchronize` non torna più. Non è un deadlock di kernel né una race di staging: le sonde,
il doppio buffer, il ring e il tipo di KV sono tutti esclusi da misure.

**Ricadute pratiche:**

- `--kv fp8` non basta a far entrare 600k posizioni: serve ridurre il footprint (posizioni backed,
  dimensione del contesto pianificato, o la lunghezza del prompt).
- Con RAM critica il motore marcava la richiesta come `halted` a metà lettura e poi proseguiva al decode
  alla posizione finale, producendo l'errore fuorviante "position P does not follow the state's N".
  Corretto: un read interrotto ora termina la richiesta come `cancelled` (niente decode su un prompt che
  non c'è), e `checkpoint_mount` ripristina la storia dei token (`set_history`).
- 33k resta sano (HTTP 200, 43,7 s) sia prima sia dopo queste modifiche.

**Da fare:** con il footprint ridotto, rimisurare 33k/222k/620k e fare il quality gate; oppure accettare
che su questa macchina il tetto pratico è ~360k token per richiesta.

## Round 18 — il tetto è il tier esperti: reserve e KV si contendono la stessa memoria

Il motore stesso, con `--vram-reserve-mib 60000`, dice:
`the latent caches cannot grow to 606208 positions: the VRAM reserve of 60000 MiB stays free; a smaller
--max-context leaves more for the expert tier` (HTTP 500 in **1,1 s**, rifiuto pulito). Quindi il
fallimento a metà prompt è la **crescita del KV che non trova memoria**, non un deadlock CUDA: quando
riesce a valutarlo all'inizio il motore rifiuta in un secondo, quando la crescita avviene a metà prompt
resta appeso (host in stream sync, GPU idle) invece di riportare lo stesso rifiuto.

**Il conflitto di configurazione (misurato):**

| reserve | esperti in VRAM | KV | esito |
|---|---|---|---|
| 700 MiB (default) | 512/512, 77,84 GiB | cresce fino a ~360k | prefill a 360448, poi appeso (117/121 GiB usati) |
| 60000 MiB | 37,2 GB spostati in RAM | non può crescere fino a 606208 | rifiuto pulito in 1,1 s |

`--vram-reserve-mib` fa due cose insieme: libera VRAM dagli esperti **e** la sottrae al KV. Su una
macchina a pool unificato da 121 GiB servirebbe "sposta gli esperti in RAM e lascia quella memoria al
KV", che oggi non è esprimibile: o il KV non cresce, o gli esperti restano wired e il KV non entra.
Il tetto pratico misurato resta **~360k token per richiesta**.

**Prossimo passo (concreto):** separare i due effetti nel tier (dare al KV la memoria liberata dagli
esperti senza contarla come reserve) e, in ogni caso, far sì che una crescita KV fallita a metà prompt
ritorni `Refused` come all'inizio, invece di restare appesa: è la differenza fra un errore in 1 s e uno
stall di 643 s.

## Round 19 — il pre-check della crescita non basta: il tetto è nel reclaim del tier

Provato a trasformare lo stall in un rifiuto pulito con un pre-check di `kv_ceiling()` prima di
`ensure_positions()` in `run_rows_impl`. **Non scatta**: il 620k resta appeso a `pp=360448` per 644 s
(503 dal server), perché `kv_ceiling()` conta come disponibili gli slot del tier esperti
(`tier_->slots() * stride_`), mentre è proprio il loro reclaim che si blocca. Il pre-check è stato
rimosso: non fa quello che dichiara.

Conclusione del filone: il blocco non è nel percorso CUDA del modello (sonde, staging, ring, tipo di KV
e sync esclusi da misure) ma nella **crescita del KV sotto pressione di memoria**, dove il reclaim del
tier esperti non torna. Il tetto misurato su questa macchina resta ~360k token per richiesta.

Le due strade rimaste, fuori dal budget di questo ciclo:
1. dare al KV la memoria liberata dagli esperti senza conteggiarla come reserve (separare i due effetti
   di `--vram-reserve-mib`), così 512 slot → 269 slot liberano ~37 GiB per il KV;
2. rendere bounded il reclaim del tier (`ExpertTier::reclaim` / `cache_.grow`) con un timeout o un
   conteggio dei tentativi, e mappare il fallimento su `Refused`.

## Round 20 — il punto esatto (traccia per layer) e il tetto a ~300k

Traccia env-gated (`BASALT_TRACE=1`, rimossa dopo la diagnosi) su chunk, layer, passo e fase: il blocco
cade in **`attn 0` → fase `cells`** del primo layer del chunk, cioè nella scrittura delle righe di cache
e nel `put_i32(meta_d_, meta, 4*arows_)` che porta `cudaStreamSynchronize` sul percorso del prompt.
Provato a togliere quello staging (copia diretta dal buffer pinned per-layer/per-step, `meta_h_`):
**nessun guadagno misurabile** (605–610 t/s contro 605–622) e il tetto resta — modifica ritirata.

Misurato che il tetto **non dipende dalla lunghezza del prompt**: un prompt da 400k muore a ~295k token
come il 620k. Il processo termina senza messaggio ("the engine ended (no exit status), nothing in its
log") con il pool a ~117/121 GiB: è la fine della memoria, non un errore CUDA riportabile.

**Patch e preset aggiornati a questo stato:**

- `patches/0001-qwen-fp16-batch.patch` — sorgenti Qwen (rigenerata dal working tree);
- `patches/0003-hy4-fp4-prompt.patch` — percorso FP4 di Hy4 (`session.cpp`, `indexer.cu`, `moe_fp4`);
- `patches/0004-qsa-seg-parity-int8.patch` — test di parità (rigenerata);
- **nuova** `patches/0005-hy4-serve-prompt-fixes.patch` — `checkpoint_mount` che ripristina la storia dei
  token, read interrotto che chiude la richiesta come `cancelled`, upload della tabella rotary da memoria
  mappata. Tutte e quattro verificate: applicate in ordine a un worktree pulito al commit pinnato
  riproducono il working tree (diff vuoto su `src/`, `include/`, `tests/`), e `git apply --reverse --check`
  conferma che corrispondono allo stato attuale.
- `presets/hy4-fp4-1m.conf` — corretto il dato irreperibile "620k = 1222,7 s", aggiunti i numeri misurati
  (33k: 45,7 s) e il tetto di memoria; `--kv fp8` documentato come opzione per prompt più lunghi.
- `install-basalt.sh` — commento dei passi aggiornato con la 0005.

**Stato finale:** 33k HTTP 200 in 45,7 s con tutte le patch applicate; >262k non raggiungibile su questa
macchina per memoria. Qwen PR #1 resta il deliverable validato.

## Round 21 — migrazione al pin c771c0a7 e patch rigenerate

Il port era rimasto a `c70e8011` e `install-basalt.sh` non poteva fare il checkout del pin con le
modifiche locali in `session.cpp`/`session_engine.cpp`. Le patch sono state riportate sul pin con un
merge a tre vie e rigenerate **contro `c771c0a7`**:

- `0001` e `0004` (Qwen) applicano invariate;
- `0003` (Hy4 FP4 prompt) assorbe anche l'upload mappato della tabella rotary; le mie costanti
  `kMaxChunk`/`kLaunchRows` sono sparite dal diff perché **#695 le ha già upstream** (32k fino a 128k,
  8k sopra 262k per slot);
- `0005` (serve) si riduce a due fix: `checkpoint_mount` che ripristina la storia dei token e una
  lettura interrotta che chiude la richiesta come `cancelled`. La mia guardia `rows_read` era
  ridondante: il pin fa avanzare `at` solo se non è stato fermato (`if (!runtime_.halted()) at = end`).

Verifica: le quattro patch applicate in ordine a un worktree pulito al pin riproducono il working tree
(diff vuoto su `src/`, `include/`, `tests/`); `./install-basalt.sh` completa gli 8 passi, costruisce il
binario per `c771c0a7` e passa lo smoke test. 33k sul build installato: **HTTP 200 in 46,0 s (656 t/s)**.
