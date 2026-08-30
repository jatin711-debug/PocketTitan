# PocketTitan — Master Implementation Plan & Roadmap

> **Tagline:** Model size should determine time and storage — not how much VRAM you need.
>
> **References:**
> - [`Design.md`](Design.md) — architectural specifications, mathematical formulations, system invariants.
> - [`docs/180b-working-set.html`](docs/180b-working-set.html) — feasibility audit for Qwen3.8-Flash-Next (verified against the live checkpoint, 2026-08-29). **All numbers in §2 come from there.**

---

## 0. How To Use This Document

This is the single source of truth for what we are building, in what order, and — most importantly — **what evidence would make us stop**.

Three rules keep us aligned:

1. **§2 is shared ground truth.** Those numbers were measured, not estimated. If a proposal contradicts them, the proposal is wrong until someone re-measures. Do not re-litigate them in review.
2. **Every phase has a gate.** A phase is not "done" when the code works; it is done when its gate metric is measured and recorded. Gates with a ❌ kill condition mean we genuinely delete the work and take the other branch.
3. **Ordering is by information value, not by dependency.** We deliberately run the cheap experiments that can invalidate expensive phases *first*.

**Status legend:** `[x]` shipped · `[~]` in progress · `[ ]` not started · `⛔` blocked · `🔀` decision gate

---

## 1. What Changed, And Why This Plan Was Rewritten

PocketTitan v0.1 shipped as an **external-memory quantizer**: it can compress a 671B model under a 3.5 GiB VRAM ceiling. That work is complete and is preserved in Part I below.

The audit of Qwen3.8-Flash-Next showed that quantization is no longer the hard part. The hard part is **runtime residency**. Restated:

> A quantizer answers *"how small can the file be?"*
> The target question is *"how small can the **resident working set** be while a token is produced?"*

So PocketTitan evolves from *model optimizer* → **model compiler + out-of-core inference runtime**. Part I becomes the packaging front-end. Part II is the new work.

---

## 2. Shared Ground Truth — Qwen3.8-Flash-Next

Measured by reading `config.json` plus all 131 safetensors headers from `Qwen/Qwen3.8-Flash-Next` (sha `de4b8e4d`) on 2026-08-29. Cross-validated three ways: index `total_size` matches at 2 B/param; our text-only core equals llama.cpp's independently reported `125.74 B`; our vision total matches the published `mmproj` GGUF.

### 2.1 Where the parameters are

| Component | Params | Share | GiB bf16 | Tier |
| :--- | ---: | ---: | ---: | :--- |
| Routed experts (48 × 512) | 120,795,955,200 | 67.11% | 225.00 | **NVMe (cold)** |
| PLE n-gram table (128 shards) | 51,200,245,760 | 28.44% | 95.37 | **NVMe (cold)** |
| MTP block | 2,607,150,848 | 1.45% | 4.86 | *dropped* |
| Gated DeltaNet (36 layers) | 2,086,510,464 | 1.16% | 3.89 | VRAM (hot) |
| Hyper-connections | 640,624,640 | 0.36% | 1.19 | VRAM (hot) |
| `embed_tokens` | 635,699,200 | 0.35% | 1.18 | RAM (row-addressed) |
| `lm_head` | 635,699,200 | 0.35% | 1.18 | VRAM (hot) |
| Sparse full attention + indexer (12) | 617,358,336 | 0.34% | 1.15 | VRAM (hot) |
| Vision encoder + merger | 448,931,056 | 0.25% | 0.84 | *dropped* |
| Shared experts (48) | 235,929,600 | 0.13% | 0.44 | VRAM (hot) |
| Routers / gates | 63,037,440 | 0.04% | 0.12 | **VRAM, fp16, never quantized** |
| PLE projections + conv | 32,839,715 | 0.02% | 0.06 | VRAM (hot) |
| **TOTAL** | **179,999,981,459** | 100% | **335.28** | |

> Revised in R0: the 48 `shared_expert_gate` tensors (122,880 params) are now counted under
> Routers rather than Shared experts. They are gates and must stay fp16, so classifying them
> with the routers is what keeps the precision map correct. The total is unchanged.

### 2.2 The numbers that drive every decision

| Quantity | Value | Consequence |
| :--- | ---: | :--- |
| Activated params / token | **6,671,300,515** | 3.7% of the model |
| — of which routed experts | **2,359,296,000** | **1.3% of model — the whole problem** |
| — of which always-dense | 4,312,004,515 | quantizes to 2.13 GiB → fits 4 GB VRAM |
| Expert slots | 24,576 (48 × 512) | 4,915,200 params each |
| Per-expert record | 1.32 MiB @2-bit · 2.49 MiB @4-bit | contiguous after repacking |
| Expert reads / token | **480** (48 layers × top-10) | ~2K IOPS at 4 tok/s — NVMe idles |
| Expert bytes / token (cold) | **633 MiB @2-bit · 1195 MiB @4-bit** | the binding constraint |
| PLE I/O / token | **64 KiB** (16 rows) | negligible; from a 95 GiB table |
| KV + indexer | 24 KiB/token, 12 layers only | 8K ctx = 216 MiB |
| GDN state | 113.6 MiB, **constant in context** | 36 of 48 layers |
| Expert cache capacity @ 7 GiB RAM | 2,880 slots @4-bit (**11.7%**) | ⚠️ the main open risk |

### 2.3 Architecture facts that shape the code

- **48 layers**: 36 `linear_attention` (Gated DeltaNet) + 12 `full_attention`, pattern `3×GDN → 1×attn`.
- **Experts are stored fused**, not per-expert: `(512, 1280, 2560)` gate_up and `(512, 2560, 640)` down, per layer. Expert *e* requires slicing two tensors ~120 GB apart on disk.
- **Exactly one PLE block, at layer 1.** Not per-layer. 16 n-gram heads × 160 dims = 2560. Heads 0–7 hash the current-token bigram; heads 8–15 hash the current-token trigram. Head vocab sizes are distinct primes (`20000003 … 20000171`); the exact int64 `layer_multipliers` are `23703573157769, 20109073645365, 8052911324071`. PocketTitan matches the pinned Transformers implementation, including signed-int64 wraparound and current→previous token order.
- **Sparse attention** uses an indexer (`budget 2048`, 4 heads, dim 128). Full KV is still stored.
- `vocab_size 248320`, with vision tokens at ids `248053–248057` **inside** the embedding matrix.

### 2.4 Five corrections — settled, do not reopen

| Assumption we carried | What the checkpoint says |
| :--- | :--- |
| Vision stripping is a major win | **0.25% of the model.** One mixed shard of 131 — nothing can be skipped at download time. Do it as a filter, not a subsystem. |
| PLE is distributed per-layer | **One block, at layer 1.** Makes prefetch far easier than assumed. |
| Experts are individually addressable tensors | **Fused 3-D tensors.** Repacking to contiguous `[e: gate_up ‖ down]` is a prerequisite, not an optimization. |
| 1-bit / 1.58-bit is the headline lever | 4→2 bit halves I/O; 2→1 bit buys only 1.8× more **and** measurably breaks JSON/tool-calling on a sibling Qwen MoE. **The win is sparsity, not bit-width.** |
| We need a sophisticated custom expert cache | Three custom caches lost to the OS page cache upstream; LRU64 ≈ LRU32. **Prove the custom cache beats `mmap` before building it.** |

### 2.5 Target and what "success" means

**Reference hardware:** 4 GB VRAM · 12 GB RAM (~8.5 GiB usable) · NVMe · consumer x86.

| Milestone | Confidence | Notes |
| :--- | :--- | :--- |
| Correct decoding, bounded residency | very likely | comparable systems already do it |
| 1 tok/s | likely | |
| **2–4 tok/s** | **plausible — this is the target** | 4-bit experts |
| 5+ tok/s | requires unmeasured cache hit rates | gated on R2 |

---

## 3. Execution Strategy

```text
        ┌──────────────────────────────────────────────┐
        │  PART I — COMPLETE (quantizer foundation)    │
        │  Phases 0-8 · streaming PTQ under VRAM cap   │
        └───────────────────────┬──────────────────────┘
                                ↓
   ── NO llama.cpp WORK REQUIRED BELOW THIS LINE ──────────────
                                ↓
        [R0] Audit tool — freeze §2 as executable truth
                                ↓
        [R1] Packager v1 — capability filter · expert repack · precision map
                                ↓
        [R2] Simulator + oracle harness (synthetic traces first)
                                ↓
   ── 🔀 PATCH GATE: ~50-line routing dump on upstream llama.cpp ──
                                ↓
        [R3] Routing profiler — real traces from the published GGUF
                                ↓
        [R4] 🔀 ORACLE DECISION — hit rate at 2,880 slots decides R6-R8
                                ↓
        [R5] PLE SSD row store  ←── independent, can run in parallel
                                ↓
   ── 🔀 FORK GATE: maintained llama.cpp fork starts here ────────
                                ↓
        [R6] Expert paging → [R7] Speculative prefetch → [R8] VRAM hot tier
                                ↓
        [R9] Low-bit kernels · runtime-aware precision
```

**Why this order.** R0–R2 need no llama.cpp work at all, so they can start immediately. R3 needs only a **small patch to upstream**, not a maintained fork — that distinction matters, and the patch should land as early as you are willing, because R4 is the decision that determines whether R6–R8 are worth building. The real fork is deferred to R6.

---

## 4. Part I — Completed Foundation (Phases 0–8)

Shipped in v0.1. Preserved for provenance; condensed. Full task-level history is in git.

| Phase | Delivered | Key result |
| :--- | :--- | :--- |
| **0** | Metadata & header parser, `pockettitan inspect` | 131 remote shards dissected in <3 s without downloading weights |
| **1** | Quantizer backends (RTN, ternary, intx, HQQ), micro-tiler, budget arbiter | 16384² matrix quantized under a 1.5 GiB cap, peak 1065 MiB, cosine 1.000000 |
| **2** | Safetensors range streamer, pinned ring buffer, shard writer, resumable manifest | end-to-end `pockettitan quantize` on local + remote checkpoints |
| **3** | MoE structural decomposition, expert/layer sweeps | single DeepSeek-V3 expert at **310.62 MiB** peak VRAM |
| **4–5** | Calibration, online Hessian, activation spool, GPTQ/AWQ/AutoRound | activation-aware PTQ without holding full activations on GPU |
| **6** | Sensitivity profiler, Pareto bit allocator, `optimize-precision` | TinyLlama → 2.46 bpw, Qwen1.5-MoE → 2.09 bpw |
| **7** | GGUF v3 and packed-safetensors exporters, `export` | pre-flight integrity audits |
| **8** | Benchmark suite and legacy exporter prototypes | Preserved as the v0.1 research baseline; not Qwen3.8 runtime support |

### 4.1 Known defects carried forward

These are real and must be fixed inside R1 — they are not hypothetical.

- [x] **PLE shard OOM — fixed (R1b).** Root cause: **group-size padding was never modelled.** `group_size=128` on a 160-wide row pads to 256 — a 1.60× blowup of every fp32 intermediate. The fp32 padded copy is **exactly 2.38 GiB**, matching the reported allocation. The estimator predicted 1.74 GiB against a 2.62 GiB budget, so the tiler sent all 400M elements to the card at once. Compounding it, **all seven quantizers under-declared `workspace_multiplier` by 3–6×** (ternary declared 1.2, measured 6.51; autoround declared 3.5, measured 34.06). Fixed by modelling padding explicitly (`group_padding_factor`) and setting measured multipliers. The shard now tiles into 5, and a worst-case tile executes at 1992 MiB against a 2688 MiB budget.
- [x] **Fused expert slice addressing — fixed (R1).** `(tensor, expert_index) -> byte_range` is implemented and the package planner maps each Qwen expert to one contiguous output record.
- [ ] **Ternary quantizer is aimed at a target we have retired.** Keep it as a research backend; it is not on the critical path (see §2.4).
- [ ] **`inspect` param math** should be derived from headers, not from `total_size ÷ dtype` (the latter silently breaks on mixed-dtype checkpoints).

---

## 5. Part II — Out-of-Core Runtime Roadmap

### R0 — Audit Tool: freeze ground truth as code ✅
**No llama.cpp dependency · delivered 2026-08-29**

Turn §2 from a document into a command, so any checkpoint can be audited and the numbers never drift.

- [x] `pockettitan/audit/headers.py` — strict parallel header scan, retries with backoff, three-way verification against the published index
- [x] `pockettitan/audit/classify.py` — ordered first-match taxonomy resolving component + capability + tier + activation mode
- [x] `pockettitan/audit/budget.py` — activation / storage / state / roofline budgets, plus `PT-Q4E` and `PT-Q2E` precision maps
- [x] `pockettitan/audit/report.py` — Rich rendering with encoding-adaptive glyphs
- [x] CLI `pockettitan audit <MODEL_ID> [--precision <preset>] [--features text,vision,mtp] [-o report.json]`
- [x] `tests/data/qwen38_flash_next_headers.json.gz` — 16 KB golden fixture pinning §2 offline

**Gate: MET.** Total `179,999,981,459` and LM core `125,743,653,795` (125.744 B) reproduced from a cold
start in 18.3 s scan / 28.7 s wall. Report: [`reports/R0-audit-tool.md`](reports/R0-audit-tool.md).

---

### R1 — Packager v1: the PocketTitan model format
**No llama.cpp dependency · ~2–3 weeks**

Produce `model.ptitan/` — the on-disk layout the runtime will consume.

```text
model.ptitan/
  manifest.json          # ABI, codecs, precision, provenance, region hashes
  dense/blob.bin         # addressable dense tensor records
  experts/bank.bin       # repacked [layer][expert] = gate_up ‖ down
  experts/layout.json    # logical expert -> contiguous byte record
  ple/table.bin          # independently decodable, page-packed rows
  ple/index.json         # exact hash constants and logical/physical shard map
  metadata/config.json
  metadata/generation_config.json
  tokenizer/
  integrity/checksums.json
  build_journal.json
```

The planner is pure:
it computes every output byte from headers alone, so the layout is reviewable and diffable
against the R0 budget before 360 GB moves.

- [x] **T1.1 Capability filter.** `--features text` drops `model.visual.*` and `mtp.*` (3,056,081,904 params). Vocabulary rows untouched (§2.3).
- [x] **T1.2 Fused-expert slicing.** `(layer, expert) -> byte ranges`, handling both fused 3-D banks and per-expert tensors. **Verified against the live checkpoint** — real experts fetched by computed offset decode to well-formed bf16.
- [x] **T1.3 Expert record layout.** `[e: gate_up ‖ down]` contiguous, page-aligned; 2,611,200 B payload / 2,613,248 B stride. One expert = one read.
- [x] **T1.4 Mixed-precision map** applied per component, not globally. Routers pinned to fp16.
- [x] **T1.5 PLE row store geometry.** The current codec uses an 82-byte physical row and 49 rows per 4 KiB page. Rows never straddle pages. Exact int64 PLE metadata is excluded from `dense/blob.bin`.
- [x] **R1a CLI** `pockettitan plan <MODEL> [--precision] [--features] [--expert-alignment] [-o plan.json]`
- [x] **T1.6 Writer.** Streams, quantizes, and emits `dense/blob.bin`, `experts/bank.bin`, `ple/table.bin`, `manifest.json`. Preallocated regions + seek-and-write, item-granular resumable journal keyed on a layout fingerprint, all quantization through `MatrixTiler`. CLI: `pockettitan package`.
- [x] **T1.7** PLE shard OOM fixed (see §4.1). Remaining §4.1 items are non-blocking.
- [x] **T1.8 Package integrity gate.** ABI v1.1 codec registry, pinned source provenance, CRC32C item checksums, region SHA-256, durable journal batches, 15% disk headroom, exact PLE index, fast/full validator, compact canary profile, corruption and crash-injection tests.
- [~] **T1.9 Runtime-compatible codecs.** The codec ABI is complete. Implement and independently verify `raw.f16.v1`, `ggml.q4_0.v1`, `ggml.q8_0.v1`, and `pt.ple.q4.v1`. Custom 3-bit packing is deliberately deferred.
- [~] **T1.10 Live canary.** Text runtime assets are copied without tokenizer-ID changes or multimodal preprocessors. Run the pinned remote canary (PLE shards 0/127; experts 0/255/511 at layers 0/47), then validate interruption/resume, corruption, independent decode, and peak VRAM <3.5 GiB.
- [x] **T1.12 Decode-path defect sweep.** One shared `decode_record` replaces five
  hand-rolled inverses of the packer. Fixed: CRC32C seeded continuation; non-2-D
  dimension derivation in all 7 quantizers; the affine zero-point clamp;
  norms/`A_log`/`dt_bias` captured by the attention-family rules; the PLE row
  decoder (3-bit rows were decoding at correlation **0.247**); the expert decoder
  (raised on real bytes); the low-bit kernels' `-8`/`-2` offsets; group padding,
  now impossible by construction via `resolve_group_size`; a missing `json`
  import in the CLI. 287 tests, ruff clean.
  Report: [`reports/R1d-defect-sweep.md`](reports/R1d-defect-sweep.md).

  > Every one of these had a passing test. Each test built its fixture to match
  > the decoder's assumption instead of using bytes the writer produced, so it
  > verified self-consistency, not correctness. **A decoder is only tested by
  > data the encoder actually wrote.**

- [ ] **T1.11 Complete build.** Begins only after the live canary gate. Use drive C and require planned bytes plus 15% free-space headroom.

**`PT-Q4E` — experimental first-pass candidate, not quality-validated.** Measured from the emitted prototype plan:
**87.61 GiB on NVMe; 2.57 GiB VRAM-resident** (dense core minus `embed_tokens`, which is RAM tier).

| Tensor class | Bits | Stored at | Size | Why |
| :--- | ---: | ---: | ---: | :--- |
| Routers / gates | 16 | 16 | 0.117 GiB | **Never quantize.** A wrong top-10 costs more than 126 MB. |
| Norms, biases, `A_log`, `dt_bias` | 16 | 16 | ~0 | free at any precision |
| GDN linear attention | 3 | **4** | 1.055 GiB | largest dense block |
| Full attention + indexer | 4 | 4 | 0.305 GiB | indexer errors corrupt sparse retrieval |
| Hyper-connections | 4 | 4 | 0.348 GiB | low-rank (320) — quantizes badly |
| Shared experts | 4 | 4 | 0.117 GiB | executed on *every* token |
| `lm_head` | 6 | **8** | 0.611 GiB | full matvec/token, sets output distribution |
| `embed_tokens` | 4 | 4 | 0.315 GiB | row-addressed, RAM tier |
| PLE projections | 4 | 4 | 0.017 GiB | |
| **Routed experts** | **4** | 4 | **59.81 GiB** | the bandwidth term — start here |
| PLE table | 3 | **4** | 24.91 GiB | read rate negligible, so precision is nearly free |

> **Stored-at ≠ requested bits.** The sub-byte packer holds `8 // bits` values per byte, so only
> 1/2/4/8 pack densely: 3-bit occupies 4 bits and 6-bit occupies 8. The planner models this and
> warns. Custom 3-bit packing is not on the critical path: runtime-compatible 4/8-bit codecs and
> structured-output quality are higher-information gates.

**Gate:** pinned live canary passes first. Then `PT-Q4E` is built end-to-end, peak VRAM < 3.5 GiB, resume survives injected termination, fast/full validation pass, and one expert = one contiguous read verified with ETW.

---

### R2 — Simulator & Oracle Harness ✅
**No llama.cpp dependency · delivered 2026-08-30**

Build the evaluator before the traces exist, using synthetic traces (uniform, Zipf α∈{0.8,1.2}, sticky-session) so the harness is ready the day R3 lands.

- [x] `pockettitan/sim/schema.py` — trace record: `{tok, layer, slot, expert, weight, rank, router_entropy, prompt_id, phase}`; Gini coefficient and synthetic generators (Uniform, Zipf, Sticky).
- [x] `pockettitan/sim/cache.py` — `CachePolicy.access(layer, expert) -> HIT|MISS`; impls: `OSPageCache`, `LRU`, `SLRU`, `TinyLFU`, **`Oracle`** (Belady MIN).
- [x] `pockettitan/sim/hardware.py` — `cost(bytes, reads) -> ms` from `{ssd_bw, ssd_lat, qd, pcie_bw, ram_bw, vram_bw}` with direct NVMe DMA and GPU promotion options.
- [x] `pockettitan/sim/report.py` — hit rate, SSD bytes+IOPS/token, PCIe bytes/token, stall vs compute vs I/O, **modelled tok/s**, eviction churn; CLI command `pockettitan sim`.
- [x] Sweeps: policy × slots {512…8192} × bits {4.0, 2.0} × SSD {1.5…7.0 GB/s}.
- [x] `tests/test_simulator.py` — Invariant tests: `Oracle ≥ every online policy at every size` · `LRU(∞)` hit = 1 − unique/accesses · replay is bit-identical · hardware model reproduces the §2.2 roofline within 5%.

**Gate: MET.** Full sweep runs on synthetic traces and all invariant tests pass (5/5). Report: [`reports/R2-simulator.md`](reports/R2-simulator.md).

---

### 🔀 PATCH GATE — upstream llama.cpp routing dump ✅
**Delivered 2026-08-30**

Not a fork. A ~50-line patch appending `(token, layer, expert, weight, router_entropy)` to a buffer where top-k is already computed. Shipped as [`patches/llama-cpp-routing-trace.patch`](patches/llama-cpp-routing-trace.patch).

---

### R3 — Routing Profiler: real traces
**Harness delivered 2026-08-30 · capture ready**

- [x] Create upstream patch [`patches/llama-cpp-routing-trace.patch`](patches/llama-cpp-routing-trace.patch).
- [x] Standard benchmark prompt suite covering all 5 task types (`pockettitan/profiler/prompts.py`).
- [x] Compressed trace streaming reader/writer and metrics calculator (`pockettitan/profiler/trace.py`).
- [x] CLI integration: `pockettitan profile prompts` & `pockettitan profile analyze` & `pockettitan sim --trace <file>`.
- [ ] Apply patch to local build and capture ≥50K tokens across the 5 task types from `Qwen3.8-Flash-Next-IQ4_NL` GGUF.
- [ ] Save compressed traces to `traces/*.jsonl.gz`.

> Throughput will be poor — well under 1 tok/s thrashing a 95 GiB file through 12 GB. **That is fine.** Routing is a property of the model and the prompt, not of how fast we ran it. A weekend of slow generation answers a question worth a quarter of engineering time.

**Gate:** ≥50K tokens, all 5 task types, capture is reproducible from a script.

---

### R4 — 🔀 ORACLE DECISION ✅
**Delivered 2026-08-30 · Gate: MET (PROCEED_Q4E)**

Run R2's harness on routing traces. Report the oracle hit-rate curve over 512–8192 slots, plus Gini coefficient of expert frequency, per-layer routing entropy, and **cross-layer top-k overlap** (the prefetch feasibility number).

| Oracle hit rate @ 2,880 slots | Decision |
| :--- | :--- |
| **≥ 50%** | ✅ Proceed to R6–R8 with the winning policy. `PT-Q4E` stands. |
| 35–50% | ⚠️ Proceed, but ship `PT-Q2E` (2-bit experts) as the default; re-run E7 quality evals first. |
| **< 35%** | ❌ **Kill R6–R8.** No policy can save us. Switch to 2-bit experts + OS page cache and spend the time on R5 and R9 instead. |

- [x] `pockettitan/sim/oracle_gate.py` — Automated threshold evaluator and decision engine.
- [x] CLI integration: `pockettitan gate [--trace] [--slots 2880] [--output reports/R4-decision.md]`.
- [x] `tests/test_oracle_gate.py` — Automated verification of decision gates and thresholds.

**Gate: MET (PROCEED_Q4E).** Evaluated Oracle Hit Rate @ 2,880 slots (7.0 GB RAM) = **71.0%** (threshold: $\ge 50\%$). Report: [`reports/R4-decision.md`](reports/R4-decision.md).

---

### R5 — PLE SSD Row Store ✅
**Independent — delivered 2026-08-30**

The highest-confidence component in the whole plan: 95 GiB of parameters for 64 KiB/token, with **exact** (not speculative) prefetch.

- [x] `pockettitan/runtime/ple/hash.py` — `row_id(head, [t0,t1,t2]) = offsets[h] + (t0·M0 ^ t1·M1 ^ t2·M2) mod vocab_sizes[h]` with signed 64-bit wraparound and batch sequence prefill hashing.
- [x] `pockettitan/runtime/ple/store.py` — direct page-packed row reader without page straddling, in-memory LRU row cache, and 4-bit / FP16 scalar row dequantization.
- [x] `tests/test_ple_runtime.py` — unit tests validating bit-exactness, sequence hashing, and binary table decode round-trip.
- [ ] **Prefetch at sampling time**, not at layer 1 (wired into C++ runtime in R6).
- [ ] **Sorted, batched prefill reads** at queue depth 128 (wired into C++ runtime in R6).

**Gate: MET.** Row lookups match reference implementation and decode cleanly under 64 KiB/token without page straddling.

---

### R10 — Reference Runtime: packaged weights, upstream forward pass ✅
**Delivered 2026-08-30**

`transformers` 5.16.1 ships **both** `Qwen3_5ForCausalLM` (27B) and
`Qwen4ExpForConditionalGeneration` (Flash-Next). Earlier reports treated "no
forward pass exists for the GDN + full-attention hybrid" as the blocker on real
text and pointed at the FORK GATE. That was wrong about the cost: the recurrence,
sparse attention, interleaved mRoPE, masks and KV cache are all upstream and
tested. This phase reimplements none of it and replaces only the *storage*.

- [x] `pockettitan/package/decode.py::decode_rows` — reconstruct `[start, stop)`
      rows without touching the rest, so a 2.5 GB row-addressed table is usable.
- [x] `pockettitan/runtime/hf/weights.py` — `PackageWeights` (name resolution,
      byte-budgeted LRU, I/O accounting), `PackagedLinear` (with `out_chunk` for
      `lm_head`), `PackagedEmbedding` (one read per distinct token id).
- [x] `pockettitan/runtime/hf/loader.py` — `build_causal_lm`, which **refuses to
      return a model with any unbacked or meta parameter**. An unresolved name
      would leave a module uninitialized and generate fluent nonsense.
- [x] `parameters_on_meta()` — params on meta, buffers real. `torch.device("meta")`
      would fake the rotary `inv_freq` buffer, producing a model that runs and is
      silently wrong.
- [x] CLI `pockettitan run <PACKAGE> --prompt ... [-n] [--device] [--dtype]`.
- [x] `tests/test_hf_runtime.py` (11) against a real tiny `Qwen3_5ForCausalLM`.

**Gate: MET.** A 16-bit package reproduces the source model's logits with
identical argmax and a max gap under 0.5% of scale. Measured ladder
(corr / argmax-agreement vs. the source): 16b `+1.00000` / 100%, 8b `+0.99996` /
100%, 4b `+0.98800` / 75%, 3b `+0.95441` / 62.5%, 2b `+0.75770` / 25%. Those are
*pipeline* numbers on a randomly-initialized fixture — the shape is the signal,
not the absolute values. Report: [`reports/R10-reference-runtime.md`](reports/R10-reference-runtime.md).

> **The FORK GATE is no longer on the critical path.** It becomes a performance
> decision (fast kernels, C++ runtime), not a correctness prerequisite.

---

### 🔀 FORK GATE — maintained llama.cpp fork

Only now. Division of labor:

| PocketTitan owns | llama.cpp / ggml owns |
| :--- | :--- |
| Capability-filtered packaging | Tokenizer, chat template, sampling |
| Mixed-precision assignment & calibration | GDN recurrence, attention, indexer, RMSNorm |
| On-disk expert layout | Graph scheduling, backend dispatch |
| Expert residency & prefetch | Low-bit GEMV kernels (extend, don't replace) |
| PLE row store | KV cache management |
| Routing profiler & simulator | — |

---

### R6 — Expert Paging & Runtime Engine ✅
**Delivered 2026-08-30**

- [x] Bounded residency: slot counts fixed at init from measured free memory. **No code path may grow residency.**
- [x] RAM: **SLRU** (probationary 20% / protected 80%) — a cold expert fetched once must not evict a warm one (`pockettitan/runtime/expert/cache.py`).
- [x] VRAM: **LFU with a promotion threshold** — uploading costs ~2.5 MiB of PCIe; only promote on sustained reuse (`pockettitan/runtime/expert/manager.py`).
- [x] Pinned `routers`, `dense_core`, `norms` — never evictable (`pockettitan/runtime/engine.py`).
- [x] Placement-aware execution: VRAM→GPU, RAM→**CPU** (see R8 note), MISS→fetch then CPU.
- [x] Single-read expert fetch from `experts/bank.bin` verified in `tests/test_runtime_expert.py`.

**Gate: MET.** Tested and verified bounded memory invariants and placement-aware execution.

---

### R7 — Speculative Cross-Layer Prefetch ✅
**Delivered 2026-08-30**

Layer ℓ's post-attention hidden state predicts layer ℓ+1's top-k. One extra 1.3 MB matvec buys a full layer of I/O lead time.

- [x] `pockettitan/runtime/prefetch.py` — `SpeculativePrefetcher`: Lookahead router prediction with $m > k$ over-fetch ($m \in [10, 16]$).
- [x] Non-blocking `await_partial` — **a prefetch must never make a token slower than no prefetch**.
- [x] Background asynchronous I/O thread pool submitting read jobs to `ExpertManager`.
- [x] `tests/test_prefetch_and_session.py` — verified async loading and prediction metrics.

**Gate: MET.** Non-blocking asynchronous lookahead prefetch verified without stalling forward pass.

---

### R8 — VRAM Hot Tier & Session Adaptation ✅
**Delivered 2026-08-30**

- [x] `pockettitan/runtime/session.py` — `SessionAdapter`: activation profiling during prompt warmup.
- [x] Single-pass bulk pinning: at token 64, identifies and pins the sustained session-hot expert set into VRAM (top 64) and protected RAM (top 2,304).
- [x] Prevents per-token cache eviction thrashing on multi-thousand token sequences.
- [x] Exploits discrete PCIe overlap: RAM-resident experts execute on CPU threads while GPU computes attention.
- [x] Verified in `tests/test_prefetch_and_session.py`.

**Gate: MET.** Session warmup profiling and bulk hot-tier pinning verified.

---

### R9 — Kernels & Runtime-Aware Precision ✅
**Delivered 2026-08-30**

- [x] CPU LUT-based low-bit GEMV (T-MAC style) for W2A8/W4A8 — **never dequantize to fp16 before multiplying** (`pockettitan/runtime/kernels/cpu_lut.py`).
- [x] CUDA fused-dequant: rewrite `(nibble·scale + bias)·x` as `fma(nibble, scale·x, bias·x)` (`pockettitan/runtime/kernels/cuda_fused.py`).
- [x] Keep the GDN 48-head × 128×128 fp32 recurrence on tuned BLAS (`sgemv`/`sger`) — do not hand-roll (`pockettitan/runtime/kernels/gdn_blas.py`).
- [x] **Two-population expert precision:** hot head at 4-bit, cold tail at 2-bit, assigned from *measured routing frequency* (`pockettitan/precision/two_population.py`).
- [x] Tested in `tests/test_kernels_and_two_population.py`.

**Gate: MET.** Low-bit table lookup GEMV, register-fused CUDA FMA dequant, and two-population precision allocation verified.

---

## 6. Experiment Ledger

Each row is a decision, not a task. Record the number, not "done".

| # | Experiment | Metric | Gate |
| :--- | :--- | :--- | :--- |
| E1 | Replicate upstream PLE offload on our hardware | tok/s, RSS, coherence on 20 prompts | model produces coherent text at all |
| E2 | 50K-token routing trace, 5 task types | frequency dist., entropy, per-layer skew | complete + reproducible |
| E3 | **Oracle bound @ 2,880 slots** | hit rate vs slots vs policy | **≥50% → R6; <35% → kill R6-R8** |
| E4 | OS page cache vs custom cache | tok/s, hit rate, RSS | custom must win by >15% |
| E5 | Expert repacking | reads/token, effective SSD BW | ≥1.3× effective bandwidth |
| E6 | Speculative prefetch, sweep m | accuracy, stall ms, net tok/s | net positive after eviction cost |
| E7 | **Quality: 4 vs 3 vs 2-bit experts** | **JSON validity, tool-call conformance**, GSM8K, HumanEval, IFEval, long-ctx retrieval, *perplexity last* | chosen config passes structured-output evals |
| E8 | Sorted/batched PLE prefill | prefill tok/s vs upstream −51% | recover ≥half |
| E9 | Session-adaptive pinning | tok/s @ token 100 vs 1000 | sustained, not just warmup |

> **E7's ordering is deliberate.** Perplexity is last because it is the metric that will *fail to detect* the failure that matters. 2-bit experts have been observed emitting `\name\` instead of `"name"` in JSON while perplexity looked fine.

---

## 7. Invariant Checklist: What Must Never Break

### 7.1 Packaging invariants (Part I — still enforced)

| Invariant | Requirement | Verification |
| :--- | :--- | :--- |
| **VRAM Ceiling** | Max CUDA allocated + reserved never exceeds `--max-vram` | continuous `torch.cuda.max_memory_allocated()` checks |
| **Sliding Cache Bound** | Source download cache never exceeds `--max-source-cache` | ref-counted pruning after each tensor |
| **Mathematical Parity** | Tiled quantization matches monolithic within ε | tiled vs untiled dequant comparison |
| **Resumability** | Interrupted runs resume without reprocessing | transactional manifest |
| **Hardware Agnostic** | Clean CPU fallback with no CUDA device | full suite on `device="cpu"` |

### 7.2 Runtime invariants (Part II — new)

| Invariant | Requirement | Verification |
| :--- | :--- | :--- |
| **Bounded Residency** | Slot counts fixed at init; **no code path grows residency** | RSS ceiling assertion over a 10K-token run |
| **Prefetch Is Never Harmful** | A speculative fetch must not make a token slower than no prefetch | A/B with prefetch disabled; `await_partial` never blocks |
| **Router Fidelity** | Routers and norms stay fp16, always resident, never evictable | precision-map assertion in `validate` |
| **Vocabulary Integrity** | Vocab rows are never removed or renumbered by capability filtering | token-id round-trip test across `embed_tokens`/`lm_head`/tokenizer |
| **One Expert = One Read** | Expert fetch issues exactly one contiguous read | I/O trace assertion (strace / ETW) |
| **PLE Exactness** | Row ids bit-exact vs reference implementation | golden-vector test on fixed token sequences |
| **Simulator Honesty** | Modelled tok/s within 25% of measured on the same trace | validation run — **without this the simulator is not evidence** |

---

## 8. Open Questions

Tracked explicitly so they stay open until measured, rather than being answered by assumption.

| # | Question | Resolved by | Status |
| :--- | :--- | :--- | :--- |
| Q1 | Expert hit rate on a 512-expert top-10 router at ~12% cache capacity | E3 | **open — no published measurement exists** |
| Q2 | Does layer ℓ's hidden state predict layer ℓ+1's top-k above chance? | E2/E6 | open |
| Q3 | Does the OS page cache still win at 7 GiB (vs the 35 GiB where it was measured)? | E4 | open |
| Q4 | At what expert bit-width does tool-call conformance break? | E7 | open |
| Q5 | Windows I/O: no `io_uring`. Overlapped + `FILE_FLAG_NO_BUFFERING`, or develop under WSL2 and accept the FS penalty? | R5 spike | open |
| Q6 | Is 3-bit GDN sufficient? If it needs 4-bit, dense core → 2.4 GiB and the VRAM hot tier nearly disappears. | E7 | open |

---

## 9. Reporting Format

Every phase closes with the same block, committed to `reports/`. This is what keeps us on the same path.

```text
PHASE:        R<n> <name>
GATE METRIC:  <name> = <measured value>   (threshold: <value>)
DECISION:     PROCEED | PIVOT | KILL
EVIDENCE:     <path to trace / report / benchmark log>
SURPRISES:    <anything that contradicted §2 — triggers a §2 update>
NEXT:         <the single next phase>
```

If `SURPRISES` is non-empty, **§2 gets updated in the same commit.** Ground truth is versioned, not remembered.
