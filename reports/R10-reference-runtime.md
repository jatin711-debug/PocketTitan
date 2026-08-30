# R10 — Reference Runtime: a `.ptitan` package that actually generates text

```text
PHASE:        R10 - packaged weights through the reference HF module tree
GATE METRIC:  a 16-bit package reproduces the source model's logits exactly
              (argmax identical, max gap < 0.5% of scale)
DECISION:     MET. The FORK GATE is no longer on the critical path.
EVIDENCE:     tests/test_hf_runtime.py (11) - 298 offline tests total, ruff clean
SURPRISES:    transformers 5.16.1 already ships BOTH qwen3_5 and qwen4_exp
NEXT:         build Qwen3.8-27B at int8, measure, then step precision down
```

## The thing that changed the plan

`transformers` 5.16.1 — already installed — contains **both** reference
implementations:

| Architecture | Class | Model |
| :--- | :--- | :--- |
| `qwen3_5` | `Qwen3_5ForCausalLM` | Qwen3.8-27B |
| `qwen4_exp` | `Qwen4ExpForConditionalGeneration` | Qwen3.8-Flash-Next |

Earlier reports treated "no forward pass exists for the GDN + full-attention
hybrid" as the blocker on seeing real text, and named the llama.cpp FORK GATE as
the way through. That was wrong about the cost. Nothing in *this repository*
implements Gated DeltaNet, but there is no need to write it: the recurrence, the
sparse attention, the interleaved mRoPE, the mask construction and the KV cache
are all present and tested upstream.

So this phase reimplements none of it. It keeps HF's module tree exactly as it is
and replaces only the **storage**.

## Design

Every large parameter becomes a module that decodes itself out of the package
when used, through `decode_record` — the writer's own inverse:

| Module | Replaces | Behaviour |
| :--- | :--- | :--- |
| `PackagedLinear` | `nn.Linear` | Decodes its weight per call; `out_chunk` splits a wide head so `lm_head` (248,320 x 5,120 = 2.5 GB) is never fully resident |
| `PackagedEmbedding` | `nn.Embedding` | Reads only the rows the batch references, one read per *distinct* token id |
| `PackageWeights` | — | Name resolution, byte-budgeted LRU, I/O accounting |

Small tensors — norms, `A_log`, `dt_bias`, `conv1d` — are materialized eagerly;
they are about 0.007% of the parameters.

`decode_rows` was added to `package/decode.py` for this: it reconstructs
`[row_start, row_stop)` without touching the rest, which is what makes a
row-addressed 2.5 GB table usable on a 12 GB machine. It falls back to a full
decode when a row would not start on a byte boundary.

### Coverage is the load-bearing safety property

An unresolved parameter name would leave the module holding whatever the
skeleton was initialized with, and the model would generate fluent nonsense —
precisely the failure this project already spent a session diagnosing. So
`build_causal_lm` **refuses to return a model with any unbacked parameter**, and
separately refuses any parameter still on the `meta` device.

`parameters_on_meta()` exists for the same reason. Building under
`torch.device("meta")` would also fake the *buffers*, and the rotary embedding's
`inv_freq` is a buffer: a model built that way runs, produces numbers, and is
silently wrong. Buffers are tiny, so they are built for real.

## Verification

The fixture is a genuine `Qwen3_5ForCausalLM` with the real hybrid layer pattern,
saved in the published layout (language tower nested under
`model.language_model.`), then packaged and run.

| Test | Property |
| :--- | :--- |
| `test_fp16_package_reproduces_the_source_models_logits` | **A 16-bit package computes what the source model computes.** Identical argmax, max gap under 0.5% of scale |
| `test_every_parameter_is_backed_by_the_package` | No parameter runs uninitialized |
| `test_missing_tensor_is_refused_not_silently_ignored` | Deleting one manifest entry fails the load |
| `test_large_weights_are_never_resident` | Resident parameters < 10% of packaged; forward still runs at `cache_bytes=0` |
| `test_embedding_reads_only_the_rows_it_needs` | Two distinct ids decode two rows, not the table |
| `test_generate_produces_text_and_reports_its_io` | Package + tokenizer -> text, with real I/O accounting |

## Measured precision ladder

Same weights, same prompt, packaged at each width and run through the same
forward pass. `corr` is against the source model's logits; `agree` is argmax
agreement; `KL` is against the source distribution.

| bits | corr | argmax agree | KL | package |
| ---: | ---: | ---: | ---: | ---: |
| 16 | **+1.00000** | **100.0%** | 0.00000 | 448.1 KiB |
| 8 | +0.99996 | 100.0% | 0.00001 | 286.5 KiB |
| 4 | +0.98800 | 75.0% | 0.00230 | 188.9 KiB |
| 3 | +0.95441 | 62.5% | 0.00901 | 188.9 KiB |
| 2 | +0.75770 | 25.0% | 0.05339 | 140.1 KiB |

**These are pipeline numbers, not quality predictions.** The fixture is randomly
initialized, and random weights are the worst case for quantization — they have
none of the redundancy or low-rank structure that makes trained weights
compress. Read the *shape*: 16-bit is exactly lossless, degradation is monotone,
and 2-bit collapses. A broken pipeline does not produce this curve; it produces
near-zero correlation everywhere.

Note that 3-bit and 4-bit produce **identically sized packages**. That is
`storage_bits` — 3-bit occupies 4 bits on disk — visible as a measurement rather
than an argument (T1.9).

## Using it

```bash
pockettitan run ./qwen27b_int8 --prompt "Explain what a memory hierarchy is." -n 64
```

Reports tokens/s, GiB decoded, read count, and cache residency. It is a
correctness tool: weights are decoded per use, so it is slow by construction.
The fast path is R6-R9's paging runtime, which this now gives a reference to
check against.

## What this does and does not settle

**Settles:** a `.ptitan` package is a complete, runnable model. Packing,
addressing, decoding and precision are now measurable end to end rather than by
proxy.

**Does not settle:** the out-of-core thesis. A dense 27B has no routed experts
and no PLE table, so the expert bank, SLRU paging, prefetch and the row store
remain exercised only on synthetic fixtures. That is the correct order — prove
the quantizer on 52 GiB of downloads before betting 360 GiB on the model that
also needs paging to work.
