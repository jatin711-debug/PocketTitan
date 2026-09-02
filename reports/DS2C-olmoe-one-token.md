# DS2-C — End-to-End Paged OLMoE One-Token Forward

```text
PHASE:        DS2-C sequential full-model forward
GATE MET:     one valid vocabulary ID traverses 16 layers and produces finite logits
REPLAY:       bit-identical logits with zero remote bytes
DECISION:     PROCEED to independent real-checkpoint parity and multi-token generation
LIMITATION:   this is not yet a natural-language generation or throughput claim
```

## Setup

- Date: 2026-08-31
- Model: `allenai/OLMoE-1B-7B-0924-Instruct`
- Revision: `7f1c97f440f06ce36705e4f2b843edb5925f4498`
- Input: valid vocabulary ID `1`, sequence length one, position zero
- Layers: all 16 real checkpoint decoder layers
- Precision: source-native BF16
- Expert routing: real top-8 router at every layer
- Hardware: AMD Ryzen 7 6800H; NVIDIA RTX 3050 Laptop GPU, 4 GiB
- Limits: 3.5 GiB CUDA allocation budget, 12 GiB process-RSS budget
- Paging: three range workers, 4 GiB NVMe page-cache budget
- GPU staging: one 4 MiB expert projection or one 8 MiB LM-head chunk

The first pass started with an empty cache. The second pass rebuilt and executed
all 16 layers from the completed local page cache in the same process. No KV
cache was requested because the experiment covers a single position.

## Result

| Metric | Network-cold first pass | Local warm replay |
| :--- | ---: | ---: |
| Result | PASS | bit-identical logits |
| Wall time | 820.888 s | 23.115 s |
| Effective rate | 0.00122 token/s | 0.04326 token/s |
| Remote payload | 2,564,034,560 B (2.39 GiB) | 0 B |
| Global pages | 3 faults | 3 hits |
| Layer-backbone pages | 144 faults | 144 hits |
| Routed-expert pages | 128 faults | 128 hits |
| Experts executed | 128 | 128 |
| Logical page bytes accessed | 2.39 GiB | 2.39 GiB |
| Peak expert projection | 4.00 MiB | 4.00 MiB |
| Peak LM-head chunk | 8.00 MiB | 8.00 MiB |
| Peak CUDA allocation | 48.13 MiB | 48.13 MiB |
| Peak sampled process RSS | 1.65 GiB | 1.38 GiB |
| Output argmax | token 309, logit 7.75 | token 309, logit 7.75 |
| Maximum logit delta | — | 0.0 |

The output-logit SHA-256 was
`1539381af541dd69cfc113988fdcd4c1b598b43955078bdd98c9bd0b1bd6eb20`
on both passes. Routing selections were also identical at every layer.

The 2.39 GiB cold payload divides into:

- global embedding, final norm, and LM head: 412,094,464 bytes;
- 16 layers of mandatory attention/router/norm weights: 541,327,360 bytes;
- 128 selected expert pages: 1,610,612,736 bytes.

## What this validates

PocketTitan can now execute a complete causal-LM forward path without ever
constructing the full 7B-parameter BF16 model in RAM or VRAM. The embedding row
is selected from a memory-mapped page, one decoder layer is resident at a time,
only the eight experts chosen by that layer's real router execute, and the
untied LM head is projected in bounded row chunks. Every completed page is
revision-addressed, checksum-verified, and reusable after restart.

This closes the storage/residency part of DS2 for a one-token forward. Both the
4 GiB VRAM and 12 GiB RAM targets were met with wide margins for this specific
experiment.

## What this does not validate

- The live result has not yet been compared against logits from an independently
  loaded full upstream checkpoint. Exact upstream parity is established for the
  expert block and the small full-model fixture, while the real 16-layer run is
  validated by finite logits and exact cold/warm replay.
- Token ID `1` is a deterministic vocabulary input, not a tokenized natural-
  language prompt and not generated text.
- There is no KV cache, sampling loop, second generated position, or quality
  evaluation yet.
- Process RSS is not total machine memory use. CUDA allocation is not the same
  as driver-level reserved VRAM.
- Logical page bytes are not a physical-NVMe counter; the operating-system file
  cache can serve warm reads.
- Warm performance is only 0.043 token/s. It is below the first 0.1 token/s
  performance gate and is not practically interactive.

## Decision and next experiment

**PROCEED, with performance and independent parity still open.** DS2-D should:

1. obtain an independent upstream/native-BF16 logit oracle for a deterministic
   tokenized prompt;
2. add KV-cache-aware multi-token generation and verify the second position;
3. amortize checksum validation instead of rehashing every immutable page on
   every lookup;
4. measure actual SSD traffic and overlap layer `n+1` page fetches with layer
   `n` execution;
5. cross 0.1 token/s warm before claiming a useful runtime.

Raw measurements are preserved in `reports/DS2C-olmoe-one-token.json`.

## Reproduction

```powershell
python -m pockettitan.cli domainslice test-model `
  allenai/OLMoE-1B-7B-0924-Instruct `
  --revision 7f1c97f440f06ce36705e4f2b843edb5925f4498 `
  --input-token-id 1 --device cuda `
  --cache-dir ./olmoe-full-pages --download-workers 3 `
  --max-cache 4GB --max-vram 3584MB --max-ram 12GB `
  --head-chunk 8MB --output ./ds2c-result.json
```
