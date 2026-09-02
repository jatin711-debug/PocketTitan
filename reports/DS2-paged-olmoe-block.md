# DS2 — Paged OLMoE Block Hypothesis

```text
PHASE:        DS2-A paged OLMoE expert block
GATE METRIC:  page-backed output vs upstream OlmoeExperts = bit-exact
DECISION:     PROCEED to mandatory-backbone packaging and full decoder layer
EVIDENCE:     tests/test_olmoe_paged.py; tests/test_domainslice_canary.py
SURPRISES:    warm local expert execution is cheap; cold network faults dominate
NEXT:         DS2-C full-model result in reports/DS2C-olmoe-one-token.md
```

## Setup

- Date: 2026-08-31
- Model: `allenai/OLMoE-1B-7B-0924-Instruct`
- Revision: `7f1c97f440f06ce36705e4f2b843edb5925f4498`
- Layer: 9
- Input: one deterministic synthetic hidden state, seed 42
- Router: real 64 × 2048 BF16 checkpoint tensor
- Hardware: AMD Ryzen 7 6800H; NVIDIA RTX 3050 Laptop GPU, 4 GiB
- Runtime: Python 3.12, PyTorch, Transformers 5.16.1

## Result

The real router selected experts `1, 2, 5, 28, 33, 38, 47, 55`. Each expert
was represented as a 12 MiB page containing its three native-BF16 projections.

| Metric | CPU network-cold | CPU warm NVMe cache | GPU cached first pass | GPU cached warm pass |
| :--- | ---: | ---: | ---: | ---: |
| Expert page faults | 8 | 0 | 0 | 0 |
| Expert page hits | 0 | 8 | 8 | 8 |
| Remote expert payload | 96.00 MiB | 0 B | 0 B | 0 B |
| Block runtime | 39.950 s | 0.419 s | 2.413 s | 0.566 s |
| Peak staged projection | 4.00 MiB | 4.00 MiB | 4.00 MiB | 4.00 MiB |
| Peak measured CUDA allocation | — | — | 12.14 MiB | 12.14 MiB |

PocketTitan and upstream `transformers.models.olmoe.OlmoeExperts` produced:

- maximum absolute error: `0.0`;
- mean absolute error: `0.0`;
- cosine similarity: `1.00000036` (floating reduction roundoff above one);
- argmax agreement: `100%`;
- cold and warm PocketTitan outputs: bit-identical.

## Interpretation

The DS2-A systems hypothesis passes: a real router can select remotely stored
experts, PocketTitan can fault only those experts, execute their native weights,
and reproduce the upstream expert block exactly while staging one 4 MiB
projection at a time.

This does **not** yet prove a runnable language model or useful token throughput.
The result isolates the remaining problem: cold internet faults are roughly two
orders of magnitude slower than warm local execution for this single block. A
full token crosses 16 MoE layers, so practical inference needs a populated NVMe
working set, bounded RAM caching, and overlap/prefetch. The next test must include
the mandatory attention/router/norm backbone and one complete decoder layer.

## DS2-B complete decoder layer

The follow-on test assembled real layer 9 on the 4 GiB RTX 3050 from a fresh
cache. It included input/post-attention RMSNorm, Q/K/V/O projections, Q/K norms,
RoPE, eager attention, the real router, eight paged experts, and both residual
connections.

| Metric | First execution | Warm execution |
| :--- | ---: | ---: |
| Mandatory backbone pages | 9 faults | resident |
| Mandatory backbone payload | 32.27 MiB | 0 B |
| Expert pages | 8 faults | 8 hits |
| Expert payload | 96.00 MiB | 0 B |
| Total first remote payload | 128.27 MiB | 0 B |
| Expert portion runtime | 36.082 s | 0.519 s |
| Peak staged projection | 4.00 MiB | 4.00 MiB |
| Peak CUDA allocation (cached repeat) | 44.43 MiB | 44.43 MiB |
| Output difference | — | bit-identical to first execution |

**DS2-B decision: PROCEED.** A complete real decoder layer now runs through the
remote → NVMe → resident-backbone/paged-expert path. This is still a deterministic
hidden-state test, not token generation. DS2-C must package the mandatory
full-model backbone and run embeddings through all 16 layers and the LM head.

DS2-C has since passed that one-token systems gate. See
[`DS2C-olmoe-one-token.md`](DS2C-olmoe-one-token.md) for the measured 16-layer
result and its remaining parity, generation, and performance limitations.

## Reproduction

```powershell
python -m pockettitan.cli domainslice test-block `
  allenai/OLMoE-1B-7B-0924-Instruct `
  --revision 7f1c97f440f06ce36705e4f2b843edb5925f4498 `
  --layer 9 --tokens 1 --device auto `
  --cache-dir ./olmoe-pages --download-workers 3 --max-cache 2GB
```
