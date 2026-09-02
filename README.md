# PocketTitan

> Model size should determine time and storage—not mandatory RAM or VRAM capacity.

[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-335%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

PocketTitan is a capability-aware model compiler and out-of-core inference research project for sparse models larger than physical memory. Its current target is text-only Qwen3.8-Flash-Next inference on approximately 4 GB VRAM, 12 GB RAM, and NVMe.

The project treats model weights as a virtual address space:

```text
Hugging Face checkpoint
          │
          ▼
capability selection + bounded quantization
          │
          ▼
     model.ptitan
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  NVMe   RAM   VRAM
  cold   warm   hot
```

Correct bounded inference comes first. Throughput milestones are 0.1, 0.5, 1, and then 2+ tokens/s.

## Current state

R0 checkpoint auditing and the core R1.8 package-integrity path are implemented:

- immutable Hugging Face revision pinning and tensor-level audit;
- text-only capability filtering without vocabulary renumbering;
- fused MoE expert slicing and one-record-per-expert layout;
- page-packed, independently addressable PLE rows;
- exact Qwen PLE bigram/trigram hash parity;
- self-describing `.ptitan` ABI v1.1 with codec descriptors;
- durable batched journaling, CRC32C item checksums, and region SHA-256;
- compact deterministic canary packages;
- fast/full package validation and adversarial corruption tests;
- CPU CI on Python 3.10 and 3.12.

Runtime-compatible GGML payload codecs and the pinned live Qwen canary are the next R1 gates. A complete model build is intentionally blocked until those pass.

See [Plan.md](Plan.md) for execution status and [Design.md](Design.md) for system invariants. The original quantizer-only design is preserved in [docs/legacy/quantizer-design-v0.1.md](docs/legacy/quantizer-design-v0.1.md).

## Package layout

```text
model.ptitan/
  manifest.json
  dense/blob.bin
  experts/bank.bin
  experts/layout.json
  ple/table.bin
  ple/index.json
  metadata/config.json
  metadata/generation_config.json
  tokenizer/
  integrity/checksums.json
  build_journal.json
```

Prototype v1.0 packages are intentionally incompatible. ABI v1.1 fails closed on unknown versions, missing codec metadata, invalid spans, incomplete journals, and payload corruption.

## CLI

Audit a local or remote checkpoint without downloading complete weight shards:

```bash
pockettitan audit Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --features text \
  --precision pt-q4e
```

Create a byte-exact package plan:

```bash
pockettitan plan Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --features text \
  --output plan.json
```

Build a deterministic canary:

```bash
pockettitan package Qwen/Qwen3.8-Flash-Next C:\PocketTitanModels\qwen-canary.ptitan \
  --profile canary \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --features text
```

Validate a package:

```bash
pockettitan validate C:\PocketTitanModels\qwen-canary.ptitan --mode fast
pockettitan validate C:\PocketTitanModels\qwen-canary.ptitan --mode full
```

## DomainSlice: On-Demand MoE Expert Paging & Acceleration Engine

PocketTitan includes **DomainSlice**, a virtualized out-of-core memory hierarchy for Mixture-of-Experts (MoE) architectures (such as `allenai/OLMoE-1B-7B-0924-Instruct` and sparse Qwen variants). Rather than requiring the full model weight footprint (13–30+ GB) to be resident in GPU VRAM, DomainSlice provisions model weights as immutable, independently addressable pages fetched across a 3-tier hierarchy:

```text
       ┌───────────────────────────┐
       │   Tier 1: GPU VRAM        │  (Uncompressed BF16 Working Set: ~144 experts)
       │   Compute: Tensor Cores   │  Latency: ~1.2 ms per expert GEMM
       └─────────────▲─────────────┘
                     │  PCIe DMA (0.7 ms for 3.3 MB INT4)
       ┌─────────────┴─────────────┐
       │   Tier 2: Host RAM Cache  │  (4-bit INT4 Compressed: 100% of 1,024 experts in 3.38 GB)
       │   Storage: Pinned Memory  │  Latency: 3.5 ms GPU dequantization
       └─────────────▲─────────────┘
                     │  Fast NVMe Read
       ┌─────────────┴─────────────┐
       │   Tier 3: Local NVMe SSD  │  (Cold immutable storage / byte-range HTTP fetch)
       │   Storage: olmoe-cache/   │  Cold page fault latency: ~40 ms
       └───────────────────────────┘
```

### Core Architecture & Optimizations

1. **Resident Backbone Fast-Path (`--fast`):** Keeps all non-expert parameters (Attention, RMSNorms, Router Gates, Embeddings, LM Head) permanently resident in GPU VRAM (~933 MB), executing attention and routing without layer-swapping stalls.
2. **Asynchronous DMA Speculative Batch Prefetching:** At the start of single-token decode, transfers all $k$ routed experts across PCIe in parallel using non-blocking CUDA streams before compute kernels begin, combined with pop-first LRU eviction.
3. **4-Bit INT4 Compressed Expert Memory (`--int4-cache` / `--quantize-ram`):** Compresses 12.5 MB BF16 expert slices down to **3.3 MB** via 4-bit RTN on GPU cores. This allows **100% of the entire model (1,024 experts)** to fit permanently within **3.38 GB of Host RAM**, completely eliminating disk I/O bottlenecks while maintaining full generation quality.
4. **GPU-Vectorized Fallback-Free Commit Routing (`--commit-routing`):** Leverages CommitMoE (AAAI 2026) to commit token routing to already-resident VRAM experts when gating affinity deltas fall within threshold $\delta$, executed entirely via zero-synchronization CUDA tensor operations.

### Benchmark Telemetry on Consumer Hardware

Tested on an **NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GiB VRAM)** with **12.0 GB System RAM** on Windows 11:  
*Prompt: "What do you know about Quantum Computing?"*

| Metric | Baseline | Fast-Path Initial | Phase 1 & 2 Prefetch | Phase 3 INT4-RAM | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Wall Time** | 1,850.08 s (30.8 min) | 179.05 s (2.98 min) | 103.14 s (1.72 min) | **82.47 s (1.37 min)** | **22.4x Faster** |
| **Prefill Time** | 1,482.32 s (0.01 tok/s) | 77.23 s (0.27 tok/s) | 62.80 s (0.33 tok/s) | **58.07 s (0.36 tok/s)** | **25.5x Faster** |
| **Decode Time (20 tok)** | 367.76 s (0.04 tok/s) | 101.82 s (0.20 tok/s) | 40.34 s (0.50 tok/s) | **24.40 s (0.82 tok/s)** | **20.5x Faster** |
| **Streaming Rate** | ~0.04 tok/s | ~0.20 tok/s | ~0.50 tok/s | **Up to 2.0 tok/s** | **50x Peak Burst** |
| **Direct VRAM Hits** | 0 (100% thrashing) | 1,838 hits (56.2%) | 3,217 hits (62.8%) | **4,223 hits (82.5%)** | **+4,223 GPU hits** |
| **Disk Page Faults** | 5,120 faults | 3,282 faults | 1,903 faults | **897 faults** | **-82.5% reduction** |
| **Peak CUDA VRAM** | 1.29 GiB | 2.79 GiB | 2.59 GiB | **2.64 GiB** | Safe in 4.0 GiB |
| **Peak Process RSS** | 4.24 GiB | 4.60 GiB | 5.98 GiB | **4.39 GiB** | Safe in 12.0 GiB |

### DomainSlice CLI Usage

Generate text on consumer hardware using the accelerated in-memory INT4 cache:

```bash
pockettitan domainslice generate allenai/OLMoE-1B-7B-0924-Instruct \
  --prompt "What do you know about Quantum Computing?" \
  --cache-dir ./olmoe-cache \
  --max-new-tokens 40 \
  --fast \
  --device cuda \
  --vram-experts 144 \
  --ram-experts 1024 \
  --int4-cache \
  --max-cache 14GB
```


## Important boundary

The older `quantize`, GGUF exporter, vLLM exporter, and Python inference runner are retained as legacy research prototypes. They do **not** constitute Qwen3.8 runtime support and are not the path used by `.ptitan`.

The intended integration boundary is:

- PocketTitan: model audit, capability selection, streaming packaging, expert/PLE storage, provenance, integrity, simulation, and storage management;
- llama.cpp/ggml: standard model execution and production CPU/CUDA kernels;
- specialized PocketTitan code only where ordinary tensor runtimes do not model the required access pattern.

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff check pockettitan tests
python -m pytest -q
```

Network and GPU suites remain manually selected:

```bash
python -m pytest -m network
python -m pytest -m gpu
```

PocketTitan is an experimental research system. Storage formats and APIs may change until the complete package and bounded-inference gates close.

## License

MIT. See [LICENSE](LICENSE).
