# PocketTitan

> Model size should determine time and storage—not mandatory RAM or VRAM capacity.

[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-205%20passed-brightgreen.svg)](tests/)
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
