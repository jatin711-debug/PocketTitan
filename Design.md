# PocketTitan — System Design v2

> **Mission:** Treat a sparse model as a virtual address space whose active weights
> move through NVMe, RAM, and VRAM under explicit residency bounds.

The original external-memory quantizer design is preserved at
[`docs/legacy/quantizer-design-v0.1.md`](docs/legacy/quantizer-design-v0.1.md).
This document specifies the current project: a capability-aware model compiler,
the `.ptitan` package ABI, a trace-driven memory simulator, and an out-of-core
runtime integrated with llama.cpp.

## 1. System boundary

PocketTitan owns checkpoint inspection, capability selection, bounded streaming
quantization, hierarchy-aware dense/expert/PLE layouts, package provenance and
integrity, routing/access simulation, and the storage managers used by the runtime.

llama.cpp/ggml owns standard model execution, tokenization, sampling, attention,
Gated DeltaNet state, KV cache, and production low-bit matrix kernels. PocketTitan
reuses GGML-compatible payload codecs where practical and adds specialized storage
codecs only when access granularity requires one, as with PLE rows.

## 2. End-to-end architecture

```text
Pinned Hugging Face revision
          │
          ▼
Header audit + capability graph
          │
          ▼
Build plan (all output offsets known)
          │
   ┌──────┼─────────┐
   ▼      ▼         ▼
 dense  experts     PLE
   │      │          │
   └──────┼──────────┘
          ▼
   model.ptitan package
          │
   fast/full validation
          │
          ▼
 llama.cpp runtime adapter
          │
   ┌──────┼──────────┐
   ▼      ▼          ▼
 NVMe    RAM        VRAM
 cold    warm        hot
```

The source model, output package, and runtime model are never required to be
resident in RAM or VRAM. Peak memory is bounded by one legal work item during
packaging and fixed slot counts during inference.

## 3. Compiler and packager

### 3.1 Inspection

The compiler reads `config.json`, the Safetensors index, and every shard header.
It pins one immutable source revision and constructs absolute tensor byte ranges.
Parameter totals come from tensor shapes, never `total_size / dtype`.

### 3.2 Capability selection

`--features text` retains the complete text decoder, tokenizer ID space,
embeddings, LM head, routed/shared experts, attention/GDN blocks, and PLE. It
excludes the vision tower and optional MTP block. Capability pruning never
renumbers vocabulary rows or removes arbitrary experts, heads, layers, or PLE
entries.

### 3.3 Work granularity

Source shards are storage units, not work units:

```text
tensor → fused expert slice / matrix → legal row tile
```

Quantizer contracts define legal split axes and workspace requirements. Group
padding is included in both VRAM and output-size calculations.

### 3.4 Output regions

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

- Dense entries have explicit offsets and codec references.
- One routed expert is one page-aligned gate/up + down record.
- PLE rows are independently decodable and page-packed; no row crosses 4 KiB.
- PLE hash/index tensors remain exact int64 metadata and are never quantized.

## 4. `.ptitan` ABI

The package is self-describing. A runtime must not import PocketTitan's Python
quantizer to decode it. Every payload references a descriptor containing codec
ID/version, method, logical and storage bits, block/group geometry, symmetry,
metadata dtypes, byte order, and packing order.

The manifest records source repository/revision, config and index hashes, build
profile, features, region sizes, and integrity metadata. Prototype package v1.0
is unsupported; v1.1 is the first integrity-checked ABI and fails closed on
unknown versions.

## 5. Integrity and recovery

The writer preallocates regions and writes at planned offsets. Completion is
committed in durable batches:

1. write payloads and compute checksums;
2. flush and synchronize the region;
3. atomically replace the journal.

After a crash, the final uncommitted batch may be repeated but committed data is
never skipped. Resume is rejected if revision, features, codecs, precision, or
layout differ. Fast validation checks schema, bounds, sizes, completion, and item
checksums. Full validation additionally hashes regions and decodes samples.

## 6. Runtime hierarchy

```text
VRAM: dense core, state/KV, sustained-hot experts, staging
RAM:  bounded expert/page cache and CPU execution buffers
NVMe: cold expert records and PLE row store
```

Slot counts are fixed at initialization. No path may grow residency. Routers,
norms, and recurrent state are non-evictable. RAM-resident experts execute on CPU
unless repeated use amortizes GPU upload.

PLE prefetch is exact: after sampling, all next-pass rows are known. Expert
prefetch is speculative and exists only if traces show positive net throughput
after overfetch and eviction costs.

## 7. Simulator and evidence gates

The simulator replays synthetic and real routing traces through cache policies
and a measured hardware model. It reports SSD/PCIe bytes, IOPS, hit rate, churn,
stall time, and throughput. Belady is the oracle upper bound.

Custom cache, prefetch, and VRAM-hot-tier work is conditional: each must beat the
OS-page-cache baseline by at least 15%. Simulation becomes evidence only after it
predicts measured microbenchmarks within 15%.

## 8. Runtime integration sequence

1. Package integrity and live canary.
2. Synthetic simulator and Windows storage measurements.
3. Routing-trace patch against pinned upstream llama.cpp.
4. Oracle decision on real traces.
5. Standalone native-Windows PLE reader.
6. Maintained llama.cpp fork for minimal correct expert streaming.
7. Conditional cache, prefetch, hot-tier, and kernel optimization.

Native Windows is the first optimized platform. Buffered/mapped reads are the
baseline; OVERLAPPED I/O is adopted only after measurement.

## 9. Non-negotiable invariants

- Packaging never exceeds the configured VRAM ceiling.
- Runtime residency is bounded by fixed slots.
- Source revision is immutable during a build.
- Vocabulary IDs and rows are preserved.
- Routers and structural metadata are lossless.
- One expert maps to one contiguous record.
- PLE row IDs match the reference exactly.
- Prefetch must not reduce matched-workload throughput.
- Every phase closes with measured evidence and a proceed/pivot/kill decision.

## 10. Success definition

For Qwen3.8-Flash-Next text-only inference on 4 GB VRAM, about 12 GB RAM, and
NVMe:

1. produce a complete validated package under the VRAM limit;
2. produce one correct token with bounded residency;
3. reach 0.1, 0.5, 1, then 2+ tokens/s without weakening correctness gates;
4. evaluate structured output and tool calling before perplexity drives quality
   decisions.
