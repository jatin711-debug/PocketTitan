# PocketTitan — System Design v3

> **Mission:** Treat a sparse model as a virtual address space whose active weights
> move through a remote checkpoint, NVMe, RAM, and VRAM under explicit
> residency bounds.

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
   model.ptitan package ◄──── immutable remote expert pages
          │
   fast/full validation
          │
          ▼
 llama.cpp runtime adapter
          │
   ┌──────┼──────────┐
   ▼      ▼          ▼
 remote  NVMe       RAM       VRAM
 origin  cold       warm       hot
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

### 5.1 DomainSlice page identity and recovery

DomainSlice extends the same virtual address space; it is not a second package
format or runtime. Mandatory weights (embeddings, attention, routers, norms,
shared experts, and LM head) must have an immutable local representation when
consumed by end-to-end inference. Routed experts may begin as remote-only and
fault into a bounded local page cache. DS2-C's 4 GiB cache held the complete
one-token working set, so mandatory and routed pages did not compete for
eviction; a smaller production cache must give mandatory pages a protected
residency class rather than rely on that capacity assumption.

Tensor filenames are source-layout details, not cache keys. Public identity is:

```text
ModelRevision(repository, immutable commit)
  └─ WeightID(layer, component, expert, projection)
       └─ WeightPageID(revision, page kind, logical key, codec version)
            └─ one or more exact SourceSlice ranges
```

One expert page contains all of that expert's projections using the existing
contiguous `ExpertRecordLayout`. V0 accepts source-native BF16 only; later codecs
receive distinct page IDs so raw and quantized pages cannot collide.

Completed pages are immutable and SHA-256 verified. Projection fragments and a
durable journal live outside the completed-page index. Assembly writes a new
page, synchronizes it, atomically renames it, then commits its manifest and cache
index. A crash can repeat uncommitted work but cannot publish a partial page.

Every remote payload response must be HTTP 206 with the exact requested
`Content-Range`. HTTP 200, malformed ranges, revision drift, and checksum
failure fail closed. Authentication comes only from `HF_TOKEN` and is never
included in logs, manifests, errors, or cache identity.

## 6. Runtime hierarchy

```text
VRAM: dense core, state/KV, sustained-hot experts, staging
RAM:  bounded expert/page cache and CPU execution buffers
NVMe: cold expert records and PLE row store
Remote: immutable Hugging Face checkpoint, exact expert ranges only
```

Slot counts are fixed at initialization. No path may grow residency. Routers,
norms, and recurrent state are non-evictable. RAM-resident experts execute on CPU
unless repeated use amortizes GPU upload.

PLE prefetch is exact: after sampling, all next-pass rows are known. Expert
prefetch is speculative and exists only if traces show positive net throughput
after overfetch and eviction costs. The local page cache is byte-budgeted and
evicts only completed, unleased pages; partial and in-use pages are never
victims. Three range workers are the default, each with an 8 MiB transfer
buffer, so download buffering remains about 24 MiB.

Dense and sparse sources follow different paths. Dense Qwen3.8-27B has no
router-selectable experts and therefore uses resumable sequential layer
streaming. DomainSlice expert faults apply only to routed MoE checkpoints.

For OLMoE, the runtime preserves the upstream router and replaces only its
expert collection. A selected expert page is memory-mapped from NVMe; gate, up,
and down projections are staged and executed one at a time, then the weighted
result is accumulated into the original token positions. The native-BF16 V0
never holds all top-k expert weights on GPU simultaneously. DS2-A measured a
4 MiB projection staging unit and bit-exact parity with upstream expert math.

The DS2-C reference path extends that rule across the complete model. It selects
one row from the memory-mapped embedding, constructs one upstream decoder layer
on meta parameters, replaces every non-expert parameter from verified tensor
pages, executes the layer, and disposes it before constructing the next. Final
normalization is resident only for its operation. The untied 196 MiB LM head is
never copied whole to CUDA; it is multiplied in 8 MiB row chunks and its logits
are assembled on the host.

On the pinned OLMoE revision, a fresh one-token pass through all 16 layers read
2.39 GiB remotely and took 820.888 seconds. A local-cache replay took 23.115
seconds with zero remote bytes and bit-identical routing and logits. Peak
measured CUDA allocation was 48.13 MiB and peak sampled process RSS was 1.65 GiB. These
numbers validate bounded residency, not useful throughput: 0.043 warm token/s
misses the first 0.1 token/s performance gate. Page checksum validation is
currently repeated on every lookup and physical SSD traffic is not yet measured.

## 7. Simulator and evidence gates

The simulator replays synthetic and real routing traces through cache policies
and a measured hardware model. It reports SSD/PCIe bytes, IOPS, hit rate, churn,
stall time, and throughput. Belady is the oracle upper bound.

Custom cache, prefetch, and VRAM-hot-tier work is conditional: each must beat the
OS-page-cache baseline by at least 15%. Simulation becomes evidence only after it
predicts measured microbenchmarks within 15%.

A domain profile is only a prewarm/ranking hint. It never removes an expert and
cannot make a correct forward pass depend on the workload label. Coding locality
is accepted only on held-out generated rollouts—not prompt routing—against both
adaptive SLRU and matched general-domain controls. At a 10% expert-bank budget,
the coding profile must reduce remote bytes/token by at least 20%, with a 95%
bootstrap lower bound above 10%, across at least three task/language families.
Failure removes the coding-profile claim while preserving generic demand paging.

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
- Remote range responses are exact; a full-shard fallback is forbidden.
- Cache identity includes model commit and codec version.
- Every routed expert remains reachable even when a domain profile is active.
- Vocabulary IDs and rows are preserved.
- Routers and structural metadata are lossless.
- One expert maps to one contiguous record.
- PLE row IDs match the reference exactly.
- Prefetch must not reduce matched-workload throughput.
- Every phase closes with measured evidence and a proceed/pivot/kill decision.

## 10. Success definition

The DomainSlice storage proof is the pinned OLMoE 1B-active/7B-total model:
metadata-only discovery, one exact 12 MiB BF16 expert fault, atomic local commit,
and a second request with zero shard payload bytes. The first systems proof now
also includes a valid vocabulary ID traversing embeddings, all 16 decoder
layers, final norm, and the chunked LM head under 4 GB VRAM and 12 GB process
RSS, followed by bit-identical zero-network replay. Independent full-checkpoint
logit parity and multi-token generation remain open gates.

The end target remains Qwen3.8-Flash-Next text-only inference on 4 GB VRAM,
about 12 GB RAM, and NVMe:

1. produce a complete validated package under the VRAM limit;
2. produce one correct token with bounded residency;
3. reach 0.1, 0.5, 1, then 2+ tokens/s without weakening correctness gates;
4. evaluate structured output and tool calling before perplexity drives quality
   decisions.
