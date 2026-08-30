# PocketTitan — System Design

> **Tagline:** Model size should determine time and storage — not how much VRAM you need.

## 1. Executive Summary

PocketTitan is an **external-memory post-training quantization (PTQ) engine** for extremely large language models.

Its primary goal is to make it possible to quantize models with hundreds of billions of parameters on consumer hardware by ensuring that the GPU never needs to hold the complete model, a complete checkpoint shard, or even a complete layer.

Instead, PocketTitan treats the model as a **stream of independently addressable tensors and bounded computational work units**.

The core hypothesis is:

> **Peak VRAM should scale with the current quantization working set, not with the total parameter count of the model.**

A 320B model should therefore not imply a 320B-scale GPU allocation.

```text
Remote / local checkpoint
        ↓
Read tensor metadata
        ↓
Locate next required tensor
        ↓
Fetch only required source bytes
        ↓
Stage in CPU memory
        ↓
Split into a legal quantization work unit
        ↓
Move bounded work unit to GPU
        ↓
Quantize
        ↓
Persist compressed result
        ↓
Release source memory
        ↓
Continue
```

The final quantized model is **assembled progressively** as its tensors are processed.

PocketTitan is not initially a training framework. Phase 1 focuses exclusively on **post-training quantization**.

---

## 2. Primary Aim

The first major experiment is:

> **Can a ~320B parameter model such as GLM-5.3-Flash be post-training quantized while enforcing a hard CUDA memory ceiling below 4 GiB?**

The initial success criterion is not necessarily maximum throughput or minimum final model size.

The first objective is to prove that:

```text
Total model size ≠ required GPU residency
```

and that a very large model can be quantized using a bounded accelerator working set.

---

## 3. Secondary Aims

### 3.1 Hardware-Adaptive Quantization

PocketTitan should run on different hardware without changing the underlying model or output semantics.

```text
80 GB GPU → large modules / many experts concurrently
24 GB GPU → experts / matrices
 8 GB GPU → smaller matrices / row groups
 4 GB GPU → small row groups / tiles
 2 GB GPU → very small tiles
 0 GB GPU → CPU backend
```

More VRAM should primarily improve **throughput**, not determine whether the model can be processed at all.

### 3.2 Heterogeneous Precision

PocketTitan should not assume that one quantization format is optimal for every tensor.

```text
Insensitive MoE experts       → ternary / 2-bit
Sensitive MoE experts         → 3-bit / 4-bit
Attention projections         → 4-bit / 8-bit
Dense layers                  → 4-bit / 8-bit
Routers                       → FP16 / BF16 / FP32
Norms                         → BF16 / FP32
Embeddings / LM head          → higher precision if required
```

Long-term objective:

\[
\min_Q 	ext{ModelStorage}(Q)
\]

subject to acceptable quality and distortion constraints.

The optimal result may therefore be 1.8, 2.1, or 2.4 effective bits/weight rather than a forced uniform bit-width.

### 3.3 Bounded Local Storage

PocketTitan should avoid requiring the entire original model to exist on local disk at once.

The source cache should remain bounded while the final quantized checkpoint is assembled progressively.

---

## 4. Core Design Principles

### 4.1 A Giant Model Is a Giant Stream

Traditional quantization often behaves conceptually like:

```text
load model
    ↓
prepare calibration
    ↓
quantize model
    ↓
save model
```

PocketTitan instead behaves like:

```text
inspect model
    ↓
plan work
    ↓
stream bounded source data
    ↓
quantize bounded work unit
    ↓
persist result
    ↓
release source
    ↓
repeat
```

The complete model does not need to exist in GPU memory.

Ideally, the complete source model does not even need to exist on local disk simultaneously.

### 4.2 Shards Are Storage Units, Not Quantization Units

Checkpoint shard boundaries are arbitrary storage boundaries.

A source shard may contain:

```text
end of Layer 8
part of Layer 9
part of Layer 10
```

PocketTitan must therefore reason about:

```text
tensor
module
expert
matrix
legal quantization tile
```

rather than blindly quantizing shard 1, then shard 2, then shard 3.

### 4.3 Separate I/O Granularity From GPU Granularity

Network reads and GPU tiles should not be the same thing.

PocketTitan should use three different granularities:

```text
Network / disk fetch chunk
        ↓
CPU staging buffer
        ↓
GPU quantization tile
```

Example:

```text
HTTP range fetch:     64–256 MiB
CPU staging buffer:   16–128 MiB
GPU tile:              1–32 MiB
```

General rule:

> **Fetch coarse, quantize fine.**

---

## 5. High-Level Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│ SOURCE MODEL                                                 │
│ Hugging Face Hub / local filesystem / object storage         │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Repository & Tensor Metadata Layer                        │
│ - model.safetensors.index.json                               │
│ - config.json                                                │
│ - per-shard Safetensors headers                              │
│ - tensor → shard mapping                                     │
│ - tensor → dtype / shape / byte offsets                      │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Range Planner & Virtual Tensor Streamer                   │
│ - coalesced HTTP range reads                                 │
│ - mmap for local Safetensors                                 │
│ - sliding source cache                                       │
│ - prefetching                                                │
│ - retry / resume                                             │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. CPU Staging & Memory Manager                              │
│ - bounded host buffers                                       │
│ - pinned memory                                              │
│ - optional NVMe-backed activation spool                      │
│ - tensor slicing                                             │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Work Planner & Memory Budget Scheduler                    │
│ Model → Layer → Module → Expert → Matrix → Legal Tile        │
│                                                              │
│ Chooses work size based on:                                  │
│ - available VRAM                                             │
│ - hard VRAM budget                                           │
│ - quantizer requirements                                     │
│ - tensor dimensions                                          │
│ - temporary memory multiplier                                │
│ - calibration state                                          │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Pluggable Quantization Engine                             │
│ RTN / HQQ / Ternary / INTx / GPTQ / AWQ / AutoRound / VPTQ  │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. Distortion & Sensitivity Evaluator                        │
│ - weight-space error                                         │
│ - activation-weighted output error                           │
│ - cosine similarity                                          │
│ - per-module sensitivity                                     │
│ - candidate precision comparison                             │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. Progressive Output Assembler                              │
│ - packed weights                                             │
│ - scales / zeros / codebooks                                 │
│ - quantization metadata                                      │
│ - output shard finalization                                  │
│ - resumable manifest                                         │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
               PocketTitan Checkpoint
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Runtime exporter   Runtime exporter
            vLLM          GGUF / others
              │                 │
      only when model architecture and format are supported
```

---

## 6. Repository and Tensor Metadata Layer

PocketTitan first inspects the source repository without downloading the complete model.

For a sharded Safetensors model:

```text
model.safetensors.index.json
```

normally provides:

```text
tensor name → shard filename
```

The per-shard Safetensors header then provides:

```text
tensor name
dtype
shape
data_offsets
```

PocketTitan builds a virtual address table:

```python
TensorAddress(
    name="model.layers.12.mlp.experts.gate_up_proj",
    shard="model-00018-of-00062.safetensors",
    dtype="...",
    shape=(288, 4096, 4096),
    byte_start=...,
    byte_end=...,
)
```

The repository index should not be assumed to contain raw byte offsets itself; offsets are derived from the Safetensors header.

---

## 7. Virtual Tensor Streamer

The streamer is responsible for obtaining source bytes.

### 7.1 Local Safetensors

```text
local shard
    ↓
mmap
    ↓
tensor range
    ↓
CPU staging
```

The entire shard should not be materialized into RAM.

### 7.2 Remote HTTP Range Fetching

```text
HTTP Range request
    ↓
coalesced contiguous bytes
    ↓
CPU staging
```

PocketTitan should avoid one HTTP request per GPU tile.

Nearby source ranges should be coalesced.

### 7.3 Sliding Source Cache

The source cache should contain only a bounded amount of source data.

```text
current source range / shard
next prefetched range
metadata
```

Once no remaining work depends on cached source data, it can be evicted.

### 7.4 Resume Support

Every completed tensor or work unit should be recorded in a job manifest.

```json
{
  "model.layers.12.expert.37.gate_up_proj": {
    "status": "done",
    "output_shard": "pockettitan-00014.safetensors",
    "quantizer": "hqq",
    "bits": 2
  }
}
```

A long-running job must resume without restarting from zero.

---

## 8. Memory Budget Scheduler

The scheduler receives a hard budget such as:

```text
--max-vram 3500MiB
```

and must avoid exceeding it.

The scheduler estimates:

```text
source tensor bytes
conversion bytes
quantizer temporary bytes
statistics bytes
output bytes
CUDA/runtime reserve
safety reserve
```

before scheduling work.

```python
usable_vram = configured_vram_budget - runtime_reserve - safety_margin
```

The quantizer reports an estimated working-memory requirement, and the scheduler selects the largest legal work unit.

---

## 9. Hierarchical Decomposition

PocketTitan can reduce the processing unit progressively:

```text
Model
  ↓
Layer
  ↓
Module
  ↓
MoE Expert
  ↓
Weight Matrix
  ↓
Row Group / Tile
```

Example:

```text
H100 80GB    → many experts concurrently
RTX 4090 24GB → expert-at-a-time
8GB GPU       → matrix-at-a-time
4GB GPU       → row groups / tiles
```

---

## 10. Quantizer Capability Contracts

PocketTitan must **not** assume that every quantization algorithm can operate on arbitrary tiles.

Each backend exposes capabilities.

```python
@dataclass
class QuantizerCapabilities:
    requires_calibration: bool
    legal_split_axes: tuple[str, ...]
    requires_full_input_dim: bool
    requires_full_output_dim: bool
    global_state: str | None
    supports_cpu: bool
    supports_cuda: bool
    supports_remote_streaming: bool
    estimated_workspace_multiplier: float
```

Example GPTQ-like backend:

```python
QuantizerCapabilities(
    requires_calibration=True,
    legal_split_axes=("out_features",),
    requires_full_input_dim=True,
    requires_full_output_dim=False,
    global_state="hessian",
    supports_cpu=True,
    supports_cuda=True,
    supports_remote_streaming=True,
)
```

The scheduler asks:

> **How may this quantizer legally decompose this matrix?**

rather than assuming all tensors can be sliced arbitrarily.

---

## 11. Preferred Matrix Tiling Direction

For a PyTorch linear weight:

```text
W = [out_features, in_features]
```

PocketTitan should generally prefer splitting over `out_features` while preserving the full input dimension.

```text
4096 × 4096

becomes:

256 × 4096
256 × 4096
256 × 4096
...
```

Advantages:

1. Row blocks are naturally contiguous in row-major storage.
2. Many Hessian-based methods can reuse the same input statistics across row blocks.
3. Network range planning is simpler.
4. GPU work units remain small.

Not every quantizer must use this layout, but it should be the default when mathematically legal.

---

## 12. Quantization Backends

### 12.1 Phase 1

#### RTN
Simple round-to-nearest baseline.

#### HQQ
Low-bit weight-only PTQ suitable for early streaming experiments.

#### Ternary / W1.58
Approximate weights using:

```text
{-1, 0, +1}
```

plus scaling.

#### Uniform INT2 / INT3 / INT4
Simple groupwise scalar baselines.

### 12.2 Later Activation-Aware Backends

#### GPTQ
Second-order / Hessian-aware quantization.

#### AWQ
Activation-aware channel scaling/protection.

#### AutoRound
Rounding optimization.

#### VPTQ
Vector post-training quantization.

### 12.3 Future Experimental Backends

Potential future directions:

```text
QuIP-style transforms
AQLM-style codebooks
RaBitQ-inspired weight-vector approaches
TurboQuant-inspired transforms
other vector / lattice quantizers
```

TurboQuant and RaBitQ are inspiration for future weight methods, not assumed drop-in weight quantizers.

---

## 13. Data-Free Quantization Path

This is the first full working pipeline.

```text
Repository metadata
        ↓
Task planner
        ↓
Fetch tensor bytes
        ↓
CPU staging
        ↓
Legal matrix / tile
        ↓
GPU or CPU
        ↓
RTN / HQQ / Ternary
        ↓
Pack result
        ↓
Write output
        ↓
Release source
```

No calibration dataset is required.

This path cleanly validates the 4GB architecture.

---

## 14. Activation-Aware Quantization Path

Activation-aware methods require model execution.

PocketTitan should later support this through **sequential activation propagation**.

```text
Calibration tokens
      ↓
Layer 0
      ↓
X1
      ↓
Layer 1
      ↓
X2
      ↓
...
```

For each layer:

```text
input activations Xn
       ↓
collect required statistics
       ↓
quantize current layer/module
       ↓
run quantized current layer
       ↓
produce Xn+1
       ↓
persist / spool Xn+1
       ↓
release current layer
```

Only the current execution block needs to be resident.

---

## 15. Activation Spool

Calibration activations can be large.

For BF16 hidden states:

\[
	ext{bytes} =
	ext{tokens} 	imes 	ext{hidden size} 	imes 2
\]

Activations should therefore not automatically remain on GPU.

```text
GPU transient batch
      ↓
CPU pinned memory
      ↓
optional mmap / NVMe spool
```

Only the current activation batch needs to move to GPU.

---

## 16. Statistics Reducer

Where possible, statistics should be accumulated online.

For a GPTQ-like Gram/Hessian approximation:

\[
H = X^T X
\]

Compute incrementally:

```python
H += X_batch.T @ X_batch
```

Other backends may accumulate:

```text
channel maxima
means
variances
quantiles
outlier statistics
```

---

## 17. MoE-Specific Optimization

Mixture-of-Experts models are a strong target.

Instead of viewing the model as:

```text
320B parameters
```

PocketTitan can often view the work as:

```text
one ~25M parameter expert
```

and then smaller matrices or row tiles inside that expert.

For many weight-only methods, experts can be treated as bounded micro-jobs.

---

## 18. MoE Calibration Reuse

Some input statistics can be shared across experts.

```text
MoE input X
   ↓
router
   ↓
expert gate/up projections
```

Input-side statistics for gate/up projections may be reusable across experts in a layer.

Down projections can require expert-specific statistics because their inputs depend on expert activations.

PocketTitan should explicitly model:

```text
shared calibration state
expert-local calibration state
```

---

## 19. Distortion Metrics

### 19.1 Weight Error

\[
D_W =
rac{\|W-Q(W)\|_F^2}
{\|W\|_F^2}
\]

### 19.2 Activation-Weighted Output Error

\[
D_X =
rac{\|XW-XQ(W)\|_F^2}
{\|XW\|_F^2}
\]

This is preferred when calibration activations are available.

### 19.3 Cosine Similarity

Compare original and quantized outputs.

### 19.4 Global Validation

Eventually evaluate:

```text
perplexity
reasoning benchmarks
coding benchmarks
long-context tests
task-specific evaluations
```

Local distortion alone is not enough.

---

## 20. Heterogeneous Precision Search

For each candidate module:

```text
Original W
   ├── ternary
   ├── INT2
   ├── HQQ-2
   ├── INT3
   ├── INT4
   ├── HQQ-4
   └── higher precision
```

A first version can choose the smallest representation where local error is below a threshold.

A more advanced version should solve:

\[
\min \sum_i Storage(Q_i)
\]

subject to:

\[
\sum_i s_i D_i(Q_i) \le E_{global}
\]

where:

- \(D_i\) = measured distortion
- \(s_i\) = module sensitivity
- \(E_{global}\) = model-wide error budget

PocketTitan therefore becomes a **precision allocator**, not only a quantization executor.

---

## 21. Progressive Output Assembly

PocketTitan does not mathematically merge quantized weights.

It progressively reconstructs the model parameter set:

```text
W1 → Q(W1) ───────┐
W2 → Q(W2) ───────┤
W3 → Q(W3) ───────┤
W4 → Q(W4) ───────┤
                  ▼
          output checkpoint
```

Each completed tensor or packed representation is persisted under its logical model identity.

---

## 22. Output Shard Builder

A standard Safetensors file contains metadata before tensor data, so an indefinitely append-only writer is not ideal.

PocketTitan should progressively build bounded output shards:

```text
current staging buffer
        ↓
accumulate quantized tensors
        ↓
reach target shard size
        ↓
construct metadata
        ↓
finalize shard
        ↓
start next shard
```

Suggested output shard size:

```text
2–5 GiB
```

configurable.

---

## 23. PocketTitan Canonical Checkpoint Format

Initial canonical representation:

```text
PocketTitan Safetensors
+
quantization metadata
+
model index
+
job manifest
```

Example:

```text
config.json
quantization_config.json
model.safetensors.index.json

pockettitan-00001-of-00024.safetensors
pockettitan-00002-of-00024.safetensors
...

pockettitan-manifest.json
```

Packed sub-byte weights can be represented as:

```text
uint8 packed payload
scale tensor
zero tensor
codebook tensor
metadata describing interpretation
```

Safetensors itself does not define arbitrary 1.58-bit, HQQ2, or VPTQ layouts; PocketTitan metadata does.

---

## 24. Runtime Exporters

Runtime export is separate from quantization.

```text
PocketTitan canonical model
          │
     ┌────┼────┐
     ▼    ▼    ▼
   vLLM  GGUF  other
```

An exporter is available only when:

1. the runtime supports the model architecture;
2. the runtime supports the quantization format;
3. the packed layout is compatible.

PocketTitan should not promise that arbitrary low-bit output automatically runs in llama.cpp, vLLM, Marlin, ExLlama, or another runtime.

---

## 25. Hard VRAM Enforcement

The 4GB claim must be measured.

PocketTitan should monitor:

```text
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
NVML memory usage
```

Example:

```bash
pockettitan quantize     zai-org/GLM-5.3-Flash     --quantizer hqq     --bits 2     --max-vram 3500MiB
```

If the planner predicts a work unit will violate the budget:

```text
split work unit further
```

If no legal decomposition exists:

```text
fail clearly before OOM
```

---

## 26. Memory Safety Strategy

Example 4GB policy:

```text
Configured hard limit       3584 MiB
CUDA/runtime reserve         700 MiB
Safety margin                500 MiB
Available working budget    ~2384 MiB
```

Exact numbers are measured dynamically.

PocketTitan should never target 100% of free VRAM.

---

## 27. Storage Model

### 27.1 Source Cache

Temporary original model bytes:

```text
one shard
or bounded HTTP range cache
```

Possible configured cache:

```text
5–20 GB
```

### 27.2 Final Output

The quantized model itself still needs storage.

Rough lower-bound example:

```text
320B × 2 bits ≈ 80 GB
```

before scales, metadata, and higher-precision tensors.

PocketTitan's claim is therefore:

> The full source model does not need to be stored locally at once.

It is not:

> A 320B model needs only 10GB total disk.

---

## 28. Source Precision Policy

PocketTitan explicitly records source precision.

```text
BF16 → INT2
FP16 → INT2
FP8  → INT2
BF16 → ternary
FP8  → ternary
```

These are different experiments.

For quality studies, BF16/FP16 → low-bit should be the clean reference when available.

FP8 → lower-bit is a **requantization experiment** and should be reported separately.

---

## 29. Job Planner

PocketTitan creates explicit work units.

```python
QuantizationTask(
    tensor_name="...",
    source_ranges=[...],
    logical_module="layer.12.expert.37.gate_up",
    quantizer="hqq",
    precision="2bit",
    split_axis=0,
    row_start=0,
    row_end=256,
)
```

Tasks can be:

```text
queued
running
completed
failed
retried
```

The job manifest makes work resumable and auditable.

---

## 30. Scheduling Modes

### 30.1 Storage-Sequential

Best for data-free methods:

```text
W1 → quantize → save
W2 → quantize → save
W3 → quantize → save
```

### 30.2 Model-Sequential

Required for activation-aware propagation:

```text
X0
 ↓
Layer 0
 ↓
X1
 ↓
Layer 1
 ↓
X2
```

Weights are still streamed, but work follows model execution dependencies.

---

## 31. Prefetching

I/O and compute should overlap where possible.

```text
GPU quantizes task N
        │
        └── meanwhile
            CPU/network fetches task N+1
```

Double-buffering:

```text
Buffer A → GPU
Buffer B → network / disk

swap

Buffer B → GPU
Buffer A → network / disk
```

This reduces the throughput penalty of small-VRAM execution.

---

## 32. CPU Backend

PocketTitan should support:

```text
device=cpu
```

from an early stage.

This provides:

1. a correctness reference;
2. an escape path for unsupported GPU operations;
3. the possibility of GPU-free quantization;
4. easier testing.

Project philosophy:

> **A GPU accelerates PocketTitan. A GPU does not define the maximum model size.**

---

## 33. Recommended Technology Stack

### Main Language
**Python**

### Tensor Runtime
**PyTorch**

### Checkpoint / Hub
```text
safetensors
huggingface_hub
```

### Model Metadata
```text
transformers
```

The data-free path should not require instantiating the full model as an `nn.Module`.

### Existing Quantization Infrastructure
Potentially reuse ideas/components from:

```text
LLM Compressor
HQQ
GPTQ implementations
AWQ implementations
VPTQ
```

while keeping PocketTitan's external-memory scheduler independent.

### Custom GPU Kernels
**Triton**, after correct PyTorch reference implementations exist.

Potential kernels:

```text
quantization
dequantization
packing
unpacking
statistics accumulation
fused scale + quantize + pack
```

### Native CUDA / C++
Only if Triton becomes limiting.

### CLI / Config
```text
Typer
Pydantic
```

### Testing
```text
pytest
```

### Profiling
```text
PyTorch profiler
NVML
torch.cuda memory statistics
```

---

## 34. Proposed Repository Structure

```text
pockettitan/
│
├── README.md
├── DESIGN.md
├── pyproject.toml
│
├── pockettitan/
│   ├── cli.py
│   ├── config.py
│   ├── manifest.py
│   │
│   ├── metadata/
│   │   ├── repo.py
│   │   ├── safetensors_header.py
│   │   └── tensor_index.py
│   │
│   ├── streaming/
│   │   ├── source.py
│   │   ├── http_range.py
│   │   ├── mmap.py
│   │   ├── cache.py
│   │   ├── prefetch.py
│   │   └── range_planner.py
│   │
│   ├── scheduler/
│   │   ├── budget.py
│   │   ├── planner.py
│   │   ├── tiler.py
│   │   └── capabilities.py
│   │
│   ├── quantizers/
│   │   ├── base.py
│   │   ├── rtn.py
│   │   ├── hqq.py
│   │   ├── ternary.py
│   │   ├── intx.py
│   │   ├── gptq.py
│   │   ├── awq.py
│   │   ├── autoround.py
│   │   └── vptq.py
│   │
│   ├── calibration/
│   │   ├── runner.py
│   │   ├── activation_spool.py
│   │   ├── statistics.py
│   │   └── moe_stats.py
│   │
│   ├── precision/
│   │   ├── distortion.py
│   │   ├── sensitivity.py
│   │   └── allocator.py
│   │
│   ├── kernels/
│   │   ├── reference.py
│   │   └── triton/
│   │       ├── quantize.py
│   │       ├── pack.py
│   │       └── statistics.py
│   │
│   ├── output/
│   │   ├── shard_builder.py
│   │   ├── metadata.py
│   │   └── checkpoint.py
│   │
│   ├── exporters/
│   │   ├── base.py
│   │   ├── vllm.py
│   │   └── gguf.py
│   │
│   └── models/
│       ├── generic.py
│       ├── moe.py
│       └── glm5.py
│
├── tests/
└── benchmarks/
```

---

## 35. GLM-5.3-Flash as the Primary Case Study

GLM-5.3-Flash is useful because it combines:

```text
very large total parameter count
MoE routing
hundreds of experts
large fraction of parameters inside experts
hybrid architecture
sharded checkpoint
```

Conceptually PocketTitan sees:

```text
320B global model
      ↓
one sparse layer
      ↓
one routed expert
      ↓
one expert matrix
      ↓
one row tile
```

---

## 36. GLM Quantization Strategy

The first GLM experiment should not immediately force every tensor to extreme precision.

Start with:

```text
routed expert weights
```

because they dominate storage.

Initially preserve sensitive structures at higher precision:

```text
routers
norms
attention state machinery
selected dense layers
embeddings
LM head
vision modules
```

Expand coverage only after quality measurements.

---

## 37. Milestone 0 — Metadata-Only Inspection

Build:

```bash
pockettitan inspect MODEL
```

Output:

```text
model architecture
source precision
number of shards
total tensor count
largest tensors
MoE structure
quantizable tensor count
estimated source size
estimated target sizes
available VRAM
recommended initial work unit
```

No full source tensor download is required.

---

## 38. Milestone 1 — Single Matrix Proof

Quantize one representative matrix.

Requirements:

```text
PyTorch reference RTN
PyTorch ternary
HQQ integration
hard VRAM measurement
roundtrip dequantization test
error metrics
```

Success:

```text
peak CUDA memory < configured budget
```

---

## 39. Milestone 2 — Micro-Tiler

Take a large matrix such as:

```text
16384 × 16384
```

and quantize it under a strict test budget such as:

```text
<2 GiB
```

Verify that tiled processing matches the untiled reference closely for algorithms where tiling is valid.

---

## 40. Milestone 3 — Safetensors Range Streamer

Implement:

```text
repository index parsing
Safetensors header parsing
remote byte-range reads
local mmap
range coalescing
sliding cache
resume manifest
```

Quantize a tensor without downloading its entire source shard where practical.

---

## 41. Milestone 4 — One GLM Expert

Process:

```text
gate/up
down
```

for one routed expert.

Test:

```text
RTN
HQQ
ternary
INT2
INT4
```

Measure:

```text
peak VRAM
RAM
disk cache
network bytes
quantization time
storage reduction
distortion
```

---

## 42. Milestone 5 — Entire MoE Layer

Process all routed experts sequentially while enforcing:

```text
<4 GiB CUDA memory
```

This validates expert-level external-memory processing.

---

## 43. Milestone 6 — Full Data-Free 320B Sweep

Process all target expert tensors across the complete model.

Requirements:

```text
resume support
progress manifest
output checkpoint assembly
bounded disk source cache
hard VRAM enforcement
failure recovery
```

This is the first major PocketTitan demonstration.

---

## 44. Milestone 7 — Activation Propagation

Add calibration data and sequential model execution.

Implement:

```text
activation spool
statistics reducer
GPTQ-compatible Hessian collection
AWQ statistics
model execution adapters
MoE shared statistics
```

---

## 45. Milestone 8 — Heterogeneous Precision Search

For each module:

```text
generate candidates
measure distortion
estimate storage
choose representation
```

Produce a per-module precision map.

---

## 46. Milestone 9 — Runtime-Compatible Export

Implement exporters only where the target inference runtime supports:

```text
model architecture
quantization type
packing format
```

PocketTitan's canonical checkpoint remains independent of runtime support.

---

## 47. Phase 1 Non-Goals

Phase 1 explicitly excludes:

```text
fine-tuning
LoRA
QLoRA
QAT
SFT
DPO
GRPO
RL
distributed training
custom inference engine
full BitNet training
```

The Phase 1 question is:

> **How far can external-memory PTQ push the minimum accelerator memory required to quantize gigantic models?**

---

## 48. Important Non-Claims

Do not claim before validation:

```text
"zero VRAM"
"all quantizers support arbitrary tiles"
"one exact model pass"
"10GB total disk for a 320B model"
"GGUF automatically runs every model"
"1.58-bit retains full quality"
"4GB quantization already works"
```

These remain hypotheses, goals, or backend-specific properties until benchmarked.

---

## 49. Success Criteria

### Architecture

- [ ] Quantization works without loading the complete model.
- [ ] Quantization works without loading a complete large layer where unnecessary.
- [ ] Source data can be streamed remotely or from bounded local cache.
- [ ] Output is assembled progressively.
- [ ] Jobs resume after interruption.
- [ ] Scheduler respects hard VRAM limits.

### 4GB Experiment

- [ ] Representative matrix quantized under 4 GiB.
- [ ] One full expert quantized under 4 GiB.
- [ ] One full MoE layer quantized under 4 GiB.
- [ ] Full-model expert sweep completed under 4 GiB.
- [ ] Peak VRAM independently recorded.
- [ ] Output checkpoint validated structurally.

### Quality

- [ ] Quantized tensors dequantize correctly.
- [ ] Local error metrics recorded.
- [ ] Activation-aware distortion measured where available.
- [ ] End-to-end inference quality measured when runtime is available.
- [ ] Mixed precision beats naive uniform precision at similar model size.

---

## 50. Research Questions

1. Can total model size be decoupled from accelerator residency during PTQ?
2. What is the minimum practical GPU memory required to quantize a 320B model?
3. How much throughput is lost when VRAM is reduced from 80GB → 24GB → 8GB → 4GB?
4. Which quantization algorithms remain mathematically valid under fine-grained streaming?
5. How small can GPU tiles become before I/O overhead dominates?
6. How should remote byte ranges be coalesced for large sharded checkpoints?
7. Can extreme low-bit expert quantization preserve MoE model capability?
8. Is heterogeneous precision significantly better than uniform precision?
9. Can per-module sensitivity predict end-to-end degradation?
10. Can a CPU-only system perform the same quantization given sufficient time and storage?

---

## 51. Performance Philosophy

PocketTitan intentionally trades:

```text
VRAM ↓
```

for:

```text
network I/O ↑
disk I/O ↑
CPU↔GPU transfers ↑
number of local work units ↑
wall-clock time ↑
```

This trade is acceptable.

The goal is to transform:

```text
"Impossible on this hardware"
```

into:

```text
"Possible, but slower"
```

---

## 52. Core Invariant

For supported quantization backends, peak GPU memory should be bounded by:

```text
largest legal quantization work unit
+
quantizer temporary state
+
calibration statistics
+
runtime reserve
```

rather than total model parameter count.

This is the core architectural invariant.

---

## 53. Final Vision

PocketTitan should eventually make this possible:

```bash
pockettitan quantize     zai-org/GLM-5.3-Flash     --quantizer auto     --target-bpw 2.0     --max-vram 3500MiB     --max-source-cache 12GiB     --stream
```

with the system automatically deciding:

```text
what to fetch
when to fetch it
how much to stage
how to tile it
which quantizer is legal
which precision to use
where to execute it
how to measure distortion
when source bytes can be deleted
where the output tensor belongs
```

The user sees a 320B model.

PocketTitan sees:

```text
a dependency graph of small bounded quantization jobs
```

---

## 54. Core Philosophy

Traditional approach:

> **Bring the model to the GPU.**

PocketTitan:

> **Bring only the bytes required for the current quantization operation to the GPU.**

Therefore:

> **A giant model should be a giant stream, not a giant allocation.**

That is PocketTitan.
