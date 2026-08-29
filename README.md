# 🚀 PocketTitan

> **Tagline:** *Model size should determine time and storage — not how much VRAM you need.*

[![PyPI Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/your-username/PocketTitan)
[![Status](https://img.shields.io/badge/status-active%20development%20%2F%20experimental-orange.svg)]()
[![Usage](https://img.shields.io/badge/not%20for%20production-research%20%26%20testing%20only-red.svg)]()
[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> [!WARNING]
> **Active Development & Testing Preview — Not for Production Use**  
> PocketTitan is currently under active experimental research and development. APIs, internal interfaces, and storage formats are subject to rapid iteration. Use for research, experimentation, and benchmarking only.

**PocketTitan** is an external-memory, post-training quantization engine designed to quantize extreme-scale Large Language Models (such as **Qwen3.6-27B-FP8**, **GLM-5.3-Flash 320B**, and **DeepSeek-V3 671B**) under a **strict peak CUDA VRAM budget of $< 3.5\text{ GiB}$** on standard consumer laptop GPUs (e.g. NVIDIA RTX 3050 Laptop 4GB).

---

## 🎯 The Central Hypothesis

> **VRAM requirements should scale with the size of the current quantization working set, not with the total parameter count of the model.**

A 27B, 320B, or 671B model does not need to exist in GPU memory or hard disk all at once. PocketTitan treats large neural networks as external-memory streams:
```
Virtual Tensor Index ──► HTTP Range / Zero-Copy Slice ──► Micro-Tiled Quantization ──► Pinned Host Staging ──► Multi-Shard Safetensors / GGUF / vLLM
```

---

## ⚡ Key Architectural Features

1. **Remote Virtual Header Indexing:**
   - Direct HTTP byte-range header parsing (`Range: bytes=0-131071`) with CDN 302/307 redirect preservation.
   - Dissects 160+ remote Safetensors shards and **91,000+ tensors in $< 3\text{ seconds}$** without downloading model weights.
2. **Memory-Bounded Micro-Tiler:**
   - Automatically decomposes arbitrarily large matrices (e.g. $16384 \times 16384$) into legal row tiles bounded by hardware profile formulas:
     $$\text{Tile Rows } T_r = \min\left(M, \left\lfloor \frac{\text{Usable VRAM Bytes}}{K \cdot (S_{\text{src}} + M_{\text{ws}})} \right\rfloor\right)$$
   - Mathematically verified: quantized a $16384 \times 16384$ ($512\text{MB}$ FP16) matrix under a **$1500\text{MB}$ VRAM cap** with peak memory strictly at **$1065\text{MB}$** and **$1.000000$ Cosine Similarity**.
3. **MoE Structural Granularity:**
   - Dedicated parsers for routed experts, shared experts, and router gate logits across DeepSeek-V3, GLM-5.3-Flash, Qwen-MoE, and Mixtral.
   - Quantized all 3 projection matrices of a DeepSeek-V3 expert ($44.04\text{M}$ params) in **$310.62\text{MB}$ peak VRAM**.
   - Zero monotonic memory leaks verified over sequential expert sweeps.
4. **Pluggable Quantizer Backends:**
   - **HQQ (Half-Quadratic Quantization):** Optimization-based weight-only PTQ with proximal coordinate descent for 1b, 2b, 3b, 4b, 8b with automatic group padding for arbitrary layer dimensions.
   - **BitNet 1.58b (Ternary):** Dynamic scaling $S = \text{mean}(|W|)$ and thresholding to $\{-1, 0, +1\}$ packed into 2 bits.
   - **GPTQ:** Second-order Cholesky column updates using the inverse Hessian $H^{-1}$ sliced over micro-tiles.
   - **AWQ (Activation-Weighted Quantization):** Salient channel protection via grid search scaling.
   - **AutoRound:** Sign-gradient descent weight rounding optimization.
   - **RTN & INTx:** Vectorized uniform groupwise round-to-nearest baselines with sub-byte bit packing.
5. **Pareto Bit-Allocation Solver:**
   - Multi-choice Lagrangian optimizer allocating per-tensor bit-widths ($1.58\text{b}, 2\text{b}, 3\text{b}, 4\text{b}, 8\text{b}, 16\text{b}$) based on module sensitivity scores.
6. **Inference Runtime Exporters:**
   - **GGUF Exporter:** Native binary GGUF v3 generation with canonical naming, hyperparameter injection, and 32-byte alignment for `llama.cpp`.
   - **vLLM / SGLang Exporter:** Multi-shard Safetensors emission with standard `quantization_config` metadata.
7. **Transactional Crash Recovery:**
   - Atomic `JobManifest` tracking per-tensor completion state with instant resume capabilities.

---

## 🛠️ CLI Quickstart & Usage Guide

### 1. Inspect Model Architecture & Work Unit Bounds
Inspect remote Hugging Face models without downloading weight files:
```bash
pockettitan inspect Qwen/Qwen3.6-27B-FP8 --max-vram 3500MB
```

### 2. Stream and Quantize Real-World Models
Stream, quantize, and write Safetensors shards with automatic memory bounding (e.g. quantizing 27B model in 2-bit under 3.5GB VRAM):
```bash
pockettitan quantize Qwen/Qwen3.6-27B-FP8 --output-dir ./qwen_2bit_model --method hqq --bits 2 --max-vram 3500MB
```

### 3. Validate Checkpoint Integrity
Audit the structural and mathematical validity of the output checkpoint:
```bash
pockettitan validate ./qwen_2bit_model
```

### 4. Export to GGUF (llama.cpp) or vLLM / SGLang
Export the quantized model for runtime execution:
```bash
# Export to GGUF (for llama.cpp / Ollama / LM Studio)
pockettitan export ./qwen_2bit_model --format gguf --output ./qwen3.6_2bit.gguf

# Export to vLLM / SGLang
pockettitan export ./qwen_2bit_model --format vllm --output ./qwen3.6_2bit_vllm
```

### 5. Inspect Layers on GPU
Inspect and verify dequantized layer numerical fidelity in real time on CUDA:
```bash
pockettitan inspect-layer ./qwen_2bit_model --layer 0
```

---

## 🏃 How to Run the Quantized Outputs

### Method A: Quick Test Script (`test_model.py`)
Run the included test script to verify tokenizer encoding/decoding and CUDA layer dequantization:
```bash
python test_model.py
```

### Method B: Question Answering & Text Generation (`ask_model.py`)
Run question answering using the quantized model and chat template:
```bash
python ask_model.py
```

### Method C: Serve with vLLM / SGLang
```bash
vllm serve ./qwen3.6_2bit_vllm --gpu-memory-utilization 0.85
```

---

## 🧪 Benchmark & Quality Summary

| Model / Benchmark | Configuration | Peak CUDA VRAM | Effective Size | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen3.6-27B-FP8 (474 Tensors)** | HQQ 2-bit, Group 128 | **2.85 GiB** (Cap: 3500MB) | **$3.19\text{ GiB}$ (from $27\text{GB}+$ raw)** | **PASS (100%)** |
| **$16384 \times 16384$ Matrix** | HQQ 2-bit, Group 128 | **1065 MiB** (Cap: 1500 MiB) | $9.31\text{ dB}$ / $1.000000$ | **PASS** |
| **DeepSeek-V3 Expert (44M params)** | HQQ 2-bit, Group 128 | **310 MiB** (Cap: 3584 MiB) | $9.31\text{ dB}$ / $0.940929$ | **PASS** |
| **MoE 32-Expert Sweep** | HQQ 2-bit, Sequential | **310 MiB** (Zero leak) | Stable | **PASS** |
| **Qwen1.5-MoE Pareto Search** | Target: $2.20\text{ bpw}$ | N/A (Analytical) | **$2.09\text{ bpw}$ (7.66x)** | **PASS** |

---

## 🏗️ Project Architecture

```text
pockettitan/
├── config.py                 # Core Pydantic configs (MemoryBudget, QuantConfig, TensorAddress)
├── cli.py                    # Rich CLI interface (inspect, test-matrix, quantize-expert, optimize-precision, quantize, validate, export, inspect-layer)
├── inference.py              # Native Python model runner and layer inspector
├── manifest.py               # Transactional JSON execution manifest and crash recovery
├── metadata/
│   ├── safetensors_header.py # 8-byte uint64 + JSON range parser with CDN redirect preservation
│   ├── repo.py               # Hugging Face config & index parser + MoE dimension extractor
│   └── tensor_index.py       # Virtual TensorAddressTable builder with multi-threaded shard probes
├── quantizers/
│   ├── base.py               # BaseQuantizer & QuantizerCapabilities contract
│   ├── rtn.py                # Symmetric / Asymmetric groupwise RTN quantizer with automatic padding
│   ├── ternary.py            # BitNet 1.58b / Ternary quantizer with 2-bit packing
│   ├── intx.py               # Uniform INT2 / INT3 / INT4 / INT8 quantizers
│   ├── hqq.py                # Half-Quadratic Quantization proximal coordinate descent optimizer
│   ├── gptq.py               # Second-order Cholesky column updates with inverse Hessian
│   ├── awq.py                # Activation-Weighted Quantization protecting salient channels
│   └── autoround.py          # Sign-gradient descent weight rounding optimization
├── scheduler/
│   ├── budget.py             # Hardware scanner and exact per-row VRAM budget calculator
│   └── tiler.py              # Memory-bounded Micro-Tiler with automatic row chunking and Hessian passing
├── streaming/
│   ├── reader.py             # Local zero-copy memory-mapped reader & Remote HTTP Range streamer (FP8/BF16/FP16)
│   └── ring_buffer.py        # Pinned host memory buffer pool for async GPU DMA transfers
├── models/
│   ├── generic.py            # Generic transformer layer analyzer
│   └── moe.py                # MoE layer parser (router gate, routed experts, shared experts)
├── pipeline/
│   └── layer_pipeline.py     # End-to-end streaming quantizer and multi-shard Safetensors writer
├── calibration/
│   ├── dataset.py            # Tokenized calibration dataset loader
│   ├── hessian.py            # Online second-order activation Hessian accumulator & outlier detector
│   ├── spool.py              # Pinned host buffer with disk overflow for inter-layer activation chaining
│   └── moe_stats.py          # MoE router token dispatcher and per-expert Hessian tracking
├── precision/
│   ├── distortion.py         # Frobenius norm error, SNR (dB), and Cosine Similarity metrics
│   ├── sensitivity.py        # Architectural sensitivity profiler for layer/expert modules
│   └── allocator.py          # Multi-choice Pareto Lagrangian bit-width allocation solver
├── export/
│   └── validator.py          # CheckpointValidator integrity scorecard
└── exporters/
    ├── base.py               # Base exporter interface
    ├── gguf.py               # GGUF v3 binary writer for llama.cpp with canonical naming & metadata injection
    └── vllm.py               # vLLM / SGLang compatible checkpoint emitter
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
