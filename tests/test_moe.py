"""Unit tests for MoE layer parsing and sequential expert quantization stability."""

import torch

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod, TensorAddress
from pockettitan.models.moe import parse_moe_layer_structure
from pockettitan.quantizers import get_quantizer
from pockettitan.scheduler.tiler import MatrixTiler


def test_parse_moe_layer_structure():
    tensors = [
        TensorAddress(
            name="model.layers.0.input_layernorm.weight",
            shard="s1",
            dtype="F16",
            shape=[4096],
            byte_start=0,
            byte_end=8192,
            num_params=4096,
            size_bytes=8192,
        ),
        TensorAddress(
            name="model.layers.0.self_attn.q_proj.weight",
            shard="s1",
            dtype="F16",
            shape=[4096, 4096],
            byte_start=8192,
            byte_end=33562624,
            num_params=16777216,
            size_bytes=33554432,
        ),
        TensorAddress(
            name="model.layers.0.mlp.gate.weight",
            shard="s1",
            dtype="F16",
            shape=[64, 4096],
            byte_start=33562624,
            byte_end=34086912,
            num_params=262144,
            size_bytes=524288,
        ),
        TensorAddress(
            name="model.layers.0.mlp.experts.0.gate_proj.weight",
            shard="s1",
            dtype="F16",
            shape=[1408, 4096],
            byte_start=34086912,
            byte_end=45621248,
            num_params=5767168,
            size_bytes=11534336,
        ),
        TensorAddress(
            name="model.layers.0.mlp.experts.0.up_proj.weight",
            shard="s1",
            dtype="F16",
            shape=[1408, 4096],
            byte_start=45621248,
            byte_end=57155584,
            num_params=5767168,
            size_bytes=11534336,
        ),
        TensorAddress(
            name="model.layers.0.mlp.experts.0.down_proj.weight",
            shard="s1",
            dtype="F16",
            shape=[4096, 1408],
            byte_start=57155584,
            byte_end=68689920,
            num_params=5767168,
            size_bytes=11534336,
        ),
        TensorAddress(
            name="model.layers.0.mlp.shared_experts.gate_proj.weight",
            shard="s1",
            dtype="F16",
            shape=[1408, 4096],
            byte_start=68689920,
            byte_end=80224256,
            num_params=5767168,
            size_bytes=11534336,
        ),
    ]

    struct = parse_moe_layer_structure(layer_idx=0, layer_tensors=tensors)
    assert struct.input_layernorm is not None
    assert struct.attention.q_proj is not None
    assert struct.router_gate is not None
    assert struct.shared_experts is not None
    assert 0 in struct.routed_experts
    assert struct.routed_experts[0].gate_proj is not None
    assert struct.routed_experts[0].down_proj is not None


def test_moe_sequential_expert_sweep_memory_stability():
    """Verify that looping across 32 experts in sequence does not leak memory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    budget = MemoryBudgetConfig(max_vram_mb=3584.0)
    quant_cfg = QuantConfig(method=QuantMethod.HQQ, bits=2, group_size=128, device=device)
    quantizer = get_quantizer(quant_cfg)
    tiler = MatrixTiler(budget)

    num_experts = 32
    intermediate_size = 1024
    hidden_size = 2048

    peak_memories = []

    for e in range(num_experts):
        w_gate = (
            torch.randn(intermediate_size, hidden_size, dtype=torch.float16, device="cpu") * 0.02
        )
        w_up = torch.randn(intermediate_size, hidden_size, dtype=torch.float16, device="cpu") * 0.02
        w_down = (
            torch.randn(hidden_size, intermediate_size, dtype=torch.float16, device="cpu") * 0.02
        )

        q_gate, p1 = tiler.quantize_matrix(w_gate, quantizer=quantizer, target_device=device)
        q_up, p2 = tiler.quantize_matrix(w_up, quantizer=quantizer, target_device=device)
        q_down, p3 = tiler.quantize_matrix(w_down, quantizer=quantizer, target_device=device)

        peak_memories.append(max(p1, p2, p3))
        del w_gate, w_up, w_down, q_gate, q_up, q_down

    if device == "cuda":
        # Check that peak memory for expert #31 is identical or very close to expert #0 (no monotonic memory growth)
        assert abs(peak_memories[-1] - peak_memories[0]) < 10.0, (
            "Detected memory accumulation across MoE iterations"
        )
        assert max(peak_memories) <= budget.max_vram_mb
