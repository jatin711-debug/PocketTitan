"""Unit tests for sensitivity profiling and Pareto bit allocation."""


from pockettitan.config import TensorAddress
from pockettitan.precision.allocator import ParetoBitAllocator
from pockettitan.precision.sensitivity import compute_tensor_sensitivity


def test_sensitivity_scoring():
    norm_tensor = TensorAddress(
        name="model.layers.0.input_layernorm.weight",
        shard="s1",
        dtype="F16",
        shape=[4096],
        byte_start=0,
        byte_end=8192,
        num_params=4096,
        size_bytes=8192,
    )
    router_tensor = TensorAddress(
        name="model.layers.0.mlp.gate.weight",
        shard="s1",
        dtype="F16",
        shape=[64, 4096],
        byte_start=0,
        byte_end=524288,
        num_params=262144,
        size_bytes=524288,
    )
    expert_tensor = TensorAddress(
        name="model.layers.0.mlp.experts.0.gate_proj.weight",
        shard="s1",
        dtype="F16",
        shape=[1408, 4096],
        byte_start=0,
        byte_end=11534336,
        num_params=5767168,
        size_bytes=11534336,
    )

    s_norm = compute_tensor_sensitivity(norm_tensor)
    s_router = compute_tensor_sensitivity(router_tensor)
    s_exp = compute_tensor_sensitivity(expert_tensor)

    assert s_norm.sensitivity > s_router.sensitivity > s_exp.sensitivity
    assert s_norm.recommended_min_bits == 16


def test_pareto_bit_allocator():
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
            byte_start=0,
            byte_end=33554432,
            num_params=16777216,
            size_bytes=33554432,
        ),
        TensorAddress(
            name="model.layers.0.mlp.gate.weight",
            shard="s1",
            dtype="F16",
            shape=[64, 4096],
            byte_start=0,
            byte_end=524288,
            num_params=262144,
            size_bytes=524288,
        ),
        TensorAddress(
            name="model.layers.0.mlp.experts.0.gate_proj.weight",
            shard="s1",
            dtype="F16",
            shape=[1408, 4096],
            byte_start=0,
            byte_end=11534336,
            num_params=5767168,
            size_bytes=11534336,
        ),
    ]

    allocator = ParetoBitAllocator(target_bpw=2.2)
    pmap = allocator.solve(model_id_or_path="test_model", tensor_addresses=tensors)

    assert pmap.effective_bpw <= 3.0
    assert pmap.tensor_quant_configs["model.layers.0.input_layernorm.weight"].bits == 16
    assert pmap.tensor_quant_configs["model.layers.0.mlp.experts.0.gate_proj.weight"].bits == 2
