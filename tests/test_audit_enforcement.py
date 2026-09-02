"""Comprehensive verification suite for external audit requirements and architectural invariants."""

import pytest
import torch

from pockettitan.config import (
    CalibrationRequiredError,
    MemoryBudgetConfig,
    QuantConfig,
    QuantMethod,
    StorageAccounting,
    UnsupportedSourceDTypeError,
)
from pockettitan.models.adapters import (
    DeepSeekAdapter,
    GLM5NextAdapter,
    OLMoEAdapter,
    get_model_adapter,
)
from pockettitan.models.layout import (
    FusedExperts3DLayout,
    get_layout_adapter,
)
from pockettitan.quantizers import get_quantizer
from pockettitan.quantizers.base import QuantizerCapabilities
from pockettitan.scheduler.budget import compute_work_unit_bounds
from pockettitan.streaming.reader import RemoteTensorSliceReader


# ---------------------------------------------------------------------------
# P0.3: Model Adapter Hierarchy & Nested Config Parsing (GLM-5.3, DeepSeek, etc.)
# ---------------------------------------------------------------------------


def test_glm53_nested_config_parsing():
    """Verify GLM-5.3 nested text_config and vision_config topology extraction."""
    glm53_raw_config = {
        "architectures": ["GLM5NextForCausalLM"],
        "model_type": "glm_moe",
        "torch_dtype": "bfloat16",
        "vision_config": {
            "hidden_size": 1152,
            "num_hidden_layers": 27,
        },
        "text_config": {
            "hidden_size": 4096,
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "first_k_dense_replace": 3,
            "vocab_size": 151552,
            "torch_dtype": "bfloat16",
        },
    }

    adapter = get_model_adapter(glm53_raw_config)
    assert isinstance(adapter, GLM5NextAdapter)

    dims = adapter.extract_dimensions()
    assert dims["hidden_size"] == 4096
    assert dims["num_hidden_layers"] == 48
    assert dims["num_attention_heads"] == 32
    assert dims["num_key_value_heads"] == 4
    assert dims["vocab_size"] == 151552

    moe = adapter.extract_moe_topology()
    assert moe["is_moe"] is True
    assert moe["num_experts"] == 128
    assert moe["num_experts_per_tok"] == 8
    assert moe["expert_intermediate_size"] == 2048
    assert moe["first_k_dense_replace"] == 3

    dtype_str, is_fp8 = adapter.extract_source_dtype()
    assert is_fp8 is False
    assert "bfloat16" in dtype_str


def test_deepseek_and_moe_adapters():
    """Verify DeepSeek-V3 and Qwen MoE adapter routing."""
    deepseek_cfg = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "torch_dtype": "float8_e4m3fn",
    }
    adapter = get_model_adapter(deepseek_cfg)
    assert isinstance(adapter, DeepSeekAdapter)
    assert adapter.is_moe_architecture() is True

    dtype_str, is_fp8 = adapter.extract_source_dtype()
    assert is_fp8 is True

    moe = adapter.extract_moe_topology()
    assert moe["num_experts"] == 256
    assert moe["first_k_dense_replace"] == 3
    assert moe["shared_expert_intermediate_size"] == 2048


def test_olmoe_uses_checkpoint_values_instead_of_deepseek_defaults():
    config = {
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "intermediate_size": 1024,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "vocab_size": 50304,
        "torch_dtype": "bfloat16",
    }
    adapter = get_model_adapter(config)
    assert isinstance(adapter, OLMoEAdapter)
    assert adapter.extract_dimensions()["num_hidden_layers"] == 16
    assert adapter.extract_dimensions()["hidden_size"] == 2048
    assert adapter.extract_moe_topology() == {
        "is_moe": True,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "expert_intermediate_size": 1024,
        "shared_expert_intermediate_size": None,
        "first_k_dense_replace": None,
    }


# ---------------------------------------------------------------------------
# P0.4: Strict UnsupportedSourceDTypeError & FP8 Support
# ---------------------------------------------------------------------------


def test_strict_unsupported_source_dtype():
    """Ensure UnsupportedSourceDTypeError is strictly raised on unknown dtypes."""
    with pytest.raises(UnsupportedSourceDTypeError):
        RemoteTensorSliceReader._bytes_to_tensor(
            raw_bytes=b"\x00\x00",
            dtype_str="UNKNOWN_COMPLEX_DTYPE",
            shape=[1, 1],
        )


def test_fp8_conversion_support():
    """Ensure FP8 source dtypes convert cleanly to torch tensors."""
    # 4 bytes = 4 float8 elements
    dummy_fp8_bytes = bytes([0x3C, 0x40, 0x44, 0x48])
    t = RemoteTensorSliceReader._bytes_to_tensor(
        raw_bytes=dummy_fp8_bytes,
        dtype_str="FLOAT8_E4M3FN",
        shape=[2, 2],
    )
    assert t.shape == torch.Size([2, 2])
    assert t.dtype == torch.float16  # Auto-promoted for computation


# ---------------------------------------------------------------------------
# P0.5: Fused 3-D MoE Expert Layout
# ---------------------------------------------------------------------------


def test_fused_3d_expert_layout():
    """Verify FusedExperts3DLayout correctly extracts individual 2-D expert matrices."""
    shape_3d = [64, 2048, 4096]  # [num_experts, out_features, in_features]
    layout = get_layout_adapter("model.layers.0.mlp.experts.gate_proj.weight", shape_3d)

    assert isinstance(layout, FusedExperts3DLayout)
    assert layout.get_num_subunits() == 64
    assert layout.get_subunit_shape(0) == [2048, 4096]

    dummy_bank = torch.randn(64, 2048, 4096, dtype=torch.float16)
    expert_0 = layout.extract_subunit_tensor(dummy_bank, 0)
    assert expert_0.shape == torch.Size([2048, 4096])
    assert torch.equal(expert_0, dummy_bank[0])


# ---------------------------------------------------------------------------
# P0.8 & P1.1: Quantizer Capabilities & Calibration Safety Contracts
# ---------------------------------------------------------------------------


def test_quantizer_capabilities_contracts():
    """Verify all quantizer backends declare complete QuantizerCapabilities contracts."""
    for method in [
        QuantMethod.RTN,
        QuantMethod.HQQ,
        QuantMethod.TERNARY,
        QuantMethod.INTX,
        QuantMethod.GPTQ,
        QuantMethod.AWQ,
        QuantMethod.AUTOROUND,
    ]:
        cfg = QuantConfig(method=method, bits=2, group_size=128)
        quantizer = get_quantizer(cfg)
        caps = quantizer.capabilities

        assert isinstance(caps, QuantizerCapabilities)
        assert isinstance(caps.name, str)
        assert isinstance(caps.requires_calibration, bool)
        assert isinstance(caps.legal_split_axes, tuple)
        assert isinstance(caps.supports_cpu, bool)
        assert isinstance(caps.supports_cuda, bool)
        assert isinstance(caps.workspace_multiplier, float)
        assert caps.workspace_multiplier >= 1.0


def test_calibration_required_fail_safe():
    """Verify quantizers requiring calibration raise CalibrationRequiredError when data is missing."""
    w = torch.randn(64, 128, dtype=torch.float16)

    # GPTQ without Hessian must fail
    gptq = get_quantizer(QuantConfig(method=QuantMethod.GPTQ, bits=4, group_size=64))
    with pytest.raises(CalibrationRequiredError):
        gptq.quantize(w, hessian=None)

    # AWQ without Hessian/activations must fail
    awq = get_quantizer(QuantConfig(method=QuantMethod.AWQ, bits=4, group_size=64))
    with pytest.raises(CalibrationRequiredError):
        awq.quantize(w, hessian=None)

    # AutoRound without Hessian/activations must fail
    autoround = get_quantizer(QuantConfig(method=QuantMethod.AUTOROUND, bits=4, group_size=64))
    with pytest.raises(CalibrationRequiredError):
        autoround.quantize(w, hessian=None)


# ---------------------------------------------------------------------------
# P1.5: Scientific Storage Accounting
# ---------------------------------------------------------------------------


def test_scientific_storage_accounting_ternary_and_int2():
    """Verify precise separation of theoretical entropy, payload bits, metadata overhead, and on-disk size."""
    shape = [4096, 4096]

    # Ternary: Theoretical = log2(3) ~ 1.585, Payload = 2.0
    ternary_acc = StorageAccounting.compute(
        method=QuantMethod.TERNARY,
        bits=2,
        group_size=128,
        shape=shape,
        has_zeros=False,
    )
    assert abs(ternary_acc.theoretical_bpw - 1.585) < 0.001
    assert ternary_acc.payload_bpw == 2.0
    assert ternary_acc.metadata_bpw > 0.0
    assert ternary_acc.on_disk_bpw > ternary_acc.payload_bpw
    assert ternary_acc.compression_ratio > 7.0  # >7x vs FP16

    # HQQ 2-bit with scales & zeros
    hqq_acc = StorageAccounting.compute(
        method=QuantMethod.HQQ,
        bits=2,
        group_size=128,
        shape=shape,
        has_zeros=True,
    )
    assert hqq_acc.theoretical_bpw == 2.0
    assert hqq_acc.payload_bpw == 2.0
    assert hqq_acc.metadata_bpw == round((4096 * 4096 / 128 * 4 * 8) / (4096 * 4096), 4)  # 0.25 bpw
    assert hqq_acc.on_disk_bpw == 2.25
    assert hqq_acc.compression_ratio == round(16.0 / 2.25, 2)


# ---------------------------------------------------------------------------
# P0.6: Work Unit Bounds & Tiling Invariant
# ---------------------------------------------------------------------------


def test_work_unit_bounds_strictly_respects_budget():
    """Verify matrix work unit decomposition calculates tile row bounds properly."""
    budget = MemoryBudgetConfig(
        max_vram_mb=2048.0, runtime_reserve_mb=256.0, safety_margin_mb=256.0
    )
    quant_cfg = QuantConfig(method=QuantMethod.HQQ, bits=2, group_size=128)

    # Large 16K x 16K matrix (~512 MB raw FP16, ~2.5 GB workspace)
    bounds = compute_work_unit_bounds(
        matrix_shape=[16384, 16384],
        budget=budget,
        quant_config=quant_cfg,
        source_dtype="float16",
        workspace_multiplier=5.0,
    )

    assert bounds["needs_tiling"] is True
    assert bounds["num_tiles"] > 1
    assert bounds["tile_rows"] < 16384
    assert bounds["estimated_vram_per_tile_mb"] <= budget.usable_vram_mb
