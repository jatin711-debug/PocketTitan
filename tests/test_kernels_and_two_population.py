"""Tests for Phase R9 Hardware-Accelerated Kernels and Two-Population Expert Allocation."""

import pytest
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.precision.two_population import TwoPopulationAllocator
from pockettitan.quantizers import get_quantizer
from pockettitan.runtime.kernels.codes import centred_weights
from pockettitan.runtime.kernels.cpu_lut import LUTQuantizedLinear
from pockettitan.runtime.kernels.cuda_fused import FusedDequantGEMV
from pockettitan.runtime.kernels.gdn_blas import GDNRecurrenceBLAS


@pytest.mark.parametrize("bits", [4, 2])
@pytest.mark.parametrize("zeros_given", [False, True])
def test_lut_gemv_equals_dequantize_then_matmul(bits, zeros_given):
    """The LUT decomposition must equal materializing the weights and multiplying.

    The reference is built from :func:`centred_weights` — the same unpacking the
    packer's inverse uses. The previous version of this test re-implemented the
    kernel's own ``code - 8`` arithmetic as its "reference", so it agreed with
    the kernel while both disagreed with every package on disk.
    """
    batch_size, in_features, out_features, group_size = 2, 128, 64, 64
    num_groups = in_features // group_size

    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, dtype=torch.float32)
    packed_w = torch.randint(0, 256, (out_features, in_features // (8 // bits)), dtype=torch.uint8)
    scales = torch.randn(out_features, num_groups, dtype=torch.float32) * 0.05
    zeros = (
        torch.randint(0, 1 << bits, (out_features, num_groups)).float() if zeros_given else None
    )

    kernel = (
        LUTQuantizedLinear.forward_int4_gemv if bits == 4
        else LUTQuantizedLinear.forward_int2_gemv
    )
    got = kernel(x=x, packed_weights=packed_w, scales=scales, zeros=zeros, group_size=group_size)

    w_ref = centred_weights(packed_w, bits, in_features, group_size, zeros)
    w_ref = (w_ref * scales.view(out_features, num_groups, 1)).view(out_features, in_features)
    assert torch.allclose(got, torch.matmul(x, w_ref.t()), atol=1e-4)


@pytest.mark.parametrize("zeros_given", [False, True])
def test_fused_fma_equals_dequantize_then_matmul(zeros_given):
    batch_size, in_features, out_features, group_size = 1, 256, 128, 128
    num_groups = in_features // group_size

    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, dtype=torch.float32)
    packed_w = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    scales = torch.randn(out_features, num_groups, dtype=torch.float32) * 0.02
    zeros = torch.randint(0, 16, (out_features, num_groups)).float() if zeros_given else None

    got = FusedDequantGEMV.forward_int4_fma(
        x=x, packed_weights=packed_w, scales=scales, zeros=zeros, group_size=group_size
    )

    w_ref = centred_weights(packed_w, 4, in_features, group_size, zeros)
    w_ref = (w_ref * scales.view(out_features, num_groups, 1)).view(out_features, in_features)
    assert torch.allclose(got, torch.matmul(x, w_ref.t()), atol=1e-4)


def test_gdn_blas_linear_recurrence():
    """Verify GDNRecurrenceBLAS maintains state across time steps and produces correct projections."""
    num_heads = 4
    head_dim = 16
    
    gdn = GDNRecurrenceBLAS(num_heads=num_heads, head_dim=head_dim, device="cpu", dtype=torch.float32)
    
    q = torch.randn(num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(num_heads, head_dim, dtype=torch.float32)
    v = torch.randn(num_heads, head_dim, dtype=torch.float32)
    beta = torch.full((num_heads, 1, 1), 0.95, dtype=torch.float32)

    # Step 1
    out1 = gdn.step(q, k, v, beta)
    assert out1.shape == torch.Size([num_heads, head_dim])
    assert not torch.allclose(out1, torch.zeros_like(out1))

    # Reference step 1: S1 = 0 * beta + v @ k.T = v @ k.T
    expected_S1 = torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))
    assert torch.allclose(gdn.state, expected_S1, atol=1e-5)

    # Step 2
    out2 = gdn.step(q, k, v, beta)
    expected_S2 = 0.95 * expected_S1 + expected_S1
    assert torch.allclose(gdn.state, expected_S2, atol=1e-4)
    # The recurrence carried state forward, so the same input reads out differently.
    assert out2.shape == out1.shape
    assert torch.isfinite(out2).all()
    assert not torch.allclose(out1, out2, atol=1e-6)


def test_two_population_expert_allocator():
    """Verify two-population allocator partitions 4-bit hot head and 2-bit cold tail."""
    allocator = TwoPopulationAllocator(
        num_layers=48,
        num_experts=512,
        hot_head_ratio=0.20,
        hot_bits=4.0,
        cold_bits=2.0,
    )

    # Generate synthetic frequency skew
    freqs = {}
    for layer in range(48):
        for e in range(512):
            freqs[(layer, e)] = (layer + 1) * (512 - e)  # higher frequency for early experts

    plan = allocator.allocate(freqs)

    assert plan.total_experts == 48 * 512  # 24,576 experts
    assert plan.hot_expert_count == int(24576 * 0.20)  # 4,915 hot experts
    assert plan.cold_expert_count == 24576 - 4915      # 19,661 cold experts
    assert len(plan.hot_population) == 4915
    assert len(plan.cold_population) == 19661
    assert plan.hot_population.isdisjoint(plan.cold_population)

    # Substantial compression relative to uniform 4-bit
    assert plan.estimated_bank_bytes < plan.uniform_4bit_bytes
    assert plan.compression_ratio > 1.5  # ~1.67x compression


# --------------------------------------------------------------------------- #
# Kernels vs. the format they are supposed to consume
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bits,symmetric", [(4, False), (4, True), (2, False), (2, True)])
def test_kernels_agree_with_the_packers_own_dequantization(bits, symmetric):
    """A kernel's arithmetic must match the packer's, not merely look plausible.

    Both kernels previously hardcoded ``code - 8`` (4-bit) and ``code - 2``
    (2-bit) and accepted no zero-point. RTN centres symmetric codes on
    ``max_int // 2`` — 7 and 1 — so every weight was biased by one code, and
    ``PrecisionEntry.symmetric`` defaults to ``False``, so real packages carry a
    per-group zero-point that a constant cannot express.

    The reference here is ``decode_record``: the writer's own inverse.
    """
    in_features, out_features, group_size = 128, 16, 64

    torch.manual_seed(bits)
    weight = torch.randn(out_features, in_features)
    x = torch.randn(2, in_features)

    config = QuantConfig(
        method=QuantMethod.RTN, bits=bits, group_size=group_size,
        symmetric=symmetric, device="cpu",
    )
    quantizer = get_quantizer(config)
    result = quantizer.quantize(weight)

    reference = quantizer.dequantize(result).float().view(out_features, in_features)
    expected = x @ reference.T

    packed = result.packed_weights.view(out_features, -1)
    scales = result.scales.view(out_features, -1)
    zeros = result.zeros.view(out_features, -1) if result.zeros is not None else None

    kernel = (
        LUTQuantizedLinear.forward_int4_gemv if bits == 4
        else LUTQuantizedLinear.forward_int2_gemv
    )
    got = kernel(x=x, packed_weights=packed, scales=scales, zeros=zeros, group_size=group_size)
    assert torch.allclose(got, expected, atol=1e-3), (
        f"CPU LUT differs from the packer by {(got - expected).abs().max():.5f}"
    )

    if bits == 4:
        fused = FusedDequantGEMV.forward_int4_fma(
            x=x, packed_weights=packed, scales=scales, zeros=zeros, group_size=group_size
        )
        assert torch.allclose(fused, expected, atol=1e-3), (
            f"fused CUDA path differs by {(fused - expected).abs().max():.5f}"
        )


def test_kernel_rejects_a_padded_group_rather_than_reading_past_the_row():
    """`in_features` not divisible by `group_size` is a planner bug, not a silent reshape."""
    packed = torch.zeros(4, 30, dtype=torch.uint8)
    with pytest.raises(ValueError, match="not a multiple of group_size"):
        LUTQuantizedLinear.forward_int4_gemv(
            x=torch.randn(1, 60), packed_weights=packed,
            scales=torch.ones(4, 1), group_size=128,
        )
