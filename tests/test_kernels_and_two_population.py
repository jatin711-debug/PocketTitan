"""Tests for Phase R9 Hardware-Accelerated Kernels and Two-Population Expert Allocation."""

import pytest
import torch

from pockettitan.precision.two_population import TwoPopulationAllocator, TwoPopulationPlan
from pockettitan.runtime.kernels.cpu_lut import LUTQuantizedLinear
from pockettitan.runtime.kernels.cuda_fused import FusedDequantGEMV
from pockettitan.runtime.kernels.gdn_blas import GDNRecurrenceBLAS


def test_cpu_lut_int4_gemv_parity():
    """Verify CPU LUT INT4 GEMV produces identical outputs to reference dequantized matmul."""
    batch_size = 2
    in_features = 128
    out_features = 64
    group_size = 64
    num_groups = in_features // group_size

    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, dtype=torch.float32)
    
    # 4-bit packed weights (each byte has 2 nibbles)
    packed_w = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    scales = torch.randn(out_features, num_groups, dtype=torch.float32) * 0.05

    # Run LUT GEMV
    lut_out = LUTQuantizedLinear.forward_int4_gemv(
        x=x,
        packed_weights=packed_w,
        scales=scales,
        group_size=group_size,
    )

    # Reference manual dequantization
    w_low = (packed_w & 0x0F).float() - 8.0
    w_high = ((packed_w >> 4) & 0x0F).float() - 8.0
    w_ref = torch.empty((out_features, in_features), dtype=torch.float32)
    w_ref[:, 0::2] = w_low
    w_ref[:, 1::2] = w_high
    
    w_ref_grouped = w_ref.view(out_features, num_groups, group_size)
    w_scaled = w_ref_grouped * scales.view(out_features, num_groups, 1)
    w_final = w_scaled.view(out_features, in_features)

    expected_out = torch.matmul(x, w_final.t())

    assert torch.allclose(lut_out, expected_out, atol=1e-4)


def test_cuda_fused_int4_fma_parity():
    """Verify FusedDequantGEMV produces identical outputs to reference scaling."""
    batch_size = 1
    in_features = 256
    out_features = 128
    group_size = 128
    num_groups = in_features // group_size

    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, dtype=torch.float32)
    packed_w = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
    scales = torch.randn(out_features, num_groups, dtype=torch.float32) * 0.02
    biases = torch.randn(out_features, num_groups, dtype=torch.float32) * 0.01

    fused_out = FusedDequantGEMV.forward_int4_fma(
        x=x,
        packed_weights=packed_w,
        scales=scales,
        biases=biases,
        group_size=group_size,
    )

    # Reference math
    w_low = (packed_w & 0x0F).float() - 8.0
    w_high = ((packed_w >> 4) & 0x0F).float() - 8.0
    w_ref = torch.empty((out_features, in_features), dtype=torch.float32)
    w_ref[:, 0::2] = w_low
    w_ref[:, 1::2] = w_high

    w_g = w_ref.view(out_features, num_groups, group_size)
    w_scaled = (w_g * scales.view(out_features, num_groups, 1)) + biases.view(out_features, num_groups, 1)
    w_final = w_scaled.view(out_features, in_features)

    expected_out = torch.matmul(x, w_final.t())
    assert torch.allclose(fused_out, expected_out, atol=1e-4)


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
    for l in range(48):
        for e in range(512):
            freqs[(l, e)] = (l + 1) * (512 - e)  # higher frequency for early experts

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
