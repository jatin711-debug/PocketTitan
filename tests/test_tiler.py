"""Unit tests for memory budget arbiter and micro-tiling parity."""

import pytest
import torch

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod
from pockettitan.quantizers import RTNQuantizer, TernaryQuantizer, HQQQuantizer
from pockettitan.scheduler.tiler import MatrixTiler


@pytest.mark.parametrize("method", [QuantMethod.RTN, QuantMethod.TERNARY, QuantMethod.HQQ])
def test_tiled_vs_untiled_mathematical_parity(method):
    """Verify that tiled execution produces identical results to monolithic execution."""
    torch.manual_seed(123)
    w = torch.randn(1024, 2048, dtype=torch.float16) * 0.05
    
    cfg = QuantConfig(method=method, bits=2, group_size=128)
    if method == QuantMethod.RTN:
        quantizer = RTNQuantizer(cfg)
    elif method == QuantMethod.TERNARY:
        quantizer = TernaryQuantizer(cfg)
    else:
        quantizer = HQQQuantizer(cfg, max_iters=10)
        
    # 1. Monolithic untiled baseline
    res_untiled = quantizer.quantize(w.clone())
    deq_untiled = quantizer.dequantize(res_untiled)
    
    # 2. Constrained budget forcing 4 tiles
    tight_budget = MemoryBudgetConfig(max_vram_mb=100.0, runtime_reserve_mb=10.0, safety_margin_mb=10.0)
    tiler = MatrixTiler(tight_budget)
    res_tiled, _ = tiler.quantize_matrix(w.clone(), quantizer=quantizer, target_device="cpu")
    deq_tiled = quantizer.dequantize(res_tiled)
    
    # Mathematical parity check
    diff = torch.max(torch.abs(deq_untiled.float() - deq_tiled.float())).item()
    assert diff < 1e-4, f"Tiled and untiled outputs differ by {diff} for {method}"
    assert torch.equal(res_untiled.packed_weights, res_tiled.packed_weights)
