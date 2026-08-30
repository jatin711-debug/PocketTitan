"""Unit tests for memory budget arbiter and micro-tiling parity."""

import pytest
import torch

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod
from pockettitan.quantizers import HQQQuantizer, RTNQuantizer, TernaryQuantizer, get_quantizer
from pockettitan.scheduler.budget import (
    compute_work_unit_bounds,
    estimate_tensor_vram_requirement,
    group_padding_factor,
    source_dtype_bytes,
)
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
    tight_budget = MemoryBudgetConfig(
        max_vram_mb=100.0, runtime_reserve_mb=10.0, safety_margin_mb=10.0
    )
    tiler = MatrixTiler(tight_budget)
    res_tiled, _ = tiler.quantize_matrix(w.clone(), quantizer=quantizer, target_device="cpu")
    deq_tiled = quantizer.dequantize(res_tiled)

    # Mathematical parity check
    diff = torch.max(torch.abs(deq_untiled.float() - deq_tiled.float())).item()
    assert diff < 1e-4, f"Tiled and untiled outputs differ by {diff} for {method}"
    assert torch.equal(res_untiled.packed_weights, res_tiled.packed_weights)


# --------------------------------------------------------------------------- #
# Group padding and VRAM estimation (R1 T1.7)
# --------------------------------------------------------------------------- #

# The n-gram shard that OOM'd the pipeline: (2500012, 160) BF16.
PLE_SHARD_SHAPE = [2500012, 160]


@pytest.mark.parametrize(
    "in_features,group_size,expected",
    [
        (2560, 128, 1.0),
        (160, 160, 1.0),
        (160, 128, 1.6),  # pads 160 -> 256
        (160, 0, 1.0),  # per-tensor scaling: no padding
        (160, -1, 1.0),
        (129, 128, 256 / 129),
    ],
)
def test_group_padding_factor(in_features, group_size, expected):
    assert group_padding_factor(in_features, group_size) == pytest.approx(expected)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("BF16", 2),
        ("F16", 2),
        ("float16", 2),
        ("F32", 4),
        ("float32", 4),
        ("F8_E4M3", 1),
        ("FLOAT8_E5M2", 1),
        ("I8", 1),
        ("U8", 1),
        ("I64", 8),
    ],
)
def test_source_dtype_bytes(dtype, expected):
    assert source_dtype_bytes(dtype) == expected


def test_estimator_accounts_for_group_padding():
    """A 160-wide row at group_size=128 costs 1.6x a group-aligned one."""
    padded = estimate_tensor_vram_requirement(
        PLE_SHARD_SHAPE, "BF16", QuantMethod.TERNARY, 2, workspace_multiplier=7.0, group_size=128
    )
    aligned = estimate_tensor_vram_requirement(
        PLE_SHARD_SHAPE, "BF16", QuantMethod.TERNARY, 2, workspace_multiplier=7.0, group_size=160
    )
    assert padded > aligned
    working_ratio = (padded - aligned) / (aligned - PLE_SHARD_SHAPE[0] * 160 * 2 / (1024 * 1024))
    assert working_ratio == pytest.approx(0.6, abs=0.05)


def test_ple_shard_requires_tiling():
    """Regression: this exact tensor was reported as fitting and then OOM'd.

    The estimator claimed 1.74 GiB against a 2.62 GiB budget, so the tiler sent
    all 400M elements to a 4 GB card at once. The fp32 group-padded copy alone is
    2.38 GiB - precisely the allocation the driver refused.
    """
    budget = MemoryBudgetConfig(max_vram_mb=3584.0)
    for group_size in (128, 160):
        cfg = QuantConfig(method=QuantMethod.TERNARY, bits=2, group_size=group_size)
        quantizer = get_quantizer(cfg)
        bounds = compute_work_unit_bounds(
            PLE_SHARD_SHAPE, budget, cfg, "BF16", quantizer.capabilities.workspace_multiplier
        )
        assert bounds["needs_tiling"], f"group_size={group_size} must tile"
        assert bounds["num_tiles"] > 1
        assert bounds["estimated_vram_per_tile_mb"] <= budget.usable_vram_mb

        # The fp32 group-padded working copy of one tile must fit the budget.
        padded_width = 160 * group_padding_factor(160, group_size)
        fp32_tile_mb = bounds["tile_rows"] * padded_width * 4 / (1024 * 1024)
        assert fp32_tile_mb < budget.usable_vram_mb


def test_tighter_group_size_needs_fewer_tiles():
    """Matching group_size to row width removes the padding blowup entirely."""
    budget = MemoryBudgetConfig(max_vram_mb=3584.0)
    quantizer = get_quantizer(QuantConfig(method=QuantMethod.TERNARY, bits=2, group_size=128))
    wm = quantizer.capabilities.workspace_multiplier

    padded = compute_work_unit_bounds(
        PLE_SHARD_SHAPE,
        budget,
        QuantConfig(method=QuantMethod.TERNARY, bits=2, group_size=128),
        "BF16",
        wm,
    )
    aligned = compute_work_unit_bounds(
        PLE_SHARD_SHAPE,
        budget,
        QuantConfig(method=QuantMethod.TERNARY, bits=2, group_size=160),
        "BF16",
        wm,
    )
    assert aligned["num_tiles"] < padded["num_tiles"]


@pytest.mark.parametrize(
    "method",
    [
        QuantMethod.TERNARY,
        QuantMethod.RTN,
        QuantMethod.INTX,
        QuantMethod.HQQ,
        QuantMethod.GPTQ,
        QuantMethod.AWQ,
        QuantMethod.AUTOROUND,
    ],
)
def test_declared_workspace_multiplier_is_not_optimistic(method):
    """Declared multipliers must exceed measured peaks, or the tiler will OOM.

    Values below were measured on an RTX 3050 with group-aligned matrices; the
    padding term is modelled separately. Every one of these was originally
    under-declared by 3-6x.
    """
    measured = {
        QuantMethod.TERNARY: 6.51,
        QuantMethod.RTN: 6.05,
        QuantMethod.INTX: 6.05,
        QuantMethod.HQQ: 12.09,
        QuantMethod.GPTQ: 12.39,
        QuantMethod.AWQ: 14.09,
        QuantMethod.AUTOROUND: 34.06,
    }
    quantizer = get_quantizer(QuantConfig(method=method, bits=4, group_size=128))
    assert quantizer.capabilities.workspace_multiplier > measured[method]


@pytest.mark.gpu
def test_ple_shard_tile_fits_real_vram():
    """Execute a worst-case tile on the real device under the real cap."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    budget = MemoryBudgetConfig(max_vram_mb=3584.0)
    cfg = QuantConfig(method=QuantMethod.TERNARY, bits=2, group_size=128, device="cuda")
    quantizer = get_quantizer(cfg)
    bounds = compute_work_unit_bounds(
        PLE_SHARD_SHAPE, budget, cfg, "BF16", quantizer.capabilities.workspace_multiplier
    )

    tile = torch.randn(bounds["tile_rows"], 160, dtype=torch.bfloat16) * 0.02
    _, peak_mb = MatrixTiler(budget).quantize_matrix(
        tile, quantizer=quantizer, target_device="cuda"
    )
    assert peak_mb < budget.usable_vram_mb, (
        f"peak {peak_mb:.0f} MiB exceeded {budget.usable_vram_mb:.0f} MiB"
    )
