"""Comprehensive unit tests for all pluggable quantizer backends."""

import pytest
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.precision.distortion import evaluate_quantization_quality
from pockettitan.quantizers import (
    RTNQuantizer,
    TernaryQuantizer,
    HQQQuantizer,
    get_quantizer,
)


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_rtn_quantizer_roundtrip(bits, symmetric):
    cfg = QuantConfig(method=QuantMethod.RTN, bits=bits, group_size=64, symmetric=symmetric)
    quantizer = RTNQuantizer(cfg)

    torch.manual_seed(42)
    w = torch.randn(256, 512, dtype=torch.float16) * 0.05

    q_res = quantizer.quantize(w)
    assert q_res.bit_width == float(bits)
    w_deq = quantizer.dequantize(q_res)
    assert w_deq.shape == w.shape

    report = evaluate_quantization_quality(w, w_deq)
    expected_cos_sim = 0.75 if bits == 2 else (0.95 if bits == 4 else 0.999)
    assert report.cosine_similarity > expected_cos_sim
    assert report.snr_db > (2.5 if bits == 2 else (10.0 if bits == 4 else 20.0))


def test_ternary_quantizer_roundtrip():
    cfg = QuantConfig(method=QuantMethod.TERNARY, bits=2, group_size=128)
    quantizer = TernaryQuantizer(cfg)

    torch.manual_seed(42)
    w = torch.randn(256, 512, dtype=torch.float16) * 0.05

    q_res = quantizer.quantize(w)
    assert q_res.bit_width == 1.58
    assert q_res.zeros is None  # Ternary is strictly zero-centered

    w_deq = quantizer.dequantize(q_res)
    assert w_deq.shape == w.shape

    report = evaluate_quantization_quality(w, w_deq)
    assert report.cosine_similarity > 0.85


@pytest.mark.parametrize("bits", [2, 4])
def test_hqq_quantizer_roundtrip(bits):
    cfg = QuantConfig(method=QuantMethod.HQQ, bits=bits, group_size=64)
    quantizer = HQQQuantizer(cfg, max_iters=10)

    torch.manual_seed(42)
    w = torch.randn(256, 512, dtype=torch.float16) * 0.05

    q_res = quantizer.quantize(w)
    assert q_res.bit_width == float(bits)
    assert q_res.zeros is not None

    w_deq = quantizer.dequantize(q_res)
    assert w_deq.shape == w.shape

    report = evaluate_quantization_quality(w, w_deq)
    assert report.cosine_similarity > (0.80 if bits == 2 else 0.96)
    assert report.snr_db > (3.0 if bits == 2 else 12.0)


@pytest.mark.parametrize("method", [QuantMethod.HQQ, QuantMethod.RTN, QuantMethod.TERNARY])
def test_non_divisible_shape_quantization(method):
    """Test quantization of matrices with non-standard dimensions (e.g. 4304 with group 128)."""
    cfg = QuantConfig(method=method, bits=2, group_size=128)
    quantizer = get_quantizer(cfg)

    # 4304 is not divisible by 128 (4304 % 128 == 80)
    w = torch.randn(64, 4304, dtype=torch.float16) * 0.02
    q_res = quantizer.quantize(w)
    w_deq = quantizer.dequantize(q_res)

    assert w_deq.shape == (64, 4304)
    assert w_deq.dtype == torch.float16
    report = evaluate_quantization_quality(w, w_deq)
    assert report.cosine_similarity > 0.70


# --------------------------------------------------------------------------- #
# Non-2-D weight round-trips
# --------------------------------------------------------------------------- #

NON_2D_SHAPES = [
    pytest.param((48,), id="1d-vector-A_log"),
    pytest.param((256,), id="1d-vector-q_norm"),
    pytest.param((640, 1, 4), id="3d-conv1d-kernel"),
    pytest.param((8, 16, 128), id="3d-fused-bank"),
]


@pytest.mark.parametrize("shape", NON_2D_SHAPES)
@pytest.mark.parametrize("method", [QuantMethod.RTN, QuantMethod.HQQ, QuantMethod.TERNARY])
def test_dequantize_restores_non_2d_shapes(shape, method):
    """``quantize`` flattens with ``view(-1, shape[-1])``; ``dequantize`` must agree.

    Reading the dims as ``(shape[0], shape[1])`` matches only for 2-D weights. A
    1-D vector comes back transposed and a 3-D kernel gets the wrong row width,
    so tensors such as ``linear_attn.A_log`` and ``conv1d.weight`` are written to
    a package that cannot decode them.
    """
    torch.manual_seed(0)
    weight = torch.randn(*shape, dtype=torch.float16)

    ternary = method is QuantMethod.TERNARY
    config = QuantConfig(
        method=method,
        bits=2 if ternary else 4,
        group_size=128,
        symmetric=False,
        device="cpu",
    )
    quantizer = get_quantizer(config)
    restored = quantizer.dequantize(quantizer.quantize(weight.float()))

    # Shape is the sharp assertion: a wrong row/column split either raises or
    # returns a transposed tensor. The error bound only guards against the
    # reconstruction being unrelated to the input.
    assert tuple(restored.shape) == shape
    scale = weight.float().abs().max().item()
    tolerance = 1.0 if ternary else 0.25
    assert (restored.float() - weight.float()).abs().max().item() < tolerance * scale


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((48,), (1, 48)),
        ((5120, 17408), (5120, 17408)),
        ((10240, 1, 4), (10240, 4)),
    ],
)
def test_matrix_dims_matches_the_quantize_flatten(shape, expected):
    """One definition of the flatten, shared by the planner and every backend.

    The calibrated backends (GPTQ/AWQ/AutoRound) cannot be exercised without a
    Hessian, so their dequantize path is pinned here at the helper instead.
    """
    from pockettitan.package.format import matrix_dims as planner_dims
    from pockettitan.quantizers.base import matrix_dims

    assert matrix_dims(tuple(shape)) == expected
    assert planner_dims(list(shape)) == expected
    assert torch.empty(*shape).view(-1, shape[-1]).shape == torch.Size(expected)


@pytest.mark.parametrize("method", [QuantMethod.RTN, QuantMethod.HQQ])
@pytest.mark.parametrize("bits", [3, 4])
def test_narrow_band_group_away_from_zero_keeps_resolution(method, bits):
    """A group that does not straddle zero must still use the whole code range.

    The affine zero-point is stored as fp16 and only used as ``(q - z) * s``, so
    it may legitimately fall outside ``[0, max_int]``. Clamping it there forces
    the representable interval to start at 0 and spends every code on the empty
    gap. Observed on ``linear_attn.norm.weight``: 128 values in [0.13, 0.16]
    were all written as code 7 and decoded to one constant, destroying the
    layer's per-channel gain.
    """
    torch.manual_seed(0)
    weight = torch.empty(1, 128).uniform_(0.130, 0.155)

    config = QuantConfig(
        method=method, bits=bits, group_size=128, symmetric=False, device="cpu"
    )
    quantizer = get_quantizer(config)
    restored = quantizer.dequantize(quantizer.quantize(weight)).float()

    band = (weight.max() - weight.min()).item()
    assert torch.unique(restored).numel() > 1, "collapsed to a constant"
    assert restored.std().item() > 0.25 * weight.std().item()
    assert (restored - weight).abs().max().item() < band / ((1 << bits) - 1)
