"""Comprehensive unit tests for all pluggable quantizer backends."""

import pytest
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.precision.distortion import evaluate_quantization_quality
from pockettitan.quantizers import (
    RTNQuantizer,
    TernaryQuantizer,
    INTxQuantizer,
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
    
    w_deq = quantizer.dequantize(q_res)
    assert w_deq.shape == w.shape
    
    report = evaluate_quantization_quality(w, w_deq)
    assert report.cosine_similarity > (0.90 if bits == 2 else 0.98)


def test_activation_distortion_metric():
    torch.manual_seed(42)
    w = torch.randn(128, 256, dtype=torch.float16)
    x = torch.randn(32, 256, dtype=torch.float16)
    
    cfg = QuantConfig(method=QuantMethod.HQQ, bits=4, group_size=64)
    quantizer = get_quantizer(cfg)
    
    q_res = quantizer.quantize(w)
    w_deq = quantizer.dequantize(q_res)
    
    report = evaluate_quantization_quality(w, w_deq, x=x)
    assert report.activation_distortion is not None
    assert report.activation_distortion < 0.10
