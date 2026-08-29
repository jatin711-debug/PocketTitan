"""Unit tests for GPTQ, AWQ, and AutoRound quantizer backends."""

import pytest
import torch

from pockettitan.calibration.hessian import HessianAccumulator
from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.precision.distortion import evaluate_quantization_quality
from pockettitan.quantizers import get_quantizer


def test_gptq_quantizer():
    torch.manual_seed(42)
    in_features = 256
    out_features = 128
    
    w = torch.randn(out_features, in_features, dtype=torch.float16) * 0.02
    
    # Generate synthetic activation Hessian
    acc = HessianAccumulator(in_features)
    for _ in range(5):
        x = torch.randn(16, in_features) * 0.1
        acc.add_batch(x)
    H = acc.get_normalized_hessian()
    
    quant_cfg = QuantConfig(method=QuantMethod.GPTQ, bits=4, group_size=64, device="cpu")
    quantizer = get_quantizer(quant_cfg)
    
    res = quantizer.quantize(w, hessian=H)
    w_deq = quantizer.dequantize(res)
    
    report = evaluate_quantization_quality(w, w_deq)
    assert report.cosine_similarity > 0.90
    assert report.snr_db > 10.0


def test_awq_quantizer():
    torch.manual_seed(42)
    in_features = 256
    out_features = 128
    
    w = torch.randn(out_features, in_features, dtype=torch.float16) * 0.02
    
    acc = HessianAccumulator(in_features)
    for _ in range(5):
        x = torch.randn(16, in_features) * 0.1
        acc.add_batch(x)
    H = acc.get_normalized_hessian()
    
    quant_cfg = QuantConfig(method=QuantMethod.AWQ, bits=4, group_size=64, device="cpu")
    quantizer = get_quantizer(quant_cfg)
    
    res = quantizer.quantize(w, hessian=H)
    w_deq = quantizer.dequantize(res)
    
    report = evaluate_quantization_quality(w, w_deq)
    assert report.cosine_similarity > 0.90
    assert report.snr_db > 10.0


def test_autoround_quantizer():
    torch.manual_seed(42)
    in_features = 256
    out_features = 128
    
    w = torch.randn(out_features, in_features, dtype=torch.float16) * 0.02
    
    quant_cfg = QuantConfig(method=QuantMethod.AUTOROUND, bits=4, group_size=64, device="cpu")
    quantizer = get_quantizer(quant_cfg)
    
    res = quantizer.quantize(w)
    w_deq = quantizer.dequantize(res)
    
    report = evaluate_quantization_quality(w, w_deq)
    assert report.cosine_similarity > 0.90
    assert report.snr_db > 10.0
