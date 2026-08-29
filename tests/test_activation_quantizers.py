"""Tests for second-order and activation-aware quantizers (GPTQ, AWQ, AutoRound)."""

import pytest
import torch

from pockettitan.config import CalibrationRequiredError, QuantConfig, QuantMethod
from pockettitan.precision.distortion import evaluate_quantization_quality
from pockettitan.quantizers import get_quantizer


def test_gptq_quantizer():
    torch.manual_seed(42)
    in_features = 256
    out_features = 128
    
    w = torch.randn(out_features, in_features, dtype=torch.float16) * 0.02
    H = torch.eye(in_features, dtype=torch.float32)
    
    quant_cfg = QuantConfig(method=QuantMethod.GPTQ, bits=4, group_size=64, device="cpu")
    quantizer = get_quantizer(quant_cfg)
    
    # Assert missing Hessian raises CalibrationRequiredError
    with pytest.raises(CalibrationRequiredError):
        quantizer.quantize(w)
        
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
    H = torch.diag(torch.rand(in_features) + 0.5)
    
    quant_cfg = QuantConfig(method=QuantMethod.AWQ, bits=4, group_size=64, device="cpu")
    quantizer = get_quantizer(quant_cfg)
    
    # Assert missing Hessian raises CalibrationRequiredError
    with pytest.raises(CalibrationRequiredError):
        quantizer.quantize(w)
        
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
    H = torch.eye(in_features, dtype=torch.float32)
    
    quant_cfg = QuantConfig(method=QuantMethod.AUTOROUND, bits=4, group_size=64, device="cpu")
    quantizer = get_quantizer(quant_cfg)
    
    # Assert missing Hessian raises CalibrationRequiredError
    with pytest.raises(CalibrationRequiredError):
        quantizer.quantize(w)
        
    res = quantizer.quantize(w, hessian=H)
    w_deq = quantizer.dequantize(res)
    
    report = evaluate_quantization_quality(w, w_deq)
    assert report.cosine_similarity > 0.90
    assert report.snr_db > 10.0
