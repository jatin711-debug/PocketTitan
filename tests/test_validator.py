"""Unit tests for checkpoint integrity auditor and validator."""

import pytest
import safetensors.torch
import torch

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod
from pockettitan.export.validator import CheckpointValidator
from pockettitan.pipeline.layer_pipeline import QuantizationPipeline


def test_checkpoint_validator(dummy_transformer_model, tmp_path):
    output_dir = tmp_path / "valid_model"
    
    quant_cfg = QuantConfig(method=QuantMethod.HQQ, bits=2, group_size=64, device="cpu")
    budget = MemoryBudgetConfig(max_vram_mb=2000.0, max_cpu_staging_mb=10.0)
    
    pipeline = QuantizationPipeline(
        model_id_or_path=str(dummy_transformer_model),
        output_dir=str(output_dir),
        quant_config=quant_cfg,
        budget_config=budget,
    )
    pipeline.run()
    
    validator = CheckpointValidator(output_dir)
    scorecard = validator.validate()
    
    assert scorecard.is_valid is True
    assert scorecard.total_shards >= 1
    assert scorecard.total_tensors == 49
    assert scorecard.quantized_tensor_count == 14
    assert scorecard.passthrough_tensor_count == 7
    assert len(scorecard.errors) == 0
