"""Integration tests for end-to-end local model streaming quantization pipeline."""

import json
import shutil
from pathlib import Path
import pytest
import safetensors.torch
import torch

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod
from pockettitan.pipeline.layer_pipeline import QuantizationPipeline, ShardWriter


def test_quantization_pipeline_execution(dummy_transformer_model, tmp_path):
    output_dir = tmp_path / "quantized_output"
    
    quant_cfg = QuantConfig(method=QuantMethod.HQQ, bits=2, group_size=64, device="cpu")
    budget = MemoryBudgetConfig(max_vram_mb=2000.0, max_cpu_staging_mb=10.0)
    
    pipeline = QuantizationPipeline(
        model_id_or_path=str(dummy_transformer_model),
        output_dir=str(output_dir),
        quant_config=quant_cfg,
        budget_config=budget,
    )
    
    index_data = pipeline.run()
    
    assert output_dir.exists()
    assert (output_dir / "model.safetensors.index.json").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "quant_config.json").exists()
    
    # Verify quantized tensors exist in output
    weight_map = index_data["weight_map"]
    assert "model.layers.0.self_attn.q_proj.packed_weight" in weight_map
    assert "model.layers.0.self_attn.q_proj.scales" in weight_map
    # Verify unquantized embeddings are preserved
    assert "model.embed_tokens.weight" in weight_map
    
    # Load and verify Safetensors shard is valid
    first_shard = list(set(weight_map.values()))[0]
    with safetensors.torch.safe_open(str(output_dir / first_shard), framework="pt") as f:
        tensor_keys = f.keys()
        assert len(tensor_keys) > 0


def test_manifest_resume_recovery(dummy_transformer_model, tmp_path):
    output_dir = tmp_path / "resume_output"
    
    quant_cfg = QuantConfig(method=QuantMethod.RTN, bits=4, group_size=64, device="cpu")
    budget = MemoryBudgetConfig(max_vram_mb=2000.0, max_cpu_staging_mb=10.0)
    
    # 1. Run pipeline
    pipeline1 = QuantizationPipeline(
        model_id_or_path=str(dummy_transformer_model),
        output_dir=str(output_dir),
        quant_config=quant_cfg,
        budget_config=budget,
    )
    pipeline1.run()
    
    # 2. Run second time on same output dir -> should detect all completed and finish instantly
    pipeline2 = QuantizationPipeline(
        model_id_or_path=str(dummy_transformer_model),
        output_dir=str(output_dir),
        quant_config=quant_cfg,
        budget_config=budget,
    )
    res = pipeline2.run()
    assert len(res["weight_map"]) > 0
