"""Unit tests for GGUF and vLLM format exporters."""

import pytest
import struct
from pathlib import Path

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod
from pockettitan.exporters.gguf import GGUFExporter
from pockettitan.exporters.vllm import VLLMExporter
from pockettitan.pipeline.layer_pipeline import QuantizationPipeline


def test_gguf_and_vllm_export(dummy_transformer_model, tmp_path):
    # 1. First create a quantized checkpoint
    quantized_dir = tmp_path / "quantized_model"
    pipeline = QuantizationPipeline(
        model_id_or_path=str(dummy_transformer_model),
        output_dir=str(quantized_dir),
        quant_config=QuantConfig(method=QuantMethod.HQQ, bits=2, group_size=64, device="cpu"),
        budget_config=MemoryBudgetConfig(max_vram_mb=2000.0, max_cpu_staging_mb=10.0),
    )
    pipeline.run()

    # 2. Test GGUF Export
    gguf_output_file = tmp_path / "model.gguf"
    gguf_exporter = GGUFExporter(quantized_dir)
    res_gguf = gguf_exporter.export(gguf_output_file)
    
    assert res_gguf.status == "success"
    assert gguf_output_file.exists()
    assert gguf_output_file.stat().st_size > 0
    
    # Check GGUF Magic Header
    with open(gguf_output_file, "rb") as f:
        magic = f.read(4)
        version = struct.unpack("<I", f.read(4))[0]
        assert magic == b"GGUF"
        assert version == 3

    # 3. Test vLLM Export
    vllm_output_dir = tmp_path / "vllm_model"
    vllm_exporter = VLLMExporter(quantized_dir)
    res_vllm = vllm_exporter.export(vllm_output_dir)
    
    assert res_vllm.status == "success"
    assert (vllm_output_dir / "config.json").exists()
    assert (vllm_output_dir / "model.safetensors.index.json").exists()
