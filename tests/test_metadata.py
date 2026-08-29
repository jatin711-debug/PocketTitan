"""Unit tests for metadata inspection and Safetensors header parsing."""

import json
import struct
import pytest
from pathlib import Path
import torch

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod
from pockettitan.metadata.safetensors_header import (
    parse_safetensors_header_from_bytes,
    parse_local_safetensors_header,
)
from pockettitan.metadata.repo import extract_moe_specs
from pockettitan.scheduler.budget import (
    compute_work_unit_bounds,
    estimate_tensor_vram_requirement,
    get_hardware_profile,
)


def test_parse_safetensors_header_from_bytes():
    header_dict = {
        "model.layers.0.weight": {
            "dtype": "BF16",
            "shape": [4096, 4096],
            "data_offsets": [0, 33554432],
        }
    }
    json_bytes = json.dumps(header_dict).encode("utf-8")
    header_len = len(json_bytes)
    prefix = struct.pack("<Q", header_len)
    raw_payload = prefix + json_bytes + b"\x00" * 1024
    
    parsed_dict, total_header_bytes = parse_safetensors_header_from_bytes(raw_payload)
    assert total_header_bytes == 8 + header_len
    assert "model.layers.0.weight" in parsed_dict
    assert parsed_dict["model.layers.0.weight"]["shape"] == [4096, 4096]
    assert parsed_dict["model.layers.0.weight"]["dtype"] == "BF16"


def test_extract_moe_specs():
    # DeepSeek / GLM style config
    config_deepseek = {
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
    }
    specs = extract_moe_specs(config_deepseek)
    assert specs["is_moe"] is True
    assert specs["num_experts"] == 256
    assert specs["num_experts_per_tok"] == 8
    assert specs["expert_intermediate_size"] == 2048
    assert specs["shared_expert_intermediate_size"] == 2048

    # Dense model config
    config_dense = {
        "hidden_size": 4096,
        "intermediate_size": 11008,
    }
    dense_specs = extract_moe_specs(config_dense)
    assert dense_specs["is_moe"] is False
    assert dense_specs["num_experts"] is None


def test_hardware_profile():
    hw = get_hardware_profile()
    assert hw.system_ram_total_mb > 0
    assert hw.disk_free_mb > 0


def test_work_unit_bounds_tiling():
    budget = MemoryBudgetConfig(max_vram_mb=3500.0, runtime_reserve_mb=500.0, safety_margin_mb=500.0)
    cfg = QuantConfig(method=QuantMethod.HQQ, bits=2, group_size=128)
    
    # 1. Moderate matrix: 4096 x 4096 in float16 (32MB) -> should fit in single pass
    moderate_shape = [4096, 4096]
    bounds_mod = compute_work_unit_bounds(moderate_shape, budget, cfg, source_dtype="float16")
    assert bounds_mod["needs_tiling"] is False
    assert bounds_mod["num_tiles"] == 1
    assert bounds_mod["estimated_vram_per_tile_mb"] < budget.usable_vram_mb
    
    # 2. Giant matrix: 129280 x 7168 in float16 (1.73 GB) -> should be tiled
    giant_shape = [129280, 7168]
    bounds_giant = compute_work_unit_bounds(giant_shape, budget, cfg, source_dtype="float16")
    assert bounds_giant["needs_tiling"] is True
    assert bounds_giant["num_tiles"] > 1
    assert bounds_giant["estimated_vram_per_tile_mb"] <= budget.usable_vram_mb
    assert bounds_giant["tile_rows"] % 128 == 0 or bounds_giant["tile_rows"] % 64 == 0
