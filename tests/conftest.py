"""Shared pytest fixtures."""

import json
from pathlib import Path
import pytest
import safetensors.torch
import torch


@pytest.fixture
def dummy_transformer_model(tmp_path):
    """Create a realistic miniature transformer checkpoint with Safetensors index."""
    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir()
    
    hidden_size = 256
    intermediate_size = 512
    num_layers = 2
    
    config_dict = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "torch_dtype": "float16",
        "vocab_size": 1000,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config_dict, f)
        
    tensors_shard_1 = {
        "model.embed_tokens.weight": torch.randn(1000, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.0.input_layernorm.weight": torch.ones(hidden_size, dtype=torch.float16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden_size, dtype=torch.float16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(intermediate_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.0.mlp.up_proj.weight": torch.randn(intermediate_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size, dtype=torch.float16) * 0.02,
    }
    
    tensors_shard_2 = {
        "model.layers.1.input_layernorm.weight": torch.ones(hidden_size, dtype=torch.float16),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.1.self_attn.k_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.1.self_attn.v_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.1.self_attn.o_proj.weight": torch.randn(hidden_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.1.post_attention_layernorm.weight": torch.ones(hidden_size, dtype=torch.float16),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(intermediate_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.1.mlp.up_proj.weight": torch.randn(intermediate_size, hidden_size, dtype=torch.float16) * 0.02,
        "model.layers.1.mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size, dtype=torch.float16) * 0.02,
        "model.norm.weight": torch.ones(hidden_size, dtype=torch.float16),
        "lm_head.weight": torch.randn(1000, hidden_size, dtype=torch.float16) * 0.02,
    }
    
    shard1_name = "model-00001-of-00002.safetensors"
    shard2_name = "model-00002-of-00002.safetensors"
    
    safetensors.torch.save_file(tensors_shard_1, str(model_dir / shard1_name))
    safetensors.torch.save_file(tensors_shard_2, str(model_dir / shard2_name))
    
    weight_map = {}
    for k in tensors_shard_1.keys():
        weight_map[k] = shard1_name
    for k in tensors_shard_2.keys():
        weight_map[k] = shard2_name
        
    index_dict = {
        "metadata": {"total_size": 1000000},
        "weight_map": weight_map,
    }
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_dict, f)
        
    return model_dir
