"""Shared pytest fixtures."""

import gzip
import json
import math
from pathlib import Path
import pytest
import safetensors.torch
import torch

from pockettitan.audit import ShardHeaderScan
from pockettitan.config import TensorAddress

QWEN_FIXTURE = Path(__file__).parent / "data" / "qwen38_flash_next_headers.json.gz"


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
        "model.layers.0.self_attn.q_proj.weight": torch.randn(
            hidden_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.0.self_attn.k_proj.weight": torch.randn(
            hidden_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.0.self_attn.v_proj.weight": torch.randn(
            hidden_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(
            hidden_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.0.post_attention_layernorm.weight": torch.ones(
            hidden_size, dtype=torch.float16
        ),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(
            intermediate_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.0.mlp.up_proj.weight": torch.randn(
            intermediate_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.0.mlp.down_proj.weight": torch.randn(
            hidden_size, intermediate_size, dtype=torch.float16
        )
        * 0.02,
    }

    tensors_shard_2 = {
        "model.layers.1.input_layernorm.weight": torch.ones(hidden_size, dtype=torch.float16),
        "model.layers.1.self_attn.q_proj.weight": torch.randn(
            hidden_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.1.self_attn.k_proj.weight": torch.randn(
            hidden_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.1.self_attn.v_proj.weight": torch.randn(
            hidden_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.1.self_attn.o_proj.weight": torch.randn(
            hidden_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.1.post_attention_layernorm.weight": torch.ones(
            hidden_size, dtype=torch.float16
        ),
        "model.layers.1.mlp.gate_proj.weight": torch.randn(
            intermediate_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.1.mlp.up_proj.weight": torch.randn(
            intermediate_size, hidden_size, dtype=torch.float16
        )
        * 0.02,
        "model.layers.1.mlp.down_proj.weight": torch.randn(
            hidden_size, intermediate_size, dtype=torch.float16
        )
        * 0.02,
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


@pytest.fixture(scope="module")
def qwen_scan():
    """Reconstruct a full scan from the checked-in header fixture (no network)."""
    with gzip.open(QWEN_FIXTURE, "rt", encoding="utf-8") as f:
        payload = json.load(f)

    tensors = {}
    for name, info in payload["tensors"].items():
        # Absolute file offsets, exactly as scan_checkpoint() produces them.
        start, end = info["byte_start"], info["byte_end"]
        shape = info["shape"]
        tensors[name] = TensorAddress(
            name=name,
            shard=info["shard"],
            dtype=info["dtype"],
            shape=shape,
            byte_start=start,
            byte_end=end,
            num_params=math.prod(shape) if shape else 0,
            size_bytes=end - start,
        )

    return ShardHeaderScan(
        model_id=payload["model_id"],
        is_local=False,
        config=payload["config"],
        shards=sorted({t.shard for t in tensors.values()}),
        tensors=tensors,
        declared_total_bytes=payload["declared_total_bytes"],
    )


@pytest.fixture
def dummy_moe_model(tmp_path):
    """Miniature MoE checkpoint using the *fused* expert layout.

    Mirrors Qwen3.8's storage convention: all experts of a layer live in two
    3-D tensors, so an expert is a slice rather than a tensor.
    """
    model_dir = tmp_path / "dummy_moe"
    model_dir.mkdir()

    hidden, inter, num_experts, num_layers = 64, 32, 8, 2

    config_dict = {
        "architectures": ["DummyMoEForCausalLM"],
        "model_type": "dummy_moe",
        "text_config": {
            "hidden_size": hidden,
            "moe_intermediate_size": inter,
            "num_hidden_layers": num_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
            "vocab_size": 512,
            "dtype": "float16",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")

    tensors = {
        "model.embed_tokens.weight": torch.randn(512, hidden, dtype=torch.float16) * 0.02,
        "lm_head.weight": torch.randn(512, hidden, dtype=torch.float16) * 0.02,
        "model.norm.weight": torch.ones(hidden, dtype=torch.float16),
    }
    for layer in range(num_layers):
        p = f"model.layers.{layer}"
        tensors[f"{p}.self_attn.q_proj.weight"] = (
            torch.randn(hidden, hidden, dtype=torch.float16) * 0.02
        )
        tensors[f"{p}.self_attn.o_proj.weight"] = (
            torch.randn(hidden, hidden, dtype=torch.float16) * 0.02
        )
        tensors[f"{p}.input_layernorm.weight"] = torch.ones(hidden, dtype=torch.float16)
        tensors[f"{p}.mlp.gate.weight"] = (
            torch.randn(num_experts, hidden, dtype=torch.float16) * 0.02
        )
        # Fused expert banks: (num_experts, out, in)
        tensors[f"{p}.mlp.experts.gate_up_proj"] = (
            torch.randn(num_experts, 2 * inter, hidden, dtype=torch.float16) * 0.02
        )
        tensors[f"{p}.mlp.experts.down_proj"] = (
            torch.randn(num_experts, hidden, inter, dtype=torch.float16) * 0.02
        )

    shard_name = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, str(model_dir / shard_name))

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": {k: shard_name for k in tensors},
            }
        ),
        encoding="utf-8",
    )
    return model_dir


@pytest.fixture
def dummy_ple_model(tmp_path):
    """Miniature checkpoint carrying a sharded n-gram (PLE) table."""
    model_dir = tmp_path / "dummy_ple"
    model_dir.mkdir()

    hidden, row_width, heads, rows_per_shard, shards = 64, 16, 4, 32, 4

    config_dict = {
        "architectures": ["DummyPleForCausalLM"],
        "model_type": "dummy_ple",
        "text_config": {
            "hidden_size": hidden,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
            "ngram_size": 3,
            "ngram_vocab_size_base": rows_per_shard,
            "ple_embed_dim": row_width * heads,
            "split_ngram_parts": shards,
            "dtype": "float16",
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    (model_dir / "generation_config.json").write_text(
        json.dumps({"max_new_tokens": 32}), encoding="utf-8"
    )
    (model_dir / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "added_tokens": []}), encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    # A text-only package must not copy this multimodal preprocessor.
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps({"do_resize": True}), encoding="utf-8"
    )

    tensors = {
        "model.embed_tokens.weight": torch.randn(256, hidden, dtype=torch.float16) * 0.02,
        "lm_head.weight": torch.randn(256, hidden, dtype=torch.float16) * 0.02,
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden, dtype=torch.float16)
        * 0.02,
        "model.layers.0.ple.key_proj.weight": torch.randn(hidden, hidden, dtype=torch.float16)
        * 0.02,
        # These are exact hash/index constants, not quantizable model weights.
        "model.layers.0.ple.ple_embedding.layer_multipliers": torch.tensor(
            [11, 17, 23], dtype=torch.int64
        ),
        "model.layers.0.ple.ple_embedding.ngram_heads_offsets": torch.tensor(
            [0, 32, 64, 96], dtype=torch.int64
        ),
        "model.layers.0.ple.ple_embedding.ngram_heads_vocab_sizes": torch.tensor(
            [32, 32, 32, 32], dtype=torch.int64
        ),
    }
    for shard in range(shards):
        tensors[f"model.layers.0.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"] = (
            torch.randn(rows_per_shard, row_width, dtype=torch.float16) * 0.05
        )

    shard_name = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, str(model_dir / shard_name))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(t.numel() * t.element_size() for t in tensors.values())
                },
                "weight_map": {k: shard_name for k in tensors},
            }
        ),
        encoding="utf-8",
    )
    return model_dir
