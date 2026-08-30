"""GGUF binary format exporter for llama.cpp execution."""

import json
from pathlib import Path
import re
import struct
from typing import Any, Dict, List, Union
import safetensors.torch
import torch

from pockettitan.exporters.base import BaseExporter, ExportResult
from pockettitan.quantizers.rtn import RTNQuantizer

# GGML Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_IQ2_XXS = 19

# GGUF Value Types
GGUF_METADATA_UINT32 = 4
GGUF_METADATA_INT32 = 5
GGUF_METADATA_FLOAT32 = 6
GGUF_METADATA_STRING = 8


def hf_to_gguf_name(name: str) -> str:
    """Map Hugging Face tensor name to standard canonical GGUF short name (<64 chars)."""
    # 1. Direct standard replacements
    if name in [
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.embed_tokens.packed_weight",
    ]:
        return "token_embd.weight"
    if name in ["lm_head.weight", "lm_head.packed_weight"]:
        return "output.weight"
    if name in ["model.norm.weight", "model.language_model.norm.weight"]:
        return "output_norm.weight"

    # 2. Transformer Layer mappings
    match = re.search(r"layers\.(\d+)\.(.+)", name)
    if match:
        layer_idx = match.group(1)
        sub = match.group(2)

        # Norms
        if "input_layernorm" in sub:
            return f"blk.{layer_idx}.attn_norm.weight"
        if "post_attention_layernorm" in sub:
            return f"blk.{layer_idx}.ffn_norm.weight"

        # Attention
        if "self_attn.q_proj" in sub or "attn.q_proj" in sub:
            return f"blk.{layer_idx}.attn_q.weight"
        if "self_attn.k_proj" in sub or "attn.k_proj" in sub:
            return f"blk.{layer_idx}.attn_k.weight"
        if "self_attn.v_proj" in sub or "attn.v_proj" in sub:
            return f"blk.{layer_idx}.attn_v.weight"
        if "self_attn.o_proj" in sub or "attn.o_proj" in sub or "self_attn.out_proj" in sub:
            return f"blk.{layer_idx}.attn_output.weight"
        if "linear_attn.in_proj_qkv" in sub:
            return f"blk.{layer_idx}.attn_qkv.weight"
        if "linear_attn.out_proj" in sub:
            return f"blk.{layer_idx}.attn_out.weight"

        # MLP
        if "mlp.gate_proj" in sub:
            return f"blk.{layer_idx}.ffn_gate.weight"
        if "mlp.up_proj" in sub:
            return f"blk.{layer_idx}.ffn_up.weight"
        if "mlp.down_proj" in sub:
            return f"blk.{layer_idx}.ffn_down.weight"

        # MoE
        if "mlp.gate" in sub or "mlp.router" in sub:
            return f"blk.{layer_idx}.ffn_gate_inp.weight"

        exp_match = re.search(r"mlp\.experts\.(\d+)\.(.+)", sub)
        if exp_match:
            exp_id = exp_match.group(1)
            exp_sub = exp_match.group(2)
            if "gate_proj" in exp_sub:
                return f"blk.{layer_idx}.ffn_gate_exp.{exp_id}.weight"
            if "up_proj" in exp_sub:
                return f"blk.{layer_idx}.ffn_up_exp.{exp_id}.weight"
            if "down_proj" in exp_sub:
                return f"blk.{layer_idx}.ffn_down_exp.{exp_id}.weight"

    # 3. Vision Tower mappings
    v_match = re.search(r"visual\.blocks\.(\d+)\.(.+)", name)
    if v_match:
        blk_idx = v_match.group(1)
        v_sub = v_match.group(2)
        if "attn.qkv" in v_sub:
            return f"v.blk.{blk_idx}.attn_qkv.w"
        if "attn.proj" in v_sub:
            return f"v.blk.{blk_idx}.attn_out.w"
        if "mlp.linear_fc1" in v_sub or "mlp.fc1" in v_sub:
            return f"v.blk.{blk_idx}.ffn_up.w"
        if "mlp.linear_fc2" in v_sub or "mlp.fc2" in v_sub:
            return f"v.blk.{blk_idx}.ffn_down.w"
        if "norm1" in v_sub:
            return f"v.blk.{blk_idx}.ln1.w"
        if "norm2" in v_sub:
            return f"v.blk.{blk_idx}.ln2.w"
        return f"v.blk.{blk_idx}.{v_sub[:20]}"

    clean = name.replace("model.language_model.", "").replace("model.", "")
    return clean[:58]


class GGUFExporter(BaseExporter):
    """Exports PocketTitan quantized model to GGUF format for llama.cpp."""

    def export(self, output_path: Union[str, Path]) -> ExportResult:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Load config.json and quant_config.json
        config_path = self.checkpoint_dir / "config.json"
        config_dict = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

        quant_config_path = self.checkpoint_dir / "quant_config.json"
        bits = 2
        group_size = 128
        if quant_config_path.exists():
            with open(quant_config_path, "r", encoding="utf-8") as f:
                q_cfg = json.load(f)
                bits = q_cfg.get("bits", 2)
                group_size = q_cfg.get("group_size", 128)

        # Resolve text architecture
        text_cfg = config_dict.get("text_config", config_dict)
        raw_arch = config_dict.get("architectures", ["llama"])[0].lower()
        if "qwen" in raw_arch:
            arch = "qwen2"
        elif "deepseek" in raw_arch:
            arch = "deepseek2"
        else:
            arch = "llama"

        # Extract model hyperparameters
        hidden_size = text_cfg.get("hidden_size", 5120)
        num_layers = text_cfg.get("num_hidden_layers", 64)
        intermediate_size = text_cfg.get("intermediate_size", 17408)
        num_heads = text_cfg.get("num_attention_heads", 24)
        num_kv_heads = text_cfg.get("num_key_value_heads", 4)
        context_len = min(text_cfg.get("max_position_embeddings", 32768), 32768)

        # Collect all tensors across shards
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        single_safetensors = self.checkpoint_dir / "model.safetensors"

        shards_to_read = []
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            shards_to_read = sorted(list(set(idx.get("weight_map", {}).values())))
        elif single_safetensors.exists():
            shards_to_read = ["model.safetensors"]
        else:
            shards_to_read = [f.name for f in self.checkpoint_dir.glob("*.safetensors")]

        # Load all raw tensor mappings from shards
        raw_shard_tensors: Dict[str, torch.Tensor] = {}
        for shard_name in shards_to_read:
            shard_path = self.checkpoint_dir / shard_name
            if not shard_path.exists():
                continue
            with safetensors.torch.safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    raw_shard_tensors[k] = f.get_tensor(k)

        # Merge quantized components into single dequantized tensors to avoid duplicates
        final_tensor_entries: List[Dict[str, Any]] = []
        seen_names = set()

        for k, tensor in raw_shard_tensors.items():
            if k.endswith(".packed_weight"):
                prefix = k[:-14]
                packed = tensor
                scales = raw_shard_tensors.get(prefix + ".scales")
                zeros = raw_shard_tensors.get(prefix + ".zeros")

                if scales is not None:
                    out_f = scales.shape[0]
                    num_g = scales.shape[1]
                    padded_in = num_g * group_size

                    unpacked = RTNQuantizer._unpack_tensor(packed, bits, (out_f, padded_in))
                    w_grouped = unpacked.view(-1, group_size).float()
                    s_g = scales.view(-1, 1).float()
                    z_g = zeros.view(-1, 1).float() if zeros is not None else 0.0
                    deq = (w_grouped - z_g) * s_g
                    deq_tensor = deq.view(out_f, padded_in).half()

                    gguf_name = hf_to_gguf_name(prefix + ".weight")
                    if gguf_name not in seen_names:
                        final_tensor_entries.append({"name": gguf_name, "tensor": deq_tensor})
                        seen_names.add(gguf_name)
            elif k.endswith(".scales") or k.endswith(".zeros") or k.endswith(".codebook"):
                continue
            else:
                gguf_name = hf_to_gguf_name(k)
                if gguf_name not in seen_names:
                    final_tensor_entries.append({"name": gguf_name, "tensor": tensor})
                    seen_names.add(gguf_name)

        rms_norm_eps = float(text_cfg.get("rms_norm_eps", 1e-6))
        rope_params = text_cfg.get("rope_parameters")
        rope_theta = float(
            rope_params.get("rope_theta", 10000000.0)
            if isinstance(rope_params, dict)
            else 10000000.0
        )

        # Build GGUF Header and Metadata
        with open(out_file, "wb") as f_out:
            # 1. Magic + Version
            f_out.write(b"GGUF")
            f_out.write(struct.pack("<I", 3))  # Version 3

            tensor_count = len(final_tensor_entries)
            metadata_kv = [
                ("general.architecture", GGUF_METADATA_STRING, arch),
                ("general.name", GGUF_METADATA_STRING, "PocketTitan-Quantized-Model"),
                (f"{arch}.block_count", GGUF_METADATA_UINT32, int(num_layers)),
                (f"{arch}.context_length", GGUF_METADATA_UINT32, int(context_len)),
                (f"{arch}.embedding_length", GGUF_METADATA_UINT32, int(hidden_size)),
                (f"{arch}.feed_forward_length", GGUF_METADATA_UINT32, int(intermediate_size)),
                (f"{arch}.attention.head_count", GGUF_METADATA_UINT32, int(num_heads)),
                (f"{arch}.attention.head_count_kv", GGUF_METADATA_UINT32, int(num_kv_heads)),
                (f"{arch}.attention.layer_norm_rms_epsilon", GGUF_METADATA_FLOAT32, rms_norm_eps),
                (f"{arch}.rope.freq_base", GGUF_METADATA_FLOAT32, rope_theta),
                ("tokenizer.ggml.model", GGUF_METADATA_STRING, "gpt2"),
            ]

            f_out.write(struct.pack("<Q", tensor_count))
            f_out.write(struct.pack("<Q", len(metadata_kv)))

            # Write metadata KV
            for k, v_type, val in metadata_kv:
                k_bytes = k.encode("utf-8")
                f_out.write(struct.pack("<Q", len(k_bytes)))
                f_out.write(k_bytes)
                f_out.write(struct.pack("<I", v_type))
                if v_type == GGUF_METADATA_STRING:
                    val_bytes = str(val).encode("utf-8")
                    f_out.write(struct.pack("<Q", len(val_bytes)))
                    f_out.write(val_bytes)
                elif v_type == GGUF_METADATA_UINT32:
                    f_out.write(struct.pack("<I", int(val)))
                elif v_type == GGUF_METADATA_FLOAT32:
                    f_out.write(struct.pack("<f", float(val)))

            # 2. Tensor Info headers (preliminary calculation of data offsets)
            alignment = 32
            current_offset = 0
            tensor_infos = []

            for item in final_tensor_entries:
                t = item["tensor"]
                name = item["name"]

                # Determine GGML type and bytes
                if t.dtype == torch.float32:
                    ggml_type = GGML_TYPE_F32
                    raw_bytes = t.cpu().numpy().tobytes()
                elif t.dtype == torch.float16 or t.dtype == torch.bfloat16:
                    ggml_type = GGML_TYPE_F16
                    raw_bytes = t.to(torch.float16).cpu().numpy().tobytes()
                elif t.dtype == torch.uint8:
                    ggml_type = GGML_TYPE_Q8_0
                    raw_bytes = t.cpu().numpy().tobytes()
                else:
                    ggml_type = GGML_TYPE_F16
                    raw_bytes = t.to(torch.float16).cpu().numpy().tobytes()

                # Align offset
                padding = (alignment - (current_offset % alignment)) % alignment
                current_offset += padding

                tensor_infos.append(
                    {
                        "name": name,
                        "shape": list(t.shape),
                        "ggml_type": ggml_type,
                        "offset": current_offset,
                        "bytes": raw_bytes,
                    }
                )
                current_offset += len(raw_bytes)

            # Write Tensor Infos
            for info in tensor_infos:
                name_bytes = info["name"].encode("utf-8")
                f_out.write(struct.pack("<Q", len(name_bytes)))
                f_out.write(name_bytes)

                shape = info["shape"]
                f_out.write(struct.pack("<I", len(shape)))
                # Write dims in reverse order (GGUF standard)
                for dim in reversed(shape):
                    f_out.write(struct.pack("<Q", dim))

                f_out.write(struct.pack("<I", info["ggml_type"]))
                f_out.write(struct.pack("<Q", info["offset"]))

            # Pad header to 32-byte alignment before data section
            header_size = f_out.tell()
            pad_before_data = (alignment - (header_size % alignment)) % alignment
            if pad_before_data > 0:
                f_out.write(b"\x00" * pad_before_data)

            # 3. Write Tensor Data Payloads
            for info in tensor_infos:
                raw_bytes = info["bytes"]
                f_out.write(raw_bytes)
                # Pad to alignment
                pad = (alignment - (len(raw_bytes) % alignment)) % alignment
                if pad > 0:
                    f_out.write(b"\x00" * pad)

        total_bytes = out_file.stat().st_size
        return ExportResult(
            format_name="gguf",
            output_path=str(out_file),
            total_tensors=len(final_tensor_entries),
            output_size_bytes=total_bytes,
            status="success",
        )
