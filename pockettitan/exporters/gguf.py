"""GGUF binary format exporter for llama.cpp execution."""

import io
import json
from pathlib import Path
import struct
from typing import Any, Dict, List, Union
import numpy as np
import safetensors.torch
import torch

from pockettitan.exporters.base import BaseExporter, ExportResult

# GGML Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_IQ2_XXS = 19

# GGUF Value Types
GGUF_METADATA_UINT32 = 4
GGUF_METADATA_STRING = 8


class GGUFExporter(BaseExporter):
    """Exports PocketTitan quantized model to GGUF format for llama.cpp."""

    def export(self, output_path: Union[str, Path]) -> ExportResult:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load config.json if present
        config_path = self.checkpoint_dir / "config.json"
        config_dict = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
                
        arch = config_dict.get("architectures", ["llama"])[0].lower().replace("forcausallm", "")
        
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

        tensor_entries: List[Dict[str, Any]] = []
        
        # Read tensors
        for shard_name in shards_to_read:
            shard_path = self.checkpoint_dir / shard_name
            if not shard_path.exists():
                continue
            with safetensors.torch.safe_open(str(shard_path), framework="pt") as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    tensor_entries.append({"name": k, "tensor": t})

        # Build GGUF Header and Metadata
        with open(out_file, "wb") as f_out:
            # 1. Magic + Version
            f_out.write(b"GGUF")
            f_out.write(struct.pack("<I", 3))  # Version 3
            
            tensor_count = len(tensor_entries)
            metadata_kv = [
                ("general.architecture", GGUF_METADATA_STRING, arch),
                ("general.name", GGUF_METADATA_STRING, "PocketTitan-Quantized-Model"),
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
                    val_bytes = val.encode("utf-8")
                    f_out.write(struct.pack("<Q", len(val_bytes)))
                    f_out.write(val_bytes)
                    
            # 2. Tensor Info headers (preliminary calculation of data offsets)
            alignment = 32
            current_offset = 0
            tensor_infos = []
            
            for item in tensor_entries:
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
                
                tensor_infos.append({
                    "name": name,
                    "shape": list(t.shape),
                    "ggml_type": ggml_type,
                    "offset": current_offset,
                    "bytes": raw_bytes,
                })
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
            total_tensors=len(tensor_entries),
            output_size_bytes=total_bytes,
            status="success",
        )
