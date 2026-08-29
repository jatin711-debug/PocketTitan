"""vLLM and SGLang format exporter for GPU high-throughput inference."""

import json
from pathlib import Path
import shutil
from typing import Dict, Union
import safetensors.torch
import torch

from pockettitan.exporters.base import BaseExporter, ExportResult


class VLLMExporter(BaseExporter):
    """Exports PocketTitan quantized model to vLLM / SGLang compatible Safetensors layout."""

    def export(self, output_path: Union[str, Path]) -> ExportResult:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Copy and patch config.json
        config_path = self.checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["quantization_config"] = {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
            }
            with open(out_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        # 2. Copy Safetensors shards and index
        shard_files = list(self.checkpoint_dir.glob("*.safetensors"))
        total_size = 0
        total_tensors = 0
        
        for sf in shard_files:
            dest = out_dir / sf.name
            shutil.copy2(sf, dest)
            total_size += dest.stat().st_size
            with safetensors.torch.safe_open(str(dest), framework="pt") as f:
                total_tensors += len(f.keys())
                
        index_file = self.checkpoint_dir / "model.safetensors.index.json"
        if index_file.exists():
            shutil.copy2(index_file, out_dir / "model.safetensors.index.json")

        return ExportResult(
            format_name="vllm",
            output_path=str(out_dir),
            total_tensors=total_tensors,
            output_size_bytes=total_size,
            status="success",
        )
