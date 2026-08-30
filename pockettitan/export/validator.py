"""Milestone 7: Checkpoint validator and integrity auditor."""

import json
from pathlib import Path
from typing import Dict, List, Union
from pydantic import BaseModel, Field
import safetensors.torch


class ValidationScorecard(BaseModel):
    is_valid: bool
    total_shards: int
    total_tensors: int
    total_size_bytes: int
    quantized_tensor_count: int
    passthrough_tensor_count: int
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CheckpointValidator:
    """Validates the structural and mathematical integrity of an exported PocketTitan checkpoint."""

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)

    def validate(self) -> ValidationScorecard:
        """Run full integrity check suite on output folder."""
        errors: List[str] = []
        warnings: List[str] = []

        if not self.checkpoint_dir.exists():
            return ValidationScorecard(
                is_valid=False,
                total_shards=0,
                total_tensors=0,
                total_size_bytes=0,
                quantized_tensor_count=0,
                passthrough_tensor_count=0,
                errors=[f"Directory does not exist: {self.checkpoint_dir}"],
            )

        # 1. Check index.json
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        single_file = self.checkpoint_dir / "model.safetensors"

        weight_map: Dict[str, str] = {}
        total_size_bytes = 0

        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                weight_map = idx_data.get("weight_map", {})
                total_size_bytes = idx_data.get("metadata", {}).get("total_size", 0)
            except Exception as e:
                errors.append(f"Invalid model.safetensors.index.json: {e}")
        elif single_file.exists():
            weight_map = {"model": "model.safetensors"}
        else:
            errors.append("No model.safetensors.index.json or model.safetensors found in directory")

        # 2. Check config.json and quant_config.json
        if not (self.checkpoint_dir / "config.json").exists():
            warnings.append("Missing config.json in output directory")
        if not (self.checkpoint_dir / "quant_config.json").exists():
            warnings.append("Missing quant_config.json in output directory")

        # 3. Check shard files exist and are readable
        referenced_shards = set(weight_map.values())
        total_tensors = len(weight_map)
        quantized_count = 0
        passthrough_count = 0

        for shard_name in referenced_shards:
            shard_file = self.checkpoint_dir / shard_name
            if not shard_file.exists():
                errors.append(f"Referenced shard file missing: {shard_name}")
                continue

            if shard_file.stat().st_size == 0:
                errors.append(f"Shard file is empty: {shard_name}")
                continue

            # Verify safe_open can inspect shard
            try:
                with safetensors.torch.safe_open(str(shard_file), framework="pt") as f:
                    keys = f.keys()
                    for k in keys:
                        if ".packed_weight" in k:
                            quantized_count += 1
                        elif not any(s in k for s in [".scales", ".zeros", ".codebook"]):
                            passthrough_count += 1
            except Exception as e:
                errors.append(f"Failed to parse Safetensors shard {shard_name}: {e}")

        is_valid = len(errors) == 0 and total_tensors > 0

        return ValidationScorecard(
            is_valid=is_valid,
            total_shards=len(referenced_shards),
            total_tensors=total_tensors,
            total_size_bytes=total_size_bytes,
            quantized_tensor_count=quantized_count,
            passthrough_tensor_count=passthrough_count,
            errors=errors,
            warnings=warnings,
        )
