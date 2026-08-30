"""Transactional execution manifest for crash recovery and resumable quantization jobs."""

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class TensorStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TensorJobRecord(BaseModel):
    name: str
    shard: str
    status: TensorStatus = TensorStatus.PENDING
    bit_width: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    output_shard: Optional[str] = None
    error_message: Optional[str] = None


class JobManifest(BaseModel):
    model_id_or_path: str
    output_dir: str
    quant_method: str
    bits: int
    group_size: int
    total_tensors: int
    completed_tensors: int = 0
    records: Dict[str, TensorJobRecord] = Field(default_factory=dict)

    def is_complete(self) -> bool:
        return self.completed_tensors >= self.total_tensors

    def get_pending_tensors(self) -> List[str]:
        return [
            name
            for name, rec in self.records.items()
            if rec.status in (TensorStatus.PENDING, TensorStatus.FAILED)
        ]


class ManifestManager:
    """Manages atomic reads and writes of the quantization manifest on disk."""

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.manifest_file = self.output_dir / "pockettitan_manifest.json"

    def exists(self) -> bool:
        return self.manifest_file.exists()

    def load(self) -> JobManifest:
        """Load manifest from disk."""
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JobManifest.model_validate(data)

    def save(self, manifest: JobManifest) -> None:
        """Atomically write manifest to disk using temporary file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = self.output_dir / "pockettitan_manifest.json.tmp"

        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        if tmp_file.exists():
            tmp_file.replace(self.manifest_file)

    def create_initial(
        self,
        model_id_or_path: str,
        tensor_names_with_shards: List[tuple[str, str]],
        quant_method: str,
        bits: int,
        group_size: int,
    ) -> JobManifest:
        """Create new manifest initialized with pending records."""
        records = {}
        for name, shard in tensor_names_with_shards:
            records[name] = TensorJobRecord(name=name, shard=shard, status=TensorStatus.PENDING)

        manifest = JobManifest(
            model_id_or_path=model_id_or_path,
            output_dir=str(self.output_dir),
            quant_method=quant_method,
            bits=bits,
            group_size=group_size,
            total_tensors=len(records),
            completed_tensors=0,
            records=records,
        )
        self.save(manifest)
        return manifest
