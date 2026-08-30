"""Out-of-core inference engine coordinating VRAM dense core, NVMe PLE, and RAM SLRU experts (Phase R6)."""

import json
import mmap
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import torch

from pockettitan.package.format import PackageManifest
from pockettitan.runtime.expert.manager import ExpertManager
from pockettitan.runtime.ple.store import PleRowStore


class DenseBlobReader:
    """Zero-copy reader for quantized dense core weights in dense/blob.bin."""

    def __init__(self, blob_path: Union[str, Path], manifest: PackageManifest, device: str = "cpu"):
        self.blob_path = Path(blob_path)
        self.manifest = manifest
        self.device = device
        self.dense_entries = {entry.name: entry for entry in manifest.dense}
        
        self._fd = os.open(str(self.blob_path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
        file_size = os.path.getsize(str(self.blob_path))
        self._mmap = mmap.mmap(self._fd, length=0, access=mmap.ACCESS_READ) if file_size > 0 else None

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if hasattr(self, "_fd") and self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def get_tensor_bytes(self, name: str) -> bytes:
        if name not in self.dense_entries:
            raise KeyError(f"Dense tensor '{name}' not found in package manifest")
        entry = self.dense_entries[name]
        return self._mmap[entry.byte_offset : entry.byte_offset + entry.length]


class PocketTitanEngine:
    """Unified out-of-core runtime engine executing .ptitan packages."""

    def __init__(
        self,
        package_dir: Union[str, Path],
        ram_budget_slots: int = 2880,  # ~7.0 GB RAM for 4-bit experts
        vram_budget_slots: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.package_dir = Path(package_dir)
        self.device = device
        
        # 1. Load manifest
        manifest_path = self.package_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self.manifest = PackageManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

        # 2. Initialize Dense Blob Reader
        blob_path = self.package_dir / "dense" / "blob.bin"
        self.dense_reader = DenseBlobReader(blob_path, self.manifest, device=device)

        # 3. Initialize PLE Row Store (R5)
        ple_table = self.package_dir / "ple" / "table.bin"
        ple_index_path = self.package_dir / "ple" / "index.json"
        if ple_table.exists() and ple_index_path.exists():
            from pockettitan.package.format import PleIndex
            ple_index = PleIndex.model_validate_json(ple_index_path.read_text(encoding="utf-8"))
            self.ple_store = PleRowStore(ple_table, ple_index)
        else:
            self.ple_store = None

        # 4. Initialize Expert Manager (R6)
        bank_path = self.package_dir / "experts" / "bank.bin"
        if bank_path.exists() and self.manifest.expert_layout:
            self.expert_manager = ExpertManager(
                bank_path=bank_path,
                layout=self.manifest.expert_layout,
                ram_capacity_slots=ram_budget_slots,
                vram_capacity_slots=vram_budget_slots,
                device=device,
            )
            from pockettitan.runtime.prefetch import SpeculativePrefetcher
            from pockettitan.runtime.session import SessionAdapter
            self.prefetcher = SpeculativePrefetcher(self.expert_manager)
            self.session_adapter = SessionAdapter(self.expert_manager)
        else:
            self.expert_manager = None
            self.prefetcher = None
            self.session_adapter = None

    def close(self) -> None:
        if self.dense_reader:
            self.dense_reader.close()
        if self.ple_store:
            self.ple_store.close()
        if self.prefetcher:
            self.prefetcher.close()
        if self.expert_manager:
            self.expert_manager.close()

    def __enter__(self) -> "PocketTitanEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_memory_profile(self) -> Dict[str, Any]:
        """Inspect active memory residency across VRAM, RAM, and NVMe."""
        stats = {
            "model_id": self.manifest.source_model,
            "architecture": self.manifest.architecture,
            "dense_vram_bytes": self.manifest.totals.dense_bytes,
            "ram_resident_experts": (
                self.expert_manager.ram_cache.total_resident if self.expert_manager else 0
            ),
            "vram_resident_experts": (
                len(self.expert_manager.vram_hot_tier) if self.expert_manager else 0
            ),
            "ram_capacity_slots": (
                self.expert_manager.ram_cache.capacity_slots if self.expert_manager else 0
            ),
            "ram_cache_hit_rate": (
                (
                    self.expert_manager.ram_cache.hits
                    / (self.expert_manager.ram_cache.hits + self.expert_manager.ram_cache.misses)
                )
                if self.expert_manager and (self.expert_manager.ram_cache.hits + self.expert_manager.ram_cache.misses) > 0
                else 0.0
            ),
        }
        return stats
