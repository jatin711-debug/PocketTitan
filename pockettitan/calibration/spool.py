"""Activation spooling between layers for single-pass forward calibration chaining."""

import os
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Union
import torch


class ActivationSpool:
    """Manages layer activations in pinned host memory with optional disk spooling."""

    def __init__(
        self,
        max_in_memory_mb: float = 1024.0,
        spool_dir: Optional[Union[str, Path]] = None,
    ):
        self.max_bytes = max_in_memory_mb * 1024 * 1024
        self.spool_dir = Path(spool_dir) if spool_dir else Path(tempfile.gettempdir()) / "pockettitan_spool"
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        
        self._in_memory_buffers: List[torch.Tensor] = []
        self._disk_spool_files: List[Path] = []
        self._current_in_memory_bytes = 0

    def add_activation_batch(self, x: torch.Tensor) -> None:
        """Add activation batch [batch_size, seq_len, hidden_dim]."""
        x_cpu = x.detach().cpu().contiguous()
        num_bytes = x_cpu.nbytes
        
        if self._current_in_memory_bytes + num_bytes > self.max_bytes:
            # Spool to temporary file
            idx = len(self._disk_spool_files)
            file_path = self.spool_dir / f"act_batch_{idx}.pt"
            torch.save(x_cpu, str(file_path))
            self._disk_spool_files.append(file_path)
        else:
            self._in_memory_buffers.append(x_cpu)
            self._current_in_memory_bytes += num_bytes

    def get_batches(self) -> Iterator[torch.Tensor]:
        """Iterate over all stored activation batches."""
        for buf in self._in_memory_buffers:
            yield buf
            
        for file_path in self._disk_spool_files:
            if file_path.exists():
                tensor = torch.load(str(file_path), map_location="cpu", weights_only=True)
                yield tensor

    def clear(self) -> None:
        """Clear memory buffers and delete spooled disk files."""
        self._in_memory_buffers.clear()
        self._current_in_memory_bytes = 0
        
        for file_path in self._disk_spool_files:
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
        self._disk_spool_files.clear()
