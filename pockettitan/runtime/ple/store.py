"""Out-of-core PLE row store reader with page-packed indexing and row cache (R5)."""

import mmap
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence, Union
import torch

from pockettitan.config import QuantMethod
from pockettitan.package.decode import decode_record
from pockettitan.package.format import PleIndex
from pockettitan.runtime.ple.hash import PleHasher


class PleRowStore:
    """Async/Mmap reader for the 51.2B-parameter PLE n-gram table on NVMe SSD."""

    def __init__(
        self,
        table_path: Union[str, Path],
        index: PleIndex,
        cache_capacity_rows: int = 65536,  # ~5.3 MB RAM cache
        quant_method: QuantMethod = QuantMethod.RTN,
    ):
        self.table_path = Path(table_path)
        self.index = index
        self.row_layout = index.row
        self.quant_method = quant_method
        self.hasher = PleHasher(index)
        
        self.cache_capacity = cache_capacity_rows
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        
        self._file_handle: Optional[os.PathLike] = None
        self._mmap: Optional[mmap.mmap] = None
        self._open_table()

    def _open_table(self) -> None:
        """Open and memory-map the binary PLE table for zero-copy random access."""
        if not self.table_path.exists():
            raise FileNotFoundError(f"PLE table file not found: {self.table_path}")
            
        self._fd = os.open(str(self.table_path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
        file_size = os.path.getsize(str(self.table_path))
        if file_size > 0:
            self._mmap = mmap.mmap(self._fd, length=0, access=mmap.ACCESS_READ)
        else:
            self._mmap = None

    def close(self) -> None:
        """Release file descriptors and memory mappings."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if hasattr(self, "_fd") and self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "PleRowStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def read_row_raw(self, row_id: int) -> bytes:
        """Read exact row bytes from page-packed layout without straddling page boundaries."""
        if self._mmap is None:
            raise RuntimeError("PLE table is not open")
            
        byte_offset = self.row_layout.row_offset(row_id)
        payload_len = self.row_layout.payload_bytes
        return self._mmap[byte_offset : byte_offset + payload_len]

    def decode_row(self, row_bytes: bytes) -> torch.Tensor:
        """Decode one packed row into an FP16 vector.

        Geometry comes from the index and the reconstruction runs through
        :func:`decode_record`, so this cannot drift from what the writer emitted.
        """
        return decode_record(
            row_bytes,
            shape=(self.row_layout.row_width,),
            bits=self.row_layout.bits,
            group_size=self.row_layout.group_size,
            symmetric=self.row_layout.symmetric,
            method=self.quant_method,
        )

    def fetch_row(self, logical_row_id: int) -> torch.Tensor:
        """Fetch and dequantize a row with LRU caching."""
        # 1. Check in-memory row cache
        if logical_row_id in self.cache:
            self.cache.move_to_end(logical_row_id)
            return self.cache[logical_row_id]
            
        # 2. Resolve logical-to-physical row address if sparse package
        physical_id = (
            self.index.physical_row_id(logical_row_id) if self.index.shards else logical_row_id
        )
        
        # 3. Read and decode
        row_bytes = self.read_row_raw(physical_id)
        tensor = self.decode_row(row_bytes)
        
        # 4. Insert into LRU cache
        if len(self.cache) >= self.cache_capacity:
            self.cache.popitem(last=False)
        self.cache[logical_row_id] = tensor
        return tensor

    def fetch_step_embedding(self, token_history: Sequence[int]) -> torch.Tensor:
        """Resolve all 16 head rows for the current token and return stacked [16, 160] tensor."""
        row_ids = self.hasher.hash_all_heads(token_history)
        tensors = [self.fetch_row(r) for r in row_ids]
        return torch.stack(tensors, dim=0)  # Shape: [16, 160]
