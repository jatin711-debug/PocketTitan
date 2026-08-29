"""Zero-copy memory-mapped local reader and direct HTTP Range tensor slice streaming reader."""

import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import safetensors.torch
import torch
from huggingface_hub import hf_hub_url

from pockettitan.config import TensorAddress
from pockettitan.metadata.safetensors_header import (
    RedirectRangeHandler,
    fetch_remote_bytes,
)


DTYPE_MAP_TORCH = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

DTYPE_MAP_NUMPY = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # NumPy does not natively support bfloat16 in older versions, view as uint16
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


class LocalTensorReader:
    """Zero-copy local Safetensors reader using OS memory mapping."""

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self._handles: Dict[str, Any] = {}

    def _get_handle(self, shard_name: str):
        if shard_name not in self._handles:
            shard_path = self.root_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            self._handles[shard_name] = safetensors.torch.safe_open(
                str(shard_path), framework="pt", device="cpu"
            )
        return self._handles[shard_name]

    def read_tensor(self, tensor_addr: TensorAddress) -> torch.Tensor:
        """Read full tensor via zero-copy memory mapping."""
        handle = self._get_handle(tensor_addr.shard)
        return handle.get_tensor(tensor_addr.name)

    def read_slice(
        self,
        tensor_addr: TensorAddress,
        row_start: int,
        row_end: int,
    ) -> torch.Tensor:
        """Read a sub-slice of a 2D matrix directly without loading the full tensor."""
        handle = self._get_handle(tensor_addr.shard)
        # safetensors get_slice supports slice indexing
        tensor_slice = handle.get_slice(tensor_addr.name)
        return tensor_slice[row_start:row_end, :]

    def close(self) -> None:
        """Close open file handles."""
        self._handles.clear()


class RemoteTensorSliceReader:
    """Direct HTTP Range tensor slice streamer fetching only exact byte ranges across CDN."""

    def __init__(
        self,
        model_id: str,
        token: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.model_id = model_id
        self.token = token
        self.headers = headers or {}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def read_tensor(self, tensor_addr: TensorAddress) -> torch.Tensor:
        """Fetch exact tensor bytes over network using HTTP Range header."""
        url = hf_hub_url(repo_id=self.model_id, filename=tensor_addr.shard)
        raw_bytes = fetch_remote_bytes(
            url,
            byte_start=tensor_addr.byte_start,
            byte_end=tensor_addr.byte_end - 1,
            headers=self.headers,
        )
        return self._bytes_to_tensor(raw_bytes, tensor_addr.dtype, tensor_addr.shape)

    def read_slice(
        self,
        tensor_addr: TensorAddress,
        row_start: int,
        row_end: int,
    ) -> torch.Tensor:
        """Fetch exact row chunk over network using computed byte offsets."""
        shape = tensor_addr.shape
        if len(shape) != 2:
            # Fallback to full tensor
            full_t = self.read_tensor(tensor_addr)
            return full_t[row_start:row_end]
            
        out_features, in_features = shape[0], shape[1]
        bytes_per_row = (tensor_addr.size_bytes) // max(1, out_features)
        
        slice_byte_start = tensor_addr.byte_start + (row_start * bytes_per_row)
        slice_byte_end = tensor_addr.byte_start + (row_end * bytes_per_row)
        
        url = hf_hub_url(repo_id=self.model_id, filename=tensor_addr.shard)
        raw_bytes = fetch_remote_bytes(
            url,
            byte_start=slice_byte_start,
            byte_end=slice_byte_end - 1,
            headers=self.headers,
        )
        slice_shape = [row_end - row_start, in_features]
        return self._bytes_to_tensor(raw_bytes, tensor_addr.dtype, slice_shape)

    @staticmethod
    def _bytes_to_tensor(
        raw_bytes: bytes,
        dtype_str: str,
        shape: List[int],
    ) -> torch.Tensor:
        """Convert raw bytes to PyTorch tensor matching dtype and shape."""
        np_dtype = DTYPE_MAP_NUMPY.get(dtype_str, np.float16)
        arr = np.frombuffer(raw_bytes, dtype=np_dtype)
        
        if dtype_str == "BF16":
            # Reinterpret uint16 array as bfloat16 tensor
            torch_tensor = torch.from_numpy(arr.copy()).view(torch.bfloat16)
        else:
            torch_tensor = torch.from_numpy(arr.copy())
            
        return torch_tensor.view(*shape)
