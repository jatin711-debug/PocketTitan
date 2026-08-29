"""Virtual Tensor Address Table builder for local and remote sharded Safetensors checkpoints."""

import concurrent.futures
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from huggingface_hub import hf_hub_url

from pockettitan.config import ModelMetadata, TensorAddress
from pockettitan.metadata.repo import fetch_model_config, fetch_model_index, inspect_model_repository
from pockettitan.metadata.safetensors_header import (
    parse_local_safetensors_header,
    parse_remote_safetensors_header,
)


DTYPE_BYTE_SIZES: Dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "FLOAT8_E4M3FN": 1,
    "FLOAT8_E5M2": 1,
}


class TensorAddressTable:
    """In-memory index of all addressable tensors across checkpoint shards."""

    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.tensors: Dict[str, TensorAddress] = metadata.tensors
        self._shard_to_tensors: Dict[str, List[TensorAddress]] = {}
        self._parsed_shards: Set[str] = set()
        self._reindex()

    def _reindex(self) -> None:
        self._shard_to_tensors.clear()
        for tensor in self.tensors.values():
            self._shard_to_tensors.setdefault(tensor.shard, []).append(tensor)

    def add_tensor(self, tensor: TensorAddress) -> None:
        self.tensors[tensor.name] = tensor
        self._shard_to_tensors.setdefault(tensor.shard, []).append(tensor)

    def get_tensor(self, name: str) -> TensorAddress:
        if name not in self.tensors:
            raise KeyError(f"Tensor '{name}' not found in address table")
        return self.tensors[name]

    def get_tensors_in_shard(self, shard: str) -> List[TensorAddress]:
        return self._shard_to_tensors.get(shard, [])

    def get_tensors_for_layer(self, layer_idx: int) -> List[TensorAddress]:
        """Find all tensors belonging to a specific transformer layer."""
        patterns = [f"layers.{layer_idx}.", f"layer.{layer_idx}.", f"h.{layer_idx}.", f"blk.{layer_idx}."]
        return [
            t for t in self.tensors.values()
            if any(pat in t.name for pat in patterns)
        ]

    def get_tensors_for_expert(self, layer_idx: int, expert_idx: int) -> List[TensorAddress]:
        """Find all tensors belonging to a specific MoE expert in a layer."""
        layer_tensors = self.get_tensors_for_layer(layer_idx)
        expert_patterns = [
            f"expert.{expert_idx}.",
            f"experts.{expert_idx}.",
            f"experts.{expert_idx}_",
            f"mlp.experts.{expert_idx}.",
        ]
        return [
            t for t in layer_tensors
            if any(pat in t.name for pat in expert_patterns)
        ]

    def get_non_moe_tensors(self) -> List[TensorAddress]:
        """Return non-expert tensors (attention, norms, routers, embeddings, lm_head)."""
        return [
            t for t in self.tensors.values()
            if not any(exp in t.name for exp in [".expert.", ".experts.", "mlp.experts"])
        ]

    @property
    def total_params(self) -> int:
        if self.metadata.total_params > 0:
            return self.metadata.total_params
        return sum(t.num_params for t in self.tensors.values())

    @property
    def total_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors.values())

    def largest_tensors(self, top_n: int = 10) -> List[TensorAddress]:
        return sorted(self.tensors.values(), key=lambda t: t.size_bytes, reverse=True)[:top_n]

    def compute_active_params(self) -> int:
        """Compute active parameter count per token taking MoE sparsity into account."""
        total = self.total_params
        if not self.metadata.is_moe or not self.metadata.num_experts or not self.metadata.num_experts_per_tok:
            return total
        
        moe_ratio = self.metadata.num_experts_per_tok / self.metadata.num_experts
        dense_est = total * 0.12
        routed_est = (total * 0.88) * moe_ratio
        return int(dense_est + routed_est)


def _fetch_single_shard(
    shard: str,
    model_id_or_path: str,
    is_local: bool,
    headers: Optional[Dict[str, str]],
) -> Tuple[str, Dict[str, Any], int]:
    """Fetch header for a single shard."""
    if is_local:
        shard_path = Path(model_id_or_path) / shard
        if not shard_path.exists():
            return shard, {}, 0
        header_dict, header_bytes = parse_local_safetensors_header(shard_path)
        return shard, header_dict, header_bytes
    else:
        try:
            url = hf_hub_url(repo_id=model_id_or_path, filename=shard)
            header_dict, header_bytes = parse_remote_safetensors_header(
                url, headers=headers
            )
            return shard, header_dict, header_bytes
        except Exception:
            return shard, {}, 0


def build_tensor_address_table(
    model_id_or_path: str,
    token: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    fast_inspect: bool = False,
    max_shards_to_probe: Optional[int] = None,
    max_workers: int = 16,
) -> TensorAddressTable:
    """Build virtual address table fast using index.json and parallel shard header probes.
    
    Args:
        model_id_or_path: Local directory or Hugging Face repository ID.
        token: Optional HF auth token.
        headers: Optional HTTP headers.
        fast_inspect: If True, samples a small subset of shards (e.g. 8) for fast CLI inspect preview.
                      If False (default for quantization & full execution), probes 100% of all shards.
        max_shards_to_probe: Explicit limit on number of shards to probe if fast_inspect is active.
        max_workers: Parallel worker thread count for remote HTTP range header requests.
    """
    model_meta = inspect_model_repository(model_id_or_path, token=token)
    table = TensorAddressTable(model_meta)
    
    path = Path(model_id_or_path)
    is_local = path.exists() and path.is_dir()
    
    target_shards = model_meta.shards
    if fast_inspect and not is_local:
        limit = max_shards_to_probe if max_shards_to_probe is not None else 8
        if len(target_shards) > limit:
            target_shards = target_shards[:limit - 1] + [target_shards[-1]]
            
    num_workers = min(max_workers, max(1, len(target_shards)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_fetch_single_shard, shard, model_id_or_path, is_local, headers)
            for shard in target_shards
        ]
        
        for future in concurrent.futures.as_completed(futures):
            shard, header_dict, header_bytes = future.result()
            table._parsed_shards.add(shard)
            for tensor_name, tensor_info in header_dict.items():
                if tensor_name == "__metadata__":
                    continue
                
                dtype = tensor_info.get("dtype", "F16")
                shape = tensor_info.get("shape", [])
                data_offsets = tensor_info.get("data_offsets", [0, 0])
                
                byte_start = header_bytes + data_offsets[0]
                byte_end = header_bytes + data_offsets[1]
                size_bytes = data_offsets[1] - data_offsets[0]
                num_params = math.prod(shape) if shape else 0
                
                addr = TensorAddress(
                    name=tensor_name,
                    shard=shard,
                    dtype=dtype,
                    shape=shape,
                    byte_start=byte_start,
                    byte_end=byte_end,
                    num_params=num_params,
                    size_bytes=size_bytes,
                )
                table.add_tensor(addr)
                
    if model_meta.total_params == 0:
        model_meta.total_params = table.total_params
    model_meta.active_params = table.compute_active_params()
    model_meta.tensors = table.tensors
    return table
