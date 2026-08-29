"""Repository discovery and configuration inspection for Hugging Face and local models."""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub import HfApi, hf_hub_url

from pockettitan.config import ModelMetadata, TensorAddress
from pockettitan.models.adapters import get_model_adapter


def is_remote_repo(model_id_or_path: str) -> bool:
    """Check if model target is a remote HF repo or local directory."""
    path = Path(model_id_or_path)
    return not path.exists()


def _fetch_remote_json(url: str, token: Optional[str] = None) -> Dict[str, Any]:
    """Fetch remote JSON payload directly into memory via HTTP."""
    headers = {"User-Agent": "PocketTitan/0.1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_repository_files(
    model_id_or_path: str,
    token: Optional[str] = None,
) -> List[str]:
    """List all filenames available in a local directory or remote HF repository."""
    path = Path(model_id_or_path)
    if path.exists() and path.is_dir():
        return [p.name for p in path.iterdir()]
    
    api = HfApi(token=token)
    return api.list_repo_files(repo_id=model_id_or_path)


def fetch_model_config(
    model_id_or_path: str,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch config.json from local path or Hugging Face Hub directly into memory."""
    path = Path(model_id_or_path)
    if path.exists() and path.is_dir():
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    url = hf_hub_url(repo_id=model_id_or_path, filename="config.json")
    return _fetch_remote_json(url, token=token)


def fetch_model_index(
    model_id_or_path: str,
    available_files: Optional[List[str]] = None,
    token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch model.safetensors.index.json if present (for sharded models)."""
    if available_files is None:
        available_files = list_repository_files(model_id_or_path, token=token)
        
    if "model.safetensors.index.json" not in available_files:
        return None
        
    path = Path(model_id_or_path)
    if path.exists() and path.is_dir():
        index_path = path / "model.safetensors.index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    url = hf_hub_url(repo_id=model_id_or_path, filename="model.safetensors.index.json")
    try:
        return _fetch_remote_json(url, token=token)
    except Exception:
        return None


def extract_moe_specs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract MoE routing and expert dimensions via model adapter."""
    adapter = get_model_adapter(config)
    return adapter.extract_moe_topology()


def inspect_model_repository(
    model_id_or_path: str,
    token: Optional[str] = None,
) -> ModelMetadata:
    """Inspect repository metadata and build complete model descriptor using model adapter."""
    repo_files = list_repository_files(model_id_or_path, token=token)
    config = fetch_model_config(model_id_or_path, token=token)
    index = fetch_model_index(model_id_or_path, available_files=repo_files, token=token)
    
    adapter = get_model_adapter(config)
    dims = adapter.extract_dimensions()
    moe_specs = adapter.extract_moe_topology()
    architecture = adapter.extract_architecture_name()
    source_dtype, is_fp8 = adapter.extract_source_dtype()
    
    shards: List[str] = []
    
    if index and "weight_map" in index:
        tensor_map = index["weight_map"]
        shards = sorted(list(set(tensor_map.values())))
    else:
        # Single or multi-shard safetensors without index.json
        shards = sorted([f for f in repo_files if f.endswith(".safetensors")])
        if not shards:
            shards = ["model.safetensors"]
        
    total_bytes = index.get("metadata", {}).get("total_size", 0) if index else 0
    bytes_per_elem = 1 if is_fp8 else (4 if "32" in str(source_dtype) else 2)
    total_params = int(total_bytes // bytes_per_elem) if total_bytes > 0 else 0
    
    return ModelMetadata(
        model_id_or_path=model_id_or_path,
        architecture=architecture,
        hidden_size=dims["hidden_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=dims.get("num_key_value_heads", dims["num_attention_heads"]),
        intermediate_size=dims.get("intermediate_size"),
        vocab_size=dims.get("vocab_size", 32000),
        is_moe=moe_specs["is_moe"],
        num_experts=moe_specs["num_experts"],
        num_experts_per_tok=moe_specs["num_experts_per_tok"],
        expert_intermediate_size=moe_specs["expert_intermediate_size"],
        shared_expert_intermediate_size=moe_specs["shared_expert_intermediate_size"],
        first_k_dense_replace=moe_specs.get("first_k_dense_replace"),
        source_dtype=str(source_dtype),
        is_fp8_source=is_fp8,
        total_params=total_params,
        active_params=total_params,
        shards=shards,
        tensors={},
    )
