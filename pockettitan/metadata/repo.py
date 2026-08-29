"""Repository discovery and configuration inspection for Hugging Face and local models."""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub import HfApi, hf_hub_url

from pockettitan.config import ModelMetadata, TensorAddress


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
    """Extract MoE routing and expert dimensions across diverse model families."""
    is_moe = False
    num_experts = None
    num_experts_per_tok = None
    expert_intermediate_size = None
    shared_expert_intermediate_size = None
    
    # Check common MoE keys (Mixtral, DeepSeek, Qwen-MoE, GLM, DBRX, JetMoE)
    moe_keys = [
        "n_routed_experts", "num_local_experts", "num_experts", "moe_num_experts",
        "n_experts", "num_routed_experts"
    ]
    for key in moe_keys:
        if key in config and config[key] is not None:
            num_experts = int(config[key])
            is_moe = True
            break
            
    # Active experts per token
    top_k_keys = [
        "num_experts_per_tok", "n_active_experts", "num_experts_per_token",
        "top_k", "moe_top_k", "moe_num_experts_per_tok"
    ]
    for key in top_k_keys:
        if key in config and config[key] is not None:
            num_experts_per_tok = int(config[key])
            is_moe = True
            break
            
    # Expert intermediate sizes
    exp_size_keys = [
        "moe_intermediate_size", "expert_intermediate_size",
        "intermediate_size_moe", "moe_dim"
    ]
    for key in exp_size_keys:
        if key in config and config[key] is not None:
            expert_intermediate_size = int(config[key])
            break
            
    # Shared experts (DeepSeek / GLM style)
    shared_keys = ["n_shared_experts", "num_shared_experts"]
    for key in shared_keys:
        if key in config and config[key] is not None:
            num_shared = int(config[key])
            if num_shared > 0:
                is_moe = True
                if "shared_expert_intermediate_size" in config:
                    shared_expert_intermediate_size = int(config["shared_expert_intermediate_size"])
                elif expert_intermediate_size is not None:
                    shared_expert_intermediate_size = expert_intermediate_size * num_shared
            break

    return {
        "is_moe": is_moe,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "expert_intermediate_size": expert_intermediate_size,
        "shared_expert_intermediate_size": shared_expert_intermediate_size,
    }


def inspect_model_repository(
    model_id_or_path: str,
    token: Optional[str] = None,
) -> ModelMetadata:
    """Inspect repository metadata and build complete model descriptor."""
    repo_files = list_repository_files(model_id_or_path, token=token)
    config = fetch_model_config(model_id_or_path, token=token)
    index = fetch_model_index(model_id_or_path, available_files=repo_files, token=token)
    moe_specs = extract_moe_specs(config)
    
    architectures = config.get("architectures", ["UnknownModel"])
    architecture = architectures[0] if isinstance(architectures, list) and architectures else "UnknownModel"
    
    hidden_size = config.get("hidden_size", config.get("d_model", 4096))
    num_hidden_layers = config.get("num_hidden_layers", config.get("n_layer", config.get("num_layers", 32)))
    num_attention_heads = config.get("num_attention_heads", config.get("n_head", 32))
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    intermediate_size = config.get("intermediate_size", config.get("n_inner", None))
    vocab_size = config.get("vocab_size", 32000)
    source_dtype = config.get("torch_dtype", "bfloat16")
    
    shards: List[str] = []
    
    if index and "weight_map" in index:
        tensor_map = index["weight_map"]
        shards = sorted(list(set(tensor_map.values())))
    else:
        # Single or multi-shard safetensors without index.json
        shards = sorted([f for f in repo_files if f.endswith(".safetensors")])
        if not shards:
            shards = ["model.safetensors"]
        
    total_params = index.get("metadata", {}).get("total_size", 0) if index else 0
    
    return ModelMetadata(
        model_id_or_path=model_id_or_path,
        architecture=architecture,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        is_moe=moe_specs["is_moe"],
        num_experts=moe_specs["num_experts"],
        num_experts_per_tok=moe_specs["num_experts_per_tok"],
        expert_intermediate_size=moe_specs["expert_intermediate_size"],
        shared_expert_intermediate_size=moe_specs["shared_expert_intermediate_size"],
        source_dtype=str(source_dtype),
        total_params=total_params,
        active_params=total_params,
        shards=shards,
        tensors={},
    )
