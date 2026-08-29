"""Core configuration models and data structures for PocketTitan."""

from enum import Enum
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"
    ROCM = "rocm"
    MPS = "mps"


class QuantMethod(str, Enum):
    RTN = "rtn"
    HQQ = "hqq"
    TERNARY = "ternary"
    INTX = "intx"
    GPTQ = "gptq"
    AWQ = "awq"
    AUTOROUND = "autoround"
    AUTO = "auto"


def parse_memory_to_mb(val: Union[str, float, int]) -> float:
    """Parse user memory string into MiB float.
    
    Supports: "4GB", "4GiB", "4G", "2048MB", "2048MiB", "2048M", "1.5GB", "1500", 1500
    """
    if isinstance(val, (int, float)):
        return float(val)
        
    s = str(val).strip().upper()
    if not s:
        return 3584.0
        
    # Match number and optional unit
    match = re.match(r"^([0-9.]+)\s*([A-Z]*)$", s)
    if not match:
        try:
            return float(s)
        except ValueError:
            return 3584.0
            
    num = float(match.group(1))
    unit = match.group(2)
    
    if unit in ["GB", "GIB", "G"]:
        return num * 1024.0
    elif unit in ["TB", "TIB", "T"]:
        return num * 1024.0 * 1024.0
    elif unit in ["MB", "MIB", "M", ""]:
        return num
    elif unit in ["KB", "KIB", "K"]:
        return num / 1024.0
    elif unit in ["B"]:
        return num / (1024.0 * 1024.0)
    return num


class MemoryBudgetConfig(BaseModel):
    """Memory budget specifications in Megabytes."""
    max_vram_mb: float = Field(default=3584.0, description="Hard cap on CUDA VRAM in MiB (default ~3.5 GiB for 4GB GPUs)")
    runtime_reserve_mb: float = Field(default=512.0, description="Reserved for PyTorch/CUDA runtime overhead in MiB")
    safety_margin_mb: float = Field(default=384.0, description="Safety buffer before triggering OOM in MiB")
    max_source_cache_mb: float = Field(default=10240.0, description="Max local disk cache for source shards (default 10 GiB)")
    max_cpu_staging_mb: float = Field(default=2048.0, description="Max pinned host CPU staging buffer in MiB")

    @property
    def usable_vram_mb(self) -> float:
        # Dynamically scale reserves if user requested a very small budget (e.g. 1000 MB or 1500 MB)
        if self.max_vram_mb <= 2048.0:
            reserve = min(self.runtime_reserve_mb, self.max_vram_mb * 0.15)
            margin = min(self.safety_margin_mb, self.max_vram_mb * 0.10)
            return max(256.0, self.max_vram_mb - reserve - margin)
        usable = self.max_vram_mb - self.runtime_reserve_mb - self.safety_margin_mb
        return max(256.0, usable)


class QuantConfig(BaseModel):
    """Configuration for quantization backends."""
    method: QuantMethod = Field(default=QuantMethod.HQQ, description="Quantization algorithm")
    bits: int = Field(default=2, description="Target bit-width (e.g. 1, 2, 3, 4, 8)")
    group_size: int = Field(default=128, description="Group size for groupwise quantization (-1 for channel/tensor-wise)")
    symmetric: bool = Field(default=False, description="Symmetric vs asymmetric quantization")
    scale_dtype: str = Field(default="float16", description="Data type for quantization scales")
    zero_dtype: str = Field(default="float16", description="Data type for quantization zero-points")
    device: str = Field(default="cuda", description="Execution device (cuda/cpu)")


class TensorAddress(BaseModel):
    """Virtual address mapping for a tensor in a sharded checkpoint."""
    name: str
    shard: str
    dtype: str
    shape: List[int]
    byte_start: int
    byte_end: int
    num_params: int
    size_bytes: int


class ModelMetadata(BaseModel):
    """Normalized metadata for any LLM architecture."""
    architecture: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    total_params: int
    active_params: int
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    expert_intermediate_size: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    source_dtype: str = "float16"
    shards: List[str] = Field(default_factory=list)
    tensors: Dict[str, TensorAddress] = Field(default_factory=dict)
