"""Core configuration models, exceptions, and data structures for PocketTitan."""

from enum import Enum
import math
import re
from typing import Dict, List, Optional, Union
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


# --- Custom Domain Exceptions ---


class UnsupportedSourceDTypeError(Exception):
    """Raised when an unrecognized source data type is encountered."""

    pass


class TruncatedTensorError(IOError):
    """Fewer bytes arrived than the tensor's shape and dtype require.

    Distinguished from a shape bug because the fix is different: retry the
    transfer, do not go looking at the layout.
    """


class CalibrationRequiredError(Exception):
    """Raised when an algorithm (e.g., GPTQ, AWQ) requires calibration data that was not provided."""

    pass


class InfeasibleBudgetError(Exception):
    """Raised when a matrix cannot be legally tiled within the configured memory budget."""

    pass


def parse_memory_to_mb(val: Union[str, float, int]) -> float:
    """Parse user memory string into MiB float.

    Supports: "4GB", "4GiB", "4G", "2048MB", "2048MiB", "2048M", "1.5GB", "1500", 1500
    """
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().upper()
    if not s:
        return 3584.0

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

    max_vram_mb: float = Field(
        default=3584.0, description="Hard cap on CUDA VRAM in MiB (default ~3.5 GiB for 4GB GPUs)"
    )
    runtime_reserve_mb: float = Field(
        default=512.0, description="Reserved for PyTorch/CUDA runtime overhead in MiB"
    )
    safety_margin_mb: float = Field(
        default=384.0, description="Safety buffer before triggering OOM in MiB"
    )
    max_source_cache_mb: float = Field(
        default=10240.0, description="Max local disk cache for source shards (default 10 GiB)"
    )
    max_cpu_staging_mb: float = Field(
        default=2048.0, description="Max pinned host CPU staging buffer in MiB"
    )

    @property
    def usable_vram_mb(self) -> float:
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
    group_size: int = Field(
        default=128,
        description="Group size for groupwise quantization (-1 for channel/tensor-wise)",
    )
    symmetric: bool = Field(default=False, description="Symmetric vs asymmetric quantization")
    scale_dtype: str = Field(default="float16", description="Data type for quantization scales")
    zero_dtype: str = Field(default="float16", description="Data type for quantization zero-points")
    device: str = Field(default="cuda", description="Execution device (cuda/cpu)")


class StorageAccounting(BaseModel):
    """Scientific storage accounting distinguishing theoretical, payload, and on-disk metrics."""

    theoretical_bpw: float = Field(
        description="Information-theoretic entropy limit (e.g. 1.585 for ternary)"
    )
    payload_bpw: float = Field(
        description="Bit-width of physical packed tensor elements (e.g. 2.0)"
    )
    metadata_bpw: float = Field(
        description="Overhead of scales, zero-points, and codebooks in bits/weight"
    )
    on_disk_bpw: float = Field(description="Actual physical storage on disk per parameter")
    compression_ratio: float = Field(
        description="Compression factor relative to FP16 source baseline"
    )

    @classmethod
    def compute(
        cls,
        method: QuantMethod,
        bits: int,
        group_size: int,
        shape: List[int],
        has_zeros: bool = True,
        scale_bytes_per_elem: int = 2,
    ) -> "StorageAccounting":
        num_params = math.prod(shape) if shape else 1
        if method == QuantMethod.TERNARY:
            theoretical_bpw = math.log2(3)  # ~1.58496
            payload_bpw = 2.0  # Packed 4 values per uint8
        else:
            theoretical_bpw = float(bits)
            payload_bpw = float(bits)

        # Scale & zero metadata calculations
        num_groups = math.ceil(num_params / group_size) if group_size > 0 else 1
        metadata_bytes = num_groups * scale_bytes_per_elem * (2 if has_zeros else 1)
        metadata_bpw = (metadata_bytes * 8.0) / num_params

        payload_bytes = math.ceil(num_params * payload_bpw / 8.0)
        total_on_disk_bytes = payload_bytes + metadata_bytes
        on_disk_bpw = (total_on_disk_bytes * 8.0) / num_params

        fp16_bytes = num_params * 2
        compression_ratio = fp16_bytes / max(1, total_on_disk_bytes)

        return cls(
            theoretical_bpw=round(theoretical_bpw, 4),
            payload_bpw=round(payload_bpw, 4),
            metadata_bpw=round(metadata_bpw, 4),
            on_disk_bpw=round(on_disk_bpw, 4),
            compression_ratio=round(compression_ratio, 2),
        )


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
    num_key_value_heads: int = 32
    intermediate_size: Optional[int] = None
    vocab_size: int = 32000
    total_params: int = 0
    active_params: int = 0
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    expert_intermediate_size: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    first_k_dense_replace: Optional[int] = None
    source_dtype: str = "float16"
    is_fp8_source: bool = False
    shards: List[str] = Field(default_factory=list)
    tensors: Dict[str, TensorAddress] = Field(default_factory=dict)
