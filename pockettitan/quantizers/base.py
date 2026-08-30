"""Base interfaces, data structures, and capability contracts for quantization backends."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch

from pockettitan.config import QuantConfig


def matrix_dims(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """``(out_features, in_features)`` exactly as ``quantize`` flattens a weight.

    Every ``quantize`` does ``weight.view(-1, shape[-1])``, so the row count is
    the product of the leading dimensions, not ``shape[0]``. Reading it as
    ``(shape[0], shape[1])`` happens to agree only for 2-D weights, and silently
    transposes 1-D vectors and 3-D convolution kernels.
    """
    if not shape:
        return 1, 1
    if len(shape) == 1:
        return 1, shape[0]
    return math.prod(shape[:-1]), shape[-1]


@dataclass
class QuantizerCapabilities:
    """Explicit mathematical and architectural constraints for a quantizer backend."""

    name: str
    requires_calibration: bool
    legal_split_axes: Tuple[str, ...]  # ("out_features",) means can split on out_features rows
    requires_full_input_dim: (
        bool  # True if algorithm requires seeing the complete in_features column
    )
    requires_full_output_dim: (
        bool  # True if algorithm requires seeing the complete out_features row
    )
    global_state: Optional[str]  # e.g. 'hessian', 'activation_max', None
    supports_cpu: bool
    supports_cuda: bool
    supports_remote_streaming: bool
    workspace_multiplier: float  # Memory factor during quantization (e.g. 2.0x)


@dataclass
class QuantizedResult:
    """Output container for a quantized weight matrix or tile."""

    packed_weights: torch.Tensor  # Packed integer or sub-byte codes (e.g. uint8)
    scales: torch.Tensor  # Quantization scales
    zeros: Optional[torch.Tensor]  # Zero-point offsets (for asymmetric quantization)
    codebook: Optional[torch.Tensor]  # Optional codebook (for vector / lattice quantizers)
    quant_config: QuantConfig
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    bit_width: float  # Effective bits per weight
    device: str

    def size_bytes(self) -> int:
        total = self.packed_weights.nbytes + self.scales.nbytes
        if self.zeros is not None:
            total += self.zeros.nbytes
        if self.codebook is not None:
            total += self.codebook.nbytes
        return total

    def to_packed_tensors(self, name_prefix: str = "weight") -> Dict[str, torch.Tensor]:
        """Convert quantized result to standard named tensors for Safetensors serialization."""
        base_name = (
            name_prefix.rsplit(".weight", 1)[0] if name_prefix.endswith(".weight") else name_prefix
        )
        tensors = {
            f"{base_name}.packed_weight": self.packed_weights,
            f"{base_name}.scales": self.scales,
        }
        if self.zeros is not None:
            tensors[f"{base_name}.zeros"] = self.zeros
        if self.codebook is not None:
            tensors[f"{base_name}.codebook"] = self.codebook
        return tensors


class BaseQuantizer(ABC):
    """Abstract base class for all pluggable PocketTitan quantizers."""

    def __init__(self, config: QuantConfig):
        self.config = config

    @property
    @abstractmethod
    def capabilities(self) -> QuantizerCapabilities:
        """Return the mathematical capability contract of this quantizer."""
        pass

    @abstractmethod
    def quantize(
        self,
        weight: torch.Tensor,
        hessian: Optional[torch.Tensor] = None,
        outlier_indices: Optional[torch.Tensor] = None,
    ) -> QuantizedResult:
        """Quantize a full weight matrix or micro-tile."""
        pass

    @abstractmethod
    def dequantize(self, quantized: QuantizedResult) -> torch.Tensor:
        """Reconstruct float approximation from quantized representations."""
        pass
