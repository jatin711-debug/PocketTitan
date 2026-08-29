"""Base interfaces, data structures, and capability contracts for quantization backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from pockettitan.config import QuantConfig, QuantMethod


@dataclass
class QuantizerCapabilities:
    """Explicit mathematical and architectural constraints for a quantizer backend."""
    name: str
    requires_calibration: bool
    legal_split_axes: Tuple[int, ...]  # (0,) means can split on out_features, (0, 1) means arbitrary 2D tiling
    requires_full_input_dim: bool     # True if must see entire in_features column
    requires_full_output_dim: bool    # True if must see entire out_features row
    global_state: Optional[str]        # e.g. 'hessian', 'activation_max', None
    supports_cpu: bool
    supports_cuda: bool
    workspace_multiplier: float        # Temp memory factor during quantization (e.g. 2.0x)


@dataclass
class QuantizedResult:
    """Output container for a quantized weight matrix or tile."""
    packed_weights: torch.Tensor       # Packed integer or sub-byte codes (e.g. uint8)
    scales: torch.Tensor               # Quantization scales
    zeros: Optional[torch.Tensor]      # Zero-point offsets (for asymmetric quantization)
    codebook: Optional[torch.Tensor]   # Optional codebook (for vector / lattice quantizers)
    quant_config: QuantConfig
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    bit_width: float                   # Effective bits per weight
    device: str

    def size_bytes(self) -> int:
        total = self.packed_weights.nbytes + self.scales.nbytes
        if self.zeros is not None:
            total += self.zeros.nbytes
        if self.codebook is not None:
            total += self.codebook.nbytes
        return total


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
    ) -> QuantizedResult:
        """Quantize a 2D weight matrix (or sliced tile) under memory constraints.
        
        Args:
            weight: Input weight tensor of shape [out_features, in_features] on target device.
            hessian: Optional second-order Hessian matrix [in_features, in_features] for activation-aware methods.
            
        Returns:
            QuantizedResult containing packed weights, scales, zeros, and metadata.
        """
        pass

    @abstractmethod
    def dequantize(self, quantized: QuantizedResult) -> torch.Tensor:
        """Reconstruct float/half weight tensor from quantized representation for verification."""
        pass
