"""Tensor Layout Adapters for decomposing diverse tensor geometries into legal 2D quantization units."""

from abc import ABC, abstractmethod
from typing import List
import torch


class TensorLayoutAdapter(ABC):
    """Abstract adapter defining how a tensor layout is decomposed into legal 2-D work units."""

    def __init__(self, name: str, shape: List[int], dtype_str: str):
        self.name = name
        self.shape = shape
        self.dtype_str = dtype_str

    @abstractmethod
    def get_num_subunits(self) -> int:
        """Number of independent 2-D matrices contained in this layout."""
        pass

    @abstractmethod
    def get_subunit_shape(self, index: int = 0) -> List[int]:
        """2-D shape [out_features, in_features] for a given subunit."""
        pass

    @abstractmethod
    def extract_subunit_tensor(self, tensor: torch.Tensor, index: int) -> torch.Tensor:
        """Extract a 2-D subunit tensor from a materialized full tensor."""
        pass


class Dense2DLayout(TensorLayoutAdapter):
    """Standard 2-D linear weight matrix [out_features, in_features]."""

    def get_num_subunits(self) -> int:
        return 1

    def get_subunit_shape(self, index: int = 0) -> List[int]:
        return [self.shape[0], self.shape[1]]

    def extract_subunit_tensor(self, tensor: torch.Tensor, index: int = 0) -> torch.Tensor:
        if tensor.ndim != 2:
            return tensor.view(self.shape[0], self.shape[1])
        return tensor


class FusedExperts3DLayout(TensorLayoutAdapter):
    """3-D fused MoE expert bank [num_experts, out_features, in_features]."""

    def __init__(self, name: str, shape: List[int], dtype_str: str):
        super().__init__(name, shape, dtype_str)
        self.num_experts = shape[0]
        self.out_features = shape[1]
        self.in_features = shape[2]

    def get_num_subunits(self) -> int:
        return self.num_experts

    def get_subunit_shape(self, index: int = 0) -> List[int]:
        return [self.out_features, self.in_features]

    def extract_subunit_tensor(self, tensor: torch.Tensor, index: int) -> torch.Tensor:
        return tensor[index, :, :]


class ConvLayout(TensorLayoutAdapter):
    """1-D or 2-D Convolution weight reshaped to legal 2-D matrix."""

    def get_num_subunits(self) -> int:
        return 1

    def get_subunit_shape(self, index: int = 0) -> List[int]:
        out_channels = self.shape[0]
        in_dim = 1
        for d in self.shape[1:]:
            in_dim *= d
        return [out_channels, in_dim]

    def extract_subunit_tensor(self, tensor: torch.Tensor, index: int = 0) -> torch.Tensor:
        out_channels = self.shape[0]
        return tensor.view(out_channels, -1)


def get_layout_adapter(name: str, shape: List[int], dtype_str: str = "F16") -> TensorLayoutAdapter:
    """Factory creating appropriate TensorLayoutAdapter based on tensor dimensions and naming."""
    if len(shape) == 2:
        return Dense2DLayout(name, shape, dtype_str)
    elif len(shape) == 3 and any(
        k in name for k in ["expert", "mlp.experts", "gate_up_proj", "down_proj", "w1", "w2", "w3"]
    ):
        return FusedExperts3DLayout(name, shape, dtype_str)
    elif len(shape) >= 3:
        return ConvLayout(name, shape, dtype_str)
    elif len(shape) == 1:
        # Vector / bias / scale (1D)
        return Dense2DLayout(name, [shape[0], 1], dtype_str)
    return Dense2DLayout(name, shape, dtype_str)
