"""Uniform INT2 / INT3 / INT4 / INT8 groupwise quantizers."""

from typing import Optional
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult
from pockettitan.quantizers.rtn import RTNQuantizer


class INTxQuantizer(BaseQuantizer):
    """Uniform INTx groupwise quantizer supporting 2-bit, 3-bit, 4-bit, and 8-bit precision."""

    def __init__(self, config: QuantConfig):
        super().__init__(config)
        self._rtn = RTNQuantizer(config)

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name=f"int{self.config.bits}",
            requires_calibration=False,
            legal_split_axes=("out_features",),
            requires_full_input_dim=True,
            requires_full_output_dim=False,
            global_state=None,
            supports_cpu=True,
            supports_cuda=True,
            supports_remote_streaming=True,
            # Measured peak/source on a group-aligned matrix: 6.05x.
            # Padding overhead is modelled separately by group_padding_factor().
            workspace_multiplier=6.5,
        )

    def quantize(
        self,
        weight: torch.Tensor,
        hessian: Optional[torch.Tensor] = None,
        outlier_indices: Optional[torch.Tensor] = None,
    ) -> QuantizedResult:
        return self._rtn.quantize(weight, hessian)

    def dequantize(self, quantized: QuantizedResult) -> torch.Tensor:
        return self._rtn.dequantize(quantized)
