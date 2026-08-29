"""BitNet 1.58b / Ternary {-1, 0, +1} groupwise quantizer."""

import math
from typing import Optional, Tuple
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult


class TernaryQuantizer(BaseQuantizer):
    """BitNet-style 1.58b ternary {-1, 0, +1} quantizer with 2-bit packing."""

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name="ternary",
            requires_calibration=False,
            legal_split_axes=(0, 1),
            requires_full_input_dim=False,
            requires_full_output_dim=False,
            global_state=None,
            supports_cpu=True,
            supports_cuda=True,
            workspace_multiplier=1.2,
        )

    def quantize(
        self,
        weight: torch.Tensor,
        hessian: Optional[torch.Tensor] = None,
    ) -> QuantizedResult:
        orig_shape = weight.shape
        orig_dtype = weight.dtype
        device = str(weight.device)
        
        w_2d = weight.view(-1, orig_shape[-1]).to(torch.float32)
        out_features, in_features = w_2d.shape
        group_size = self.config.group_size if self.config.group_size > 0 else in_features
        
        if in_features % group_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")
            
        num_groups_per_row = in_features // group_size
        w_grouped = w_2d.view(-1, group_size)
        
        # BitNet scale: mean absolute value per group (or per matrix)
        scale = torch.mean(torch.abs(w_grouped), dim=-1, keepdim=True).clamp(min=1e-5)
        # Quantize to {-1, 0, +1}
        q_ternary = torch.round(w_grouped / scale).clamp(-1.0, 1.0)
        
        # Map {-1, 0, 1} -> {0, 1, 2} in uint8 for 2-bit packing
        q_mapped = (q_ternary + 1.0).to(torch.uint8)
        
        scale = scale.view(out_features, num_groups_per_row).to(torch.float16)
        q_reshaped = q_mapped.view(out_features, in_features)
        
        # Pack 4 ternary values per byte (2-bit packing)
        flat = q_reshaped.contiguous().view(-1)
        b0 = flat[0::4]
        b1 = flat[1::4]
        b2 = flat[2::4]
        b3 = flat[3::4]
        packed = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)
        
        return QuantizedResult(
            packed_weights=packed,
            scales=scale,
            zeros=None,
            codebook=None,
            quant_config=self.config,
            original_shape=orig_shape,
            original_dtype=orig_dtype,
            bit_width=1.58,
            device=device,
        )

    def dequantize(self, quantized: QuantizedResult) -> torch.Tensor:
        orig_shape = quantized.original_shape
        out_features, in_features = orig_shape[0], orig_shape[1]
        group_size = quantized.quant_config.group_size if quantized.quant_config.group_size > 0 else in_features
        num_elements = math.prod(orig_shape)
        
        # Unpack 2-bit representations
        packed = quantized.packed_weights
        unpacked = torch.zeros(num_elements, dtype=torch.uint8, device=packed.device)
        unpacked[0::4] = packed & 0x03
        unpacked[1::4] = (packed >> 2) & 0x03
        unpacked[2::4] = (packed >> 4) & 0x03
        unpacked[3::4] = (packed >> 6) & 0x03
        
        # Map {0, 1, 2} back to {-1.0, 0.0, +1.0}
        q_ternary = unpacked.view(-1, group_size).to(torch.float32) - 1.0
        scale_grouped = quantized.scales.view(-1, 1).to(torch.float32)
        
        deq = q_ternary * scale_grouped
        return deq.view(orig_shape).to(quantized.original_dtype)
