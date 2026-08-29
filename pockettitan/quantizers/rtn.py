"""Round-To-Nearest (RTN) groupwise uniform quantizer."""

import math
from typing import Optional, Tuple
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult


class RTNQuantizer(BaseQuantizer):
    """Symmetric/Asymmetric groupwise Round-To-Nearest quantizer."""

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name="rtn",
            requires_calibration=False,
            legal_split_axes=(0, 1),  # RTN can be sliced along rows OR columns with zero dependency
            requires_full_input_dim=False,
            requires_full_output_dim=False,
            global_state=None,
            supports_cpu=True,
            supports_cuda=True,
            workspace_multiplier=1.5,
        )

    def quantize(
        self,
        weight: torch.Tensor,
        hessian: Optional[torch.Tensor] = None,
    ) -> QuantizedResult:
        orig_shape = weight.shape
        orig_dtype = weight.dtype
        device = str(weight.device)
        
        # Flatten to 2D [out_features, in_features]
        w_2d = weight.view(-1, orig_shape[-1]).to(torch.float32)
        out_features, in_features = w_2d.shape
        group_size = self.config.group_size if self.config.group_size > 0 else in_features
        
        if in_features % group_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")
            
        num_groups_per_row = in_features // group_size
        # Reshape to [out_features * num_groups_per_row, group_size]
        w_grouped = w_2d.view(-1, group_size)
        
        bits = self.config.bits
        max_int = (1 << bits) - 1
        
        if self.config.symmetric:
            max_val = torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True).clamp(min=1e-5)
            scale = max_val / ((1 << (bits - 1)) - 1 if bits > 1 else 1.0)
            zero = None
            q_val = torch.round(w_grouped / scale).clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
            # Offset to unsigned int for storage
            q_uint = (q_val + (1 << (bits - 1))).to(torch.uint8)
        else:
            min_val = torch.amin(w_grouped, dim=-1, keepdim=True)
            max_val = torch.amax(w_grouped, dim=-1, keepdim=True)
            scale = ((max_val - min_val) / max(1, max_int)).clamp(min=1e-5)
            zero = torch.round(-min_val / scale).clamp(0, max_int)
            q_uint = torch.round((w_grouped - min_val) / scale).clamp(0, max_int).to(torch.uint8)
            zero = zero.view(out_features, num_groups_per_row).to(torch.float16)

        scale = scale.view(out_features, num_groups_per_row).to(torch.float16)
        q_reshaped = q_uint.view(out_features, in_features)
        
        # Pack to bit width
        packed = self._pack_tensor(q_reshaped, bits)
        
        return QuantizedResult(
            packed_weights=packed,
            scales=scale,
            zeros=zero,
            codebook=None,
            quant_config=self.config,
            original_shape=orig_shape,
            original_dtype=orig_dtype,
            bit_width=float(bits),
            device=device,
        )

    def dequantize(self, quantized: QuantizedResult) -> torch.Tensor:
        bits = quantized.quant_config.bits
        orig_shape = quantized.original_shape
        out_features, in_features = orig_shape[0], orig_shape[1]
        group_size = quantized.quant_config.group_size if quantized.quant_config.group_size > 0 else in_features
        num_groups = in_features // group_size
        
        unpacked = self._unpack_tensor(quantized.packed_weights, bits, (out_features, in_features))
        w_grouped = unpacked.view(-1, group_size).to(torch.float32)
        scale_grouped = quantized.scales.view(-1, 1).to(torch.float32)
        
        if quantized.zeros is None:
            # Symmetric
            q_signed = w_grouped - (1 << (bits - 1))
            deq = q_signed * scale_grouped
        else:
            # Asymmetric
            zero_grouped = quantized.zeros.view(-1, 1).to(torch.float32)
            deq = (w_grouped - zero_grouped) * scale_grouped
            
        return deq.view(orig_shape).to(quantized.original_dtype)

    @staticmethod
    def _pack_tensor(q_tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack uint8 tensor into compact bits representation."""
        if bits == 8:
            return q_tensor.contiguous()
        elif bits == 4:
            # 2 elements per byte
            flat = q_tensor.contiguous().view(-1)
            even = flat[0::2]
            odd = flat[1::2]
            packed = even | (odd << 4)
            return packed
        elif bits == 2:
            # 4 elements per byte
            flat = q_tensor.contiguous().view(-1)
            b0 = flat[0::4]
            b1 = flat[1::4]
            b2 = flat[2::4]
            b3 = flat[3::4]
            packed = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)
            return packed
        elif bits == 1:
            # 8 elements per byte
            flat = q_tensor.contiguous().view(-1)
            packed = torch.zeros(len(flat) // 8, dtype=torch.uint8, device=q_tensor.device)
            for i in range(8):
                packed |= (flat[i::8] << i)
            return packed
        else:
            # Unpacked fallback for non-power-of-2 (e.g. 3-bit or 5-bit stored as uint8)
            return q_tensor.contiguous()

    @staticmethod
    def _unpack_tensor(packed: torch.Tensor, bits: int, shape: Tuple[int, ...]) -> torch.Tensor:
        """Unpack compact bits representation into uint8 tensor."""
        num_elements = math.prod(shape)
        if bits == 8:
            return packed.view(shape)
        elif bits == 4:
            unpacked = torch.zeros(num_elements, dtype=torch.uint8, device=packed.device)
            unpacked[0::2] = packed & 0x0F
            unpacked[1::2] = (packed >> 4) & 0x0F
            return unpacked.view(shape)
        elif bits == 2:
            unpacked = torch.zeros(num_elements, dtype=torch.uint8, device=packed.device)
            unpacked[0::4] = packed & 0x03
            unpacked[1::4] = (packed >> 2) & 0x03
            unpacked[2::4] = (packed >> 4) & 0x03
            unpacked[3::4] = (packed >> 6) & 0x03
            return unpacked.view(shape)
        elif bits == 1:
            unpacked = torch.zeros(num_elements, dtype=torch.uint8, device=packed.device)
            for i in range(8):
                unpacked[i::8] = (packed >> i) & 0x01
            return unpacked.view(shape)
        else:
            return packed.view(shape)
