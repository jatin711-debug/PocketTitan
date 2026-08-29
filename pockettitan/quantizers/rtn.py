"""Vectorized Round-to-Nearest (RTN) Quantizer with sub-byte bit packing (Milestone 1)."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult


class RTNQuantizer(BaseQuantizer):
    """Vectorized Uniform Groupwise Round-to-Nearest Quantizer supporting 1-8 bits."""

    def __init__(self, config: QuantConfig):
        super().__init__(config)

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name="rtn",
            requires_calibration=False,
            legal_split_axes=(0,),
            requires_full_input_dim=True,
            requires_full_output_dim=False,
            global_state=None,
            supports_cpu=True,
            supports_cuda=True,
            workspace_multiplier=2.0,
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
        
        pad_k = (group_size - (in_features % group_size)) % group_size
        padded_in_features = in_features + pad_k
        if pad_k > 0:
            w_2d = F.pad(w_2d, (0, pad_k))
            
        num_groups_per_row = padded_in_features // group_size
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
        q_reshaped = q_uint.view(out_features, padded_in_features)
        
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
        
        pad_k = (group_size - (in_features % group_size)) % group_size
        padded_in_features = in_features + pad_k
        
        unpacked = self._unpack_tensor(quantized.packed_weights, bits, (out_features, padded_in_features))
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
            
        deq_2d = deq.view(out_features, padded_in_features)[:, :in_features]
        return deq_2d.view(orig_shape).to(quantized.original_dtype)

    @staticmethod
    def _pack_tensor(q_tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack uint8 tensor into compact bits representation."""
        if bits == 8:
            return q_tensor.contiguous()
            
        elements_per_byte = 8 // bits
        flat = q_tensor.reshape(-1)
        num_elements = flat.numel()
        
        pad_len = (elements_per_byte - (num_elements % elements_per_byte)) % elements_per_byte
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=flat.device)])
            
        reshaped = flat.view(-1, elements_per_byte)
        packed = torch.zeros(reshaped.shape[0], dtype=torch.uint8, device=flat.device)
        
        for i in range(elements_per_byte):
            packed = packed | (reshaped[:, i] << (i * bits))
            
        return packed

    @staticmethod
    def _unpack_tensor(packed: torch.Tensor, bits: int, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Unpack compact bits representation back to uint8 tensor."""
        if bits == 8:
            return packed.view(original_shape)
            
        elements_per_byte = 8 // bits
        mask = (1 << bits) - 1
        
        unpacked_chunks = []
        for i in range(elements_per_byte):
            chunk = (packed >> (i * bits)) & mask
            unpacked_chunks.append(chunk)
            
        # Interleave unpacked chunks
        interleaved = torch.stack(unpacked_chunks, dim=1).view(-1)
        total_elements = math.prod(original_shape)
        return interleaved[:total_elements].view(original_shape)
