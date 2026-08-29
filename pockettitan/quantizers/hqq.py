"""Half-Quadratic Quantization (HQQ) Backend with Proximal Coordinate Descent (Milestone 1)."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult
from pockettitan.quantizers.rtn import RTNQuantizer


class HQQQuantizer(BaseQuantizer):
    """HQQ Quantizer using proximal alternating optimization for scale and zero points."""

    def __init__(self, config: QuantConfig, max_iters: int = 5):
        super().__init__(config)
        self.max_iters = max_iters

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name="hqq",
            requires_calibration=False,
            legal_split_axes=(0,),
            requires_full_input_dim=True,
            requires_full_output_dim=False,
            global_state=None,
            supports_cpu=True,
            supports_cuda=True,
            workspace_multiplier=5.0,  # Accurate accounting for FP32 coordinate descent workspace
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
        
        pad_k = (group_size - (in_features % group_size)) % group_size
        padded_in_features = in_features + pad_k
        if pad_k > 0:
            w_2d = F.pad(w_2d, (0, pad_k))
            
        num_groups_per_row = padded_in_features // group_size
        w_grouped = w_2d.view(-1, group_size)
        
        bits = self.config.bits
        max_int = (1 << bits) - 1
        
        # Step 1: Initial scale and zero estimates via min-max
        min_val = torch.amin(w_grouped, dim=-1, keepdim=True)
        max_val = torch.amax(w_grouped, dim=-1, keepdim=True)
        scale = ((max_val - min_val) / max(1, max_int)).clamp(min=1e-5)
        zero = torch.round(-min_val / scale).clamp(0, max_int)
        
        # Step 2: Memory-efficient coordinate descent loop
        for _ in range(self.max_iters):
            # Quantize: q_float in [0, max_int]
            q_float = torch.clamp(torch.round(w_grouped / scale + zero), 0, max_int)
            # Centered weights
            q_centered = q_float - zero
            # Least-squares update for scale S
            num = torch.sum(q_centered * w_grouped, dim=-1, keepdim=True)
            denom = torch.clamp(torch.sum(q_centered * q_centered, dim=-1, keepdim=True), min=1e-5)
            scale = torch.clamp(num / denom, min=1e-5)
            # Update zero point Z
            zero = torch.clamp(torch.mean(q_float - w_grouped / scale, dim=-1, keepdim=True), 0, max_int)
            del q_float, q_centered, num, denom

        # Final integer codes
        q_uint = torch.round(w_grouped / scale + zero).clamp(0, max_int).to(torch.uint8)
        del w_grouped, w_2d
        
        scale = scale.view(out_features, num_groups_per_row).to(torch.float16)
        zero = zero.view(out_features, num_groups_per_row).to(torch.float16)
        q_reshaped = q_uint.view(out_features, padded_in_features)
        
        packed = RTNQuantizer._pack_tensor(q_reshaped, bits)
        del q_reshaped, q_uint
        
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
        
        unpacked = RTNQuantizer._unpack_tensor(quantized.packed_weights, bits, (out_features, padded_in_features))
        w_grouped = unpacked.view(-1, group_size).to(torch.float32)
        scale_grouped = quantized.scales.view(-1, 1).to(torch.float32)
        zero_grouped = quantized.zeros.view(-1, 1).to(torch.float32)
        
        deq = (w_grouped - zero_grouped) * scale_grouped
        deq_2d = deq.view(out_features, padded_in_features)[:, :in_features]
        return deq_2d.view(orig_shape).to(quantized.original_dtype)
