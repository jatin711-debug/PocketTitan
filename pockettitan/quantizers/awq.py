"""Activation-Weighted Quantization (AWQ) backend."""

import math
from typing import Optional, Tuple
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult
from pockettitan.quantizers.rtn import RTNQuantizer


class AWQQuantizer(BaseQuantizer):
    """Activation-Weighted Quantization (AWQ) protecting salient weight channels."""

    def __init__(self, config: QuantConfig, grid_search_steps: int = 10):
        super().__init__(config)
        self.grid_search_steps = grid_search_steps

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name="awq",
            requires_calibration=True,
            legal_split_axes=(0,),
            requires_full_input_dim=True,
            requires_full_output_dim=False,
            global_state="hessian",
            supports_cpu=True,
            supports_cuda=True,
            workspace_multiplier=2.5,
        )

    def quantize(
        self,
        weight: torch.Tensor,
        hessian: Optional[torch.Tensor] = None,
    ) -> QuantizedResult:
        orig_shape = weight.shape
        orig_dtype = weight.dtype
        device = str(weight.device)
        
        w_2d = weight.view(-1, orig_shape[-1]).clone().float()
        out_features, in_features = w_2d.shape
        group_size = self.config.group_size if self.config.group_size > 0 else in_features
        
        # 1. Extract activation scales from diagonal of Hessian: s = diag(H)^0.5
        if hessian is not None:
            act_scales = torch.sqrt(torch.clamp(torch.diag(hessian.float()), min=1e-5)).to(weight.device)
        else:
            act_scales = torch.ones(in_features, device=weight.device)
            
        act_scales = act_scales / torch.mean(act_scales).clamp(min=1e-5)
        
        # 2. Grid search optimal alpha
        best_error = float("inf")
        best_scales = torch.ones_like(act_scales)
        bits = self.config.bits
        max_int = (1 << bits) - 1
        
        for step in range(self.grid_search_steps):
            ratio = (step + 1) / self.grid_search_steps
            cand_s = torch.pow(act_scales, ratio).clamp(min=1e-4)
            
            # Scale weights
            w_scaled = w_2d * cand_s.unsqueeze(0)
            
            # Simple groupwise quantize
            w_grouped = w_scaled.view(-1, group_size)
            min_g = torch.amin(w_grouped, dim=-1, keepdim=True)
            max_g = torch.amax(w_grouped, dim=-1, keepdim=True)
            scale = ((max_g - min_g) / max(1, max_int)).clamp(min=1e-5)
            zero = torch.round(-min_g / scale).clamp(0, max_int)
            
            q_float = torch.clamp(torch.round(w_grouped / scale + zero), 0, max_int)
            q_deq_scaled = (q_float - zero) * scale
            q_deq = q_deq_scaled.view(out_features, in_features) / cand_s.unsqueeze(0)
            
            err = torch.sum((w_2d - q_deq) ** 2).item()
            if err < best_error:
                best_error = err
                best_scales = cand_s

        # Apply best scales
        w_best_scaled = w_2d * best_scales.unsqueeze(0)
        num_groups = in_features // group_size
        w_grouped = w_best_scaled.view(-1, group_size)
        
        min_g = torch.amin(w_grouped, dim=-1, keepdim=True)
        max_g = torch.amax(w_grouped, dim=-1, keepdim=True)
        scale = ((max_g - min_g) / max(1, max_int)).clamp(min=1e-5)
        zero = torch.round(-min_g / scale).clamp(0, max_int)
        
        q_uint = torch.clamp(torch.round(w_grouped / scale + zero), 0, max_int).to(torch.uint8)
        q_reshaped = q_uint.view(out_features, in_features)
        
        scale = scale.view(out_features, num_groups).to(torch.float16)
        zero = zero.view(out_features, num_groups).to(torch.float16)
        
        packed = RTNQuantizer._pack_tensor(q_reshaped, bits)
        
        return QuantizedResult(
            packed_weights=packed,
            scales=scale,
            zeros=zero,
            codebook=best_scales.to(torch.float16),  # Store AWQ channel multipliers in codebook field
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
        
        unpacked = RTNQuantizer._unpack_tensor(quantized.packed_weights, bits, (out_features, in_features))
        w_grouped = unpacked.view(-1, group_size).to(torch.float32)
        scale_grouped = quantized.scales.view(-1, 1).to(torch.float32)
        zero_grouped = quantized.zeros.view(-1, 1).to(torch.float32) if quantized.zeros is not None else 0.0
        
        deq_scaled = (w_grouped - zero_grouped) * scale_grouped
        deq_2d = deq_scaled.view(out_features, in_features)
        
        if quantized.codebook is not None:
            awq_scales = quantized.codebook.float().to(quantized.packed_weights.device)
            deq_2d = deq_2d / awq_scales.unsqueeze(0).clamp(min=1e-5)
            
        return deq_2d.view(orig_shape).to(quantized.original_dtype)
