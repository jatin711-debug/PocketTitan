"""Activation-Weighted Quantization (AWQ) Backend (Milestone 1)."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult
from pockettitan.quantizers.rtn import RTNQuantizer


class AWQQuantizer(BaseQuantizer):
    """Activation-Weighted Quantizer protecting salient weight channels via grid search scaling."""

    def __init__(self, config: QuantConfig, grid_search_steps: int = 20):
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
        
        pad_k = (group_size - (in_features % group_size)) % group_size
        padded_in_features = in_features + pad_k
        if pad_k > 0:
            w_2d = F.pad(w_2d, (0, pad_k))
            
        num_groups = padded_in_features // group_size
        
        # 1. Extract activation scales from diagonal of Hessian: s = diag(H)^0.5
        if hessian is not None:
            act_scales = torch.sqrt(torch.clamp(torch.diag(hessian.float()), min=1e-5)).to(weight.device)
            if pad_k > 0:
                act_scales = F.pad(act_scales, (0, pad_k), value=1.0)
        else:
            act_scales = torch.ones(padded_in_features, device=weight.device)
            
        act_scales = act_scales / torch.mean(act_scales).clamp(min=1e-5)
        
        # 2. Grid search optimal alpha
        best_error = float("inf")
        best_scales = torch.ones_like(act_scales)
        bits = self.config.bits
        max_int = (1 << bits) - 1
        
        for step in range(self.grid_search_steps):
            ratio = step / max(1, self.grid_search_steps - 1)
            # Candidate scales: s = act_scales ^ ratio
            scales = (act_scales ** ratio).clamp(min=1e-4)
            
            # Scale weights
            w_scaled = w_2d * scales.view(1, -1)
            w_grouped = w_scaled.view(-1, group_size)
            
            # Quantize & dequantize candidate
            min_val = torch.amin(w_grouped, dim=-1, keepdim=True)
            max_val = torch.amax(w_grouped, dim=-1, keepdim=True)
            scale = ((max_val - min_val) / max(1, max_int)).clamp(min=1e-5)
            zero = torch.round(-min_val / scale).clamp(0, max_int)
            
            q_candidate = torch.round(w_grouped / scale + zero).clamp(0, max_int)
            w_deq_scaled = (q_candidate - zero) * scale
            w_deq = w_deq_scaled.view(out_features, padded_in_features) / scales.view(1, -1)
            
            # Weighted L2 error against activations
            err = torch.sum((w_2d - w_deq) ** 2 * act_scales.view(1, -1))
            if err < best_error:
                best_error = err
                best_scales = scales
                
        # 3. Apply optimal scales
        w_scaled = w_2d * best_scales.view(1, -1)
        w_grouped = w_scaled.view(-1, group_size)
        
        min_val = torch.amin(w_grouped, dim=-1, keepdim=True)
        max_val = torch.amax(w_grouped, dim=-1, keepdim=True)
        scale_grouped = ((max_val - min_val) / max(1, max_int)).clamp(min=1e-5)
        zero_grouped = torch.round(-min_val / scale_grouped).clamp(0, max_int)
        
        q_uint = torch.round(w_grouped / scale_grouped + zero_grouped).clamp(0, max_int).to(torch.uint8)
        packed = RTNQuantizer._pack_tensor(q_uint.view(out_features, padded_in_features), bits)
        
        return QuantizedResult(
            packed_weights=packed,
            scales=scale_grouped.view(out_features, num_groups).to(torch.float16),
            zeros=zero_grouped.view(out_features, num_groups).to(torch.float16),
            codebook=best_scales.to(torch.float16),  # Store AWQ channel scales in codebook field
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
        
        deq_scaled = (w_grouped - zero_grouped) * scale_grouped
        deq = deq_scaled.view(out_features, padded_in_features)
        
        # Undo AWQ channel scales
        if quantized.codebook is not None:
            deq = deq / quantized.codebook.float().view(1, -1)
            
        deq_2d = deq[:, :in_features]
        return deq_2d.view(orig_shape).to(quantized.original_dtype)
