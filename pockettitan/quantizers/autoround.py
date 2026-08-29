"""AutoRound Gradient-Optimized Weight Rounding Quantizer (Milestone 1)."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult
from pockettitan.quantizers.rtn import RTNQuantizer


class AutoRoundQuantizer(BaseQuantizer):
    """AutoRound: Optimizing weight rounding via sign-gradient descent on local calibration."""

    def __init__(self, config: QuantConfig, iters: int = 50, lr: float = 0.05):
        super().__init__(config)
        self.iters = iters
        self.lr = lr

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name="autoround",
            requires_calibration=True,
            legal_split_axes=(0,),
            requires_full_input_dim=True,
            requires_full_output_dim=False,
            global_state="hessian",
            supports_cpu=True,
            supports_cuda=True,
            workspace_multiplier=3.5,
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
            
        bits = self.config.bits
        max_int = (1 << bits) - 1
        num_groups = padded_in_features // group_size
        
        w_grouped = w_2d.view(-1, group_size)
        min_g = torch.amin(w_grouped, dim=-1, keepdim=True)
        max_g = torch.amax(w_grouped, dim=-1, keepdim=True)
        scale = ((max_g - min_g) / max(1, max_int)).clamp(min=1e-5)
        zero = torch.round(-min_g / scale).clamp(0, max_int)
        
        # Initial continuous code
        w_cont = w_grouped / scale + zero
        w_floor = torch.floor(w_cont)
        # Learnable soft rounding parameter V: Q = w_floor + sigmoid(V)
        # Initialize V such that sigmoid(V) matches fractional part
        frac = (w_cont - w_floor).clamp(1e-4, 1.0 - 1e-4)
        v = torch.logit(frac)
        v.requires_grad_(True)
        
        optimizer = torch.optim.Adam([v], lr=self.lr)
        
        # Optimization loop minimizing reconstruction loss
        for _ in range(self.iters):
            optimizer.zero_grad()
            # Rectified soft rounding: h(v) = clamp(sigmoid(v) * (zeta - gamma) + gamma, 0, 1)
            soft_q = torch.clamp(torch.sigmoid(v) * 1.2 - 0.1, 0.0, 1.0)
            q_candidate = (w_floor + soft_q).clamp(0, max_int)
            w_deq = (q_candidate - zero) * scale
            
            # Loss = || W - W_deq ||_F^2
            loss = torch.sum((w_grouped - w_deq) ** 2)
            loss.backward()
            optimizer.step()
            
        # Hard rounding based on trained parameter V
        with torch.no_grad():
            final_soft = torch.clamp(torch.sigmoid(v) * 1.2 - 0.1, 0.0, 1.0)
            final_q = torch.round(w_floor + final_soft).clamp(0, max_int).to(torch.uint8)
            
        q_reshaped = final_q.view(out_features, padded_in_features)
        packed = RTNQuantizer._pack_tensor(q_reshaped, bits)
        
        return QuantizedResult(
            packed_weights=packed,
            scales=scale.view(out_features, num_groups).to(torch.float16),
            zeros=zero.view(out_features, num_groups).to(torch.float16),
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
