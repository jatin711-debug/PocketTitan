"""AutoRound (Gradient-based Weight Rounding Optimization) backend."""

import math
from typing import Optional, Tuple
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult
from pockettitan.quantizers.rtn import RTNQuantizer


class AutoRoundQuantizer(BaseQuantizer):
    """AutoRound: Sign-gradient descent weight rounding optimization on bounded memory tiles."""

    def __init__(self, config: QuantConfig, iters: int = 30, lr: float = 0.05):
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
        
        bits = self.config.bits
        max_int = (1 << bits) - 1
        num_groups = in_features // group_size
        
        w_grouped = w_2d.view(-1, group_size)
        min_g = torch.amin(w_grouped, dim=-1, keepdim=True)
        max_g = torch.amax(w_grouped, dim=-1, keepdim=True)
        scale = ((max_g - min_g) / max(1, max_int)).clamp(min=1e-5)
        zero = torch.round(-min_g / scale).clamp(0, max_int)
        
        # Initial continuous code
        w_cont = w_grouped / scale + zero
        w_floor = torch.floor(w_cont)
        # Learnable soft rounding parameter V: Q = w_floor + sigmoid(V)
        v = torch.zeros_like(w_grouped, requires_grad=True)
        
        # If Hessian is provided, use it to weight columns; else uniform weighting
        if hessian is not None:
            diag_h = torch.diag(hessian.float()).clamp(min=1e-5).to(weight.device)
            weight_factor = (diag_h / torch.mean(diag_h)).view(1, in_features).view(-1, group_size)
        else:
            weight_factor = torch.ones_like(w_grouped)
            
        optimizer = torch.optim.Adam([v], lr=self.lr)
        
        for _ in range(self.iters):
            optimizer.zero_grad()
            # Rectified sigmoid rounding
            h_v = torch.clamp(torch.sigmoid(v) * 1.2 - 0.1, 0.0, 1.0)
            q_est = torch.clamp(w_floor + h_v, 0, max_int)
            w_rec = (q_est - zero) * scale
            
            # Loss: activation-weighted L2 error
            diff = (w_grouped - w_rec) * torch.sqrt(weight_factor)
            loss = torch.mean(diff ** 2)
            loss.backward()
            optimizer.step()

        # Hard integer codes
        with torch.no_grad():
            h_v = torch.clamp(torch.sigmoid(v) * 1.2 - 0.1, 0.0, 1.0)
            q_final = torch.clamp(torch.round(w_floor + h_v), 0, max_int).to(torch.uint8)

        scale = scale.view(out_features, num_groups).to(torch.float16)
        zero = zero.view(out_features, num_groups).to(torch.float16)
        q_reshaped = q_final.view(out_features, in_features)
        
        packed = RTNQuantizer._pack_tensor(q_reshaped, bits)
        
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
        
        unpacked = RTNQuantizer._unpack_tensor(quantized.packed_weights, bits, (out_features, in_features))
        w_grouped = unpacked.view(-1, group_size).to(torch.float32)
        scale_grouped = quantized.scales.view(-1, 1).to(torch.float32)
        zero_grouped = quantized.zeros.view(-1, 1).to(torch.float32) if quantized.zeros is not None else 0.0
        
        deq = (w_grouped - zero_grouped) * scale_grouped
        return deq.view(orig_shape).to(quantized.original_dtype)
