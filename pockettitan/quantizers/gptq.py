"""Second-order GPTQ (Generalized Post-Training Quantization) backend."""

import math
from typing import Optional, Tuple
import torch

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult
from pockettitan.quantizers.rtn import RTNQuantizer


class GPTQQuantizer(BaseQuantizer):
    """Activation-aware second-order GPTQ quantizer with row-tiled Cholesky updates."""

    def __init__(self, config: QuantConfig, block_size: int = 128, percdamp: float = 0.01):
        super().__init__(config)
        self.block_size = block_size
        self.percdamp = percdamp

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name="gptq",
            requires_calibration=True,
            legal_split_axes=(0,),  # GPTQ must see the full in_features dimension to invert Hessian
            requires_full_input_dim=True,
            requires_full_output_dim=False,
            global_state="hessian",
            supports_cpu=True,
            supports_cuda=True,
            workspace_multiplier=3.0,
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
        num_groups = in_features // group_size
        bits = self.config.bits
        max_int = (1 << bits) - 1
        
        # If no Hessian provided, fallback to identity Hessian
        if hessian is None:
            H = torch.eye(in_features, dtype=torch.float32, device=weight.device)
        else:
            H = hessian.clone().float().to(weight.device)
            
        # 1. Dampening on diagonal: H += percdamp * mean(diag(H)) * I
        damp = self.percdamp * torch.mean(torch.diag(H)).item()
        H.diagonal().add_(damp)
        
        # 2. Invert Hessian via Cholesky
        try:
            H_inv = torch.linalg.inv(H)
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)
        except Exception:
            H_inv = torch.eye(in_features, dtype=torch.float32, device=weight.device)
            H_inv_chol = H_inv

        q_weights = torch.zeros_like(w_2d, dtype=torch.uint8)
        scale_final = torch.zeros(out_features, num_groups, dtype=torch.float32, device=weight.device)
        zero_final = torch.zeros(out_features, num_groups, dtype=torch.float32, device=weight.device)
        
        # Precompute groupwise scales and zeros
        for g in range(num_groups):
            g_start = g * group_size
            g_end = min(in_features, (g + 1) * group_size)
            w_grp = w_2d[:, g_start:g_end]
            min_g = torch.amin(w_grp, dim=-1, keepdim=True)
            max_g = torch.amax(w_grp, dim=-1, keepdim=True)
            s_g = ((max_g - min_g) / max(1, max_int)).clamp(min=1e-5)
            z_g = torch.round(-min_g / s_g).clamp(0, max_int)
            scale_final[:, g] = s_g.squeeze(-1)
            zero_final[:, g] = z_g.squeeze(-1)
        
        # 3. Block-by-block column quantization with second-order error propagation
        for i1 in range(0, in_features, self.block_size):
            i2 = min(i1 + self.block_size, in_features)
            count = i2 - i1
            
            W_block = w_2d[:, i1:i2].clone()
            Q_block = torch.zeros_like(W_block)
            Err_block = torch.zeros_like(W_block)
            H_inv_block = H_inv_chol[i1:i2, i1:i2]
            
            for j in range(count):
                col_idx = i1 + j
                g_idx = min(num_groups - 1, col_idx // group_size)
                
                scale = scale_final[:, g_idx]
                zero = zero_final[:, g_idx]
                
                w_col = W_block[:, j]
                d = H_inv_block[j, j]
                
                # Quantize column against its group scale and zero
                q = torch.clamp(torch.round(w_col / scale + zero), 0, max_int)
                q_deq = (q - zero) * scale
                
                Q_block[:, j] = q
                err = (w_col - q_deq) / d
                Err_block[:, j] = err
                
                # Update remaining columns in block
                W_block[:, j:] -= err.unsqueeze(1) @ H_inv_block[j:j+1, j:]
                
            q_weights[:, i1:i2] = Q_block.to(torch.uint8)
            
            # Update remaining weight matrix columns outside current block
            if i2 < in_features:
                w_2d[:, i2:] -= Err_block @ H_inv_chol[i1:i2, i2:]

        packed = RTNQuantizer._pack_tensor(q_weights, bits)
        
        return QuantizedResult(
            packed_weights=packed,
            scales=scale_final.to(torch.float16),
            zeros=zero_final.to(torch.float16),
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
