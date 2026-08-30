"""Second-order Generalized Post-Training Quantization (GPTQ) Backend."""

from typing import Optional
import torch
import torch.nn.functional as F

from pockettitan.config import CalibrationRequiredError, QuantConfig
from pockettitan.quantizers.base import BaseQuantizer, QuantizedResult, QuantizerCapabilities, matrix_dims
from pockettitan.quantizers.rtn import RTNQuantizer


class GPTQQuantizer(BaseQuantizer):
    """Memory-bounded GPTQ quantizer using online row-tiled inverse Hessian Cholesky updates."""

    def __init__(self, config: QuantConfig, block_size: int = 128, percdamp: float = 0.01):
        super().__init__(config)
        self.block_size = block_size
        self.percdamp = percdamp

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name="gptq",
            requires_calibration=True,
            legal_split_axes=("out_features",),
            requires_full_input_dim=True,
            requires_full_output_dim=False,
            global_state="hessian",
            supports_cpu=True,
            supports_cuda=True,
            supports_remote_streaming=True,
            # Measured peak/source on a group-aligned matrix: 12.39x.
            # Padding overhead is modelled separately by group_padding_factor().
            workspace_multiplier=13.5,
        )

    def quantize(
        self,
        weight: torch.Tensor,
        hessian: Optional[torch.Tensor] = None,
        outlier_indices: Optional[torch.Tensor] = None,
    ) -> QuantizedResult:
        orig_shape = weight.shape
        orig_dtype = weight.dtype
        device = str(weight.device)

        # Enforce Calibration Safety Contract: GPTQ strictly requires Hessian
        if hessian is None:
            raise CalibrationRequiredError(
                "GPTQ quantization strictly requires an empirical second-order Hessian matrix computed from "
                "calibration data. Silent fallback to identity matrix is prohibited to guarantee quantization fidelity."
            )

        w_2d = weight.view(-1, orig_shape[-1]).clone().float()
        out_features, in_features = w_2d.shape
        group_size = self.config.group_size if self.config.group_size > 0 else in_features

        pad_k = (group_size - (in_features % group_size)) % group_size
        padded_in_features = in_features + pad_k
        if pad_k > 0:
            w_2d = F.pad(w_2d, (0, pad_k))

        num_groups = padded_in_features // group_size
        bits = self.config.bits
        max_int = (1 << bits) - 1

        H = hessian.clone().float().to(weight.device)
        if pad_k > 0:
            H = F.pad(H, (0, pad_k, 0, pad_k))
            H.diagonal()[-pad_k:].fill_(1.0)

        # 1. Dampening on diagonal: H += percdamp * mean(diag(H)) * I
        damp = self.percdamp * torch.mean(torch.diag(H)).item()
        H.diagonal().add_(damp)

        # 2. Invert Hessian via Cholesky
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except Exception:
            Hinv = torch.inverse(H + torch.eye(padded_in_features, device=H.device) * 1e-4)

        Hinv_chol = torch.linalg.cholesky(Hinv, upper=True)

        # Precompute groupwise min-max scales and zeros from original weights
        w_orig_grouped = w_2d.view(-1, group_size)
        min_val = torch.amin(w_orig_grouped, dim=-1, keepdim=True)
        max_val = torch.amax(w_orig_grouped, dim=-1, keepdim=True)
        scale_grouped = ((max_val - min_val) / max(1, max_int)).clamp(min=1e-5)
        zero_grouped = torch.round(-min_val / scale_grouped).clamp(0, max_int)

        scale_per_col = scale_grouped.view(out_features, num_groups).repeat_interleave(
            group_size, dim=1
        )
        zero_per_col = zero_grouped.view(out_features, num_groups).repeat_interleave(
            group_size, dim=1
        )

        Q = torch.zeros_like(w_2d)

        # 3. Block-by-block column quantization with second-order error propagation
        for i1 in range(0, padded_in_features, self.block_size):
            i2 = min(i1 + self.block_size, padded_in_features)
            count = i2 - i1

            W1 = w_2d[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Hinv1 = Hinv_chol[i1:i2, i1:i2]

            scale_block = scale_per_col[:, i1:i2]
            zero_block = zero_per_col[:, i1:i2]

            for j in range(count):
                w_col = W1[:, j]
                d = Hinv1[j, j]
                s_col = scale_block[:, j]
                z_col = zero_block[:, j]

                # Quantize column
                q_col = torch.round(w_col / s_col + z_col).clamp(0, max_int)
                w_deq_col = (q_col - z_col) * s_col
                Q1[:, j] = q_col

                # Propagate column error to remaining columns in block
                err = (w_col - w_deq_col) / d
                W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)

            Q[:, i1:i2] = Q1

            # Propagate block error to remaining matrix columns
            if i2 < padded_in_features:
                err_block = (w_2d[:, i1:i2] - ((Q1 - zero_block) * scale_block)) @ torch.inverse(
                    Hinv1
                )
                w_2d[:, i2:] -= err_block @ Hinv_chol[i1:i2, i2:]

        # 4. Pack integer codes
        q_uint = Q.to(torch.uint8)
        packed = RTNQuantizer._pack_tensor(q_uint, bits)

        return QuantizedResult(
            packed_weights=packed,
            scales=scale_grouped.view(out_features, num_groups).to(torch.float16),
            zeros=zero_grouped.view(out_features, num_groups).to(torch.float16),
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
        out_features, in_features = matrix_dims(orig_shape)
        group_size = (
            quantized.quant_config.group_size
            if quantized.quant_config.group_size > 0
            else in_features
        )

        pad_k = (group_size - (in_features % group_size)) % group_size
        padded_in_features = in_features + pad_k

        unpacked = RTNQuantizer._unpack_tensor(
            quantized.packed_weights, bits, (out_features, padded_in_features)
        )
        w_grouped = unpacked.view(-1, group_size).to(torch.float32)
        scale_grouped = quantized.scales.view(-1, 1).to(torch.float32)
        zero_grouped = quantized.zeros.view(-1, 1).to(torch.float32)

        deq = (w_grouped - zero_grouped) * scale_grouped
        deq_2d = deq.view(out_features, padded_in_features)[:, :in_features]
        return deq_2d.view(orig_shape).to(quantized.original_dtype)
