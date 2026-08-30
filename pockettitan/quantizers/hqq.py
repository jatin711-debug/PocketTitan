"""Half-Quadratic Quantization (HQQ) Backend with Proximal Coordinate Descent."""

from typing import Optional
import torch
import torch.nn.functional as F

from pockettitan.config import QuantConfig
from pockettitan.quantizers.base import BaseQuantizer, QuantizedResult, QuantizerCapabilities, matrix_dims
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
            legal_split_axes=("out_features",),
            requires_full_input_dim=True,
            requires_full_output_dim=False,
            global_state=None,
            supports_cpu=True,
            supports_cuda=True,
            supports_remote_streaming=True,
            # Measured peak/source on a group-aligned matrix: 12.09x.
            # Padding overhead is modelled separately by group_padding_factor().
            workspace_multiplier=13.0,
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

        num_groups = padded_in_features // group_size
        bits = self.config.bits
        max_int = (1 << bits) - 1

        w_grouped = w_2d.view(out_features, num_groups, group_size)

        # 1. Initialize scales and zeros with RTN
        w_min = torch.amin(w_grouped, dim=-1, keepdim=True)
        w_max = torch.amax(w_grouped, dim=-1, keepdim=True)
        scales = torch.clamp((w_max - w_min) / float(max_int), min=1e-8)
        # Not clamped into the code range: see the note in RTNQuantizer.quantize.
        zeros = torch.round(-w_min / scales)

        # 2. Proximal Coordinate Descent Loop
        for _ in range(self.max_iters):
            q_grouped = torch.clamp(torch.round(w_grouped / scales) + zeros, 0, max_int)
            deq_centered = q_grouped - zeros

            numerator = torch.sum(w_grouped * deq_centered, dim=-1, keepdim=True)
            denominator = torch.clamp(torch.sum(deq_centered**2, dim=-1, keepdim=True), min=1e-8)
            scales = torch.clamp(numerator / denominator, min=1e-8)

            q_scaled = q_grouped * scales
            zeros = torch.round(
                torch.mean(q_scaled - w_grouped, dim=-1, keepdim=True) / scales
            )

        q_grouped = torch.clamp(torch.round(w_grouped / scales) + zeros, 0, max_int)
        q_flat = q_grouped.view(out_features, padded_in_features).to(torch.uint8)

        packed_weights = RTNQuantizer._pack_tensor(q_flat, bits)

        return QuantizedResult(
            packed_weights=packed_weights,
            scales=scales.squeeze(-1).to(torch.float16),
            zeros=zeros.squeeze(-1).to(torch.float16),
            codebook=None,
            quant_config=self.config,
            original_shape=orig_shape,
            original_dtype=orig_dtype,
            bit_width=float(bits),
            device=device,
        )

    def dequantize(self, quantized: QuantizedResult) -> torch.Tensor:
        orig_shape = quantized.original_shape
        out_features, in_features = matrix_dims(orig_shape)
        bits = quantized.quant_config.bits
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
        num_groups = padded_in_features // group_size

        q_grouped = unpacked.view(out_features, num_groups, group_size).to(torch.float32)
        scales = quantized.scales.view(out_features, num_groups, 1).to(torch.float32)
        zeros = quantized.zeros.view(out_features, num_groups, 1).to(torch.float32)

        w_deq = (q_grouped - zeros) * scales
        w_flat = w_deq.view(out_features, padded_in_features)[:, :in_features]
        return w_flat.view(orig_shape).to(quantized.original_dtype)
