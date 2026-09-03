"""Vectorized Round-to-Nearest (RTN) Quantizer with sub-byte bit packing."""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from pockettitan.config import QuantConfig
from pockettitan.quantizers.base import BaseQuantizer, QuantizedResult, QuantizerCapabilities, matrix_dims


class RTNQuantizer(BaseQuantizer):
    """Vectorized Uniform Groupwise Round-to-Nearest Quantizer supporting 1-8 bits."""

    def __init__(self, config: QuantConfig):
        super().__init__(config)

    @property
    def capabilities(self) -> QuantizerCapabilities:
        return QuantizerCapabilities(
            name="rtn",
            requires_calibration=False,
            legal_split_axes=("out_features",),
            requires_full_input_dim=True,
            requires_full_output_dim=False,
            global_state=None,
            supports_cpu=True,
            supports_cuda=True,
            supports_remote_streaming=True,
            # Measured peak/source on a group-aligned matrix: 6.05x.
            # Padding overhead is modelled separately by group_padding_factor().
            workspace_multiplier=6.5,
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

        num_groups = padded_in_features // group_size
        bits = self.config.bits
        max_int = (1 << bits) - 1

        w_grouped = w_2d.view(out_features, num_groups, group_size)

        if self.config.symmetric:
            w_max = torch.amax(torch.abs(w_grouped), dim=-1, keepdim=True)
            scales = torch.clamp(w_max / (max_int / 2.0), min=1e-8)
            zeros = None
            q_grouped = torch.clamp(
                torch.round(w_grouped / scales), -(max_int // 2), (max_int // 2)
            )
            q_grouped = q_grouped + (max_int // 2)
        else:
            w_min = torch.amin(w_grouped, dim=-1, keepdim=True)
            w_max = torch.amax(w_grouped, dim=-1, keepdim=True)
            scales = torch.clamp((w_max - w_min) / float(max_int), min=1e-8)
            # The affine zero-point is stored as fp16 and only ever used as
            # `(q - zero) * scale`, so it must NOT be clamped into the code
            # range. A group that does not straddle zero has its correct
            # zero-point outside [0, max_int]; clamping forces the
            # representable interval to include 0 and spends the entire
            # budget on the empty gap. An all-positive vector clustered away
            # from zero then collapses to a single code.
            zeros = torch.round(-w_min / scales)
            q_grouped = torch.clamp(torch.round(w_grouped / scales) + zeros, 0, max_int)

        q_flat = q_grouped.view(out_features, padded_in_features).to(torch.uint8)
        packed_weights = self._pack_tensor(q_flat, bits)

        return QuantizedResult(
            packed_weights=packed_weights,
            scales=scales.squeeze(-1).to(torch.float16),
            zeros=zeros.squeeze(-1).to(torch.float16) if zeros is not None else None,
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

        unpacked = self._unpack_tensor(
            quantized.packed_weights, bits, (out_features, padded_in_features)
        )
        num_groups = padded_in_features // group_size

        target_dtype = quantized.original_dtype
        q_grouped = unpacked.view(out_features, num_groups, group_size).to(target_dtype)
        scales = quantized.scales.view(out_features, num_groups, 1).to(target_dtype)

        if quantized.zeros is not None:
            zeros = quantized.zeros.view(out_features, num_groups, 1).to(target_dtype)
            w_deq = (q_grouped - zeros) * scales
        else:
            max_int = (1 << bits) - 1
            w_deq = (q_grouped - (max_int // 2)) * scales

        w_flat = w_deq.view(out_features, padded_in_features)[:, :in_features]
        return w_flat.reshape(orig_shape)

    def dequantize_to(self, quantized: QuantizedResult, out: torch.Tensor) -> torch.Tensor:
        """Dequantize directly into an existing pre-allocated output tensor."""
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

        unpacked = self._unpack_tensor(
            quantized.packed_weights, bits, (out_features, padded_in_features)
        )
        num_groups = padded_in_features // group_size
        target_dtype = out.dtype

        q_grouped = unpacked.view(out_features, num_groups, group_size).to(target_dtype)
        scales = quantized.scales.view(out_features, num_groups, 1).to(target_dtype)

        if quantized.zeros is not None:
            zeros = quantized.zeros.view(out_features, num_groups, 1).to(target_dtype)
            w_deq = (q_grouped - zeros) * scales
        else:
            max_int = (1 << bits) - 1
            w_deq = (q_grouped - (max_int // 2)) * scales

        w_flat = w_deq.view(out_features, padded_in_features)[:, :in_features]
        out.view(-1).copy_(w_flat.reshape(-1))
        return out

    @staticmethod
    def _pack_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Packs sub-byte integer tensor into uint8 byte array."""
        if bits >= 16 or bits <= 0:
            raise ValueError(
                f"bits={bits} is not a packable width; 16-bit and wider tensors are "
                "stored verbatim and never reach the packer"
            )
        flat = tensor.contiguous().view(-1)
        vals_per_byte = 8 // bits
        if flat.numel() % vals_per_byte != 0:
            pad_len = vals_per_byte - (flat.numel() % vals_per_byte)
            flat = F.pad(flat, (0, pad_len))

        packed = torch.zeros(flat.numel() // vals_per_byte, dtype=torch.uint8, device=tensor.device)
        for i in range(vals_per_byte):
            shift = i * bits
            packed |= flat[i::vals_per_byte].to(torch.uint8) << shift
        return packed

    @staticmethod
    def _unpack_tensor(
        packed: torch.Tensor, bits: int, target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Unpacks uint8 byte array back into integer codes."""
        vals_per_byte = 8 // bits
        mask = (1 << bits) - 1
        num_elements = math.prod(target_shape)

        unpacked = torch.zeros(
            packed.numel() * vals_per_byte, dtype=torch.uint8, device=packed.device
        )
        for i in range(vals_per_byte):
            shift = i * bits
            unpacked[i::vals_per_byte] = (packed >> shift) & mask

        return unpacked[:num_elements].view(target_shape)
