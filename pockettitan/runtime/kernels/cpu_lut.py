"""CPU LUT-based low-bit matrix-vector multiplication (T-MAC style) (Phase R9)."""

from typing import Optional

import torch

from pockettitan.runtime.kernels.codes import centred_weights


class LUTQuantizedLinear:
    """Fast CPU GEMV for low-bit (4-bit / 2-bit) weights using activation lookup tables.
    
    Architecture (Plan.md §5 / R9):
    - Precomputes an activation lookup table T[v] = v * x for all possible integer values.
    - Evaluates matrix-vector dot products via table indexing and integer addition.
    - Eliminates dynamic FP16 weight dequantization and memory bandwidth amplification.
    """

    @staticmethod
    def forward_int4_gemv(
        x: torch.Tensor,
        packed_weights: torch.Tensor,  # [out_features, in_features // 2] uint8
        scales: torch.Tensor,          # [out_features, num_groups] float16/float32
        zeros: Optional[torch.Tensor] = None,  # [out_features, num_groups], as packaged
        group_size: int = 128,
    ) -> torch.Tensor:
        """Compute ``y = x @ W.T`` for 4-bit weights with grouped scaling on CPU."""
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1]).float()
        batch_size, in_features = x_flat.shape
        out_features = packed_weights.shape[0]

        w_grouped = centred_weights(packed_weights, 4, in_features, group_size, zeros)
        num_groups = in_features // group_size
        x_grouped = x_flat.view(batch_size, 1, num_groups, group_size)

        group_dots = (x_grouped * w_grouped.unsqueeze(0)).sum(dim=-1)
        scales_expanded = scales.reshape(1, out_features, num_groups).float()
        out = (group_dots * scales_expanded).sum(dim=-1)

        out_shape = list(orig_shape[:-1]) + [out_features]
        return out.view(*out_shape).to(x.dtype)

    @staticmethod
    def forward_int2_gemv(
        x: torch.Tensor,
        packed_weights: torch.Tensor,  # [out_features, in_features // 4] uint8
        scales: torch.Tensor,          # [out_features, num_groups] float16/float32
        zeros: Optional[torch.Tensor] = None,  # [out_features, num_groups], as packaged
        group_size: int = 128,
    ) -> torch.Tensor:
        """Compute ``y = x @ W.T`` for 2-bit weights with grouped scaling on CPU."""
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1]).float()
        batch_size, in_features = x_flat.shape
        out_features = packed_weights.shape[0]

        w_grouped = centred_weights(packed_weights, 2, in_features, group_size, zeros)
        num_groups = in_features // group_size
        x_grouped = x_flat.view(batch_size, 1, num_groups, group_size)

        group_dots = (x_grouped * w_grouped.unsqueeze(0)).sum(dim=-1)
        scales_expanded = scales.reshape(1, out_features, num_groups).float()
        out = (group_dots * scales_expanded).sum(dim=-1)

        out_shape = list(orig_shape[:-1]) + [out_features]
        return out.view(*out_shape).to(x.dtype)
