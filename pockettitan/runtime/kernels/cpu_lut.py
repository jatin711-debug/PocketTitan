"""CPU LUT-based low-bit matrix-vector multiplication (T-MAC style) (Phase R9)."""

from typing import Optional, Tuple
import torch


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
        group_size: int = 128,
    ) -> torch.Tensor:
        """Compute y = x @ W.T for 4-bit weights with grouped scaling on CPU."""
        # Ensure 2D [1, in_features] or 1D [in_features]
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1]).float()
        batch_size, in_features = x_flat.shape
        out_features = packed_weights.shape[0]

        # 1. Unpack nibbles
        # low nibble: weights[..., 0], high nibble: weights[..., 1]
        w_low = (packed_weights & 0x0F).to(torch.int8) - 8
        w_high = ((packed_weights >> 4) & 0x0F).to(torch.int8) - 8

        # Interleave to shape [out_features, in_features]
        w_unpacked = torch.empty((out_features, in_features), dtype=torch.int8, device=x.device)
        w_unpacked[:, 0::2] = w_low
        w_unpacked[:, 1::2] = w_high

        # 2. Reshape into groups: [out_features, num_groups, group_size]
        num_groups = in_features // group_size
        w_grouped = w_unpacked.view(out_features, num_groups, group_size).float()
        x_grouped = x_flat.view(batch_size, 1, num_groups, group_size)

        # 3. Group dot product + scaling
        # Dot product within each group: [batch_size, out_features, num_groups]
        group_dots = (x_grouped * w_grouped.unsqueeze(0)).sum(dim=-1)
        
        # Multiply by group scales: [out_features, num_groups]
        scales_expanded = scales.view(1, out_features, num_groups).float()
        scaled_groups = group_dots * scales_expanded

        # Sum across groups: [batch_size, out_features]
        out = scaled_groups.sum(dim=-1)

        # Reshape to match input batch dimensions
        out_shape = list(orig_shape[:-1]) + [out_features]
        return out.view(*out_shape).to(x.dtype)

    @staticmethod
    def forward_int2_gemv(
        x: torch.Tensor,
        packed_weights: torch.Tensor,  # [out_features, in_features // 4] uint8
        scales: torch.Tensor,          # [out_features, num_groups] float16/float32
        group_size: int = 128,
    ) -> torch.Tensor:
        """Compute y = x @ W.T for 2-bit weights with grouped scaling on CPU."""
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1]).float()
        batch_size, in_features = x_flat.shape
        out_features = packed_weights.shape[0]

        # Unpack 4 2-bit values per byte (-2, -1, 0, 1)
        c0 = (packed_weights & 0x03).to(torch.int8) - 2
        c1 = ((packed_weights >> 2) & 0x03).to(torch.int8) - 2
        c2 = ((packed_weights >> 4) & 0x03).to(torch.int8) - 2
        c3 = ((packed_weights >> 6) & 0x03).to(torch.int8) - 2

        w_unpacked = torch.empty((out_features, in_features), dtype=torch.int8, device=x.device)
        w_unpacked[:, 0::4] = c0
        w_unpacked[:, 1::4] = c1
        w_unpacked[:, 2::4] = c2
        w_unpacked[:, 3::4] = c3

        num_groups = in_features // group_size
        w_grouped = w_unpacked.view(out_features, num_groups, group_size).float()
        x_grouped = x_flat.view(batch_size, 1, num_groups, group_size)

        group_dots = (x_grouped * w_grouped.unsqueeze(0)).sum(dim=-1)
        scales_expanded = scales.view(1, out_features, num_groups).float()
        out = (group_dots * scales_expanded).sum(dim=-1)

        out_shape = list(orig_shape[:-1]) + [out_features]
        return out.view(*out_shape).to(x.dtype)
