"""Shared unpacking for the low-bit kernels.

The kernels consume bytes produced by :class:`RTNQuantizer`, so they must use
that packer's conventions rather than a plausible-looking constant of their own.
Two conventions matter and both were previously wrong here:

* **The symmetric offset is ``max_int // 2``, not ``2 ** (bits - 1)``.** The
  packer centres codes on ``(1 << bits) - 1) // 2`` — 7 at 4 bits and 1 at
  2 bits. The kernels subtracted 8 and 2, biasing every weight by one code.
* **Asymmetric records carry a per-group zero-point.**
  ``PrecisionEntry.symmetric`` defaults to ``False``, so a packaged expert has a
  ``ZEROS`` section that a fixed offset cannot represent at all.
"""

from typing import Optional

import torch


def symmetric_offset(bits: int) -> int:
    """The zero-point RTN uses when no ``ZEROS`` section is stored."""
    return ((1 << bits) - 1) // 2


def unpack_codes(packed: torch.Tensor, bits: int, in_features: int) -> torch.Tensor:
    """Expand packed bytes into ``[rows, in_features]`` unsigned codes.

    Element ``i`` occupies bit ``i * bits`` of its byte, matching
    ``RTNQuantizer._pack_tensor``.
    """
    values_per_byte = 8 // bits
    mask = (1 << bits) - 1
    rows = packed.shape[0]
    out = torch.empty((rows, in_features), dtype=torch.uint8, device=packed.device)
    for i in range(values_per_byte):
        out[:, i::values_per_byte] = (packed >> (i * bits)) & mask
    return out


def centred_weights(
    packed: torch.Tensor,
    bits: int,
    in_features: int,
    group_size: int,
    zeros: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``codes - zero_point`` as float, grouped ``[rows, num_groups, group_size]``."""
    if in_features % group_size:
        raise ValueError(
            f"in_features={in_features} is not a multiple of group_size={group_size}; "
            "the planner resolves group sizes to divisors so records are never padded"
        )
    codes = unpack_codes(packed, bits, in_features).float()
    rows = codes.shape[0]
    num_groups = in_features // group_size
    grouped = codes.view(rows, num_groups, group_size)
    if zeros is None:
        return grouped - float(symmetric_offset(bits))
    return grouped - zeros.reshape(rows, num_groups, 1).float()
