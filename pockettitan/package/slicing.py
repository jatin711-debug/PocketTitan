"""Slice-granular addressing into source checkpoints (R1, task T1.2).

Qwen3.8-Flash-Next packs all 512 experts of a layer into two fused tensors,
``(512, 1280, 2560)`` and ``(512, 2560, 640)``. An expert is therefore a *slice*,
not a tensor, and ``TensorAddress`` alone cannot name it. This module resolves
``(layer, expert)`` to the exact byte ranges that must be read, for both fused
and per-expert checkpoint layouts.

Byte arithmetic is derived from the element width implied by
``size_bytes / num_params``, so FP8 and BF16 banks are handled without a dtype
table lookup.
"""

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from pockettitan.audit.classify import Component, classify_tensor
from pockettitan.config import TensorAddress

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_INDEX_RE = re.compile(r"\.experts\.(\d+)\.")

# Ordered so that a repacked record places gate/up before down, matching
# execution order in the FFN.
_PROJECTION_PRIORITY = ("gate_up_proj", "gate_proj", "up_proj", "down_proj")


class SliceError(ValueError):
    """Raised when a slice cannot be addressed unambiguously."""


class SourceSlice(BaseModel):
    """A contiguous byte range inside one source shard."""

    tensor: str
    shard: str
    projection: str = Field(description="Logical projection name, e.g. 'gate_up_proj'")
    dtype: str
    shape: List[int] = Field(description="Shape of the slice, expert axis removed")
    byte_start: int
    byte_end: int
    expert_index: Optional[int] = Field(
        default=None, description="Index into a fused bank; None when the slice is a whole tensor"
    )

    @property
    def num_params(self) -> int:
        return math.prod(self.shape) if self.shape else 0

    @property
    def size_bytes(self) -> int:
        return self.byte_end - self.byte_start


class ExpertSlice(BaseModel):
    """Every byte belonging to one expert, across however many source tensors."""

    layer: int
    expert: int
    projections: List[SourceSlice] = Field(default_factory=list)

    @property
    def num_params(self) -> int:
        return sum(p.num_params for p in self.projections)

    @property
    def source_bytes(self) -> int:
        return sum(p.size_bytes for p in self.projections)

    @property
    def source_reads(self) -> int:
        """Reads required from the *source* checkpoint. The packaged bank
        reduces this to 1 (Plan.md T1.3)."""
        return len(self.projections)


def layer_index(name: str) -> Optional[int]:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _projection_name(tensor_name: str) -> str:
    """Derive the logical projection name from a tensor name."""
    leaf = tensor_name.rsplit(".", 1)[-1]
    if leaf == "weight":
        parts = tensor_name.split(".")
        leaf = parts[-2] if len(parts) >= 2 else leaf
    return leaf


def _element_bytes(address: TensorAddress) -> float:
    if address.num_params <= 0:
        raise SliceError(f"Tensor '{address.name}' reports zero parameters")
    return address.size_bytes / address.num_params


def slice_expert_from_bank(address: TensorAddress, expert_index: int, num_experts: int) -> SourceSlice:
    """Address one expert inside a fused ``(num_experts, ...)`` bank tensor.

    Raises:
        SliceError: If the tensor is not a bank, or its leading axis disagrees
            with the routing topology. Both indicate the caller is about to read
            the wrong bytes, so neither is recoverable.
    """
    if len(address.shape) < 2:
        raise SliceError(
            f"Tensor '{address.name}' has shape {address.shape}; a fused expert bank needs >= 2 dims"
        )
    if address.shape[0] != num_experts:
        raise SliceError(
            f"Tensor '{address.name}' leading axis is {address.shape[0]}, "
            f"but routing config declares {num_experts} experts"
        )
    if not 0 <= expert_index < num_experts:
        raise SliceError(f"Expert {expert_index} out of range [0, {num_experts})")

    per_expert_shape = list(address.shape[1:])
    per_expert_params = math.prod(per_expert_shape)
    element_bytes = _element_bytes(address)
    per_expert_bytes = int(round(per_expert_params * element_bytes))

    if per_expert_bytes * num_experts != address.size_bytes:
        raise SliceError(
            f"Tensor '{address.name}' is not evenly divisible into {num_experts} experts "
            f"({address.size_bytes} bytes / {num_experts})"
        )

    start = address.byte_start + expert_index * per_expert_bytes
    return SourceSlice(
        tensor=address.name,
        shard=address.shard,
        projection=_projection_name(address.name),
        dtype=address.dtype,
        shape=per_expert_shape,
        byte_start=start,
        byte_end=start + per_expert_bytes,
        expert_index=expert_index,
    )


def _sort_projections(slices: List[SourceSlice]) -> List[SourceSlice]:
    def key(s: SourceSlice) -> tuple:
        try:
            return (_PROJECTION_PRIORITY.index(s.projection), s.projection)
        except ValueError:
            return (len(_PROJECTION_PRIORITY), s.projection)

    return sorted(slices, key=key)


def expert_bank_tensors(tensors: Dict[str, TensorAddress]) -> Dict[int, List[TensorAddress]]:
    """Group routed-expert tensors by source layer index."""
    banks: Dict[int, List[TensorAddress]] = {}
    for address in tensors.values():
        if classify_tensor(address.name).component is not Component.EXPERTS_ROUTED:
            continue
        layer = layer_index(address.name)
        if layer is None:
            continue
        banks.setdefault(layer, []).append(address)
    return banks


def build_expert_slices(
    tensors: Dict[str, TensorAddress],
    num_experts: int,
    layers: Optional[Sequence[int]] = None,
    experts: Optional[Sequence[int]] = None,
) -> List[ExpertSlice]:
    """Resolve every ``(layer, expert)`` to its source byte ranges.

    Handles both storage conventions:

    * **Fused** — ``layers.L.mlp.experts.gate_up_proj`` with shape
      ``(num_experts, out, in)``; each expert is a sub-range.
    * **Per-expert** — ``layers.L.mlp.experts.E.gate_proj.weight``; each expert
      already owns whole tensors.

    Args:
        tensors: Full tensor inventory from an R0 scan.
        num_experts: Routing width, from config.
        layers: Restrict to these layers. Defaults to all layers with experts.
        experts: Restrict to these expert indices. Defaults to all.

    Returns:
        Slices ordered layer-major then expert-major, matching bank order.
    """
    banks = expert_bank_tensors(tensors)
    target_layers = sorted(banks) if layers is None else [l for l in layers if l in banks]
    target_experts = range(num_experts) if experts is None else experts

    out: List[ExpertSlice] = []
    for layer in target_layers:
        addresses = banks[layer]
        fused = [a for a in addresses if _EXPERT_INDEX_RE.search(a.name) is None]
        per_expert: Dict[int, List[TensorAddress]] = {}
        for a in addresses:
            match = _EXPERT_INDEX_RE.search(a.name)
            if match is not None:
                per_expert.setdefault(int(match.group(1)), []).append(a)

        if fused and per_expert:
            raise SliceError(
                f"Layer {layer} mixes fused and per-expert tensors; refusing to guess the layout"
            )

        for expert in target_experts:
            if fused:
                slices = [slice_expert_from_bank(a, expert, num_experts) for a in fused]
            else:
                owned = per_expert.get(expert)
                if not owned:
                    raise SliceError(f"Layer {layer} expert {expert} has no tensors")
                slices = [
                    SourceSlice(
                        tensor=a.name,
                        shard=a.shard,
                        projection=_projection_name(a.name),
                        dtype=a.dtype,
                        shape=list(a.shape),
                        byte_start=a.byte_start,
                        byte_end=a.byte_end,
                    )
                    for a in owned
                ]
            out.append(
                ExpertSlice(layer=layer, expert=expert, projections=_sort_projections(slices))
            )

    return out


def projection_signature(slices: Iterable[ExpertSlice]) -> List[Dict[str, object]]:
    """Shared projection geometry across experts, for building a record layout.

    Raises:
        SliceError: If experts disagree on geometry. A single record layout
            cannot then describe the bank, and the packager must not proceed.
    """
    signature: Optional[List[Dict[str, object]]] = None
    for expert_slice in slices:
        current = [
            {"name": p.projection, "shape": list(p.shape), "dtype": p.dtype}
            for p in expert_slice.projections
        ]
        if signature is None:
            signature = current
        elif current != signature:
            raise SliceError(
                f"Layer {expert_slice.layer} expert {expert_slice.expert} has geometry {current}, "
                f"expected {signature}"
            )
    if signature is None:
        raise SliceError("No expert slices supplied")
    return signature
