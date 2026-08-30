"""Placement-aware expert bank manager with SLRU residency and VRAM promotion (Phase R6)."""

from collections import Counter
import mmap
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import torch

from pockettitan.config import QuantMethod
from pockettitan.package.decode import decode_record
from pockettitan.package.format import ExpertLayout
from pockettitan.runtime.expert.cache import BoundedSLRUCache


class DecodedExpert:
    """In-memory decoded expert projections ready for GEMV execution."""

    def __init__(self, gate_up_weight: torch.Tensor, down_weight: torch.Tensor, device: str = "cpu"):
        self.gate_up = gate_up_weight.to(device)
        self.down = down_weight.to(device)
        self.device = device

    def to(self, target_device: str) -> "DecodedExpert":
        if self.device == target_device:
            return self
        return DecodedExpert(
            gate_up_weight=self.gate_up.to(target_device),
            down_weight=self.down.to(target_device),
            device=target_device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SwiGLU forward pass for this single expert: down(silu(gate) * up)."""
        x_dev = x.to(self.device)
        # gate_up projection: [in_features] -> [2 * intermediate_size]
        h = torch.matmul(x_dev, self.gate_up.t())
        dim = h.shape[-1] // 2
        gate, up = h[..., :dim], h[..., dim:]
        act = torch.nn.functional.silu(gate) * up
        out = torch.matmul(act, self.down.t())
        return out.to(x.device)


class ExpertManager:
    """Manages out-of-core expert streaming, SLRU RAM residency, and GPU VRAM promotion."""

    def __init__(
        self,
        bank_path: Union[str, Path],
        layout: ExpertLayout,
        ram_capacity_slots: int = 2880,  # ~7.0 GB RAM for 4-bit experts
        vram_capacity_slots: int = 64,  # Up to 64 sustained-hot experts in VRAM (~160 MB)
        vram_promotion_threshold: int = 8,  # Promote to VRAM on 8th access
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        quant_method: QuantMethod = QuantMethod.RTN,
    ):
        self.bank_path = Path(bank_path)
        self.layout = layout
        self.device = device
        self.quant_method = quant_method

        # 1. RAM SLRU Cache (20% probationary / 80% protected)
        self.ram_cache = BoundedSLRUCache(capacity_slots=ram_capacity_slots, probationary_ratio=0.20)

        # 2. VRAM Hot Tier (Pinned LFU)
        self.vram_capacity = vram_capacity_slots
        self.vram_promotion_threshold = vram_promotion_threshold
        self.vram_hot_tier: Dict[Tuple[int, int], DecodedExpert] = {}
        self.access_frequency: Counter[Tuple[int, int]] = Counter()

        # 3. Binary bank memory map
        self._fd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None
        self._open_bank()

    def _open_bank(self) -> None:
        if not self.bank_path.exists():
            raise FileNotFoundError(f"Expert bank file not found: {self.bank_path}")
        self._fd = os.open(str(self.bank_path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
        file_size = os.path.getsize(str(self.bank_path))
        if file_size > 0:
            self._mmap = mmap.mmap(self._fd, length=0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "ExpertManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def read_expert_record(self, layer: int, expert: int) -> bytes:
        """Perform a single contiguous read from the page-aligned bank."""
        if self._mmap is None:
            raise RuntimeError("Expert bank mmap is not open")
        offset, length = self.layout.byte_range(layer, expert)
        return self._mmap[offset : offset + length]

    def decode_expert_payload(self, raw_bytes: bytes) -> DecodedExpert:
        """Dequantize one expert record's projections on CPU.

        Both projections are decoded from the layout the planner recorded, at
        whatever bit width and symmetry the precision map assigned. The previous
        implementation hardcoded 4-bit nibbles with a ``-8`` offset and never
        read the ``ZEROS`` section, so it silently discarded the zero-point that
        ``PrecisionEntry.symmetric = False`` (the default) makes mandatory, and
        it sliced rows at the unpadded width so every row after the first was
        misaligned under group padding.
        """
        record = self.layout.record
        decoded = {}
        for index, key in enumerate(("gate_up_proj", "down_proj")):
            spec = record.projection(key) or record.projections[index]
            decoded[key] = decode_record(
                raw_bytes,
                shape=spec.shape,
                bits=spec.bits,
                group_size=spec.group_size,
                symmetric=spec.symmetric,
                spans=spec.spans,
                method=self.quant_method,
                base_offset=spec.offset,
            )
        return DecodedExpert(
            gate_up_weight=decoded["gate_up_proj"],
            down_weight=decoded["down_proj"],
            device="cpu",
        )

    def fetch_expert(self, layer: int, expert: int) -> DecodedExpert:
        """Retrieve an expert, managing VRAM hot tier, RAM SLRU, or on-demand NVMe read."""
        key = (layer, expert)
        self.access_frequency[key] += 1
        freq = self.access_frequency[key]

        # 1. Check GPU VRAM Hot Tier
        if key in self.vram_hot_tier:
            return self.vram_hot_tier[key]

        # 2. Check RAM SLRU Cache
        cached_expert = self.ram_cache.get(key)
        if cached_expert is None:
            # 3. Cache Miss -> Read single contiguous record from NVMe bank.bin
            raw_record = self.read_expert_record(layer, expert)
            cached_expert = self.decode_expert_payload(raw_record)
            self.ram_cache.put(key, cached_expert)

        # 4. Check for GPU VRAM Promotion
        if (
            self.device == "cuda"
            and freq >= self.vram_promotion_threshold
            and len(self.vram_hot_tier) < self.vram_capacity
        ):
            vram_expert = cached_expert.to("cuda")
            self.vram_hot_tier[key] = vram_expert
            return vram_expert

        return cached_expert

    def forward_layer_experts(
        self,
        layer: int,
        top_k_indices: Sequence[int],
        routing_weights: Sequence[float],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute and accumulate outputs for the top-k selected experts of a layer."""
        accumulated_output = torch.zeros_like(hidden_states)
        
        for exp_idx, weight in zip(top_k_indices, routing_weights):
            if weight <= 1e-6:
                continue
            expert = self.fetch_expert(layer, exp_idx)
            # Forward computation (GPU if in VRAM, CPU if in RAM)
            exp_out = expert.forward(hidden_states)
            accumulated_output += exp_out * weight

        return accumulated_output
