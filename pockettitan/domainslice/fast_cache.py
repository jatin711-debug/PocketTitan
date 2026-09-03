"""High-performance two-tier in-memory expert cache for DomainSlice."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import torch

from pockettitan.domainslice.types import (
    ModelRevision,
    ProgressCallback,
    WeightPageID,
    WeightStore,
)
from pockettitan.runtime.hf.olmoe_paged import ExpertPageTensors


@dataclass
class CachedExpert:
    gate_up: Any  # torch.Tensor or QuantizedResult
    down: Any     # torch.Tensor or QuantizedResult
    tier: str     # "vram" or "ram"
    is_quantized: bool = False
    arena_slot: Optional[int] = None


class ExpertVRAMArena:
    """Pre-allocated contiguous GPU VRAM buffer pool for expert tensors.

    Eliminates cudaMalloc / cudaFree memory churn and allocator overhead during
    on-demand expert paging.
    """

    def __init__(
        self,
        capacity: int,
        gate_up_shape: Tuple[int, ...],
        down_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.capacity = max(1, capacity)
        self.device = device
        self.dtype = dtype
        self.gate_up_shape = gate_up_shape
        self.down_shape = down_shape

        self.gate_up_pool = torch.empty((self.capacity, *gate_up_shape), dtype=dtype, device=device)
        self.down_pool = torch.empty((self.capacity, *down_shape), dtype=dtype, device=device)

        self.free_slots: List[int] = list(reversed(range(self.capacity)))
        self.key_to_slot: dict[Tuple[int, int], int] = {}
        self.slot_to_key: dict[int, Tuple[int, int]] = {}

    def allocate_slot(self, key: Tuple[int, int]) -> int:
        if key in self.key_to_slot:
            return self.key_to_slot[key]
        if not self.free_slots:
            # Auto-recycle the oldest allocated slot
            oldest_key = next(iter(self.key_to_slot))
            self.free_slot(oldest_key)
        slot = self.free_slots.pop()
        self.key_to_slot[key] = slot
        self.slot_to_key[slot] = key
        return slot

    def free_slot(self, key: Tuple[int, int]) -> Optional[int]:
        slot = self.key_to_slot.pop(key, None)
        if slot is not None:
            self.slot_to_key.pop(slot, None)
            self.free_slots.append(slot)
        return slot

    def get_views(self, slot: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gate_up_pool[slot], self.down_pool[slot]

    def reset(self) -> None:
        self.free_slots = list(reversed(range(self.capacity)))
        self.key_to_slot.clear()
        self.slot_to_key.clear()


class ExpertMemoryCache:
    """Two-tier LRU memory cache for expert weight projections.

    Tier 1 (GPU VRAM): Keeps up to `vram_capacity` hottest experts directly on CUDA.
                       Latency: < 0.1 ms (zero copy).
    Tier 2 (Host RAM): Keeps up to `ram_capacity` warm experts in pinned host memory.
                       When `quantize_ram=True`, experts are stored in 4-bit INT4 (3.3 MB each),
                       allowing all 1,024 experts (the entire model) to fit into 3.38 GB of RAM.
                       Latency: ~0.7 ms DMA transfer + 3.5 ms CUDA dequantization.
    Tier 3 (Disk/Store): Fallback to WeightStore (SQLite + SSD file read).
    """

    def __init__(
        self,
        vram_capacity: int = 144,
        ram_capacity: int = 384,
        quantize_ram: bool = False,
        quant_bits: int = 4,
        quant_group_size: int = 128,
        use_arena: bool = True,
    ):
        self.vram_capacity = max(0, int(vram_capacity))
        self.ram_capacity = max(1, int(ram_capacity))
        self.quantize_ram = bool(quantize_ram)
        self.quant_bits = int(quant_bits)
        self.quant_group_size = int(quant_group_size)
        self.use_arena = bool(use_arena)
        # When RAM quantization is active, all 1024 experts fit into ~3.38 GB Host RAM!
        if self.quantize_ram and ram_capacity == 384:
            self.ram_capacity = 1024
        else:
            self.ram_capacity = max(1, int(ram_capacity))
        self._vram_cache: OrderedDict[Tuple[int, int], CachedExpert] = OrderedDict()
        self._ram_cache: OrderedDict[Tuple[int, int], CachedExpert] = OrderedDict()
        self.vram_hits = 0
        self.ram_hits = 0
        self.disk_faults = 0
        self._quantizer = None
        self._arena: Optional[ExpertVRAMArena] = None

        if self.quantize_ram:
            from pockettitan.config import QuantConfig, QuantMethod
            from pockettitan.quantizers.rtn import RTNQuantizer

            quant_dev = "cuda" if torch.cuda.is_available() else "cpu"
            cfg = QuantConfig(
                method=QuantMethod.RTN,
                bits=self.quant_bits,
                group_size=self.quant_group_size,
                device=quant_dev,
            )
            self._quantizer = RTNQuantizer(cfg)
        self._prefetch_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

    def _ensure_arena(
        self,
        gate_up_shape: Tuple[int, ...],
        down_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if (
            self._arena is None
            and self.use_arena
            and device.type == "cuda"
            and self.vram_capacity > 0
        ):
            try:
                self._arena = ExpertVRAMArena(
                    self.vram_capacity,
                    gate_up_shape,
                    down_shape,
                    dtype,
                    device,
                )
            except torch.cuda.OutOfMemoryError:
                # If pre-allocating full arena exceeds VRAM ceiling, fallback to dynamic allocation
                self._arena = None
                self.use_arena = False

    def clear(self) -> None:
        self._vram_cache.clear()
        self._ram_cache.clear()
        if self._arena is not None:
            self._arena.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evict_oldest_vram(self) -> None:
        if len(self._vram_cache) >= self.vram_capacity:
            old_key, old_expert = self._vram_cache.popitem(last=False)
            if self._arena is not None and old_expert.arena_slot is not None:
                self._arena.free_slot(old_key)
            del old_expert

    def prefetch_batch(
        self,
        model_revision: ModelRevision,
        layer_idx: int,
        expert_ids: List[int],
        *,
        target_device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Asynchronously initiate non-blocking PCIe DMA transfers for upcoming experts."""
        if target_device.type != "cuda" or self.vram_capacity <= 0:
            return
        stream_ctx = (
            torch.cuda.stream(self._prefetch_stream)
            if self._prefetch_stream is not None
            else nullcontext()
        )
        with stream_ctx:
            for expert_id in expert_ids:
                key = (layer_idx, expert_id)
                if key not in self._vram_cache and key in self._ram_cache:
                    expert = self._ram_cache[key]
                    self._evict_oldest_vram()

                    # Infer shapes for arena initialization if not ready
                    gu_shape = expert.gate_up.original_shape if expert.is_quantized else expert.gate_up.shape
                    dn_shape = expert.down.original_shape if expert.is_quantized else expert.down.shape
                    self._ensure_arena(gu_shape, dn_shape, dtype, target_device)

                    if self._arena is not None:
                        slot = self._arena.allocate_slot(key)
                        gu_view, dn_view = self._arena.get_views(slot)
                        if expert.is_quantized and self._quantizer is not None:
                            gu_dev = expert.gate_up.to(device=target_device, non_blocking=True)
                            dn_dev = expert.down.to(device=target_device, non_blocking=True)
                            self._quantizer.dequantize_to(gu_dev, gu_view)
                            self._quantizer.dequantize_to(dn_dev, dn_view)
                        else:
                            gu_view.copy_(expert.gate_up, non_blocking=True)
                            dn_view.copy_(expert.down, non_blocking=True)
                        self._vram_cache[key] = CachedExpert(
                            gate_up=gu_view, down=dn_view, tier="vram", is_quantized=False, arena_slot=slot
                        )
                    else:
                        if expert.is_quantized and self._quantizer is not None:
                            gu_dev = expert.gate_up.to(device=target_device, non_blocking=True)
                            dn_dev = expert.down.to(device=target_device, non_blocking=True)
                            gate_up = self._quantizer.dequantize(gu_dev).to(dtype=dtype)
                            down = self._quantizer.dequantize(dn_dev).to(dtype=dtype)
                        else:
                            gate_up = expert.gate_up.to(device=target_device, dtype=dtype, non_blocking=True)
                            down = expert.down.to(device=target_device, dtype=dtype, non_blocking=True)
                        self._vram_cache[key] = CachedExpert(
                            gate_up=gate_up, down=down, tier="vram", is_quantized=False
                        )

    def get_or_load(
        self,
        model_revision: ModelRevision,
        layer_idx: int,
        expert_id: int,
        store: WeightStore,
        *,
        target_device: torch.device,
        dtype: torch.dtype,
        progress: Optional[ProgressCallback] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Fetch expert tensors (gate_up, down) from VRAM, RAM, or SSD storage."""
        key = (layer_idx, expert_id)

        # 1. Check Tier 1: CUDA VRAM Cache
        if target_device.type == "cuda" and key in self._vram_cache:
            if self._prefetch_stream is not None:
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            self.vram_hits += 1
            expert = self._vram_cache[key]
            self._vram_cache.move_to_end(key)
            return expert.gate_up, expert.down, "vram"

        # 2. Check Tier 2: Host RAM Cache
        if key in self._ram_cache:
            self.ram_hits += 1
            expert = self._ram_cache[key]
            self._ram_cache.move_to_end(key)
            
            # Promote to Tier 1 VRAM if target is CUDA (evict oldest first)
            if target_device.type == "cuda" and self.vram_capacity > 0:
                self._evict_oldest_vram()
                gu_shape = expert.gate_up.original_shape if expert.is_quantized else expert.gate_up.shape
                dn_shape = expert.down.original_shape if expert.is_quantized else expert.down.shape
                self._ensure_arena(gu_shape, dn_shape, dtype, target_device)

                if self._arena is not None:
                    slot = self._arena.allocate_slot(key)
                    gu_view, dn_view = self._arena.get_views(slot)
                    if expert.is_quantized and self._quantizer is not None:
                        gu_dev = expert.gate_up.to(device=target_device, non_blocking=True)
                        dn_dev = expert.down.to(device=target_device, non_blocking=True)
                        self._quantizer.dequantize_to(gu_dev, gu_view)
                        self._quantizer.dequantize_to(dn_dev, dn_view)
                    else:
                        gu_view.copy_(expert.gate_up, non_blocking=True)
                        dn_view.copy_(expert.down, non_blocking=True)
                    self._vram_cache[key] = CachedExpert(
                        gate_up=gu_view, down=dn_view, tier="vram", is_quantized=False, arena_slot=slot
                    )
                    return gu_view, dn_view, "ram"

                if expert.is_quantized and self._quantizer is not None:
                    gu_dev = expert.gate_up.to(device=target_device, non_blocking=True)
                    dn_dev = expert.down.to(device=target_device, non_blocking=True)
                    gate_up = self._quantizer.dequantize(gu_dev).to(dtype=dtype)
                    down = self._quantizer.dequantize(dn_dev).to(dtype=dtype)
                else:
                    gate_up = expert.gate_up.to(device=target_device, dtype=dtype, non_blocking=True)
                    down = expert.down.to(device=target_device, dtype=dtype, non_blocking=True)
                self._vram_cache[key] = CachedExpert(gate_up=gate_up, down=down, tier="vram", is_quantized=False)
                return gate_up, down, "ram"

            if expert.is_quantized and self._quantizer is not None:
                gu_dev = expert.gate_up.to(device=target_device, non_blocking=True)
                dn_dev = expert.down.to(device=target_device, non_blocking=True)
                gate_up = self._quantizer.dequantize(gu_dev).to(dtype=dtype)
                down = self._quantizer.dequantize(dn_dev).to(dtype=dtype)
            else:
                gate_up = expert.gate_up.to(device=target_device, dtype=dtype, non_blocking=True)
                down = expert.down.to(device=target_device, dtype=dtype, non_blocking=True)
            return gate_up, down, "ram"

        # 3. Tier 3: Disk / WeightStore Fallback
        self.disk_faults += 1
        page_id = WeightPageID.expert(model_revision, layer_idx, expert_id)
        descriptor = store.resolve(page_id)
        handle = store.materialize(page_id, progress=progress)
        try:
            page = ExpertPageTensors(handle, descriptor)
            gate = page.tensor("gate_proj").to(dtype=dtype)
            up = page.tensor("up_proj").to(dtype=dtype)
            down = page.tensor("down_proj").to(dtype=dtype)
            gate_up = torch.cat([gate, up], dim=0)

            # Stage in Host RAM (compressed to INT4 in 1ms on GPU)
            if self.quantize_ram and self._quantizer is not None:
                if target_device.type == "cuda":
                    gate_up_dev = gate_up.to(device=target_device, dtype=dtype, non_blocking=True)
                    down_dev = down.to(device=target_device, dtype=dtype, non_blocking=True)
                    gu_q = self._quantizer.quantize(gate_up_dev.float()).to("cpu")
                    dn_q = self._quantizer.quantize(down_dev.float()).to("cpu")
                    try:
                        gu_q = gu_q.pin_memory()
                        dn_q = dn_q.pin_memory()
                    except Exception:
                        pass
                    if len(self._ram_cache) >= self.ram_capacity:
                        self._ram_cache.popitem(last=False)
                    self._ram_cache[key] = CachedExpert(gate_up=gu_q, down=dn_q, tier="ram", is_quantized=True)
                    if self.vram_capacity > 0:
                        self._evict_oldest_vram()
                        self._ensure_arena(gate_up.shape, down.shape, dtype, target_device)
                        if self._arena is not None:
                            slot = self._arena.allocate_slot(key)
                            gu_v, dn_v = self._arena.get_views(slot)
                            gu_v.copy_(gate_up_dev, non_blocking=True)
                            dn_v.copy_(down_dev, non_blocking=True)
                            self._vram_cache[key] = CachedExpert(
                                gate_up=gu_v, down=dn_v, tier="vram", is_quantized=False, arena_slot=slot
                            )
                            return gu_v, dn_v, "disk"
                        self._vram_cache[key] = CachedExpert(gate_up=gate_up_dev, down=down_dev, tier="vram", is_quantized=False)
                    return gate_up_dev, down_dev, "disk"
                else:
                    gu_q = self._quantizer.quantize(gate_up.float()).to("cpu")
                    dn_q = self._quantizer.quantize(down.float()).to("cpu")
                    if len(self._ram_cache) >= self.ram_capacity:
                        self._ram_cache.popitem(last=False)
                    self._ram_cache[key] = CachedExpert(gate_up=gu_q, down=dn_q, tier="ram", is_quantized=True)
                    return gate_up, down, "disk"
            else:
                can_pin = torch.cuda.is_available() and target_device.type == "cuda"
                gate_up_ram = gate_up.detach().cpu()
                down_ram = down.detach().cpu()
                if can_pin:
                    try:
                        gate_up_ram = gate_up_ram.pin_memory()
                        down_ram = down_ram.pin_memory()
                    except Exception:
                        pass  # Non-fatal if host pinning is unavailable

                if len(self._ram_cache) >= self.ram_capacity:
                    self._ram_cache.popitem(last=False)
                self._ram_cache[key] = CachedExpert(gate_up=gate_up_ram, down=down_ram, tier="ram", is_quantized=False)

                # Evict from VRAM before allocating new CUDA tensor
                if target_device.type == "cuda" and self.vram_capacity > 0:
                    self._evict_oldest_vram()
                    self._ensure_arena(gate_up.shape, down.shape, dtype, target_device)
                    if self._arena is not None:
                        slot = self._arena.allocate_slot(key)
                        gu_v, dn_v = self._arena.get_views(slot)
                        gu_v.copy_(gate_up, non_blocking=True)
                        dn_v.copy_(down, non_blocking=True)
                        self._vram_cache[key] = CachedExpert(
                            gate_up=gu_v, down=dn_v, tier="vram", is_quantized=False, arena_slot=slot
                        )
                        return gu_v, dn_v, "disk"

                gate_up_dev = gate_up.to(device=target_device, dtype=dtype, non_blocking=True)
                down_dev = down.to(device=target_device, dtype=dtype, non_blocking=True)

                if target_device.type == "cuda" and self.vram_capacity > 0:
                    self._vram_cache[key] = CachedExpert(gate_up=gate_up_dev, down=down_dev, tier="vram", is_quantized=False)

                return gate_up_dev, down_dev, "disk"
        finally:
            store.release(handle)
