"""Pinned host memory ring buffers and async H2D transfer queue."""

from typing import Dict, List, Optional, Tuple
import torch


class PinnedHostRingBuffer:
    """Pre-allocated pinned host memory buffer pool for asynchronous GPU transfers."""

    def __init__(self, buffer_size_mb: int = 512, num_slots: int = 2):
        self.buffer_size_bytes = buffer_size_mb * 1024 * 1024
        self.num_slots = num_slots
        self.cuda_available = torch.cuda.is_available()
        self._slots: List[Optional[torch.Tensor]] = [None] * num_slots
        self._current_slot = 0
        
        # Pre-allocate pinned memory byte buffers if CUDA is active
        if self.cuda_available:
            for i in range(num_slots):
                try:
                    self._slots[i] = torch.empty(
                        self.buffer_size_bytes,
                        dtype=torch.uint8,
                        pin_memory=True,
                    )
                except Exception:
                    self._slots[i] = None

    def transfer_to_device(
        self,
        tensor: torch.Tensor,
        device: str = "cuda",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Transfer tensor to CUDA device using non-blocking DMA if possible."""
        if device == "cpu" or not self.cuda_available:
            return tensor.to(device)
            
        target_device = torch.device(device)
        
        # If tensor is already contiguous and on CPU, check if we can pin and stream
        if not tensor.is_pinned():
            try:
                pinned_tensor = tensor.pin_memory()
            except Exception:
                pinned_tensor = tensor
        else:
            pinned_tensor = tensor

        if stream is not None:
            with torch.cuda.stream(stream):
                return pinned_tensor.to(target_device, non_blocking=True)
        else:
            return pinned_tensor.to(target_device, non_blocking=True)

    def cleanup(self) -> None:
        """Release pinned buffers."""
        for i in range(self.num_slots):
            self._slots[i] = None
        if self.cuda_available:
            torch.cuda.empty_cache()
