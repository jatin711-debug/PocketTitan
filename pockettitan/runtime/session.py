"""Session-adaptive expert pinning and warmup stabilization (Phase R8)."""

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple
import torch

from pockettitan.runtime.expert.manager import DecodedExpert, ExpertManager


class SessionAdapter:
    """Stabilizes long-context inference by profiling initial warmup and pinning the session-hot set.
    
    Architecture (Plan.md §5 / R8):
    - Profiles expert usage over the first 64 tokens of a session.
    - At token 64, freezes the session-hot working set and bulk-promotes it to VRAM / protected RAM.
    - Eliminates per-token cache eviction churn across multi-thousand token generations.
    """

    def __init__(
        self,
        expert_manager: ExpertManager,
        warmup_token_threshold: int = 64,
        vram_pin_count: int = 64,  # ~160 MB VRAM
    ):
        self.manager = expert_manager
        self.warmup_threshold = warmup_token_threshold
        self.vram_pin_count = vram_pin_count
        
        self.current_session_token = 0
        self.session_frequencies: Counter[Tuple[int, int]] = Counter()
        self.pinned_vram_keys: Set[Tuple[int, int]] = set()
        self.is_session_pinned = False

    def record_routing_step(self, layer: int, active_experts: List[int]) -> None:
        """Track expert activations during token generation."""
        for exp in active_experts:
            self.session_frequencies[(layer, exp)] += 1

    def step_token(self) -> None:
        """Increment token step and trigger session pinning once warmup is reached."""
        self.current_session_token += 1
        
        if not self.is_session_pinned and self.current_session_token >= self.warmup_threshold:
            self._pin_session_hot_tier()

    def _pin_session_hot_tier(self) -> None:
        """Extract the top sustained experts from warmup and pin them into VRAM / protected RAM."""
        if not self.session_frequencies:
            return

        # 1. Identify most frequent experts
        most_common = self.session_frequencies.most_common()
        
        # 2. Promote top-N into GPU VRAM Hot Tier if CUDA available
        if self.manager.device == "cuda":
            top_vram = most_common[: self.vram_pin_count]
            for (layer, exp), count in top_vram:
                try:
                    expert = self.manager.fetch_expert(layer, exp)
                    vram_expert = expert.to("cuda")
                    self.manager.vram_hot_tier[(layer, exp)] = vram_expert
                    self.pinned_vram_keys.add((layer, exp))
                except Exception:
                    pass

        # 3. Promote next tier into protected RAM SLRU partition
        protected_target = self.manager.ram_cache.protected_capacity
        top_ram = most_common[len(self.pinned_vram_keys) : len(self.pinned_vram_keys) + protected_target]
        for (layer, exp), count in top_ram:
            try:
                # Accessing twice ensures promotion into protected partition
                self.manager.ram_cache.get((layer, exp))
            except Exception:
                pass

        self.is_session_pinned = True

    def reset_session(self) -> None:
        """Reset session state for a new conversation context."""
        self.current_session_token = 0
        self.session_frequencies.clear()
        self.pinned_vram_keys.clear()
        self.is_session_pinned = False
