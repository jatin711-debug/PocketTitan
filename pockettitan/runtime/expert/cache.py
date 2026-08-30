"""Bounded Segmented-LRU (SLRU) cache for out-of-core MoE expert residency (Phase R6)."""

from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import torch


class CachePartition(str, Enum):
    PROBATIONARY = "probationary"
    PROTECTED = "protected"


class BoundedSLRUCache:
    """Bounded Segmented-LRU (SLRU) cache with strict residency invariants.
    
    Architecture (Plan.md §5 / R6):
    - 20% Probationary Partition: newly fetched cold experts land here.
    - 80% Protected Partition: experts accessed at least twice are promoted here.
    - A cold expert fetched once cannot evict a frequently used warm expert.
    - Slot counts are fixed at initialization. No allocation may grow capacity.
    """

    def __init__(self, capacity_slots: int = 2880, probationary_ratio: float = 0.20):
        if capacity_slots <= 0:
            raise ValueError(f"capacity_slots must be positive, got {capacity_slots}")
            
        self.capacity_slots = capacity_slots
        self.probationary_capacity = max(1, int(capacity_slots * probationary_ratio))
        self.protected_capacity = capacity_slots - self.probationary_capacity

        # Key: (layer_idx, expert_idx) -> Value: torch.Tensor or payload
        self.probationary: OrderedDict[Tuple[int, int], Any] = OrderedDict()
        self.protected: OrderedDict[Tuple[int, int], Any] = OrderedDict()

        # Cumulative performance counters
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.promotions = 0
        self.demotions = 0

    @property
    def total_resident(self) -> int:
        return len(self.probationary) + len(self.protected)

    def contains(self, key: Tuple[int, int]) -> bool:
        return key in self.protected or key in self.probationary

    def get(self, key: Tuple[int, int]) -> Optional[Any]:
        """Lookup an expert in the cache and update SLRU recency & promotion state."""
        if key in self.protected:
            # Hit in protected tier -> refresh to MRU
            self.protected.move_to_end(key)
            self.hits += 1
            return self.protected[key]

        if key in self.probationary:
            # Hit in probationary tier -> promote to protected tier
            value = self.probationary.pop(key)
            self.hits += 1
            self.promotions += 1

            if len(self.protected) >= self.protected_capacity:
                # Demote protected LRU back to probationary MRU
                demoted_key, demoted_val = self.protected.popitem(last=False)
                self.probationary[demoted_key] = demoted_val
                self.demotions += 1

            self.protected[key] = value
            return value

        self.misses += 1
        return None

    def put(self, key: Tuple[int, int], value: Any) -> Optional[Tuple[Tuple[int, int], Any]]:
        """Insert a newly fetched cold expert into the probationary tier.
        
        Returns the evicted (key, value) pair if eviction occurred, else None.
        """
        if self.contains(key):
            # Already present, update and refresh
            self.get(key)
            if key in self.protected:
                self.protected[key] = value
            else:
                self.probationary[key] = value
            return None

        evicted = None
        # If probationary tier is full, evict its LRU item
        if len(self.probationary) >= self.probationary_capacity:
            evicted_k, evicted_v = self.probationary.popitem(last=False)
            self.evictions += 1
            evicted = (evicted_k, evicted_v)

        self.probationary[key] = value
        
        # Enforce strict capacity invariant
        assert self.total_resident <= self.capacity_slots, "Residency bound violated!"
        return evicted

    def clear(self) -> None:
        self.probationary.clear(    )
        self.protected.clear()
