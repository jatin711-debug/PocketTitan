"""Cache replacement policy implementations for out-of-core MoE expert paging (R2).

Includes online policies (OS Page Cache, LRU, Segmented-LRU, TinyLFU) and the
offline Belady's Optimal Replacement (Oracle) algorithm as an upper bound.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Dict, List, Sequence, Set, Tuple


class CachePolicy(ABC):
    """Abstract base class for expert cache simulation."""

    def __init__(self, capacity_slots: int):
        self.capacity = max(1, capacity_slots)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def total_accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    @abstractmethod
    def access(self, layer: int, expert: int, step: int) -> bool:
        """Record an access to (layer, expert). Returns True on HIT, False on MISS."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal cache state and access counters."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0


class LRUCache(CachePolicy):
    """Standard Least-Recently-Used (LRU) cache."""

    def __init__(self, capacity_slots: int):
        super().__init__(capacity_slots)
        self.cache: OrderedDict[Tuple[int, int], None] = OrderedDict()

    def access(self, layer: int, expert: int, step: int) -> bool:
        key = (layer, expert)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True
        
        self.misses += 1
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # Evict oldest
            self.evictions += 1
            
        self.cache[key] = None
        return False

    def reset(self) -> None:
        super().reset()
        self.cache.clear()


class OSPageCache(LRUCache):
    """Simulates the OS kernel page cache with global page-stealing LRU."""
    pass


class SLRUCache(CachePolicy):
    """Segmented LRU (SLRU) cache with Probationary and Protected tiers.
    
    Default split: 20% Probationary / 80% Protected.
    New items enter Probationary. Repeated hits promote to Protected.
    Protected overflow demotes to the head of Probationary.
    """

    def __init__(self, capacity_slots: int, protected_ratio: float = 0.8):
        super().__init__(capacity_slots)
        self.protected_cap = max(1, int(capacity_slots * protected_ratio))
        self.probationary_cap = max(1, capacity_slots - self.protected_cap)
        self.probationary: OrderedDict[Tuple[int, int], None] = OrderedDict()
        self.protected: OrderedDict[Tuple[int, int], None] = OrderedDict()

    def access(self, layer: int, expert: int, step: int) -> bool:
        key = (layer, expert)
        
        # 1. Protected Hit
        if key in self.protected:
            self.protected.move_to_end(key)
            self.hits += 1
            return True
            
        # 2. Probationary Hit -> Promote to Protected
        if key in self.probationary:
            del self.probationary[key]
            if len(self.protected) >= self.protected_cap:
                # Demote LRU of protected to probationary
                demoted_key, _ = self.protected.popitem(last=False)
                if len(self.probationary) >= self.probationary_cap:
                    self.probationary.popitem(last=False)
                    self.evictions += 1
                self.probationary[demoted_key] = None
            self.protected[key] = None
            self.hits += 1
            return True
            
        # 3. Cache Miss -> Insert into Probationary
        self.misses += 1
        if len(self.probationary) >= self.probationary_cap:
            self.probationary.popitem(last=False)
            self.evictions += 1
            
        self.probationary[key] = None
        return False

    def reset(self) -> None:
        super().reset()
        self.probationary.clear()
        self.protected.clear()


class TinyLFUCache(CachePolicy):
    """Frequency-based admission filter over an LRU cache."""

    def __init__(self, capacity_slots: int, sample_window: int = 10000):
        super().__init__(capacity_slots)
        self.cache: OrderedDict[Tuple[int, int], None] = OrderedDict()
        self.freqs: Dict[Tuple[int, int], int] = defaultdict(int)
        self.sample_window = sample_window
        self.total_count = 0

    def _record_freq(self, key: Tuple[int, int]) -> None:
        self.freqs[key] += 1
        self.total_count += 1
        if self.total_count >= self.sample_window:
            # Decay frequencies (halving)
            for k in list(self.freqs.keys()):
                self.freqs[k] //= 2
                if self.freqs[k] == 0:
                    del self.freqs[k]
            self.total_count = 0

    def access(self, layer: int, expert: int, step: int) -> bool:
        key = (layer, expert)
        self._record_freq(key)
        
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return True
            
        self.misses += 1
        if len(self.cache) < self.capacity:
            self.cache[key] = None
            return False
            
        # Admission check: compare frequency of candidate vs LRU victim
        victim_key, _ = next(iter(self.cache.items()))
        if self.freqs[key] >= self.freqs[victim_key]:
            self.cache.popitem(last=False)
            self.cache[key] = None
            self.evictions += 1
        else:
            # Candidate rejected, victim retained
            self.evictions += 1
            
        return False

    def reset(self) -> None:
        super().reset()
        self.cache.clear()
        self.freqs.clear()
        self.total_count = 0


class OracleCache(CachePolicy):
    """Belady's Optimal Replacement Algorithm (MIN / Oracle).
    
    Given the complete access history in advance, evicts the item whose next
    reference is furthest in the future. Represents the mathematical upper bound.
    """

    def __init__(self, capacity_slots: int, full_trace: Sequence[Tuple[int, int]]):
        super().__init__(capacity_slots)
        self.full_trace = list(full_trace)
        self.cache: Set[Tuple[int, int]] = set()
        
        # Build future access lookup: key -> list of future step indices
        self.future_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx, key in enumerate(self.full_trace):
            self.future_indices[key].append(idx)
        # Reverse so we can pop from the back in O(1)
        for key in self.future_indices:
            self.future_indices[key].reverse()

    def access(self, layer: int, expert: int, step: int) -> bool:
        key = (layer, expert)
        
        # Pop current occurrence
        if self.future_indices[key] and self.future_indices[key][-1] == step:
            self.future_indices[key].pop()
            
        if key in self.cache:
            self.hits += 1
            return True
            
        self.misses += 1
        if len(self.cache) < self.capacity:
            self.cache.add(key)
            return False
            
        # Evict item with furthest next access
        furthest_key = None
        furthest_step = -1
        
        for resident in self.cache:
            indices = self.future_indices[resident]
            if not indices:
                furthest_key = resident
                break
            next_step = indices[-1]
            if next_step > furthest_step:
                furthest_step = next_step
                furthest_key = resident
                
        self.cache.remove(furthest_key)
        self.cache.add(key)
        self.evictions += 1
        return False

    def reset(self) -> None:
        super().reset()
        self.cache.clear()
        self.future_indices = defaultdict(list)
        for idx, key in enumerate(self.full_trace):
            self.future_indices[key].append(idx)
        for key in self.future_indices:
            self.future_indices[key].reverse()
