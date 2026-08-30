"""Two-population expert precision allocator (Phase R9)."""

from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class TwoPopulationPlan(BaseModel):
    """Execution plan for two-population expert precision assignment."""

    num_layers: int
    num_experts_per_layer: int
    total_experts: int
    hot_expert_count: int
    cold_expert_count: int
    hot_bits: float = 4.0
    cold_bits: float = 2.0
    hot_population: Set[Tuple[int, int]] = Field(default_factory=set)
    cold_population: Set[Tuple[int, int]] = Field(default_factory=set)
    estimated_bank_bytes: int
    uniform_4bit_bytes: int
    compression_ratio: float


class TwoPopulationAllocator:
    """Assigns 4-bit precision to the frequently-activated hot head and 2-bit to the cold tail.
    
    Architecture (Plan.md §5 / R9):
    - Hot head (e.g. top 20% activated experts): 4-bit precision (PT-Q4E) for maximal output quality.
    - Cold tail (remaining 80%): 2-bit precision (PT-Q2E) for maximal storage compaction.
    - Doubles RAM cache slot capacity from 2,880 to ~5,000+ slots in the same 7.0 GB budget.
    """

    def __init__(
        self,
        num_layers: int = 48,
        num_experts: int = 512,
        hot_head_ratio: float = 0.20,
        hot_bits: float = 4.0,
        cold_bits: float = 2.0,
        bytes_per_expert_16bit: int = 10_452_992,  # ~10 MB for unquantized expert
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.total_experts = num_layers * num_experts
        self.hot_ratio = hot_head_ratio
        self.hot_bits = hot_bits
        self.cold_bits = cold_bits
        self.bytes_16bit = bytes_per_expert_16bit

    def allocate(
        self,
        expert_frequencies: Dict[Tuple[int, int], int],
    ) -> TwoPopulationPlan:
        """Partition experts into hot and cold populations based on measured routing frequency."""
        # Ensure all experts exist in frequency map
        all_experts: List[Tuple[Tuple[int, int], int]] = []
        for l in range(self.num_layers):
            for e in range(self.num_experts):
                count = expert_frequencies.get((l, e), 0)
                all_experts.append(((l, e), count))

        # Sort by activation frequency descending
        all_experts.sort(key=lambda x: x[1], reverse=True)

        hot_count = int(self.total_experts * self.hot_ratio)
        cold_count = self.total_experts - hot_count

        hot_keys = {item[0] for item in all_experts[:hot_count]}
        cold_keys = {item[0] for item in all_experts[hot_count:]}

        # Calculate sizes
        hot_expert_bytes = int(self.bytes_16bit * (self.hot_bits / 16.0))
        cold_expert_bytes = int(self.bytes_16bit * (self.cold_bits / 16.0))

        total_bytes = (hot_count * hot_expert_bytes) + (cold_count * cold_expert_bytes)
        uniform_4bit = self.total_experts * hot_expert_bytes
        compression_ratio = uniform_4bit / total_bytes if total_bytes > 0 else 1.0

        return TwoPopulationPlan(
            num_layers=self.num_layers,
            num_experts_per_layer=self.num_experts,
            total_experts=self.total_experts,
            hot_expert_count=hot_count,
            cold_expert_count=cold_count,
            hot_bits=self.hot_bits,
            cold_bits=self.cold_bits,
            hot_population=hot_keys,
            cold_population=cold_keys,
            estimated_bank_bytes=total_bytes,
            uniform_4bit_bytes=uniform_4bit,
            compression_ratio=compression_ratio,
        )
