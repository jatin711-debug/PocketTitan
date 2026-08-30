"""Trace data models and synthetic trace generators for the memory simulator (R2)."""

import math
import random
from enum import Enum
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from pydantic import BaseModel, Field


class DistributionType(str, Enum):
    UNIFORM = "uniform"
    ZIPF = "zipf"
    STICKY = "sticky"


class RoutingEvent(BaseModel):
    """A single expert routing activation during a forward pass."""
    
    token_id: int = Field(alias="tok", default=0)
    layer_idx: int = Field(alias="layer", default=0)
    slot_idx: int = Field(alias="slot", default=0)  # 0 .. top_k-1
    expert_idx: int = Field(alias="expert", default=0)  # 0 .. num_experts-1
    weight: float = 1.0
    router_entropy: float = 0.0
    prompt_id: int = 0
    phase: str = "decode"  # "prefill" or "decode"

    model_config = {"populate_by_name": True}

    @property
    def tok(self) -> int:
        return self.token_id

    @property
    def layer(self) -> int:
        return self.layer_idx

    @property
    def slot(self) -> int:
        return self.slot_idx

    @property
    def expert(self) -> int:
        return self.expert_idx


class TraceSummary(BaseModel):
    """Statistical summary of a captured or generated routing trace."""
    
    num_tokens: int
    num_layers: int
    num_experts: int
    top_k: int
    total_accesses: int
    unique_experts_accessed: int
    expert_frequency: Dict[int, int] = Field(default_factory=dict)
    layer_expert_frequency: Dict[int, Dict[int, int]] = Field(default_factory=dict)
    gini_coefficient: float = 0.0


def compute_gini(frequencies: Sequence[int]) -> float:
    """Calculate Gini coefficient of expert usage frequencies (0 = equal, 1 = concentrated)."""
    if not frequencies or sum(frequencies) == 0:
        return 0.0
    sorted_freqs = sorted(frequencies)
    n = len(sorted_freqs)
    index = range(1, n + 1)
    return float(sum((2 * i - n - 1) * freq for i, freq in zip(index, sorted_freqs)) / (n * sum(sorted_freqs)))


def summarize_trace(events: Sequence[RoutingEvent], num_layers: int = 48, num_experts: int = 512) -> TraceSummary:
    """Compute aggregate access statistics and Gini distribution from an event stream."""
    expert_freq: Dict[int, int] = {e: 0 for e in range(num_experts)}
    layer_freq: Dict[int, Dict[int, int]] = {l: {e: 0 for e in range(num_experts)} for l in range(num_layers)}
    
    tokens = set()
    top_k_set = set()
    
    for ev in events:
        tokens.add((ev.prompt_id, ev.token_id))
        top_k_set.add(ev.slot_idx)
        expert_freq[ev.expert_idx] = expert_freq.get(ev.expert_idx, 0) + 1
        layer_freq[ev.layer_idx][ev.expert_idx] = layer_freq[ev.layer_idx].get(ev.expert_idx, 0) + 1
        
    num_tokens = len(tokens)
    top_k = len(top_k_set) if top_k_set else 10
    total_accesses = len(events)
    unique_accessed = sum(1 for count in expert_freq.values() if count > 0)
    gini = compute_gini(list(expert_freq.values()))
    
    return TraceSummary(
        num_tokens=num_tokens,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        total_accesses=total_accesses,
        unique_experts_accessed=unique_accessed,
        expert_frequency=expert_freq,
        layer_expert_frequency=layer_freq,
        gini_coefficient=gini,
    )


def generate_synthetic_trace(
    num_tokens: int = 1000,
    num_layers: int = 48,
    num_experts: int = 512,
    top_k: int = 10,
    distribution: DistributionType = DistributionType.ZIPF,
    alpha: float = 1.0,
    sticky_prob: float = 0.7,
    seed: int = 42,
) -> List[RoutingEvent]:
    """Generate synthetic routing event stream across multiple token steps.
    
    Distributions:
    - UNIFORM: Equal probability of routing to any expert.
    - ZIPF: Power-law probability P(rank k) ~ 1 / k^alpha (standard MoE empirical model).
    - STICKY: Temporal Markov locality where consecutive tokens retain previous active experts with sticky_prob.
    """
    rng = random.Random(seed)
    events: List[RoutingEvent] = []
    
    # Precompute probabilities
    if distribution == DistributionType.UNIFORM:
        weights = [1.0] * num_experts
    elif distribution == DistributionType.ZIPF:
        weights = [1.0 / math.pow(i + 1, alpha) for i in range(num_experts)]
    else:  # Sticky fallback base weights
        weights = [1.0 / math.pow(i + 1, 0.8) for i in range(num_experts)]
        
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    
    # State for sticky-session Markov generator
    last_active_per_layer: Dict[int, List[int]] = {l: [] for l in range(num_layers)}
    
    for tok in range(num_tokens):
        for l in range(num_layers):
            if distribution == DistributionType.STICKY and last_active_per_layer[l] and rng.random() < sticky_prob:
                # Keep some previous experts, sample remaining
                prev = last_active_per_layer[l]
                num_keep = rng.randint(max(1, top_k // 2), top_k)
                chosen = set(rng.sample(prev, min(len(prev), num_keep)))
                while len(chosen) < top_k:
                    exp = rng.choices(range(num_experts), weights=probs, k=1)[0]
                    chosen.add(exp)
                selected = list(chosen)
            else:
                # Sample without replacement using weighted probabilities
                selected = []
                pool_weights = list(probs)
                for _ in range(top_k):
                    w_sum = sum(pool_weights)
                    if w_sum <= 0:
                        break
                    norm_w = [w / w_sum for w in pool_weights]
                    pick = rng.choices(range(num_experts), weights=norm_w, k=1)[0]
                    selected.append(pick)
                    pool_weights[pick] = 0.0
                    
            last_active_per_layer[l] = selected
            
            for slot, exp in enumerate(selected):
                events.append(
                    RoutingEvent(
                        token_id=tok,
                        layer_idx=l,
                        slot_idx=slot,
                        expert_idx=exp,
                        weight=1.0 / top_k,
                        router_entropy=0.0,
                        prompt_id=0,
                        phase="decode",
                    )
                )
                
    return events
