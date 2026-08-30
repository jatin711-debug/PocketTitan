"""Trace reader, writer, and statistical profiling engine for MoE routing (R3)."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
from pydantic import BaseModel, Field

from pockettitan.sim.schema import RoutingEvent, compute_gini


class TraceMetrics(BaseModel):
    """Deep statistical breakdown of an MoE routing trace."""

    total_tokens: int = 0
    total_events: int = 0
    unique_experts_accessed: int = 0
    gini_coefficient: float = 0.0
    mean_router_entropy: float = 0.0
    layer_entropy: Dict[int, float] = Field(default_factory=dict)
    cross_layer_overlap_mean: float = Field(
        default=0.0,
        description="Average top-k expert overlap between adjacent layers (L and L+1)",
    )
    layer_top_experts: Dict[int, List[int]] = Field(default_factory=dict)


class TraceWriter:
    """Streams routing events directly to disk with transparent compression."""

    def __init__(self, output_path: Union[str, Path]):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.is_gz = self.path.name.endswith(".gz")
        
        if self.is_gz:
            self._file = gzip.open(self.path, "wt", encoding="utf-8")
        else:
            self._file = open(self.path, "w", encoding="utf-8")

    def write_event(self, event: RoutingEvent) -> None:
        self._file.write(event.model_dump_json() + "\n")

    def write_events(self, events: Sequence[RoutingEvent]) -> None:
        for ev in events:
            self.write_event(ev)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class TraceReader:
    """Reads routing events from raw or compressed JSONL files."""

    def __init__(self, input_path: Union[str, Path]):
        self.path = Path(input_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.path}")
        self.is_gz = self.path.name.endswith(".gz")

    def stream_events(self) -> Iterator[RoutingEvent]:
        """Iterate over events one by one to avoid large memory footprints."""
        if self.is_gz:
            with gzip.open(self.path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield RoutingEvent.model_validate_json(line)
        else:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield RoutingEvent.model_validate_json(line)

    def read_all_events(self) -> List[RoutingEvent]:
        return list(self.stream_events())


def analyze_routing_trace(events: Sequence[RoutingEvent]) -> TraceMetrics:
    """Compute comprehensive statistical metrics from a sequence of routing events."""
    if not events:
        return TraceMetrics()

    expert_counts: Counter[Tuple[int, int]] = Counter()
    layer_entropies: Dict[int, List[float]] = defaultdict(list)
    tokens_seen: Set[int] = set()
    
    # Track top-k per (token, layer) for cross-layer overlap analysis
    token_layer_experts: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    max_layer = 0

    for ev in events:
        tokens_seen.add(ev.tok)
        expert_counts[(ev.layer, ev.expert)] += 1
        layer_entropies[ev.layer].append(ev.router_entropy)
        token_layer_experts[(ev.tok, ev.layer)].add(ev.expert)
        if ev.layer > max_layer:
            max_layer = ev.layer

    # 1. Gini skew
    gini = compute_gini(list(expert_counts.values()))

    # 2. Per-layer entropy
    mean_layer_entropy: Dict[int, float] = {}
    all_entropies: List[float] = []
    for l_idx, ents in layer_entropies.items():
        avg = sum(ents) / len(ents) if ents else 0.0
        mean_layer_entropy[l_idx] = avg
        all_entropies.extend(ents)

    overall_mean_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else 0.0

    # 3. Cross-layer top-k overlap: Overlap(L, L+1) = |E_L ∩ E_{L+1}| / k
    overlaps: List[float] = []
    for tok in tokens_seen:
        for l in range(max_layer):
            set_l = token_layer_experts.get((tok, l), set())
            set_next = token_layer_experts.get((tok, l + 1), set())
            if set_l and set_next:
                intersect = len(set_l.intersection(set_next))
                k = len(set_l)
                overlaps.append(intersect / float(k))

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    # 4. Top 3 most frequent experts per layer
    top_experts: Dict[int, List[int]] = {}
    for l in range(max_layer + 1):
        l_counts = {exp: cnt for (layer, exp), cnt in expert_counts.items() if layer == l}
        sorted_exp = sorted(l_counts.keys(), key=lambda e: l_counts[e], reverse=True)[:3]
        top_experts[l] = sorted_exp

    return TraceMetrics(
        total_tokens=len(tokens_seen),
        total_events=len(events),
        unique_experts_accessed=len(expert_counts),
        gini_coefficient=gini,
        mean_router_entropy=overall_mean_entropy,
        layer_entropy=mean_layer_entropy,
        cross_layer_overlap_mean=mean_overlap,
        layer_top_experts=top_experts,
    )
