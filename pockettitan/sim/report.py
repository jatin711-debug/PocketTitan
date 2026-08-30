"""Simulation execution, policy comparison sweeps, and Rich reporting (R2)."""

from typing import Dict, List, Optional, Sequence
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pockettitan.sim.cache import (
    CachePolicy,
    LRUCache,
    OracleCache,
    OSPageCache,
    SLRUCache,
    TinyLFUCache,
)
from pockettitan.sim.hardware import HardwareProfile, HardwareSimulator, LatencyBreakdown
from pockettitan.sim.schema import RoutingEvent, TraceSummary, summarize_trace


class PolicyMetric(BaseModel):
    """Aggregate simulation results for a specific cache policy and configuration."""

    policy_name: str
    capacity_slots: int
    bits_per_weight: float
    total_accesses: int
    hits: int
    misses: int
    evictions: int
    hit_rate: float
    avg_ssd_mb_per_token: float
    avg_stall_time_ms: float
    avg_tokens_per_second: float


class SimulationReport(BaseModel):
    """Complete multi-policy sweep report over a routing trace."""

    model_id: str = "Qwen/Qwen3.8-Flash-Next"
    trace_summary: TraceSummary
    hardware_name: str
    ssd_bandwidth_gbps: float
    results: List[PolicyMetric] = Field(default_factory=list)

    def best_policy(self, capacity_slots: int) -> Optional[PolicyMetric]:
        matching = [r for r in self.results if r.capacity_slots == capacity_slots]
        if not matching:
            return None
        return max(matching, key=lambda m: m.hit_rate)


def run_simulation(
    events: Sequence[RoutingEvent],
    capacity_slots: int = 2880,
    bits_per_weight: float = 4.0,
    hardware: Optional[HardwareProfile] = None,
    policies: Optional[Sequence[str]] = None,
) -> List[PolicyMetric]:
    """Run trace replay across multiple caching policies under the specified hardware model."""
    hw_sim = HardwareSimulator(hardware)
    chosen_policies = list(policies or ["Oracle", "OSPageCache", "SLRU", "TinyLFU", "LRU"])
    
    trace_pairs = [(ev.layer_idx, ev.expert_idx) for ev in events]
    
    metrics: List[PolicyMetric] = []
    
    for pol_name in chosen_policies:
        # Instantiate policy
        if pol_name == "Oracle":
            policy: CachePolicy = OracleCache(capacity_slots, trace_pairs)
        elif pol_name == "SLRU":
            policy = SLRUCache(capacity_slots)
        elif pol_name == "TinyLFU":
            policy = TinyLFUCache(capacity_slots)
        elif pol_name == "OSPageCache":
            policy = OSPageCache(capacity_slots)
        else:
            policy = LRUCache(capacity_slots)
            
        # Group events by token_id to simulate token-by-token latencies
        events_by_token: Dict[int, List[RoutingEvent]] = {}
        for ev in events:
            events_by_token.setdefault(ev.token_id, []).append(ev)
            
        step = 0
        token_latencies: List[LatencyBreakdown] = []
        
        for tok_id, tok_events in sorted(events_by_token.items()):
            token_hits = 0
            token_misses = 0
            
            for ev in tok_events:
                is_hit = policy.access(ev.layer_idx, ev.expert_idx, step)
                if is_hit:
                    token_hits += 1
                else:
                    token_misses += 1
                step += 1
                
            lat = hw_sim.simulate_token(
                token_id=tok_id,
                expert_misses=token_misses,
                expert_hits=token_hits,
                bits_per_weight=bits_per_weight,
            )
            token_latencies.append(lat)
            
        # Aggregate statistics
        total_tokens = max(1, len(token_latencies))
        avg_ssd_mb = sum(lat.ssd_bytes_read for lat in token_latencies) / (total_tokens * 1024 * 1024)
        avg_stall = sum(lat.stall_time_ms for lat in token_latencies) / total_tokens
        avg_tok_s = sum(lat.tokens_per_second for lat in token_latencies) / total_tokens
        
        metrics.append(
            PolicyMetric(
                policy_name=pol_name,
                capacity_slots=capacity_slots,
                bits_per_weight=bits_per_weight,
                total_accesses=policy.total_accesses,
                hits=policy.hits,
                misses=policy.misses,
                evictions=policy.evictions,
                hit_rate=policy.hit_rate,
                avg_ssd_mb_per_token=avg_ssd_mb,
                avg_stall_time_ms=avg_stall,
                avg_tokens_per_second=avg_tok_s,
            )
        )
        
    return metrics


def run_capacity_sweep(
    events: Sequence[RoutingEvent],
    capacities: Sequence[int] = (512, 1024, 2048, 2880, 4096, 5437),
    bits_per_weight: float = 4.0,
    hardware: Optional[HardwareProfile] = None,
    policies: Optional[Sequence[str]] = None,
    model_id: str = "Qwen/Qwen3.8-Flash-Next",
) -> SimulationReport:
    """Run a multi-capacity sweep and generate an aggregate SimulationReport."""
    hw = hardware or HardwareProfile()
    summary = summarize_trace(events)
    
    all_metrics: List[PolicyMetric] = []
    for cap in capacities:
        cap_results = run_simulation(
            events,
            capacity_slots=cap,
            bits_per_weight=bits_per_weight,
            hardware=hw,
            policies=policies,
        )
        all_metrics.extend(cap_results)
        
    return SimulationReport(
        model_id=model_id,
        trace_summary=summary,
        hardware_name=hw.name,
        ssd_bandwidth_gbps=hw.ssd_bandwidth_gbps,
        results=all_metrics,
    )


def render_simulation_report(console: Console, report: SimulationReport) -> None:
    """Render Rich summary tables and oracle evaluation."""
    # 1. Masthead
    summary = report.trace_summary
    masthead = (
        f"[bold]Trace Tokens:[/bold] {summary.num_tokens:,} | "
        f"[bold]Total Expert Accesses:[/bold] {summary.total_accesses:,}\n"
        f"[bold]Active Experts / Token:[/bold] {summary.top_k} × {summary.num_layers} layers = {summary.top_k * summary.num_layers}\n"
        f"[bold]Expert Gini Skew:[/bold] {summary.gini_coefficient:.3f} | "
        f"[bold]Hardware Profile:[/bold] {report.hardware_name} ({report.ssd_bandwidth_gbps} GB/s SSD)"
    )
    console.print(Panel(masthead, title="PocketTitan MoE Out-of-Core Memory Simulator (R2)", border_style="cyan"))

    # 2. Main Comparison Table
    table = Table(title="Cache Policy & Throughput Evaluation", header_style="bold magenta")
    table.add_column("Policy", style="bold")
    table.add_column("Slots", justify="right")
    table.add_column("Cache Share", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("SSD Traffic / tok", justify="right")
    table.add_column("Stall (ms)", justify="right")
    table.add_column("Modeled tok/s", justify="right", style="green bold")

    for m in report.results:
        slot_share = (m.capacity_slots / 24576.0) * 100.0
        hit_str = f"{m.hit_rate * 100.0:.1f}%"
        traffic_str = f"{m.avg_ssd_mb_per_token:.1f} MB"
        stall_str = f"{m.avg_stall_time_ms:.1f} ms"
        toks_str = f"{m.avg_tokens_per_second:.2f} tok/s"
        
        is_best = m.policy_name == "Oracle"
        style = "bold cyan" if is_best else None
        
        table.add_row(
            m.policy_name,
            f"{m.capacity_slots:,}",
            f"{slot_share:.1f}%",
            hit_str,
            traffic_str,
            stall_str,
            toks_str,
            style=style,
        )

    console.print(table)
