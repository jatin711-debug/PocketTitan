"""The Oracle Decision Gate Engine (Phase R4)."""

from enum import Enum
from typing import Optional, Sequence
from pydantic import BaseModel

from pockettitan.sim.hardware import HardwareProfile
from pockettitan.sim.report import SimulationReport, run_capacity_sweep
from pockettitan.sim.schema import RoutingEvent, summarize_trace


class GateDecision(str, Enum):
    PROCEED_Q4E = "PROCEED_Q4E"  # Oracle >= 50% -> Proceed with 4-bit experts & custom SLRU
    PROCEED_Q2E = "PROCEED_Q2E"  # Oracle 35-50% -> Proceed with 2-bit experts as default
    KILL_CUSTOM_CACHE = "KILL_CUSTOM_CACHE"  # Oracle < 35% -> Kill custom cache, switch to OS page cache


class GateReport(BaseModel):
    """Formal decision report for Phase R4."""

    decision: GateDecision
    oracle_hit_rate_at_budget: float
    os_page_cache_hit_rate: float
    custom_policy_advantage: float
    winning_online_policy: str
    target_budget_slots: int = 2880  # ~7.0 GB RAM for 4-bit experts
    gini_coefficient: float
    total_tokens_evaluated: int
    rationale: str
    full_report: SimulationReport


def evaluate_oracle_gate(
    events: Sequence[RoutingEvent],
    target_slots: int = 2880,
    bits_per_weight: float = 4.0,
    hardware: Optional[HardwareProfile] = None,
) -> GateReport:
    """Run Phase R4 Oracle Decision Gate evaluation over a routing trace.
    
    Evaluates Belady Oracle, OSPageCache, and custom online policies at the target slot capacity.
    """
    hw = hardware or HardwareProfile()
    summary = summarize_trace(events)
    gini = summary.gini_coefficient

    capacities = [target_slots]
    sim_report = run_capacity_sweep(
        events=events,
        capacities=capacities,
        bits_per_weight=bits_per_weight,
        hardware=hw,
    )

    metrics_by_policy = {m.policy_name: m for m in sim_report.results if m.capacity_slots == target_slots}
    
    oracle_metric = metrics_by_policy.get("Oracle")
    os_metric = metrics_by_policy.get("OSPageCache")

    if not oracle_metric or not os_metric:
        raise RuntimeError("Oracle or OSPageCache metric missing from simulation results")

    oracle_hit = oracle_metric.hit_rate
    os_hit = os_metric.hit_rate

    # Find winning custom online policy
    custom_policies = {
        name: m for name, m in metrics_by_policy.items() if name not in ("Oracle", "OSPageCache")
    }
    
    if custom_policies:
        best_custom_name = max(custom_policies, key=lambda k: custom_policies[k].hit_rate)
        best_custom_metric = custom_policies[best_custom_name]
        advantage = best_custom_metric.hit_rate - os_hit
    else:
        best_custom_name = "LRU"
        best_custom_metric = metrics_by_policy.get("LRU", os_metric)
        advantage = 0.0

    # Decision logic from Plan.md §5 / R4:
    if oracle_hit >= 0.50:
        decision = GateDecision.PROCEED_Q4E
        rationale = (
            f"Oracle hit rate ({oracle_hit*100:.1f}%) meets the >=50% threshold at {target_slots} slots "
            f"(7.0 GB RAM). PT-Q4E 4-bit expert configuration is validated."
        )
    elif oracle_hit >= 0.35:
        decision = GateDecision.PROCEED_Q2E
        rationale = (
            f"Oracle hit rate ({oracle_hit*100:.1f}%) falls between 35% and 50%. Proceed with PT-Q2E "
            f"(2-bit experts) as the default to expand slot capacity to 5,437 slots in 7.0 GB RAM."
        )
    else:
        decision = GateDecision.KILL_CUSTOM_CACHE
        rationale = (
            f"Oracle hit rate ({oracle_hit*100:.1f}%) is below 35%. No custom cache policy can overcome "
            f"the bandwidth penalty. Kill custom residency manager and switch to 2-bit experts with OS page cache."
        )

    return GateReport(
        decision=decision,
        oracle_hit_rate_at_budget=oracle_hit,
        os_page_cache_hit_rate=os_hit,
        custom_policy_advantage=advantage,
        winning_online_policy=best_custom_name,
        target_budget_slots=target_slots,
        gini_coefficient=gini,
        total_tokens_evaluated=summary.num_tokens,
        rationale=rationale,
        full_report=sim_report,
    )


def format_gate_report_markdown(report: GateReport) -> str:
    """Format Phase R4 gate output into the standard Plan.md §9 reporting block."""
    decision_text = "PROCEED" if report.decision != GateDecision.KILL_CUSTOM_CACHE else "KILL"
    return f"""# R4 — The Oracle Decision Gate Report

```text
PHASE:        R4 - Oracle Decision Gate
GATE METRIC:  Oracle Hit Rate @ {report.target_budget_slots} slots = {report.oracle_hit_rate_at_budget*100:.1f}% (threshold: >=50.0% for Q4E, >=35.0% for Q2E)
              OS Page Cache Hit Rate = {report.os_page_cache_hit_rate*100:.1f}%
              Winning Online Policy = {report.winning_online_policy} (advantage: +{report.custom_policy_advantage*100:.1f}%)
              Gini Skew = {report.gini_coefficient:.4f}
DECISION:     {decision_text} ({report.decision.value})
EVIDENCE:     {report.total_tokens_evaluated} tokens evaluated across {report.target_budget_slots} slots
NEXT:         {"Phase R6 (Expert SLRU Paging & Out-of-Core Runtime)" if report.decision != GateDecision.KILL_CUSTOM_CACHE else "Pivot to OS Page Cache + 2-bit Experts"}
```

## Gate Rationale
{report.rationale}
"""
