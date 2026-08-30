"""PocketTitan Memory and Cache Simulator (R2)."""

from pockettitan.sim.cache import (
    CachePolicy,
    LRUCache,
    OracleCache,
    OSPageCache,
    SLRUCache,
    TinyLFUCache,
)
from pockettitan.sim.hardware import HardwareProfile, HardwareSimulator, LatencyBreakdown
from pockettitan.sim.oracle_gate import (
    GateDecision,
    GateReport,
    evaluate_oracle_gate,
    format_gate_report_markdown,
)
from pockettitan.sim.report import (
    PolicyMetric,
    SimulationReport,
    render_simulation_report,
    run_capacity_sweep,
    run_simulation,
)
from pockettitan.sim.schema import (
    DistributionType,
    RoutingEvent,
    TraceSummary,
    compute_gini,
    generate_synthetic_trace,
    summarize_trace,
)

__all__ = [
    # schema
    "RoutingEvent",
    "TraceSummary",
    "DistributionType",
    "compute_gini",
    "summarize_trace",
    "generate_synthetic_trace",
    # cache
    "CachePolicy",
    "LRUCache",
    "OSPageCache",
    "SLRUCache",
    "TinyLFUCache",
    "OracleCache",
    # hardware
    "HardwareProfile",
    "HardwareSimulator",
    "LatencyBreakdown",
    # report
    "PolicyMetric",
    "SimulationReport",
    "run_simulation",
    "run_capacity_sweep",
    "render_simulation_report",
]
