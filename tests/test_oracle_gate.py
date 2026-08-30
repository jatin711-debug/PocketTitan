"""Tests for Phase R4 The Oracle Decision Gate."""

import pytest
from pockettitan.sim.oracle_gate import (
    GateDecision,
    GateReport,
    evaluate_oracle_gate,
    format_gate_report_markdown,
)
from pockettitan.sim.schema import DistributionType, generate_synthetic_trace


def test_oracle_gate_proceed_q4e_on_skewed_trace():
    """Verify that a standard Zipf skewed trace passes the >=50% Oracle threshold at 2,880 slots."""
    events = generate_synthetic_trace(
        num_tokens=50,
        num_layers=48,
        num_experts=512,
        top_k=10,
        distribution=DistributionType.ZIPF,
        alpha=1.0,
    )
    
    report = evaluate_oracle_gate(events, target_slots=2880, bits_per_weight=4.0)
    assert report.decision == GateDecision.PROCEED_Q4E
    assert report.oracle_hit_rate_at_budget >= 0.50
    assert report.winning_online_policy in ("TinyLFU", "LRU", "SLRU")


def test_oracle_gate_kill_on_uniform_trace_with_tiny_capacity():
    """Verify that a uniform random trace with tiny cache triggers KILL_CUSTOM_CACHE."""
    events = generate_synthetic_trace(
        num_tokens=50,
        num_layers=48,
        num_experts=512,
        top_k=10,
        distribution=DistributionType.UNIFORM,
    )
    
    # 64 slots is ~0.2% cache capacity
    report = evaluate_oracle_gate(events, target_slots=64, bits_per_weight=4.0)
    assert report.decision == GateDecision.KILL_CUSTOM_CACHE
    assert report.oracle_hit_rate_at_budget < 0.35


def test_format_gate_report_markdown():
    """Verify markdown output adheres to Plan.md §9 format."""
    events = generate_synthetic_trace(
        num_tokens=20,
        num_layers=12,
        num_experts=128,
        top_k=6,
        distribution=DistributionType.ZIPF,
    )
    
    report = evaluate_oracle_gate(events, target_slots=256, bits_per_weight=4.0)
    md = format_gate_report_markdown(report)
    
    assert "PHASE:        R4 - Oracle Decision Gate" in md
    assert "GATE METRIC:" in md
    assert "DECISION:" in md
    assert "EVIDENCE:" in md
