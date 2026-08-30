"""Tests for MoE routing profiler, prompt suite, trace engine, and patch gate (R3)."""

from pathlib import Path
import tempfile
import pytest

from pockettitan.profiler.prompts import TaskType, generate_benchmark_suite
from pockettitan.profiler.trace import (
    TraceReader,
    TraceWriter,
    analyze_routing_trace,
)
from pockettitan.sim.cache import OracleCache
from pockettitan.sim.report import run_capacity_sweep
from pockettitan.sim.schema import DistributionType, generate_synthetic_trace


def test_patch_file_exists_and_is_valid():
    """Verify that the upstream llama.cpp patch is present and well-formed."""
    patch_path = Path("patches/llama-cpp-routing-trace.patch")
    assert patch_path.exists(), "Patch file missing at patches/llama-cpp-routing-trace.patch"
    content = patch_path.read_text(encoding="utf-8")
    assert "LLAMA_ROUTING_TRACE_PATH" in content
    assert "log_routing_event" in content
    assert "diff --git a/src/llama-context.cpp" in content


def test_benchmark_prompts_cover_all_5_tasks():
    """Verify prompt suite generator covers the 5 canonical evaluation tasks."""
    suite = generate_benchmark_suite(num_samples_per_task=3)
    assert len(suite) == 15  # 5 tasks * 3 samples
    
    categories = {p.task_type for p in suite}
    assert categories == {
        TaskType.CHAT,
        TaskType.CODE,
        TaskType.MATH,
        TaskType.RETRIEVAL,
        TaskType.TOOL_CALLING,
    }


def test_trace_writer_and_reader_roundtrip_gz():
    """Verify streaming compressed .jsonl.gz trace write and read."""
    trace_events = generate_synthetic_trace(
        num_tokens=10,
        num_layers=4,
        num_experts=64,
        top_k=4,
        distribution=DistributionType.ZIPF,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = Path(tmpdir) / "trace.jsonl.gz"
        
        with TraceWriter(trace_path) as writer:
            writer.write_events(trace_events)
            
        assert trace_path.exists()
        
        reader = TraceReader(trace_path)
        read_events = reader.read_all_events()
        assert len(read_events) == len(trace_events)
        assert read_events[0].tok == trace_events[0].tok
        assert read_events[0].layer == trace_events[0].layer
        assert read_events[0].expert == trace_events[0].expert


def test_trace_analysis_metrics():
    """Verify Gini skew, per-layer entropy, and cross-layer overlap metrics."""
    trace_events = generate_synthetic_trace(
        num_tokens=20,
        num_layers=8,
        num_experts=128,
        top_k=8,
        distribution=DistributionType.ZIPF,
    )
    
    metrics = analyze_routing_trace(trace_events)
    assert metrics.total_tokens == 20
    assert metrics.total_events == 20 * 8 * 8
    assert 0.0 <= metrics.gini_coefficient <= 1.0
    assert 0.0 <= metrics.cross_layer_overlap_mean <= 1.0
    assert len(metrics.layer_entropy) == 8


def test_replay_trace_through_simulator_sweep():
    """Verify ingested trace can be fed directly into the capacity sweep engine."""
    trace_events = generate_synthetic_trace(
        num_tokens=25,
        num_layers=12,
        num_experts=128,
        top_k=6,
        distribution=DistributionType.ZIPF,
    )
    
    report = run_capacity_sweep(
        events=trace_events,
        capacities=[64, 128, 256],
        bits_per_weight=4.0,
    )
    
    assert len(report.results) > 0
    # Oracle hit rate monotonic with capacity
    oracle_runs = [e for e in report.results if e.policy_name == "Oracle"]
    assert len(oracle_runs) == 3
    assert oracle_runs[0].hit_rate <= oracle_runs[1].hit_rate <= oracle_runs[2].hit_rate
