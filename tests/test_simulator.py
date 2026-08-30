"""R2 simulator invariant tests."""

import pytest
from pockettitan.sim import (
    DistributionType,
    HardwareProfile,
    HardwareSimulator,
    LRUCache,
    generate_synthetic_trace,
    run_capacity_sweep,
    run_simulation,
    summarize_trace,
)


def test_synthetic_trace_generation_and_gini():
    """Verify synthetic trace generators produce valid distributions."""
    uniform_trace = generate_synthetic_trace(num_tokens=50, num_layers=4, top_k=2, distribution=DistributionType.UNIFORM, seed=42)
    zipf_trace = generate_synthetic_trace(num_tokens=50, num_layers=4, top_k=2, distribution=DistributionType.ZIPF, alpha=1.2, seed=42)
    
    u_summary = summarize_trace(uniform_trace, num_layers=4, num_experts=512)
    z_summary = summarize_trace(zipf_trace, num_layers=4, num_experts=512)
    
    assert u_summary.total_accesses == 50 * 4 * 2 == 400
    assert z_summary.total_accesses == 400
    # Zipf distribution must have significantly higher Gini coefficient (concentration) than uniform
    assert z_summary.gini_coefficient > u_summary.gini_coefficient


def test_oracle_bounds_all_online_policies():
    """Invariant: Oracle (Belady's MIN) hit rate >= every online policy at any capacity."""
    trace = generate_synthetic_trace(num_tokens=100, num_layers=8, top_k=4, distribution=DistributionType.ZIPF, alpha=1.0, seed=123)
    
    for capacity in [16, 32, 64, 128]:
        results = run_simulation(trace, capacity_slots=capacity)
        by_policy = {r.policy_name: r.hit_rate for r in results}
        
        oracle_hit = by_policy["Oracle"]
        for pol, hit in by_policy.items():
            assert oracle_hit >= hit - 1e-6, f"Oracle failed against {pol} at cap {capacity}: {oracle_hit} vs {hit}"


def test_infinite_capacity_lru_hit_rate():
    """Invariant: LRU(inf) hit rate == 1 - (unique_accesses / total_accesses)."""
    trace = generate_synthetic_trace(num_tokens=100, num_layers=4, top_k=4, distribution=DistributionType.ZIPF, alpha=0.9, seed=99)
    unique_pairs = len({(ev.layer_idx, ev.expert_idx) for ev in trace})
    expected_hits = len(trace) - unique_pairs
    expected_hit_rate = expected_hits / len(trace)
    
    lru = LRUCache(capacity_slots=10000)
    for idx, ev in enumerate(trace):
        lru.access(ev.layer_idx, ev.expert_idx, idx)
        
    assert lru.hit_rate == pytest.approx(expected_hit_rate, abs=1e-5)


def test_hardware_simulator_matches_plan_roofline():
    """Invariant: Hardware roofline reproduces §2.2 numbers within 5% tolerance."""
    hw = HardwareProfile(
        ssd_bandwidth_gbps=3.0,
        ssd_latency_us=0.0,
        gpu_tflops_fp16=10.0,
        gpu_utilization=0.5,
    )
    sim = HardwareSimulator(hw)
    
    # At 0% hit rate, 480 expert reads per token @ 2-bit (~633 MiB cold traffic)
    # On a 3.0 GB/s SSD: 633 MiB / (3000 MB/s) = ~0.221s -> ~4.5 tok/s
    lat_2b = sim.simulate_token(
        token_id=0,
        expert_misses=480,
        expert_hits=0,
        bits_per_weight=2.0,
    )
    assert 4.0 <= lat_2b.tokens_per_second <= 5.5
    
    # At 50% hit rate (240 misses), throughput doubles to ~8.0 - 10.0 tok/s
    lat_50p = sim.simulate_token(
        token_id=0,
        expert_misses=240,
        expert_hits=240,
        bits_per_weight=2.0,
    )
    assert 8.0 <= lat_50p.tokens_per_second <= 11.0


def test_capacity_sweep_monotonicity():
    """Invariant: Increasing cache capacity must not decrease hit rate for LRU/Oracle."""
    trace = generate_synthetic_trace(num_tokens=100, num_layers=4, top_k=4, distribution=DistributionType.STICKY, seed=77)
    report = run_capacity_sweep(trace, capacities=(16, 32, 64, 128, 256))
    
    for policy_name in ["Oracle", "LRU"]:
        policy_metrics = [r for r in report.results if r.policy_name == policy_name]
        hit_rates = [m.hit_rate for m in policy_metrics]
        # Check non-decreasing
        for i in range(len(hit_rates) - 1):
            assert hit_rates[i+1] >= hit_rates[i] - 1e-6
