"""Tests for out-of-core ExpertManager, BoundedSLRUCache, and runtime engine (Phase R6)."""

import os
from pathlib import Path
import random
import tempfile
import pytest
import torch

from pockettitan.package.format import ExpertLayout, ExpertRecordLayout
from pockettitan.runtime.expert.cache import BoundedSLRUCache
from pockettitan.runtime.expert.manager import DecodedExpert, ExpertManager


def test_bounded_slru_cache_transitions_and_invariants():
    """Verify SLRU probationary/protected promotions, demotions, and capacity invariants."""
    # Capacity: 10 slots (2 probationary, 8 protected)
    cache = BoundedSLRUCache(capacity_slots=10, probationary_ratio=0.20)
    assert cache.probationary_capacity == 2
    assert cache.protected_capacity == 8

    # 1. Insert item A -> lands in probationary
    cache.put((0, 1), "expert_A")
    assert (0, 1) in cache.probationary
    assert (0, 1) not in cache.protected
    assert cache.total_resident == 1

    # 2. Access item A again -> promoted to protected
    val = cache.get((0, 1))
    assert val == "expert_A"
    assert (0, 1) in cache.protected
    assert (0, 1) not in cache.probationary

    # 3. Fill probationary partition with B and C
    cache.put((0, 2), "expert_B")
    cache.put((0, 3), "expert_C")
    assert cache.total_resident == 3  # A in protected; B, C in probationary

    # 4. Insert D into full probationary -> B (LRU) is evicted
    evicted = cache.put((0, 4), "expert_D")
    assert evicted == ((0, 2), "expert_B")
    assert not cache.contains((0, 2))
    assert cache.total_resident == 3

    # 5. Stress test invariant over 500 randomized operations
    for i in range(500):
        key = (random.randint(0, 5), random.randint(0, 20))
        if random.random() < 0.6:
            cache.get(key)
        else:
            cache.put(key, f"exp_{i}")
        assert cache.total_resident <= cache.capacity_slots


def test_decoded_expert_swiglu_forward():
    """Verify SwiGLU forward pass computation matches reference math."""
    in_features = 32
    intermediate_size = 64  # gate_up is [2 * intermediate_size, in_features] = [128, 32]
    
    torch.manual_seed(42)
    gate_up = torch.randn(2 * intermediate_size, in_features, dtype=torch.float16)
    down = torch.randn(in_features, intermediate_size, dtype=torch.float16)
    
    expert = DecodedExpert(gate_up_weight=gate_up, down_weight=down, device="cpu")
    
    x = torch.randn(1, in_features, dtype=torch.float16)
    out = expert.forward(x)
    
    # Reference calculation
    h = torch.matmul(x, gate_up.t())
    gate, up = h[..., :intermediate_size], h[..., intermediate_size:]
    act = torch.nn.functional.silu(gate.float()) * up.float()
    expected_out = torch.matmul(act.to(torch.float16), down.t())
    
    assert torch.allclose(out, expected_out, atol=1e-1, rtol=1e-2)


def test_expert_manager_bank_roundtrip():
    """Verify ExpertManager reads, caches, and executes from a synthetic binary expert bank."""
    in_features = 16
    intermediate_size = 32
    
    record_layout = ExpertRecordLayout.build(
        projections=[
            {
                "name": "gate_up_proj",
                "shape": [2 * intermediate_size, in_features],
                "bits": 16.0,
                "group_size": in_features,
            },
            {
                "name": "down_proj",
                "shape": [in_features, intermediate_size],
                "bits": 16.0,
                "group_size": in_features,
            },
        ],
        alignment=4096,
    )
    
    layout = ExpertLayout(
        num_layers=2,
        num_experts=4,
        layers=[0, 1],
        record=record_layout,
    )
    
    # Create synthetic bank.bin on disk
    bank_bytes = bytearray(layout.total_bytes)
    
    # Write expert (layer 0, expert 2)
    gu_w = torch.ones(2 * intermediate_size, in_features, dtype=torch.float16)
    dn_w = torch.ones(in_features, intermediate_size, dtype=torch.float16)
    
    offset, _ = layout.byte_range(0, 2)
    gu_offset = offset + record_layout.projection("gate_up_proj").offset
    dn_offset = offset + record_layout.projection("down_proj").offset
    
    bank_bytes[gu_offset : gu_offset + gu_w.numel() * 2] = gu_w.numpy().tobytes()
    bank_bytes[dn_offset : dn_offset + dn_w.numel() * 2] = dn_w.numpy().tobytes()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        bank_path = Path(tmpdir) / "bank.bin"
        bank_path.write_bytes(bank_bytes)
        
        with ExpertManager(bank_path=bank_path, layout=layout, ram_capacity_slots=4, device="cpu") as manager:
            # 1. Fetch expert
            exp = manager.fetch_expert(0, 2)
            assert exp.gate_up.shape == torch.Size([2 * intermediate_size, in_features])
            assert exp.down.shape == torch.Size([in_features, intermediate_size])
            assert (0, 2) in manager.ram_cache.probationary
            
            # 2. Forward layer experts
            x = torch.ones(1, in_features, dtype=torch.float16)
            out = manager.forward_layer_experts(
                layer=0,
                top_k_indices=[2],
                routing_weights=[1.0],
                hidden_states=x,
            )
            assert out.shape == torch.Size([1, in_features])
            assert not torch.allclose(out, torch.zeros_like(out))
