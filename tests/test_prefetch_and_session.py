"""Tests for Speculative Lookahead Prefetcher (R7) and Session Adapter (R8)."""

from pathlib import Path
import tempfile
import time
import pytest
import torch

from pockettitan.package.format import ExpertLayout, ExpertRecordLayout
from pockettitan.runtime.expert.manager import ExpertManager
from pockettitan.runtime.prefetch import SpeculativePrefetcher
from pockettitan.runtime.session import SessionAdapter


@pytest.fixture
def dummy_expert_manager():
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
        num_layers=4,
        num_experts=8,
        layers=[0, 1, 2, 3],
        record=record_layout,
    )
    
    bank_bytes = bytearray(layout.total_bytes)
    
    # Initialize some dummy weights
    for l in range(4):
        for e in range(8):
            offset, _ = layout.byte_range(l, e)
            gu_w = torch.ones(2 * intermediate_size, in_features, dtype=torch.float16) * (e + 1)
            dn_w = torch.ones(in_features, intermediate_size, dtype=torch.float16) * (e + 1)
            
            gu_offset = offset + record_layout.projection("gate_up_proj").offset
            dn_offset = offset + record_layout.projection("down_proj").offset
            
            bank_bytes[gu_offset : gu_offset + gu_w.numel() * 2] = gu_w.numpy().tobytes()
            bank_bytes[dn_offset : dn_offset + dn_w.numel() * 2] = dn_w.numpy().tobytes()
            
    with tempfile.TemporaryDirectory() as tmpdir:
        bank_path = Path(tmpdir) / "bank.bin"
        bank_path.write_bytes(bank_bytes)
        
        manager = ExpertManager(
            bank_path=bank_path,
            layout=layout,
            ram_capacity_slots=32,
            vram_capacity_slots=4,
            device="cpu",
        )
        yield manager
        manager.close()


def test_speculative_prefetcher_prediction_and_async_io(dummy_expert_manager):
    """Verify router prediction and background async expert loading into RAM cache."""
    with SpeculativePrefetcher(dummy_expert_manager, num_prefetch_experts=4) as prefetcher:
        # Layer 0 hidden state
        h = torch.randn(1, 16)
        # Layer 1 router weight: [8 experts, 16 hidden_dim]
        next_router = torch.randn(8, 16)
        
        # 1. Predict
        predicted = prefetcher.predict_next_layer_experts(0, h, next_router)
        assert len(predicted) == 4
        
        # 2. Issue async prefetch for Layer 1
        prefetcher.issue_speculative_prefetch(1, predicted)
        
        # Allow background thread to process
        time.sleep(0.15)
        
        # 3. Synchronize
        prefetcher.await_partial(1, predicted[:2])
        assert prefetcher.prediction_accuracy > 0.0
        
        # Verify experts are now in RAM SLRU cache
        for p in predicted:
            assert dummy_expert_manager.ram_cache.contains((1, p))


def test_session_adapter_warmup_and_pinning(dummy_expert_manager):
    """Verify session adapter tracks warmup and pins the hot set at token 64."""
    adapter = SessionAdapter(
        expert_manager=dummy_expert_manager,
        warmup_token_threshold=10,  # Test with 10 tokens warmup
        vram_pin_count=2,
    )
    
    assert not adapter.is_session_pinned
    
    # Simulate 10 decoding steps activating expert 3 heavily
    for step in range(9):
        adapter.record_routing_step(layer=0, active_experts=[3, 1])
        adapter.step_token()
        assert not adapter.is_session_pinned
        
    # 10th token reaches warmup threshold
    adapter.record_routing_step(layer=0, active_experts=[3, 1])
    adapter.step_token()
    
    assert adapter.is_session_pinned
    assert (0, 3) in adapter.session_frequencies
    assert adapter.session_frequencies[(0, 3)] == 10
    
    # Reset
    adapter.reset_session()
    assert not adapter.is_session_pinned
    assert adapter.current_session_token == 0
