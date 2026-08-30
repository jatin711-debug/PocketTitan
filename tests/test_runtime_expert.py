"""Tests for out-of-core ExpertManager, BoundedSLRUCache, and runtime engine (Phase R6)."""

from pathlib import Path
import random
import tempfile
import pytest
import safetensors.torch
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


# --------------------------------------------------------------------------- #
# The quantized path, end to end through the real writer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bits,symmetric", [(4, False), (4, True), (2, False)])
def test_expert_manager_decodes_a_package_the_writer_actually_wrote(
    dummy_moe_model, tmp_path, bits, symmetric
):
    """Fetch experts from a real ``experts/bank.bin`` and compare against the source.

    Every previous expert test used ``bits=16``, so the quantized branch of
    ``decode_expert_payload`` was never executed. It hardcoded 4-bit nibbles with
    a ``-8`` offset and never read the ``ZEROS`` section — yet
    ``PrecisionEntry.symmetric`` defaults to ``False``, so every packaged expert
    carries a zero-point that was being discarded.
    """
    from pockettitan.audit import PrecisionMap, scan_checkpoint
    from pockettitan.config import MemoryBudgetConfig, QuantMethod
    from pockettitan.package import PackageWriter, plan_package
    from pockettitan.streaming.reader import LocalTensorReader

    precision = PrecisionMap.uniform(bits, 32, f"expert-{bits}b")
    for entry in list(precision.entries.values()) + [precision.default]:
        entry.symmetric = symmetric

    scan = scan_checkpoint(str(dummy_moe_model))
    plan = plan_package(scan, precision_map=precision)
    output = tmp_path / "pkg.ptitan"
    PackageWriter(
        plan,
        output,
        LocalTensorReader(dummy_moe_model),
        budget=MemoryBudgetConfig(max_vram_mb=512.0),
        method=QuantMethod.RTN,
        device="cpu",
    ).build()

    layout = plan.manifest.expert_layout
    assert layout is not None, "fixture must produce routed experts"

    source = {}
    for path in sorted(dummy_moe_model.glob("*.safetensors")):
        source.update(safetensors.torch.load_file(str(path)))

    with ExpertManager(
        bank_path=output / "experts" / "bank.bin",
        layout=layout,
        ram_capacity_slots=4,
        device="cpu",
    ) as manager:
        for layer in layout.layers:
            for expert in (0, layout.num_experts // 2, layout.num_experts - 1):
                decoded = manager.fetch_expert(layer, expert)

                gate_up = source[f"model.layers.{layer}.mlp.experts.gate_up_proj"][expert]
                down = source[f"model.layers.{layer}.mlp.experts.down_proj"][expert]

                assert decoded.gate_up.shape == gate_up.shape
                assert decoded.down.shape == down.shape
                for got, want, name in (
                    (decoded.gate_up, gate_up, "gate_up"),
                    (decoded.down, down, "down"),
                ):
                    got_f, want_f = got.float().flatten(), want.float().flatten()
                    correlation = torch.corrcoef(torch.stack([got_f, want_f]))[0, 1]
                    assert correlation > 0.9, (
                        f"L{layer}E{expert} {name} decoded at correlation "
                        f"{correlation:.3f} — the record is being read wrong, not "
                        f"merely quantized coarsely"
                    )


def test_distinct_experts_decode_to_distinct_weights(dummy_moe_model, tmp_path):
    """Guards the failure where every record resolves to one offset."""
    from pockettitan.audit import PrecisionMap, scan_checkpoint
    from pockettitan.config import MemoryBudgetConfig, QuantMethod
    from pockettitan.package import PackageWriter, plan_package
    from pockettitan.streaming.reader import LocalTensorReader

    scan = scan_checkpoint(str(dummy_moe_model))
    plan = plan_package(scan, precision_map=PrecisionMap.uniform(4, 32, "int4"))
    output = tmp_path / "pkg.ptitan"
    PackageWriter(
        plan,
        output,
        LocalTensorReader(dummy_moe_model),
        budget=MemoryBudgetConfig(max_vram_mb=512.0),
        method=QuantMethod.RTN,
        device="cpu",
    ).build()

    layout = plan.manifest.expert_layout
    with ExpertManager(
        bank_path=output / "experts" / "bank.bin",
        layout=layout,
        ram_capacity_slots=layout.num_experts * 2,
        device="cpu",
    ) as manager:
        signatures = {
            expert: manager.fetch_expert(0, expert).gate_up.float().sum().item()
            for expert in range(layout.num_experts)
        }
    assert len(set(signatures.values())) == layout.num_experts, signatures
