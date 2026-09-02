"""Tests for ExpertMemoryCache (Tier 1 VRAM & Tier 2 RAM expert caching)."""

import torch

from pockettitan.domainslice.fast_cache import ExpertMemoryCache
from pockettitan.domainslice.types import ModelRevision
from tests.test_olmoe_paged import _StaticPageStore, _build_pages


def test_expert_memory_cache_tiers_and_eviction(tmp_path):
    cache = ExpertMemoryCache(vram_capacity=2, ram_capacity=3)
    assert cache.vram_capacity == 2
    assert cache.ram_capacity == 3

    model_revision = ModelRevision(repo_id="test/model", commit_sha="abcdef")

    # Build 4 dummy expert pages: expert 0, 1, 2, 3 in layer 0 (intermediate=4, hidden=8)
    torch.manual_seed(42)
    gate_weights = torch.randn(4, 4, 8, dtype=torch.bfloat16)
    up_weights = torch.randn(4, 4, 8, dtype=torch.bfloat16)
    down_weights = torch.randn(4, 8, 4, dtype=torch.bfloat16)

    pages = _build_pages(tmp_path, model_revision, gate_weights, up_weights, down_weights, layer=0)
    store = _StaticPageStore(pages)

    # 1. First fetch (expert 0) -> should be disk fault, then cached in RAM
    gate_up_0, down_0, tier_0 = cache.get_or_load(
        model_revision,
        0,
        0,
        store,
        target_device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert tier_0 == "disk"
    assert cache.disk_faults == 1
    assert gate_up_0.shape == (8, 8)  # 2 * intermediate_dim, hidden_dim
    assert down_0.shape == (8, 4)     # hidden_dim, intermediate_dim

    # 2. Second fetch (expert 0) -> should hit RAM cache on CPU
    gate_up_0b, down_0b, tier_0b = cache.get_or_load(
        model_revision,
        0,
        0,
        store,
        target_device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert tier_0b == "ram"
    assert cache.ram_hits == 1
    assert torch.equal(gate_up_0, gate_up_0b)

    # 3. Fetch expert 1 and expert 2 -> fill RAM cache (capacity=3)
    _, _, tier_1 = cache.get_or_load(
        model_revision, 0, 1, store, target_device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert tier_1 == "disk"

    _, _, tier_2 = cache.get_or_load(
        model_revision, 0, 2, store, target_device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert tier_2 == "disk"
    assert len(cache._ram_cache) == 3

    # 4. Fetch expert 3 -> evicts oldest from RAM cache ((0, 0) was oldest)
    _, _, tier_3 = cache.get_or_load(
        model_revision, 0, 3, store, target_device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert tier_3 == "disk"
    assert len(cache._ram_cache) == 3
    # Key (0, 0) should have been evicted
    assert (0, 0) not in cache._ram_cache
    assert (0, 1) in cache._ram_cache
    assert (0, 2) in cache._ram_cache
    assert (0, 3) in cache._ram_cache

    # 5. Clear cache
    cache.clear()
    assert len(cache._ram_cache) == 0
    assert len(cache._vram_cache) == 0


def test_expert_memory_cache_prefetch_batch(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ExpertMemoryCache(vram_capacity=4, ram_capacity=4)
    model_revision = ModelRevision(repo_id="test/prefetch", commit_sha="abcdef123")

    gate_weights = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    up_weights = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    down_weights = torch.randn(2, 8, 4, dtype=torch.bfloat16)

    pages = _build_pages(tmp_path, model_revision, gate_weights, up_weights, down_weights, layer=0)
    store = _StaticPageStore(pages)

    # Populate RAM cache
    cache.get_or_load(model_revision, 0, 0, store, target_device=torch.device("cpu"), dtype=torch.bfloat16)
    cache.get_or_load(model_revision, 0, 1, store, target_device=torch.device("cpu"), dtype=torch.bfloat16)
    assert (0, 0) in cache._ram_cache
    assert (0, 1) in cache._ram_cache

    # Prefetch batch to device
    cache.prefetch_batch(model_revision, 0, [0, 1], target_device=device, dtype=torch.bfloat16)
    if device.type == "cuda":
        assert (0, 0) in cache._vram_cache
        assert (0, 1) in cache._vram_cache


def test_expert_memory_cache_quantize_ram(tmp_path):
    cache = ExpertMemoryCache(vram_capacity=2, ram_capacity=4, quantize_ram=True, quant_bits=4)
    assert cache.quantize_ram is True
    assert cache._quantizer is not None

    model_revision = ModelRevision(repo_id="test/quant_model", commit_sha="123456")

    torch.manual_seed(42)
    gate_weights = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    up_weights = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    down_weights = torch.randn(2, 8, 4, dtype=torch.bfloat16)

    pages = _build_pages(tmp_path, model_revision, gate_weights, up_weights, down_weights, layer=0)
    store = _StaticPageStore(pages)

    # 1. Fetch expert 0 -> disk fault, quantizes to INT4, stores in RAM
    gate_up_0, down_0, tier_0 = cache.get_or_load(
        model_revision, 0, 0, store, target_device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert tier_0 == "disk"
    assert (0, 0) in cache._ram_cache
    assert cache._ram_cache[(0, 0)].is_quantized is True

    # 2. Fetch expert 0 again -> hits RAM cache, dequantizes cleanly
    gate_up_0b, down_0b, tier_0b = cache.get_or_load(
        model_revision, 0, 0, store, target_device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert tier_0b == "ram"
    assert gate_up_0b.shape == (8, 8)
    assert down_0b.shape == (8, 4)
    # Dequantized values correlate closely with original weights
    cos_sim = torch.nn.functional.cosine_similarity(gate_up_0.flatten(), gate_up_0b.flatten(), dim=0)
    assert cos_sim > 0.95


