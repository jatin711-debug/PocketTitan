"""Numerical parity for OLMoE expert math executed from page files."""

import hashlib
from types import SimpleNamespace

import torch

from pockettitan.domainslice import (
    ModelRevision,
    PageDescriptor,
    PageHandle,
    StoreStats,
    WeightID,
    WeightPageID,
)
from pockettitan.domainslice.hypothesis import (
    run_olmoe_full_model_hypothesis,
    run_olmoe_full_model_reference,
    run_olmoe_two_position_generation,
)
from pockettitan.package import ExpertRecordLayout, SourceSlice
from pockettitan.runtime.hf.olmoe_paged import (
    ExpertPageTensors,
    PagedOlmoeExperts,
    replace_olmoe_sparse_moe,
)
from pockettitan.runtime.hf.olmoe_layer import build_paged_olmoe_decoder_layer
from pockettitan.runtime.hf.olmoe_model import PagedOlmoeOneTokenRunner


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint16).numpy().tobytes()


class _StaticPageStore:
    def __init__(self, pages):
        self.pages = pages
        self.seen = set()
        self.releases = 0

    def resolve(self, page_id):
        return self.pages[page_id.cache_key][0]

    def materialize(self, page_id, **_kwargs):
        descriptor, template = self.pages[page_id.cache_key]
        hit = page_id.cache_key in self.seen
        self.seen.add(page_id.cache_key)
        return template.model_copy(
            update={
                "cache_hit": hit,
                "bytes_fetched": 0 if hit else descriptor.expected_bytes,
            }
        )

    def prefetch(self, page_ids):
        return []

    def release(self, _handle):
        self.releases += 1

    def stats(self):
        return StoreStats(cached_pages=len(self.seen))


def _build_pages(tmp_path, revision, gate_weights, up_weights, down_weights, layer=0):
    pages = {}
    hidden = gate_weights.shape[-1]
    intermediate = gate_weights.shape[-2]
    for expert in range(gate_weights.shape[0]):
        page_id = WeightPageID.expert(revision, layer, expert)
        tensors = [gate_weights[expert], up_weights[expert], down_weights[expert]]
        names = ["gate_proj", "up_proj", "down_proj"]
        layout = ExpertRecordLayout.build(
            [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "bits": 16,
                    "group_size": -1,
                    "symmetric": True,
                    "codec_id": "pt.raw.bf16.v1",
                }
                for name, tensor in zip(names, tensors)
            ]
        )
        payload = b"".join(_raw_bytes(tensor) for tensor in tensors)
        payload += b"\x00" * (layout.stride - len(payload))
        path = tmp_path / f"expert-{layer}-{expert}.ptpage"
        path.write_bytes(payload)
        source_slices = []
        cursor = 0
        for name, tensor in zip(names, tensors):
            size = tensor.numel() * tensor.element_size()
            source_slices.append(
                SourceSlice(
                    tensor=f"model.layers.{layer}.mlp.experts.{expert}.{name}.weight",
                    shard="fixture.safetensors",
                    projection=name,
                    dtype="BF16",
                    shape=list(tensor.shape),
                    byte_start=cursor,
                    byte_end=cursor + size,
                )
            )
            cursor += size
        descriptor = PageDescriptor(
            page_id=page_id,
            weight_ids=[
                WeightID(
                    layer=layer,
                    component="routed_expert",
                    expert_id=expert,
                    projection=name,
                )
                for name in names
            ],
            source_slices=source_slices,
            output_layout=layout,
            expected_bytes=cursor,
        )
        handle = PageHandle(
            page_id=page_id,
            path=path,
            checksum=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            cache_hit=False,
        )
        pages[page_id.cache_key] = (descriptor, handle)
    assert hidden == 8
    assert intermediate == 4
    return pages


def _add_tensor_page(tmp_path, pages, revision, tensor_name, tensor):
    page_id = WeightPageID.tensor(revision, tensor_name)
    layout = ExpertRecordLayout.build(
        [
            {
                "name": "tensor",
                "shape": list(tensor.shape),
                "bits": 16,
                "group_size": -1,
                "symmetric": True,
                "codec_id": "pt.raw.bf16.v1",
            }
        ]
    )
    raw = _raw_bytes(tensor)
    payload = raw + b"\x00" * (layout.stride - len(raw))
    path = tmp_path / f"tensor-{page_id.cache_key}.ptpage"
    path.write_bytes(payload)
    descriptor = PageDescriptor(
        page_id=page_id,
        weight_ids=[WeightID(layer=0, component="backbone", projection=tensor_name)],
        source_slices=[
            SourceSlice(
                tensor=tensor_name,
                shard="fixture.safetensors",
                projection="tensor",
                dtype="BF16",
                shape=list(tensor.shape),
                byte_start=0,
                byte_end=len(raw),
            )
        ],
        output_layout=layout,
        expected_bytes=len(raw),
    )
    handle = PageHandle(
        page_id=page_id,
        path=path,
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        cache_hit=False,
    )
    pages[page_id.cache_key] = (descriptor, handle)


def test_page_backed_olmoe_experts_match_transformers(tmp_path):
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    torch.manual_seed(7)
    revision = ModelRevision(repo_id="fixture/olmoe", commit_sha="f" * 40)
    gate_weights = (torch.randn(3, 4, 8) * 0.1).to(torch.bfloat16)
    up_weights = (torch.randn(3, 4, 8) * 0.1).to(torch.bfloat16)
    down_weights = (torch.randn(3, 8, 4) * 0.1).to(torch.bfloat16)
    store = _StaticPageStore(
        _build_pages(tmp_path, revision, gate_weights, up_weights, down_weights)
    )

    paged = PagedOlmoeExperts(
        store,
        revision,
        0,
        num_experts=3,
        hidden_dim=8,
        intermediate_dim=4,
    )
    hidden = torch.randn(3, 8).to(torch.bfloat16)
    top_k_index = torch.tensor([[0, 1], [2, 0], [1, 2]])
    top_k_weights = torch.tensor(
        [[0.7, 0.3], [0.55, 0.45], [0.6, 0.4]], dtype=torch.bfloat16
    )

    config = OlmoeConfig(
        hidden_size=8,
        intermediate_size=4,
        num_experts=3,
        num_experts_per_tok=2,
        hidden_act="silu",
        num_attention_heads=2,
    )
    reference = OlmoeExperts(config).to(dtype=torch.bfloat16)
    with torch.no_grad():
        reference.gate_up_proj.copy_(torch.cat((gate_weights, up_weights), dim=1))
        reference.down_proj.copy_(down_weights)

    expected = reference(hidden, top_k_index, top_k_weights)
    actual = paged(hidden, top_k_index, top_k_weights)
    assert torch.equal(actual, expected)
    assert paged.last_metrics.experts_executed == 3
    assert paged.last_metrics.page_faults == 3
    assert paged.last_metrics.remote_bytes == 3 * (4 * 8 + 4 * 8 + 8 * 4) * 2
    assert paged.last_metrics.peak_projection_bytes == 2 * 4 * 8 * 2

    warm = paged(hidden, top_k_index, top_k_weights)
    assert torch.equal(warm, expected)
    assert paged.last_metrics.page_hits == 3
    assert paged.last_metrics.remote_bytes == 0
    assert store.releases == 6


def test_expert_page_views_preserve_projection_shapes(tmp_path):
    revision = ModelRevision(repo_id="fixture/olmoe", commit_sha="a" * 40)
    gate = torch.arange(32).reshape(1, 4, 8).to(torch.bfloat16)
    up = (gate + 100).to(torch.bfloat16)
    down = torch.arange(32).reshape(1, 8, 4).to(torch.bfloat16)
    pages = _build_pages(tmp_path, revision, gate, up, down)
    descriptor, handle = pages[WeightPageID.expert(revision, 0, 0).cache_key]
    page = ExpertPageTensors(handle, descriptor)
    assert torch.equal(page.tensor("gate_proj"), gate[0])
    assert torch.equal(page.tensor("up_proj"), up[0])
    assert torch.equal(page.tensor("down_proj"), down[0])


def test_paged_block_preserves_the_upstream_router_contract(tmp_path):
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

    torch.manual_seed(11)
    revision = ModelRevision(repo_id="fixture/olmoe", commit_sha="b" * 40)
    config = OlmoeConfig(
        hidden_size=8,
        intermediate_size=4,
        num_experts=3,
        num_experts_per_tok=2,
        hidden_act="silu",
        num_attention_heads=2,
    )
    reference = OlmoeSparseMoeBlock(config).to(dtype=torch.bfloat16)
    gate_weights = reference.experts.gate_up_proj[:, :4].detach().clone()
    up_weights = reference.experts.gate_up_proj[:, 4:].detach().clone()
    down_weights = reference.experts.down_proj.detach().clone()
    store = _StaticPageStore(
        _build_pages(tmp_path, revision, gate_weights, up_weights, down_weights)
    )
    hidden = torch.randn(2, 3, 8).to(torch.bfloat16)
    expected = reference(hidden)
    paged = replace_olmoe_sparse_moe(
        reference,
        store,
        revision,
        0,
        intermediate_dim=4,
    )
    actual = paged(hidden)
    assert torch.equal(actual, expected)
    assert paged.last_top_k_index is not None
    assert paged.last_top_k_index.shape == (6, 2)


def test_complete_decoder_layer_loads_backbone_pages_and_matches_upstream(tmp_path):
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import (
        OlmoeDecoderLayer,
        OlmoeRotaryEmbedding,
    )

    torch.manual_seed(13)
    revision = ModelRevision(repo_id="fixture/olmoe", commit_sha="c" * 40)
    config = OlmoeConfig(
        hidden_size=8,
        intermediate_size=4,
        num_hidden_layers=1,
        num_experts=3,
        num_experts_per_tok=2,
        hidden_act="silu",
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    reference = OlmoeDecoderLayer(config, 0).to(dtype=torch.bfloat16).eval()
    gate = reference.mlp.experts.gate_up_proj[:, :4].detach().clone()
    up = reference.mlp.experts.gate_up_proj[:, 4:].detach().clone()
    down = reference.mlp.experts.down_proj.detach().clone()
    pages = _build_pages(tmp_path, revision, gate, up, down)
    for name, parameter in reference.named_parameters():
        if name.startswith("mlp.experts."):
            continue
        _add_tensor_page(
            tmp_path,
            pages,
            revision,
            f"model.layers.0.{name}",
            parameter.detach().clone(),
        )
    store = _StaticPageStore(pages)
    paged, metrics = build_paged_olmoe_decoder_layer(config, store, revision, 0)
    assert metrics.tensors == 9
    assert metrics.page_faults == 9
    assert metrics.resident_bytes == sum(
        parameter.numel() * parameter.element_size()
        for name, parameter in reference.named_parameters()
        if not name.startswith("mlp.experts.")
    )

    hidden = torch.randn(1, 3, 8).to(torch.bfloat16)
    position_ids = torch.arange(3).unsqueeze(0)
    rotary = OlmoeRotaryEmbedding(config)
    position_embeddings = rotary(hidden, position_ids)
    with torch.no_grad():
        expected = reference(
            hidden,
            attention_mask=None,
            position_embeddings=position_embeddings,
        )
        actual = paged(
            hidden,
            attention_mask=None,
            position_embeddings=position_embeddings,
        )
    assert torch.equal(actual, expected)


def test_sequential_one_token_model_matches_upstream_logits(tmp_path):
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    torch.manual_seed(17)
    revision = ModelRevision(repo_id="fixture/olmoe", commit_sha="d" * 40)
    config = OlmoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=4,
        num_hidden_layers=2,
        num_experts=3,
        num_experts_per_tok=2,
        hidden_act="silu",
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    reference = OlmoeForCausalLM(config).to(dtype=torch.bfloat16).eval()
    pages = {}
    for layer_idx, layer in enumerate(reference.model.layers):
        gate_up = layer.mlp.experts.gate_up_proj.detach().clone()
        pages.update(
            _build_pages(
                tmp_path,
                revision,
                gate_up[:, : config.intermediate_size],
                gate_up[:, config.intermediate_size :],
                layer.mlp.experts.down_proj.detach().clone(),
                layer=layer_idx,
            )
        )
        for name, parameter in layer.named_parameters():
            if name.startswith("mlp.experts."):
                continue
            _add_tensor_page(
                tmp_path,
                pages,
                revision,
                f"model.layers.{layer_idx}.{name}",
                parameter.detach().clone(),
            )
    _add_tensor_page(
        tmp_path,
        pages,
        revision,
        "model.embed_tokens.weight",
        reference.model.embed_tokens.weight.detach().clone(),
    )
    _add_tensor_page(
        tmp_path,
        pages,
        revision,
        "model.norm.weight",
        reference.model.norm.weight.detach().clone(),
    )
    _add_tensor_page(
        tmp_path,
        pages,
        revision,
        "lm_head.weight",
        reference.lm_head.weight.detach().clone(),
    )
    store = _StaticPageStore(pages)
    runner = PagedOlmoeOneTokenRunner(
        config,
        store,
        revision,
        head_chunk_bytes=32,
    )
    input_token_id = 3
    with torch.inference_mode():
        expected = reference(torch.tensor([[input_token_id]])).logits
        actual, first = runner.run(input_token_id)
        replay, warm = runner.run(input_token_id)

    assert torch.equal(actual, expected)
    assert torch.equal(replay, expected)
    assert first.global_page_faults == 3
    assert first.backbone_page_faults == 18
    assert first.expert_page_faults == 4
    assert warm.global_page_hits == 3
    assert warm.backbone_page_hits == 18
    assert warm.expert_page_hits == 4
    assert warm.total_remote_bytes == 0
    assert first.top_token_id == int(expected.argmax().item())

    hypothesis = run_olmoe_full_model_hypothesis(
        store,
        SimpleNamespace(model_revision=revision, token=None),
        input_token_id=input_token_id,
        head_chunk_bytes=32,
        max_vram_bytes=1024**3,
        max_ram_bytes=12 * 1024**3,
        config=config,
    )
    assert hypothesis.passed is True
    assert hypothesis.num_layers == 2
    assert hypothesis.logits_max_abs_delta == 0.0
    assert hypothesis.warm.total_remote_bytes == 0

    parity = run_olmoe_full_model_reference(
        store,
        SimpleNamespace(model_revision=revision, token=None),
        input_token_id=input_token_id,
        head_chunk_bytes=32,
        config=config,
    )
    assert parity.passed is True
    assert parity.bit_exact is True
    assert parity.max_abs_error == 0.0
    assert parity.routing_agreement is True

    generation = run_olmoe_two_position_generation(
        store,
        SimpleNamespace(model_revision=revision, token=None),
        input_token_id=input_token_id,
        head_chunk_bytes=32,
        max_vram_bytes=1024**3,
        max_ram_bytes=12 * 1024**3,
        config=config,
    )
    assert generation.passed is True
    assert len(generation.token_ids) == 3
    assert len(generation.positions) == 2
    assert generation.positions[0].candidate.cache_sequence_length == 1
    assert generation.positions[1].candidate.cache_sequence_length == 2
    assert generation.positions[1].bit_exact is True
    assert generation.final_kv_cache_bytes > 0


def test_commit_routing_substitutes_vram_resident_experts(tmp_path):
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    from pockettitan.domainslice.fast_cache import ExpertMemoryCache, CachedExpert

    torch.manual_seed(42)
    revision = ModelRevision(repo_id="fixture/commit_routing", commit_sha="c" * 40)
    config = OlmoeConfig(
        hidden_size=8,
        intermediate_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
        num_attention_heads=2,
    )
    reference = OlmoeSparseMoeBlock(config).to(dtype=torch.bfloat16)
    gate_weights = reference.experts.gate_up_proj[:, :4].detach().clone()
    up_weights = reference.experts.gate_up_proj[:, 4:].detach().clone()
    down_weights = reference.experts.down_proj.detach().clone()
    store = _StaticPageStore(
        _build_pages(tmp_path, revision, gate_weights, up_weights, down_weights)
    )

    cache = ExpertMemoryCache(vram_capacity=4, ram_capacity=4)
    # Pre-seed expert 0 in layer 0 into VRAM cache
    cache._vram_cache[(0, 0)] = CachedExpert(gate_up=torch.zeros(8, 8), down=torch.zeros(8, 4), tier="vram")

    paged = replace_olmoe_sparse_moe(
        reference,
        store,
        revision,
        0,
        intermediate_dim=4,
        expert_cache=cache,
        commit_routing=True,
        commit_threshold=10.0,
    )

    hidden = torch.randn(1, 1, 8).to(torch.bfloat16)
    _ = paged(hidden)
    selected = paged.last_top_k_index[0].tolist()
    assert 0 in selected

