"""Measured block-level hypothesis test for real OLMoE expert pages."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from pockettitan.metadata.repo import fetch_model_config
from pockettitan.runtime.hf.olmoe_paged import ExpertPageTensors, PagedOlmoeExperts
from pockettitan.runtime.hf.olmoe_layer import build_paged_olmoe_decoder_layer
from pockettitan.runtime.hf.olmoe_model import PagedOlmoeOneTokenRunner
from pockettitan.streaming.reader import RemoteTensorSliceReader

from .store import CompositeWeightStore, RemoteHuggingFaceStore
from .types import ModelRevision, WeightPageID
from .types import ProgressCallback


class BlockRunMetrics(BaseModel):
    elapsed_seconds: float
    page_hits: int
    page_faults: int
    remote_bytes: int
    resumed_bytes: int
    page_bytes: int
    peak_projection_bytes: int
    peak_cuda_bytes: int


class BlockHypothesisResult(BaseModel):
    model_revision: ModelRevision
    layer: int
    tokens: int
    execution_device: str
    selected_experts: list[int] = Field(default_factory=list)
    router_payload_bytes: int
    cold: BlockRunMetrics
    warm: BlockRunMetrics
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    argmax_agreement: float
    reference_scale: float
    cache_occupancy_bytes: int
    passed: bool


class LayerBackboneMetrics(BaseModel):
    tensors: int
    page_hits: int
    page_faults: int
    remote_bytes: int
    resident_bytes: int


class LayerHypothesisResult(BaseModel):
    model_revision: ModelRevision
    layer: int
    execution_device: str
    backbone: LayerBackboneMetrics
    selected_experts: list[int] = Field(default_factory=list)
    first: BlockRunMetrics
    warm: BlockRunMetrics
    output_max_abs: float
    warm_max_abs_delta: float
    cache_occupancy_bytes: int
    total_first_remote_bytes: int
    passed: bool


class FullLayerRunMetrics(BaseModel):
    layer: int
    selected_experts: list[int] = Field(default_factory=list)
    backbone_page_hits: int
    backbone_page_faults: int
    backbone_remote_bytes: int
    expert_page_hits: int
    expert_page_faults: int
    expert_remote_bytes: int
    elapsed_seconds: float
    output_sha256: str
    output_max_abs: float


class FullPassMetrics(BaseModel):
    elapsed_seconds: float
    global_page_hits: int
    global_page_faults: int
    global_remote_bytes: int
    backbone_page_hits: int
    backbone_page_faults: int
    backbone_remote_bytes: int
    expert_page_hits: int
    expert_page_faults: int
    expert_remote_bytes: int
    experts_executed: int
    total_remote_bytes: int
    logical_page_bytes: int
    peak_projection_bytes: int
    peak_head_chunk_bytes: int
    peak_cuda_bytes: int
    rss_start_bytes: int
    rss_end_bytes: int
    peak_rss_bytes: int
    top_token_id: int
    top_logit: float
    logits_sha256: str
    kv_cache_bytes: int
    cache_sequence_length: int
    layers: list[FullLayerRunMetrics] = Field(default_factory=list)


class FullModelHypothesisResult(BaseModel):
    model_revision: ModelRevision
    execution_device: str
    input_token_id: int
    num_layers: int
    first: FullPassMetrics
    warm: FullPassMetrics
    logits_max_abs_delta: float
    argmax_agreement: bool
    routing_agreement: bool
    cache_occupancy_bytes: int
    max_vram_bytes: Optional[int] = None
    max_ram_bytes: Optional[int] = None
    vram_within_budget: bool
    ram_within_budget: bool
    passed: bool


class FullModelReferenceResult(BaseModel):
    model_revision: ModelRevision
    execution_device: str
    input_token_id: int
    candidate: FullPassMetrics
    oracle: FullPassMetrics
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    argmax_agreement: bool
    routing_agreement: bool
    bit_exact: bool
    reference_scale: float
    cache_occupancy_bytes: int
    max_vram_bytes: Optional[int] = None
    max_ram_bytes: Optional[int] = None
    vram_within_budget: bool
    ram_within_budget: bool
    passed: bool


class GenerationPositionResult(BaseModel):
    position: int
    input_token_id: int
    predicted_token_id: int
    candidate: FullPassMetrics
    oracle: FullPassMetrics
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    argmax_agreement: bool
    routing_agreement: bool
    bit_exact: bool
    passed: bool


class TwoTokenGenerationResult(BaseModel):
    model_revision: ModelRevision
    execution_device: str
    token_ids: list[int] = Field(default_factory=list)
    positions: list[GenerationPositionResult] = Field(default_factory=list)
    candidate_new_remote_bytes: int
    final_kv_cache_bytes: int
    cache_occupancy_bytes: int
    max_vram_bytes: Optional[int] = None
    max_ram_bytes: Optional[int] = None
    vram_within_budget: bool
    ram_within_budget: bool
    passed: bool


def _metrics(value) -> BlockRunMetrics:
    return BlockRunMetrics(
        elapsed_seconds=value.elapsed_seconds,
        page_hits=value.page_hits,
        page_faults=value.page_faults,
        remote_bytes=value.remote_bytes,
        resumed_bytes=value.resumed_bytes,
        page_bytes=value.page_bytes,
        peak_projection_bytes=value.peak_projection_bytes,
        peak_cuda_bytes=value.peak_cuda_bytes,
    )


def _full_metrics(value) -> FullPassMetrics:
    return FullPassMetrics(
        elapsed_seconds=value.elapsed_seconds,
        global_page_hits=value.global_page_hits,
        global_page_faults=value.global_page_faults,
        global_remote_bytes=value.global_remote_bytes,
        backbone_page_hits=value.backbone_page_hits,
        backbone_page_faults=value.backbone_page_faults,
        backbone_remote_bytes=value.backbone_remote_bytes,
        expert_page_hits=value.expert_page_hits,
        expert_page_faults=value.expert_page_faults,
        expert_remote_bytes=value.expert_remote_bytes,
        experts_executed=value.experts_executed,
        total_remote_bytes=value.total_remote_bytes,
        logical_page_bytes=value.logical_page_bytes,
        peak_projection_bytes=value.peak_projection_bytes,
        peak_head_chunk_bytes=value.peak_head_chunk_bytes,
        peak_cuda_bytes=value.peak_cuda_bytes,
        rss_start_bytes=value.rss_start_bytes,
        rss_end_bytes=value.rss_end_bytes,
        peak_rss_bytes=value.peak_rss_bytes,
        top_token_id=value.top_token_id,
        top_logit=value.top_logit,
        logits_sha256=value.logits_sha256,
        kv_cache_bytes=value.kv_cache_bytes,
        cache_sequence_length=value.cache_sequence_length,
        layers=[
            FullLayerRunMetrics(
                layer=item.layer,
                selected_experts=item.selected_experts,
                backbone_page_hits=item.backbone.page_hits,
                backbone_page_faults=item.backbone.page_faults,
                backbone_remote_bytes=item.backbone.remote_bytes,
                expert_page_hits=item.experts.page_hits,
                expert_page_faults=item.experts.page_faults,
                expert_remote_bytes=item.experts.remote_bytes,
                elapsed_seconds=item.elapsed_seconds,
                output_sha256=item.output_sha256,
                output_max_abs=item.output_max_abs,
            )
            for item in value.layers
        ],
    )


def _router_address(remote: RemoteHuggingFaceStore, layer: int):
    candidates = [
        item
        for item in remote.address_table.get_tensors_for_layer(layer)
        if item.name.endswith(".mlp.gate.weight") and ".experts." not in item.name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one OLMoE router tensor at layer {layer}, found "
            f"{[item.name for item in candidates]}"
        )
    return candidates[0]


def _load_reference_experts(
    store: CompositeWeightStore,
    revision: ModelRevision,
    layer: int,
    selected: list[int],
    hidden_dim: int,
    intermediate_dim: int,
    hidden_act: str,
    execution_device: torch.device,
):
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

    config = OlmoeConfig(
        hidden_size=hidden_dim,
        intermediate_size=intermediate_dim,
        num_experts=len(selected),
        num_experts_per_tok=len(selected),
        hidden_act=hidden_act,
    )
    reference = OlmoeExperts(config).to(device=execution_device, dtype=torch.bfloat16)
    with torch.no_grad():
        for slot, expert_id in enumerate(selected):
            page_id = WeightPageID.expert(revision, layer, expert_id)
            descriptor = store.resolve(page_id)
            handle = store.materialize(page_id)
            try:
                page = ExpertPageTensors(handle, descriptor)
                gate = page.tensor("gate_proj").to(execution_device)
                up = page.tensor("up_proj").to(execution_device)
                down = page.tensor("down_proj").to(execution_device)
                reference.gate_up_proj[slot].copy_(torch.cat((gate, up), dim=0))
                reference.down_proj[slot].copy_(down)
            finally:
                store.release(handle)
    return reference


def run_olmoe_block_hypothesis(
    store: CompositeWeightStore,
    remote: RemoteHuggingFaceStore,
    *,
    layer: int = 9,
    tokens: int = 1,
    seed: int = 42,
    execution_device: str | torch.device = "cpu",
    progress: Optional[ProgressCallback] = None,
) -> BlockHypothesisResult:
    """Compare a real page-backed expert block with upstream Transformers."""
    if tokens < 1:
        raise ValueError("tokens must be positive")
    device = torch.device(execution_device)
    metadata = remote.address_table.metadata
    if metadata.num_experts is None or metadata.num_experts_per_tok is None:
        raise RuntimeError("The source checkpoint has no MoE routing topology")
    if metadata.expert_intermediate_size is None:
        raise RuntimeError("The source checkpoint has no expert intermediate width")

    raw_config = fetch_model_config(
        remote.model_revision.repo_id,
        token=remote.token,
        revision=remote.model_revision.commit_sha,
    )
    hidden_act = str(raw_config.get("hidden_act", "silu"))
    norm_topk_prob = bool(raw_config.get("norm_topk_prob", False))

    router_address = _router_address(remote, layer)
    router = RemoteTensorSliceReader(
        remote.model_revision.repo_id,
        token=remote.token,
        revision=remote.model_revision.commit_sha,
    ).read_tensor(router_address)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(
        tokens,
        metadata.hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    router_logits = F.linear(hidden, router)
    probabilities = torch.softmax(router_logits, dtype=torch.float32, dim=-1)
    top_k_weights, top_k_index = torch.topk(
        probabilities,
        metadata.num_experts_per_tok,
        dim=-1,
    )
    if norm_topk_prob:
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights.to(router_logits.dtype)
    selected = sorted(int(value) for value in torch.unique(top_k_index).tolist())

    paged = PagedOlmoeExperts(
        store,
        remote.model_revision,
        layer,
        num_experts=metadata.num_experts,
        hidden_dim=metadata.hidden_size,
        intermediate_dim=metadata.expert_intermediate_size,
        execution_device=device,
        compute_dtype=torch.bfloat16,
        hidden_act=hidden_act,
        progress=progress,
    )
    cold_output = paged(hidden, top_k_index, top_k_weights)
    cold = _metrics(paged.last_metrics)

    reference = _load_reference_experts(
        store,
        remote.model_revision,
        layer,
        selected,
        metadata.hidden_size,
        metadata.expert_intermediate_size,
        hidden_act,
        device,
    )
    remap = {expert_id: slot for slot, expert_id in enumerate(selected)}
    reference_indices = top_k_index.clone()
    for expert_id, slot in remap.items():
        reference_indices[top_k_index == expert_id] = slot
    reference_output = reference(
        hidden.to(device),
        reference_indices.to(device),
        top_k_weights.to(device),
    ).cpu()

    warm_output = paged(hidden, top_k_index, top_k_weights)
    warm = _metrics(paged.last_metrics)
    delta = cold_output.float() - reference_output.float()
    reference_float = reference_output.float()
    max_abs = float(delta.abs().max().item())
    mean_abs = float(delta.abs().mean().item())
    cosine = float(
        F.cosine_similarity(
            cold_output.float().reshape(1, -1), reference_float.reshape(1, -1)
        ).item()
    )
    argmax = float(
        (cold_output.argmax(dim=-1) == reference_output.argmax(dim=-1)).float().mean().item()
    )
    scale = float(reference_float.abs().max().item())
    tolerance = max(0.01, scale * 0.005)
    passed = (
        max_abs <= tolerance
        and cosine >= 0.99999
        and argmax == 1.0
        and torch.equal(cold_output, warm_output)
    )
    return BlockHypothesisResult(
        model_revision=remote.model_revision,
        layer=layer,
        tokens=tokens,
        execution_device=str(device),
        selected_experts=selected,
        router_payload_bytes=router_address.size_bytes,
        cold=cold,
        warm=warm,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        cosine_similarity=cosine,
        argmax_agreement=argmax,
        reference_scale=scale,
        cache_occupancy_bytes=store.stats().cache_occupancy_bytes,
        passed=passed,
    )


def run_olmoe_layer_hypothesis(
    store: CompositeWeightStore,
    remote: RemoteHuggingFaceStore,
    *,
    layer: int = 9,
    seed: int = 42,
    execution_device: str | torch.device = "cpu",
) -> LayerHypothesisResult:
    """Assemble and execute one complete real decoder layer twice."""
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding

    device = torch.device(execution_device)
    raw_config = fetch_model_config(
        remote.model_revision.repo_id,
        token=remote.token,
        revision=remote.model_revision.commit_sha,
    )
    config = OlmoeConfig(**raw_config)
    decoder, backbone = build_paged_olmoe_decoder_layer(
        config,
        store,
        remote.model_revision,
        layer,
        device=device,
        dtype=torch.bfloat16,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(
        1,
        1,
        config.hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    position_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    rotary = OlmoeRotaryEmbedding(config).to(device)
    position_embeddings = rotary(hidden, position_ids)

    with torch.no_grad():
        first_output = decoder(
            hidden,
            attention_mask=None,
            position_embeddings=position_embeddings,
        )
        first = _metrics(decoder.mlp.experts.last_metrics)
        selected = sorted(
            int(value) for value in torch.unique(decoder.mlp.last_top_k_index).tolist()
        )
        warm_output = decoder(
            hidden,
            attention_mask=None,
            position_embeddings=position_embeddings,
        )
        warm = _metrics(decoder.mlp.experts.last_metrics)

    finite = bool(torch.isfinite(first_output).all().item())
    delta = float((first_output.float() - warm_output.float()).abs().max().item())
    output_max = float(first_output.float().abs().max().item())
    passed = (
        finite
        and delta == 0.0
        and len(selected) == config.num_experts_per_tok
        and warm.remote_bytes == 0
        and warm.page_faults == 0
    )
    return LayerHypothesisResult(
        model_revision=remote.model_revision,
        layer=layer,
        execution_device=str(device),
        backbone=LayerBackboneMetrics(
            tensors=backbone.tensors,
            page_hits=backbone.page_hits,
            page_faults=backbone.page_faults,
            remote_bytes=backbone.remote_bytes,
            resident_bytes=backbone.resident_bytes,
        ),
        selected_experts=selected,
        first=first,
        warm=warm,
        output_max_abs=output_max,
        warm_max_abs_delta=delta,
        cache_occupancy_bytes=store.stats().cache_occupancy_bytes,
        total_first_remote_bytes=backbone.remote_bytes + first.remote_bytes,
        passed=passed,
    )


def run_olmoe_full_model_hypothesis(
    store: CompositeWeightStore,
    remote: RemoteHuggingFaceStore,
    *,
    input_token_id: int = 1,
    execution_device: str | torch.device = "cpu",
    head_chunk_bytes: int = 8 * 1024 * 1024,
    max_vram_bytes: Optional[int] = None,
    max_ram_bytes: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    layer_callback=None,
    config=None,
) -> FullModelHypothesisResult:
    """Execute the pinned model twice and require exact warm-cache logits."""
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig

    device = torch.device(execution_device)
    if config is None:
        raw_config = fetch_model_config(
            remote.model_revision.repo_id,
            token=remote.token,
            revision=remote.model_revision.commit_sha,
        )
        config = OlmoeConfig(**raw_config)
    elif isinstance(config, dict):
        config = OlmoeConfig(**config)

    runner = PagedOlmoeOneTokenRunner(
        config,
        store,
        remote.model_revision,
        execution_device=device,
        compute_dtype=torch.bfloat16,
        head_chunk_bytes=head_chunk_bytes,
        progress=progress,
    )
    first_logits, first_raw = runner.run(
        input_token_id,
        layer_callback=(
            (lambda item: layer_callback("first", item))
            if layer_callback is not None
            else None
        ),
    )
    warm_logits, warm_raw = runner.run(
        input_token_id,
        layer_callback=(
            (lambda item: layer_callback("warm", item))
            if layer_callback is not None
            else None
        ),
    )
    first = _full_metrics(first_raw)
    warm = _full_metrics(warm_raw)
    delta = float((first_logits.float() - warm_logits.float()).abs().max().item())
    argmax_agreement = first.top_token_id == warm.top_token_id
    routing_agreement = [item.selected_experts for item in first.layers] == [
        item.selected_experts for item in warm.layers
    ]
    vram_within_budget = max_vram_bytes is None or max(
        first.peak_cuda_bytes, warm.peak_cuda_bytes
    ) <= max_vram_bytes
    ram_within_budget = max_ram_bytes is None or max(
        first.peak_rss_bytes, warm.peak_rss_bytes
    ) <= max_ram_bytes
    complete_routing = (
        len(first.layers) == int(config.num_hidden_layers)
        and all(
            len(item.selected_experts) == int(config.num_experts_per_tok)
            for item in first.layers
        )
    )
    passed = (
        torch.equal(first_logits, warm_logits)
        and first.logits_sha256 == warm.logits_sha256
        and warm.total_remote_bytes == 0
        and argmax_agreement
        and routing_agreement
        and complete_routing
        and vram_within_budget
        and ram_within_budget
    )
    return FullModelHypothesisResult(
        model_revision=remote.model_revision,
        execution_device=str(device),
        input_token_id=input_token_id,
        num_layers=int(config.num_hidden_layers),
        first=first,
        warm=warm,
        logits_max_abs_delta=delta,
        argmax_agreement=argmax_agreement,
        routing_agreement=routing_agreement,
        cache_occupancy_bytes=store.stats().cache_occupancy_bytes,
        max_vram_bytes=max_vram_bytes,
        max_ram_bytes=max_ram_bytes,
        vram_within_budget=vram_within_budget,
        ram_within_budget=ram_within_budget,
        passed=passed,
    )


def run_olmoe_full_model_reference(
    store: CompositeWeightStore,
    remote: RemoteHuggingFaceStore,
    *,
    input_token_id: int = 1,
    execution_device: str | torch.device = "cpu",
    head_chunk_bytes: int = 8 * 1024 * 1024,
    max_vram_bytes: Optional[int] = None,
    max_ram_bytes: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    layer_callback=None,
    config=None,
) -> FullModelReferenceResult:
    """Compare PocketTitan expert math with the official HF expert implementation."""
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig

    device = torch.device(execution_device)
    if config is None:
        raw_config = fetch_model_config(
            remote.model_revision.repo_id,
            token=remote.token,
            revision=remote.model_revision.commit_sha,
        )
        config = OlmoeConfig(**raw_config)
    elif isinstance(config, dict):
        config = OlmoeConfig(**config)

    common = {
        "execution_device": device,
        "compute_dtype": torch.bfloat16,
        "head_chunk_bytes": head_chunk_bytes,
        "progress": progress,
    }
    candidate_runner = PagedOlmoeOneTokenRunner(
        config,
        store,
        remote.model_revision,
        expert_backend="paged",
        **common,
    )
    oracle_runner = PagedOlmoeOneTokenRunner(
        config,
        store,
        remote.model_revision,
        expert_backend="transformers",
        **common,
    )
    candidate_logits, candidate_raw = candidate_runner.run(
        input_token_id,
        layer_callback=(
            (lambda item: layer_callback("candidate", item))
            if layer_callback is not None
            else None
        ),
    )
    oracle_logits, oracle_raw = oracle_runner.run(
        input_token_id,
        layer_callback=(
            (lambda item: layer_callback("oracle", item))
            if layer_callback is not None
            else None
        ),
    )
    candidate = _full_metrics(candidate_raw)
    oracle = _full_metrics(oracle_raw)
    delta = candidate_logits.float() - oracle_logits.float()
    reference_float = oracle_logits.float()
    max_abs = float(delta.abs().max().item())
    mean_abs = float(delta.abs().mean().item())
    cosine = float(
        F.cosine_similarity(
            candidate_logits.float().reshape(1, -1),
            reference_float.reshape(1, -1),
        ).item()
    )
    scale = float(reference_float.abs().max().item())
    tolerance = max(0.01, scale * 0.005)
    routing_agreement = [item.selected_experts for item in candidate.layers] == [
        item.selected_experts for item in oracle.layers
    ]
    argmax_agreement = candidate.top_token_id == oracle.top_token_id
    bit_exact = torch.equal(candidate_logits, oracle_logits)
    vram_within_budget = max_vram_bytes is None or max(
        candidate.peak_cuda_bytes, oracle.peak_cuda_bytes
    ) <= max_vram_bytes
    ram_within_budget = max_ram_bytes is None or max(
        candidate.peak_rss_bytes, oracle.peak_rss_bytes
    ) <= max_ram_bytes
    passed = (
        max_abs <= tolerance
        and cosine >= 0.99999
        and argmax_agreement
        and routing_agreement
        and oracle.total_remote_bytes == 0
        and vram_within_budget
        and ram_within_budget
    )
    return FullModelReferenceResult(
        model_revision=remote.model_revision,
        execution_device=str(device),
        input_token_id=input_token_id,
        candidate=candidate,
        oracle=oracle,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        cosine_similarity=cosine,
        argmax_agreement=argmax_agreement,
        routing_agreement=routing_agreement,
        bit_exact=bit_exact,
        reference_scale=scale,
        cache_occupancy_bytes=store.stats().cache_occupancy_bytes,
        max_vram_bytes=max_vram_bytes,
        max_ram_bytes=max_ram_bytes,
        vram_within_budget=vram_within_budget,
        ram_within_budget=ram_within_budget,
        passed=passed,
    )


def run_olmoe_two_position_generation(
    store: CompositeWeightStore,
    remote: RemoteHuggingFaceStore,
    *,
    input_token_id: int = 1,
    execution_device: str | torch.device = "cpu",
    head_chunk_bytes: int = 8 * 1024 * 1024,
    max_vram_bytes: Optional[int] = None,
    max_ram_bytes: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    layer_callback=None,
    config=None,
) -> TwoTokenGenerationResult:
    """Generate across two positions and compare KV-cached execution to HF experts."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig

    device = torch.device(execution_device)
    if config is None:
        raw_config = fetch_model_config(
            remote.model_revision.repo_id,
            token=remote.token,
            revision=remote.model_revision.commit_sha,
        )
        config = OlmoeConfig(**raw_config)
    elif isinstance(config, dict):
        config = OlmoeConfig(**config)

    common = {
        "execution_device": device,
        "compute_dtype": torch.bfloat16,
        "head_chunk_bytes": head_chunk_bytes,
        "progress": progress,
    }
    candidate_runner = PagedOlmoeOneTokenRunner(
        config,
        store,
        remote.model_revision,
        expert_backend="paged",
        **common,
    )
    oracle_runner = PagedOlmoeOneTokenRunner(
        config,
        store,
        remote.model_revision,
        expert_backend="transformers",
        **common,
    )
    candidate_cache = DynamicCache(config=config)
    oracle_cache = DynamicCache(config=config)
    current_token = input_token_id
    token_ids = [input_token_id]
    positions = []
    for position in range(2):
        candidate_logits, candidate_raw = candidate_runner.run(
            current_token,
            position_id=position,
            past_key_values=candidate_cache,
            use_cache=True,
            layer_callback=(
                (lambda item, pos=position: layer_callback(f"candidate-p{pos}", item))
                if layer_callback is not None
                else None
            ),
        )
        predicted = int(candidate_logits.reshape(-1).argmax().item())
        oracle_logits, oracle_raw = oracle_runner.run(
            current_token,
            position_id=position,
            past_key_values=oracle_cache,
            use_cache=True,
            layer_callback=(
                (lambda item, pos=position: layer_callback(f"oracle-p{pos}", item))
                if layer_callback is not None
                else None
            ),
        )
        candidate = _full_metrics(candidate_raw)
        oracle = _full_metrics(oracle_raw)
        delta = candidate_logits.float() - oracle_logits.float()
        reference_float = oracle_logits.float()
        max_abs = float(delta.abs().max().item())
        mean_abs = float(delta.abs().mean().item())
        cosine = float(
            F.cosine_similarity(
                candidate_logits.float().reshape(1, -1),
                reference_float.reshape(1, -1),
            ).item()
        )
        routing_agreement = [item.selected_experts for item in candidate.layers] == [
            item.selected_experts for item in oracle.layers
        ]
        argmax_agreement = candidate.top_token_id == oracle.top_token_id
        bit_exact = torch.equal(candidate_logits, oracle_logits)
        position_passed = (
            bit_exact
            and routing_agreement
            and argmax_agreement
            and oracle.total_remote_bytes == 0
            and candidate.cache_sequence_length == position + 1
            and oracle.cache_sequence_length == position + 1
        )
        positions.append(
            GenerationPositionResult(
                position=position,
                input_token_id=current_token,
                predicted_token_id=predicted,
                candidate=candidate,
                oracle=oracle,
                max_abs_error=max_abs,
                mean_abs_error=mean_abs,
                cosine_similarity=cosine,
                argmax_agreement=argmax_agreement,
                routing_agreement=routing_agreement,
                bit_exact=bit_exact,
                passed=position_passed,
            )
        )
        token_ids.append(predicted)
        current_token = predicted

    all_metrics = [
        metric
        for position in positions
        for metric in (position.candidate, position.oracle)
    ]
    vram_within_budget = max_vram_bytes is None or max(
        metric.peak_cuda_bytes for metric in all_metrics
    ) <= max_vram_bytes
    ram_within_budget = max_ram_bytes is None or max(
        metric.peak_rss_bytes for metric in all_metrics
    ) <= max_ram_bytes
    passed = (
        all(position.passed for position in positions)
        and vram_within_budget
        and ram_within_budget
    )
    return TwoTokenGenerationResult(
        model_revision=remote.model_revision,
        execution_device=str(device),
        token_ids=token_ids,
        positions=positions,
        candidate_new_remote_bytes=sum(
            position.candidate.total_remote_bytes for position in positions
        ),
        final_kv_cache_bytes=positions[-1].candidate.kv_cache_bytes,
        cache_occupancy_bytes=store.stats().cache_occupancy_bytes,
        max_vram_bytes=max_vram_bytes,
        max_ram_bytes=max_ram_bytes,
        vram_within_budget=vram_within_budget,
        ram_within_budget=ram_within_budget,
        passed=passed,
    )


def default_hf_token() -> Optional[str]:
    """Read authentication without ever placing it in reports or identities."""
    return os.environ.get("HF_TOKEN")
