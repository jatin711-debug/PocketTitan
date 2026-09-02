"""Sequential one-token OLMoE execution backed by DomainSlice pages."""

from __future__ import annotations

import gc
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from pockettitan.domainslice import ModelRevision, WeightPageID, WeightStore
from pockettitan.domainslice.types import ProgressCallback
from pockettitan.runtime.hf.olmoe_layer import (
    BackboneLoadMetrics,
    build_paged_olmoe_decoder_layer,
)
from pockettitan.runtime.hf.olmoe_paged import ExpertPageTensors, PagedExpertMetrics


class PagedModelError(RuntimeError):
    """The sequential paged model could not execute safely."""


@dataclass
class SequentialLayerMetrics:
    layer: int
    selected_experts: list[int]
    backbone: BackboneLoadMetrics
    experts: PagedExpertMetrics
    elapsed_seconds: float
    output_sha256: str
    output_max_abs: float


@dataclass
class SequentialPassMetrics:
    elapsed_seconds: float = 0.0
    global_page_hits: int = 0
    global_page_faults: int = 0
    global_remote_bytes: int = 0
    global_page_bytes: int = 0
    backbone_page_hits: int = 0
    backbone_page_faults: int = 0
    backbone_remote_bytes: int = 0
    backbone_page_bytes: int = 0
    expert_page_hits: int = 0
    expert_page_faults: int = 0
    expert_remote_bytes: int = 0
    expert_page_bytes: int = 0
    experts_executed: int = 0
    peak_projection_bytes: int = 0
    peak_head_chunk_bytes: int = 0
    peak_cuda_bytes: int = 0
    rss_start_bytes: int = 0
    rss_end_bytes: int = 0
    peak_rss_bytes: int = 0
    top_token_id: int = -1
    top_logit: float = 0.0
    logits_sha256: str = ""
    kv_cache_bytes: int = 0
    cache_sequence_length: int = 0
    layers: list[SequentialLayerMetrics] = field(default_factory=list)

    @property
    def total_remote_bytes(self) -> int:
        return (
            self.global_remote_bytes
            + self.backbone_remote_bytes
            + self.expert_remote_bytes
        )

    @property
    def logical_page_bytes(self) -> int:
        return self.global_page_bytes + self.backbone_page_bytes + self.expert_page_bytes


def _rss_bytes() -> int:
    """Return current process RSS without making psutil a runtime dependency."""
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError):
        return 0


def _kv_cache_bytes(cache) -> int:
    if cache is None:
        return 0
    total = 0
    for layer in getattr(cache, "layers", []):
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total


class PagedOlmoeOneTokenRunner:
    """Run one token through every layer while only one decoder layer occupies VRAM."""

    def __init__(
        self,
        config,
        store: WeightStore,
        model_revision: ModelRevision,
        *,
        execution_device: str | torch.device = "cpu",
        compute_dtype: torch.dtype = torch.bfloat16,
        head_chunk_bytes: int = 8 * 1024 * 1024,
        progress: Optional[ProgressCallback] = None,
        expert_backend: str = "paged",
        resident_backbone: bool = False,
        expert_cache: Optional[Any] = None,
        commit_routing: bool = False,
        commit_threshold: float = 0.15,
    ):
        self.config = config
        self.store = store
        self.model_revision = model_revision
        self.execution_device = torch.device(execution_device)
        self.compute_dtype = compute_dtype
        self.head_chunk_bytes = int(head_chunk_bytes)
        self.progress = progress
        self.resident_backbone = bool(resident_backbone)
        self.expert_cache = expert_cache
        self.commit_routing = bool(commit_routing)
        self.commit_threshold = float(commit_threshold)
        self._resident_layers = None
        self._resident_embed = None
        self._resident_norm = None
        self._resident_lm_head = None
        self._resident_rotary = None
        if expert_backend not in {"paged", "transformers"}:
            raise PagedModelError(
                "expert_backend must be 'paged' or 'transformers'"
            )
        self.expert_backend = expert_backend
        if self.execution_device.type == "cuda" and not torch.cuda.is_available():
            raise PagedModelError("CUDA model execution requested but CUDA is unavailable")
        row_bytes = int(config.hidden_size) * torch.tensor([], dtype=compute_dtype).element_size()
        if self.head_chunk_bytes < row_bytes:
            raise PagedModelError(
                f"Output-head chunk budget must fit one row ({row_bytes:,} bytes)"
            )

    def _init_resident_backbone(self, metrics: Optional[SequentialPassMetrics] = None) -> None:
        if self._resident_layers is not None:
            return
        from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm, OlmoeRotaryEmbedding

        # 1. Embeddings
        descriptor, handle, _page, tensor = self._lease_tensor("model.embed_tokens.weight")
        try:
            if metrics:
                self._record_global(metrics, descriptor, handle)
            self._resident_embed = tensor.to(device=self.execution_device, dtype=self.compute_dtype).clone()
        finally:
            self.store.release(handle)

        # 2. Final Norm
        descriptor, handle, _page, tensor = self._lease_tensor("model.norm.weight")
        try:
            if metrics:
                self._record_global(metrics, descriptor, handle)
            norm = OlmoeRMSNorm(
                int(self.config.hidden_size), eps=float(self.config.rms_norm_eps)
            ).to(device=self.execution_device, dtype=self.compute_dtype)
            with torch.no_grad():
                norm.weight.copy_(tensor.to(device=self.execution_device, dtype=self.compute_dtype))
            self._resident_norm = norm
        finally:
            self.store.release(handle)

        # 3. LM Head
        descriptor, handle, _page, tensor = self._lease_tensor("lm_head.weight")
        try:
            if metrics:
                self._record_global(metrics, descriptor, handle)
            self._resident_lm_head = tensor.to(device=self.execution_device, dtype=self.compute_dtype).clone()
        finally:
            self.store.release(handle)

        # 4. Rotary Embedding
        self._resident_rotary = OlmoeRotaryEmbedding(self.config).to(self.execution_device)

        # 5. Pre-build all decoder layers
        self._resident_layers = []
        for layer_idx in range(int(self.config.num_hidden_layers)):
            decoder, backbone_metrics = build_paged_olmoe_decoder_layer(
                self.config,
                self.store,
                self.model_revision,
                layer_idx,
                device=self.execution_device,
                dtype=self.compute_dtype,
                progress=self.progress,
                reset_cuda_peak=False,
                expert_cache=self.expert_cache,
                commit_routing=self.commit_routing,
                commit_threshold=self.commit_threshold,
            )
            if metrics:
                metrics.backbone_page_hits += backbone_metrics.page_hits
                metrics.backbone_page_faults += backbone_metrics.page_faults
                metrics.backbone_remote_bytes += backbone_metrics.remote_bytes
                metrics.backbone_page_bytes += backbone_metrics.resident_bytes
            self._resident_layers.append(decoder)

    def _lease_tensor(self, tensor_name: str):
        page_id = WeightPageID.tensor(self.model_revision, tensor_name)
        descriptor = self.store.resolve(page_id)
        handle = self.store.materialize(page_id, progress=self.progress)
        try:
            page = ExpertPageTensors(handle, descriptor)
            tensor = page.tensor("tensor")
            return descriptor, handle, page, tensor
        except Exception:
            self.store.release(handle)
            raise

    @staticmethod
    def _record_global(metrics: SequentialPassMetrics, descriptor, handle) -> None:
        metrics.global_page_hits += int(handle.cache_hit)
        metrics.global_page_faults += int(not handle.cache_hit)
        metrics.global_remote_bytes += handle.bytes_fetched
        metrics.global_page_bytes += descriptor.expected_bytes

    @staticmethod
    def _sample_rss(metrics: SequentialPassMetrics) -> None:
        metrics.peak_rss_bytes = max(metrics.peak_rss_bytes, _rss_bytes())

    def _embed(self, input_token_id: int, metrics: SequentialPassMetrics) -> torch.Tensor:
        descriptor, handle, _page, tensor = self._lease_tensor("model.embed_tokens.weight")
        try:
            if tensor.ndim != 2 or tensor.shape[1] != int(self.config.hidden_size):
                raise PagedModelError(
                    f"Embedding has shape {list(tensor.shape)}, expected "
                    f"[vocab, {self.config.hidden_size}]"
                )
            if not 0 <= input_token_id < tensor.shape[0]:
                raise PagedModelError(
                    f"Input token {input_token_id} is outside [0, {tensor.shape[0]})"
                )
            self._record_global(metrics, descriptor, handle)
            return (
                tensor[input_token_id]
                .to(device=self.execution_device, dtype=self.compute_dtype)
                .clone()
                .reshape(1, 1, -1)
            )
        finally:
            self.store.release(handle)

    def _final_norm(
        self, hidden_states: torch.Tensor, metrics: SequentialPassMetrics
    ) -> torch.Tensor:
        from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm

        descriptor, handle, _page, tensor = self._lease_tensor("model.norm.weight")
        try:
            if list(tensor.shape) != [int(self.config.hidden_size)]:
                raise PagedModelError(
                    f"Final norm has shape {list(tensor.shape)}, expected "
                    f"[{self.config.hidden_size}]"
                )
            self._record_global(metrics, descriptor, handle)
            norm = OlmoeRMSNorm(
                int(self.config.hidden_size), eps=float(self.config.rms_norm_eps)
            ).to(device=self.execution_device, dtype=self.compute_dtype)
            with torch.no_grad():
                norm.weight.copy_(
                    tensor.to(device=self.execution_device, dtype=self.compute_dtype)
                )
                output = norm(hidden_states)
            del norm
            return output
        finally:
            self.store.release(handle)

    def _lm_head(
        self, hidden_states: torch.Tensor, metrics: SequentialPassMetrics
    ) -> torch.Tensor:
        descriptor, handle, _page, tensor = self._lease_tensor("lm_head.weight")
        try:
            expected = [int(self.config.vocab_size), int(self.config.hidden_size)]
            if list(tensor.shape) != expected:
                raise PagedModelError(
                    f"LM head has shape {list(tensor.shape)}, expected {expected}"
                )
            self._record_global(metrics, descriptor, handle)
            row_bytes = tensor.shape[1] * tensor.element_size()
            rows_per_chunk = max(1, self.head_chunk_bytes // row_bytes)
            chunks = []
            for start in range(0, tensor.shape[0], rows_per_chunk):
                weight = tensor[start : start + rows_per_chunk].to(
                    device=self.execution_device, dtype=self.compute_dtype
                )
                metrics.peak_head_chunk_bytes = max(
                    metrics.peak_head_chunk_bytes,
                    weight.numel() * weight.element_size(),
                )
                chunks.append(F.linear(hidden_states, weight).cpu())
                del weight
            return torch.cat(chunks, dim=-1)
        finally:
            self.store.release(handle)

    def run(
        self,
        input_token_id: int,
        *,
        position_id: int = 0,
        past_key_values=None,
        use_cache: bool = False,
        layer_callback: Optional[Callable[[SequentialLayerMetrics], None]] = None,
    ) -> tuple[torch.Tensor, SequentialPassMetrics]:
        """Execute a no-KV-cache, one-token causal-LM forward pass."""
        from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding

        started = time.perf_counter()
        metrics = SequentialPassMetrics()
        metrics.rss_start_bytes = _rss_bytes()
        metrics.peak_rss_bytes = metrics.rss_start_bytes
        if self.resident_backbone:
            self._init_resident_backbone(metrics)
            with torch.inference_mode():
                hidden_states = (
                    self._resident_embed[input_token_id]
                    .clone()
                    .reshape(1, 1, -1)
                )
                self._sample_rss(metrics)
                position_ids = torch.full(
                    (1, 1),
                    int(position_id),
                    dtype=torch.long,
                    device=self.execution_device,
                )
                cache_position = torch.tensor(
                    [int(position_id)], dtype=torch.long, device=self.execution_device
                )
                position_embeddings = self._resident_rotary(hidden_states, position_ids)

                for layer_idx, decoder in enumerate(self._resident_layers):
                    if layer_callback is not None:
                        layer_started = time.perf_counter()
                    hidden_states = decoder(
                        hidden_states,
                        attention_mask=None,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                    expert_metrics = decoder.mlp.experts.last_metrics
                    metrics.expert_page_hits += expert_metrics.page_hits
                    metrics.expert_page_faults += expert_metrics.page_faults
                    metrics.expert_remote_bytes += expert_metrics.remote_bytes
                    metrics.expert_page_bytes += expert_metrics.page_bytes
                    metrics.experts_executed += expert_metrics.experts_executed
                    metrics.peak_projection_bytes = max(
                        metrics.peak_projection_bytes, expert_metrics.peak_projection_bytes
                    )
                    if layer_callback is not None:
                        selected = sorted(
                            int(value)
                            for value in torch.unique(decoder.mlp.last_top_k_index).tolist()
                        )
                        layer_metrics = SequentialLayerMetrics(
                            layer=layer_idx,
                            selected_experts=selected,
                            backbone=BackboneLoadMetrics(),
                            experts=expert_metrics,
                            elapsed_seconds=time.perf_counter() - layer_started,
                            output_sha256="",
                            output_max_abs=float(hidden_states.float().abs().max().item()),
                        )
                        metrics.layers.append(layer_metrics)
                        layer_callback(layer_metrics)

                normed = self._resident_norm(hidden_states)
                logits = F.linear(normed, self._resident_lm_head)

            if self.execution_device.type == "cuda":
                metrics.peak_cuda_bytes = int(
                    torch.cuda.max_memory_allocated(self.execution_device)
                )
            metrics.rss_end_bytes = _rss_bytes()
            metrics.peak_rss_bytes = max(metrics.peak_rss_bytes, metrics.rss_end_bytes)
            metrics.elapsed_seconds = time.perf_counter() - started
            metrics.kv_cache_bytes = _kv_cache_bytes(past_key_values)
            if past_key_values is not None:
                metrics.cache_sequence_length = int(past_key_values.get_seq_length())
            return logits, metrics

        with torch.inference_mode():
            hidden_states = self._embed(input_token_id, metrics)
            self._sample_rss(metrics)
            position_ids = torch.full(
                (1, 1),
                int(position_id),
                dtype=torch.long,
                device=self.execution_device,
            )
            cache_position = torch.tensor(
                [int(position_id)], dtype=torch.long, device=self.execution_device
            )
            rotary = OlmoeRotaryEmbedding(self.config).to(self.execution_device)
            position_embeddings = rotary(hidden_states, position_ids)

            for layer_idx in range(int(self.config.num_hidden_layers)):
                layer_started = time.perf_counter()
                decoder, backbone = build_paged_olmoe_decoder_layer(
                    self.config,
                    self.store,
                    self.model_revision,
                    layer_idx,
                    device=self.execution_device,
                    dtype=self.compute_dtype,
                    progress=self.progress,
                    reset_cuda_peak=False,
                )
                if self.expert_backend == "transformers":
                    from pockettitan.runtime.hf.olmoe_reference import (
                        TransformersOlmoeExpertsFromPages,
                    )

                    decoder.mlp.experts = TransformersOlmoeExpertsFromPages(
                        self.config,
                        self.store,
                        self.model_revision,
                        layer_idx,
                        execution_device=self.execution_device,
                        compute_dtype=self.compute_dtype,
                        progress=self.progress,
                    )
                hidden_states = decoder(
                    hidden_states,
                    attention_mask=None,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
                expert_metrics = decoder.mlp.experts.last_metrics
                selected = sorted(
                    int(value)
                    for value in torch.unique(decoder.mlp.last_top_k_index).tolist()
                )
                layer_metrics = SequentialLayerMetrics(
                    layer=layer_idx,
                    selected_experts=selected,
                    backbone=backbone,
                    experts=expert_metrics,
                    elapsed_seconds=time.perf_counter() - layer_started,
                    output_sha256=hashlib.sha256(
                        hidden_states.detach()
                        .cpu()
                        .contiguous()
                        .view(torch.uint16)
                        .numpy()
                        .tobytes()
                    ).hexdigest(),
                    output_max_abs=float(hidden_states.float().abs().max().item()),
                )
                metrics.layers.append(layer_metrics)
                metrics.backbone_page_hits += backbone.page_hits
                metrics.backbone_page_faults += backbone.page_faults
                metrics.backbone_remote_bytes += backbone.remote_bytes
                metrics.backbone_page_bytes += backbone.resident_bytes
                metrics.expert_page_hits += expert_metrics.page_hits
                metrics.expert_page_faults += expert_metrics.page_faults
                metrics.expert_remote_bytes += expert_metrics.remote_bytes
                metrics.expert_page_bytes += expert_metrics.page_bytes
                metrics.experts_executed += expert_metrics.experts_executed
                metrics.peak_projection_bytes = max(
                    metrics.peak_projection_bytes, expert_metrics.peak_projection_bytes
                )
                if layer_callback is not None:
                    layer_callback(layer_metrics)
                del decoder
                gc.collect()
                if self.execution_device.type == "cuda":
                    torch.cuda.synchronize(self.execution_device)
                    torch.cuda.empty_cache()
                self._sample_rss(metrics)

            hidden_states = self._final_norm(hidden_states, metrics)
            logits = self._lm_head(hidden_states, metrics)
            self._sample_rss(metrics)

        if not torch.isfinite(logits).all():
            raise PagedModelError("Full-model logits contain NaN or infinity")
        if self.execution_device.type == "cuda":
            torch.cuda.synchronize(self.execution_device)
            metrics.peak_cuda_bytes = int(
                torch.cuda.max_memory_allocated(self.execution_device)
            )
        metrics.rss_end_bytes = _rss_bytes()
        metrics.peak_rss_bytes = max(metrics.peak_rss_bytes, metrics.rss_end_bytes)
        metrics.elapsed_seconds = time.perf_counter() - started
        flattened = logits.reshape(-1)
        metrics.top_token_id = int(flattened.argmax().item())
        metrics.top_logit = float(flattened[metrics.top_token_id].float().item())
        raw_logits = logits.contiguous().view(torch.uint16).numpy().tobytes()
        metrics.logits_sha256 = hashlib.sha256(raw_logits).hexdigest()
        metrics.kv_cache_bytes = _kv_cache_bytes(past_key_values)
        if past_key_values is not None:
            metrics.cache_sequence_length = int(past_key_values.get_seq_length())
        return logits, metrics
