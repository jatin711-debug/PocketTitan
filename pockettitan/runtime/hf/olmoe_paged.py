"""OLMoE expert execution backed by DomainSlice pages.

The router and decoder stay in the upstream Transformers module tree.  Only the
expert collection is replaced: each selected expert is leased from the local
page cache, its three BF16 projections are memory-mapped, and one projection at
a time is staged to the execution device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pockettitan.domainslice import (
    ModelRevision,
    PageDescriptor,
    PageHandle,
    WeightPageID,
    WeightStore,
)
from pockettitan.domainslice.types import ProgressCallback


class PagedExpertError(RuntimeError):
    """A cached page cannot execute as the requested OLMoE expert."""


class ExpertPageTensors:
    """Typed zero-copy views over one raw-BF16 expert page."""

    def __init__(self, handle: PageHandle, descriptor: PageDescriptor):
        if handle.page_id != descriptor.page_id:
            raise PagedExpertError("Page handle and descriptor identities differ")
        if handle.path.stat().st_size != descriptor.output_layout.stride:
            raise PagedExpertError(
                f"Expert page is {handle.path.stat().st_size:,} bytes; expected "
                f"{descriptor.output_layout.stride:,}"
            )
        self.handle = handle
        self.descriptor = descriptor
        self._arrays: Dict[str, np.memmap] = {}
        self._tensors: Dict[str, torch.Tensor] = {}

    def tensor(self, projection: str) -> torch.Tensor:
        cached = self._tensors.get(projection)
        if cached is not None:
            return cached
        layout = self.descriptor.output_layout.projection(projection)
        if layout is None:
            raise PagedExpertError(f"Expert page has no {projection!r} projection")
        if layout.bits != 16:
            raise PagedExpertError(
                f"Raw OLMoE execution requires 16-bit pages, got {layout.bits} bits"
            )
        if layout.offset % 2:
            raise PagedExpertError(f"BF16 projection {projection!r} has an unaligned offset")
        array = np.memmap(
            self.handle.path,
            mode="c",
            dtype=np.uint16,
            offset=layout.offset,
            shape=(layout.num_elements,),
        )
        tensor = torch.from_numpy(array).view(torch.bfloat16).reshape(layout.shape)
        self._arrays[projection] = array
        self._tensors[projection] = tensor
        return tensor


@dataclass
class PagedExpertMetrics:
    experts_executed: int = 0
    page_hits: int = 0
    page_faults: int = 0
    remote_bytes: int = 0
    resumed_bytes: int = 0
    page_bytes: int = 0
    peak_projection_bytes: int = 0
    peak_cuda_bytes: int = 0
    elapsed_seconds: float = 0.0


class PagedOlmoeExperts(nn.Module):
    """Drop-in expert collection for an OLMoE sparse-MoE block."""

    def __init__(
        self,
        store: WeightStore,
        model_revision: ModelRevision,
        layer_idx: int,
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        execution_device: str | torch.device = "cpu",
        compute_dtype: torch.dtype = torch.bfloat16,
        hidden_act: str = "silu",
        progress: Optional[ProgressCallback] = None,
        reset_cuda_peak: bool = True,
        expert_cache: Optional[Any] = None,
    ):
        super().__init__()
        if hidden_act not in {"silu", "swish"}:
            raise PagedExpertError(
                f"OLMoE V0 supports silu/swish experts, got hidden_act={hidden_act!r}"
            )
        self.store = store
        self.model_revision = model_revision
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.execution_device = torch.device(execution_device)
        self.compute_dtype = compute_dtype
        self.progress = progress
        self.reset_cuda_peak = reset_cuda_peak
        self.expert_cache = expert_cache
        self.last_metrics = PagedExpertMetrics()
        if self.execution_device.type == "cuda" and not torch.cuda.is_available():
            raise PagedExpertError("CUDA expert execution requested but CUDA is unavailable")

    def _stage(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == self.compute_dtype and tensor.device == self.execution_device:
            return tensor
        return tensor.to(device=self.execution_device, dtype=self.compute_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.hidden_dim:
            raise PagedExpertError(
                f"Expected flattened hidden states [tokens, {self.hidden_dim}], got "
                f"{list(hidden_states.shape)}"
            )
        if top_k_index.shape != top_k_weights.shape:
            raise PagedExpertError("top-k indices and weights must have identical shapes")
        if top_k_index.shape[0] != hidden_states.shape[0]:
            raise PagedExpertError("Routing token count does not match hidden states")

        started = time.perf_counter()
        metrics = PagedExpertMetrics()
        if self.execution_device.type == "cuda" and self.reset_cuda_peak:
            torch.cuda.reset_peak_memory_stats(self.execution_device)

        if hidden_states.shape[0] == 1:
            current = hidden_states.to(device=self.execution_device, dtype=self.compute_dtype)
            final_hidden_states = torch.zeros_like(current)
            selected_experts = top_k_index[0].tolist()
            route_weights = top_k_weights[0]

            if self.expert_cache is not None and hasattr(self.expert_cache, "prefetch_batch"):
                self.expert_cache.prefetch_batch(
                    self.model_revision,
                    self.layer_idx,
                    selected_experts,
                    target_device=self.execution_device,
                    dtype=self.compute_dtype,
                )

            for k, expert_id in enumerate(selected_experts):
                if not 0 <= expert_id < self.num_experts:
                    raise PagedExpertError(
                        f"Router selected expert {expert_id}, outside [0, {self.num_experts})"
                    )
                w_k = route_weights[k].to(device=self.execution_device, dtype=self.compute_dtype)
                if self.expert_cache is not None:
                    gate_up_weight, down_weight, tier = self.expert_cache.get_or_load(
                        self.model_revision,
                        self.layer_idx,
                        expert_id,
                        self.store,
                        target_device=self.execution_device,
                        dtype=self.compute_dtype,
                        progress=self.progress,
                    )
                    metrics.peak_projection_bytes = max(
                        metrics.peak_projection_bytes,
                        gate_up_weight.numel() * gate_up_weight.element_size(),
                        down_weight.numel() * down_weight.element_size(),
                    )
                    gate, up = F.linear(current, gate_up_weight).chunk(2, dim=-1)
                    intermediate = F.silu(gate) * up
                    del gate, up
                    output = F.linear(intermediate, down_weight)
                    del intermediate
                    final_hidden_states.add_(output * w_k)
                    metrics.experts_executed += 1
                    if tier == "disk":
                        metrics.page_faults += 1
                    else:
                        metrics.page_hits += 1
                    continue

                page_id = WeightPageID.expert(self.model_revision, self.layer_idx, expert_id)
                descriptor = self.store.resolve(page_id)
                handle = self.store.materialize(page_id, progress=self.progress)
                try:
                    page = ExpertPageTensors(handle, descriptor)
                    expected_shapes = {
                        "gate_proj": [self.intermediate_dim, self.hidden_dim],
                        "up_proj": [self.intermediate_dim, self.hidden_dim],
                        "down_proj": [self.hidden_dim, self.intermediate_dim],
                    }
                    for projection, expected_shape in expected_shapes.items():
                        layout = descriptor.output_layout.projection(projection)
                        if layout is None or layout.shape != expected_shape:
                            actual = None if layout is None else layout.shape
                            raise PagedExpertError(
                                f"Expert {expert_id} {projection} shape is {actual}; "
                                f"expected {expected_shape}"
                            )
                    gate = self._stage(page.tensor("gate_proj"))
                    up = self._stage(page.tensor("up_proj"))
                    down = self._stage(page.tensor("down_proj"))
                    metrics.page_bytes += handle.size_bytes
                    metrics.remote_bytes += handle.bytes_fetched
                    metrics.resumed_bytes += handle.bytes_resumed
                    metrics.page_hits += int(handle.cache_hit)
                    metrics.page_faults += int(not handle.cache_hit)
                    metrics.peak_projection_bytes = max(
                        metrics.peak_projection_bytes,
                        gate.numel() * gate.element_size(),
                        up.numel() * up.element_size(),
                        down.numel() * down.element_size(),
                    )
                    intermediate = F.silu(F.linear(current, gate)) * F.linear(current, up)
                    del gate, up
                    output = F.linear(intermediate, down)
                    del intermediate
                    final_hidden_states.add_(output * w_k)
                    del down
                    metrics.experts_executed += 1
                finally:
                    self.store.release(handle)

            metrics.duration_seconds = time.perf_counter() - started
            if self.execution_device.type == "cuda":
                metrics.peak_cuda_bytes = int(
                    torch.cuda.max_memory_allocated(self.execution_device)
                )
            self.last_metrics = metrics
            return final_hidden_states

        final_hidden_states = torch.zeros_like(hidden_states)
        expert_ids = sorted(int(value) for value in torch.unique(top_k_index).tolist())
        for expert_id in expert_ids:
            if not 0 <= expert_id < self.num_experts:
                raise PagedExpertError(
                    f"Router selected expert {expert_id}, outside [0, {self.num_experts})"
                )
            locations = torch.nonzero(top_k_index == expert_id, as_tuple=False)
            if not len(locations):
                continue
            token_idx = locations[:, 0]
            top_k_pos = locations[:, 1]
            current = hidden_states.index_select(0, token_idx).to(
                device=self.execution_device, dtype=self.compute_dtype
            )

            if self.expert_cache is not None:
                gate_up_weight, down_weight, tier = self.expert_cache.get_or_load(
                    self.model_revision,
                    self.layer_idx,
                    expert_id,
                    self.store,
                    target_device=self.execution_device,
                    dtype=self.compute_dtype,
                    progress=self.progress,
                )
                metrics.peak_projection_bytes = max(
                    metrics.peak_projection_bytes,
                    gate_up_weight.numel() * gate_up_weight.element_size(),
                    down_weight.numel() * down_weight.element_size(),
                )
                gate, up = F.linear(current, gate_up_weight).chunk(2, dim=-1)
                intermediate = F.silu(gate) * up
                del gate, up
                output = F.linear(intermediate, down_weight)
                del intermediate

                route_weights = top_k_weights[token_idx, top_k_pos].to(
                    device=self.execution_device, dtype=output.dtype
                )
                output = output * route_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0,
                    token_idx.to(final_hidden_states.device),
                    output.to(device=final_hidden_states.device, dtype=final_hidden_states.dtype),
                )
                metrics.experts_executed += 1
                if tier == "disk":
                    metrics.page_faults += 1
                else:
                    metrics.page_hits += 1
                continue

            page_id = WeightPageID.expert(
                self.model_revision,
                self.layer_idx,
                expert_id,
            )
            descriptor = self.store.resolve(page_id)
            handle = self.store.materialize(page_id, progress=self.progress)
            try:
                page = ExpertPageTensors(handle, descriptor)
                expected_shapes = {
                    "gate_proj": [self.intermediate_dim, self.hidden_dim],
                    "up_proj": [self.intermediate_dim, self.hidden_dim],
                    "down_proj": [self.hidden_dim, self.intermediate_dim],
                }
                for projection, expected_shape in expected_shapes.items():
                    layout = descriptor.output_layout.projection(projection)
                    if layout is None or layout.shape != expected_shape:
                        actual = None if layout is None else layout.shape
                        raise PagedExpertError(
                            f"Expert {expert_id} {projection} shape is {actual}; "
                            f"expected {expected_shape}"
                        )

                gate_up_weight = torch.empty(
                    (2 * self.intermediate_dim, self.hidden_dim),
                    device=self.execution_device,
                    dtype=self.compute_dtype,
                )
                gate_weight = self._stage(page.tensor("gate_proj"))
                gate_up_weight[: self.intermediate_dim].copy_(gate_weight)
                metrics.peak_projection_bytes = max(
                    metrics.peak_projection_bytes,
                    gate_up_weight.numel() * gate_up_weight.element_size(),
                )
                del gate_weight

                up_weight = self._stage(page.tensor("up_proj"))
                gate_up_weight[self.intermediate_dim :].copy_(up_weight)
                del up_weight

                gate, up = F.linear(current, gate_up_weight).chunk(2, dim=-1)
                del gate_up_weight
                intermediate = F.silu(gate) * up
                del gate, up
                down_weight = self._stage(page.tensor("down_proj"))
                metrics.peak_projection_bytes = max(
                    metrics.peak_projection_bytes,
                    down_weight.numel() * down_weight.element_size(),
                )
                output = F.linear(intermediate, down_weight)
                del down_weight, intermediate

                route_weights = top_k_weights[token_idx, top_k_pos].to(
                    device=self.execution_device, dtype=output.dtype
                )
                output = output * route_weights.unsqueeze(-1)
                final_hidden_states.index_add_(
                    0,
                    token_idx.to(final_hidden_states.device),
                    output.to(device=final_hidden_states.device, dtype=final_hidden_states.dtype),
                )
                metrics.experts_executed += 1
                metrics.page_hits += int(handle.cache_hit)
                metrics.page_faults += int(not handle.cache_hit)
                metrics.remote_bytes += handle.bytes_fetched
                metrics.resumed_bytes += handle.bytes_resumed
                metrics.page_bytes += handle.size_bytes
            finally:
                self.store.release(handle)

        if self.execution_device.type == "cuda":
            torch.cuda.synchronize(self.execution_device)
            metrics.peak_cuda_bytes = int(torch.cuda.max_memory_allocated(self.execution_device))
        metrics.elapsed_seconds = time.perf_counter() - started
        self.last_metrics = metrics
        return final_hidden_states


class PagedOlmoeSparseMoeBlock(nn.Module):
    """Upstream OLMoE router plus PocketTitan's page-backed experts with Commit Routing."""

    def __init__(
        self,
        gate: nn.Module,
        experts: PagedOlmoeExperts,
        commit_routing: bool = False,
        commit_threshold: float = 0.15,
    ):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.commit_routing = bool(commit_routing)
        self.commit_threshold = float(commit_threshold)
        self.last_top_k_index: Optional[torch.Tensor] = None
        self.last_top_k_weights: Optional[torch.Tensor] = None

    def _apply_commit_routing(
        self,
        router_logits: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CommitMoE: Vectorized GPU substitution without CPU-GPU synchronization stalls."""
        cache = self.experts.expert_cache
        layer_idx = self.experts.layer_idx
        vram_resident = [
            exp_id for (l_id, exp_id) in cache._vram_cache.keys() if l_id == layer_idx
        ]
        if not vram_resident:
            return top_k_index, top_k_weights

        device = router_logits.device
        vram_tensor = torch.tensor(vram_resident, device=device, dtype=torch.long)

        # Boolean mask of resident experts [num_experts]
        vram_mask = torch.zeros(self.experts.num_experts, device=device, dtype=torch.bool)
        vram_mask.scatter_(0, vram_tensor, True)

        # Mask resident experts already picked in top-k [num_tokens, num_experts]
        num_tokens, _k = top_k_index.shape
        chosen_mask = torch.zeros(num_tokens, self.experts.num_experts, device=device, dtype=torch.bool)
        chosen_mask.scatter_(1, top_k_index, True)

        available_vram = vram_mask.unsqueeze(0) & (~chosen_mask)
        if not available_vram.any():
            return top_k_index, top_k_weights

        # Best available resident candidate logit per token
        candidate_logits = torch.where(available_vram, router_logits, -1e9)
        cand_vals, cand_indices = torch.topk(candidate_logits, k=1, dim=-1)

        # Weakest selected expert in top-k
        weakest_pos = torch.argmin(top_k_weights, dim=-1, keepdim=True)
        weakest_expert = torch.gather(top_k_index, 1, weakest_pos)
        weakest_logit = torch.gather(router_logits, 1, weakest_expert)
        weakest_is_vram = vram_mask[weakest_expert]

        # Only substitute if weakest is non-resident and logit delta <= threshold
        should_swap = (~weakest_is_vram) & ((weakest_logit - cand_vals) <= self.commit_threshold)
        if not should_swap.any():
            return top_k_index, top_k_weights

        new_top_k_index = top_k_index.clone()
        new_top_k_index.scatter_(1, weakest_pos, torch.where(should_swap, cand_indices, weakest_expert))
        selected_logits = torch.gather(router_logits, 1, new_top_k_index)
        new_top_k_weights = torch.softmax(selected_logits, dim=-1)
        return new_top_k_index, new_top_k_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden_dim)
        router_logits, top_k_weights, top_k_index = self.gate(flattened)
        expert_cache = getattr(self.experts, "expert_cache", None)
        if (
            self.commit_routing
            and expert_cache is not None
            and hasattr(expert_cache, "_vram_cache")
        ):
            top_k_index, top_k_weights = self._apply_commit_routing(
                router_logits, top_k_index, top_k_weights
            )
        self.last_top_k_index = top_k_index.detach()
        self.last_top_k_weights = top_k_weights.detach()

        # APEX Inter-Layer Lookahead: Asynchronously prefetch Layer L+1 candidates on background CUDA stream
        layer_idx = getattr(self.experts, "layer_idx", None)
        if (
            expert_cache is not None
            and hasattr(expert_cache, "prefetch_batch")
            and layer_idx is not None
            and layer_idx < 15
        ):
            next_layer = layer_idx + 1
            predicted_candidates = top_k_index[0].tolist()
            expert_cache.prefetch_batch(
                self.experts.model_revision,
                next_layer,
                predicted_candidates,
                target_device=self.experts.execution_device,
                dtype=self.experts.compute_dtype,
            )

        return self.experts(flattened, top_k_index, top_k_weights).reshape(
            batch_size, sequence_length, hidden_dim
        )


def replace_olmoe_sparse_moe(
    source_block: nn.Module,
    store: WeightStore,
    model_revision: ModelRevision,
    layer_idx: int,
    *,
    intermediate_dim: int,
    execution_device: str | torch.device = "cpu",
    compute_dtype: torch.dtype = torch.bfloat16,
    hidden_act: str = "silu",
    progress: Optional[ProgressCallback] = None,
    reset_cuda_peak: bool = True,
    expert_cache: Optional[Any] = None,
    commit_routing: bool = False,
    commit_threshold: float = 0.15,
) -> PagedOlmoeSparseMoeBlock:
    """Create a paged block while preserving the source block's exact router."""
    gate = getattr(source_block, "gate", None)
    if gate is None:
        raise PagedExpertError("Source OLMoE block has no gate module")
    return PagedOlmoeSparseMoeBlock(
        gate=gate,
        experts=PagedOlmoeExperts(
            store,
            model_revision,
            layer_idx,
            num_experts=int(gate.num_experts),
            hidden_dim=int(gate.hidden_dim),
            intermediate_dim=intermediate_dim,
            execution_device=execution_device,
            compute_dtype=compute_dtype,
            hidden_act=hidden_act,
            progress=progress,
            reset_cuda_peak=reset_cuda_peak,
            expert_cache=expert_cache,
        ),
        commit_routing=commit_routing,
        commit_threshold=commit_threshold,
    )
