"""Independent Transformers expert oracle assembled from selected pages."""

from __future__ import annotations

import copy
import time
from typing import Optional

import torch
from torch import nn

from pockettitan.domainslice import ModelRevision, WeightPageID, WeightStore
from pockettitan.domainslice.types import ProgressCallback
from pockettitan.runtime.hf.loader import parameters_on_meta
from pockettitan.runtime.hf.olmoe_paged import (
    ExpertPageTensors,
    PagedExpertError,
    PagedExpertMetrics,
)


class TransformersOlmoeExpertsFromPages(nn.Module):
    """Call upstream ``OlmoeExperts`` with a compact bank of routed experts.

    This deliberately does not reuse PocketTitan's expert arithmetic. It builds
    the official Transformers expert module for the experts selected by the
    router, remaps their IDs into a compact bank, and uses the upstream forward
    method as an independent numerical oracle.
    """

    def __init__(
        self,
        config,
        store: WeightStore,
        model_revision: ModelRevision,
        layer_idx: int,
        *,
        execution_device: str | torch.device = "cpu",
        compute_dtype: torch.dtype = torch.bfloat16,
        progress: Optional[ProgressCallback] = None,
    ):
        super().__init__()
        self.config = config
        self.store = store
        self.model_revision = model_revision
        self.layer_idx = int(layer_idx)
        self.execution_device = torch.device(execution_device)
        self.compute_dtype = compute_dtype
        self.progress = progress
        self.last_metrics = PagedExpertMetrics()

    def _build_reference(self, selected: list[int], metrics: PagedExpertMetrics):
        from transformers.models.olmoe.modeling_olmoe import OlmoeExperts

        compact_config = copy.deepcopy(self.config)
        compact_config.num_experts = len(selected)
        compact_config.num_experts_per_tok = min(
            int(self.config.num_experts_per_tok), len(selected)
        )
        with parameters_on_meta():
            reference = OlmoeExperts(compact_config)

        hidden = int(self.config.hidden_size)
        intermediate = int(self.config.intermediate_size)
        gate_up = torch.empty(
            (len(selected), 2 * intermediate, hidden),
            device=self.execution_device,
            dtype=self.compute_dtype,
        )
        down = torch.empty(
            (len(selected), hidden, intermediate),
            device=self.execution_device,
            dtype=self.compute_dtype,
        )
        for slot, expert_id in enumerate(selected):
            page_id = WeightPageID.expert(
                self.model_revision,
                self.layer_idx,
                expert_id,
            )
            descriptor = self.store.resolve(page_id)
            handle = self.store.materialize(page_id, progress=self.progress)
            try:
                page = ExpertPageTensors(handle, descriptor)
                gate = page.tensor("gate_proj").to(
                    device=self.execution_device, dtype=self.compute_dtype
                )
                up = page.tensor("up_proj").to(
                    device=self.execution_device, dtype=self.compute_dtype
                )
                down_weight = page.tensor("down_proj").to(
                    device=self.execution_device, dtype=self.compute_dtype
                )
                gate_up[slot, :intermediate].copy_(gate)
                gate_up[slot, intermediate:].copy_(up)
                down[slot].copy_(down_weight)
                metrics.page_hits += int(handle.cache_hit)
                metrics.page_faults += int(not handle.cache_hit)
                metrics.remote_bytes += handle.bytes_fetched
                metrics.resumed_bytes += handle.bytes_resumed
                metrics.page_bytes += handle.size_bytes
                metrics.peak_projection_bytes = max(
                    metrics.peak_projection_bytes,
                    gate.numel() * gate.element_size(),
                    up.numel() * up.element_size(),
                    down_weight.numel() * down_weight.element_size(),
                )
            finally:
                self.store.release(handle)

        reference.register_parameter(
            "gate_up_proj", nn.Parameter(gate_up, requires_grad=False)
        )
        reference.register_parameter("down_proj", nn.Parameter(down, requires_grad=False))
        reference.eval()
        return reference

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        started = time.perf_counter()
        metrics = PagedExpertMetrics()
        selected = sorted(int(value) for value in torch.unique(top_k_index).tolist())
        if not selected:
            raise PagedExpertError("The router selected no experts")
        reference = self._build_reference(selected, metrics)
        remapped = top_k_index.clone()
        for compact_id, source_id in enumerate(selected):
            remapped[top_k_index == source_id] = compact_id
        output = reference(
            hidden_states.to(self.execution_device),
            remapped.to(self.execution_device),
            top_k_weights.to(self.execution_device),
        )
        metrics.experts_executed = len(selected)
        if self.execution_device.type == "cuda":
            torch.cuda.synchronize(self.execution_device)
            metrics.peak_cuda_bytes = int(
                torch.cuda.max_memory_allocated(self.execution_device)
            )
        metrics.elapsed_seconds = time.perf_counter() - started
        self.last_metrics = metrics
        return output.to(device=hidden_states.device, dtype=hidden_states.dtype)
