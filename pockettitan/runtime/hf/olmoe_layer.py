"""Assemble one complete OLMoE decoder layer from DomainSlice pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from pockettitan.domainslice import ModelRevision, WeightPageID, WeightStore
from pockettitan.runtime.hf.loader import parameters_on_meta
from pockettitan.runtime.hf.olmoe_paged import ExpertPageTensors, replace_olmoe_sparse_moe
from pockettitan.domainslice.types import ProgressCallback


class PagedLayerError(RuntimeError):
    """A complete decoder layer could not be backed by immutable pages."""


@dataclass
class BackboneLoadMetrics:
    tensors: int = 0
    page_hits: int = 0
    page_faults: int = 0
    remote_bytes: int = 0
    resident_bytes: int = 0


def _assign_parameter(module: nn.Module, qualified: str, value: torch.Tensor) -> None:
    parent_path, _, leaf = qualified.rpartition(".")
    parent = module.get_submodule(parent_path) if parent_path else module
    existing = getattr(parent, leaf)
    parent.register_parameter(
        leaf,
        nn.Parameter(value.detach(), requires_grad=existing.requires_grad),
    )


def load_raw_tensor_page(
    store: WeightStore,
    model_revision: ModelRevision,
    tensor_name: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    progress: Optional[ProgressCallback] = None,
) -> tuple[torch.Tensor, int, bool]:
    """Materialize and clone one immutable tensor page into resident memory."""
    page_id = WeightPageID.tensor(model_revision, tensor_name)
    descriptor = store.resolve(page_id)
    handle = store.materialize(page_id, progress=progress)
    try:
        page = ExpertPageTensors(handle, descriptor)
        value = page.tensor("tensor").to(device=device, dtype=dtype).clone()
        return value, handle.bytes_fetched, handle.cache_hit
    finally:
        store.release(handle)


def build_paged_olmoe_decoder_layer(
    config,
    store: WeightStore,
    model_revision: ModelRevision,
    layer_idx: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    progress: Optional[ProgressCallback] = None,
    reset_cuda_peak: bool = True,
    expert_cache: Optional[Any] = None,
    commit_routing: bool = False,
    commit_threshold: float = 0.15,
):
    """Build a Transformers decoder layer with resident backbone and paged experts."""
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

    execution_device = torch.device(device)
    with parameters_on_meta():
        layer = OlmoeDecoderLayer(config, layer_idx)
    layer.mlp = replace_olmoe_sparse_moe(
        layer.mlp,
        store,
        model_revision,
        layer_idx,
        intermediate_dim=int(config.intermediate_size),
        execution_device=execution_device,
        compute_dtype=dtype,
        hidden_act=str(config.hidden_act),
        progress=progress,
        reset_cuda_peak=reset_cuda_peak,
        expert_cache=expert_cache,
        commit_routing=commit_routing,
        commit_threshold=commit_threshold,
    )

    metrics = BackboneLoadMetrics()
    source_names: Dict[str, str] = {
        name: f"model.layers.{layer_idx}.{name}" for name, _parameter in layer.named_parameters()
    }
    for name, source_name in source_names.items():
        value, remote_bytes, cache_hit = load_raw_tensor_page(
            store,
            model_revision,
            source_name,
            device=execution_device,
            dtype=dtype,
            progress=progress,
        )
        expected = dict(layer.named_parameters())[name]
        if list(value.shape) != list(expected.shape):
            raise PagedLayerError(
                f"{source_name} has shape {list(value.shape)}, expected {list(expected.shape)}"
            )
        _assign_parameter(layer, name, value)
        metrics.tensors += 1
        metrics.page_hits += int(cache_hit)
        metrics.page_faults += int(not cache_hit)
        metrics.remote_bytes += remote_bytes
        metrics.resident_bytes += value.numel() * value.element_size()

    meta = [name for name, parameter in layer.named_parameters() if parameter.device.type == "meta"]
    if meta:
        raise PagedLayerError(f"Decoder layer still has meta parameters: {meta}")
    layer.eval()
    return layer, metrics
