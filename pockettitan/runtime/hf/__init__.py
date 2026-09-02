"""Run a ``.ptitan`` package through the reference Hugging Face module tree."""

from pockettitan.runtime.hf.generate import (
    GenerationResult,
    generate,
    load_tokenizer,
    resolve_device,
)
from pockettitan.runtime.hf.loader import (
    DEFAULT_HEAD_CHUNK,
    LoaderError,
    build_causal_lm,
    load_text_config,
    parameters_on_meta,
    summarize,
)
from pockettitan.runtime.hf.weights import (
    NAME_ALIASES,
    PackagedEmbedding,
    PackagedLinear,
    PackageWeights,
    WeightNotFound,
)
from pockettitan.runtime.hf.olmoe_paged import (
    ExpertPageTensors,
    PagedExpertError,
    PagedExpertMetrics,
    PagedOlmoeExperts,
    PagedOlmoeSparseMoeBlock,
    replace_olmoe_sparse_moe,
)
from pockettitan.runtime.hf.olmoe_layer import (
    BackboneLoadMetrics,
    PagedLayerError,
    build_paged_olmoe_decoder_layer,
    load_raw_tensor_page,
)
from pockettitan.runtime.hf.olmoe_model import (
    PagedModelError,
    PagedOlmoeOneTokenRunner,
    SequentialLayerMetrics,
    SequentialPassMetrics,
)
from pockettitan.runtime.hf.olmoe_reference import TransformersOlmoeExpertsFromPages

__all__ = [
    "DEFAULT_HEAD_CHUNK",
    "GenerationResult",
    "LoaderError",
    "NAME_ALIASES",
    "PackageWeights",
    "PackagedEmbedding",
    "PackagedLinear",
    "WeightNotFound",
    "ExpertPageTensors",
    "PagedExpertError",
    "PagedExpertMetrics",
    "PagedOlmoeExperts",
    "PagedOlmoeSparseMoeBlock",
    "replace_olmoe_sparse_moe",
    "BackboneLoadMetrics",
    "PagedLayerError",
    "build_paged_olmoe_decoder_layer",
    "load_raw_tensor_page",
    "PagedModelError",
    "PagedOlmoeOneTokenRunner",
    "SequentialLayerMetrics",
    "SequentialPassMetrics",
    "TransformersOlmoeExpertsFromPages",
    "build_causal_lm",
    "generate",
    "load_tokenizer",
    "resolve_device",
    "load_text_config",
    "parameters_on_meta",
    "summarize",
]
