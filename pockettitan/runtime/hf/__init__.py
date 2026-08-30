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

__all__ = [
    "DEFAULT_HEAD_CHUNK",
    "GenerationResult",
    "LoaderError",
    "NAME_ALIASES",
    "PackageWeights",
    "PackagedEmbedding",
    "PackagedLinear",
    "WeightNotFound",
    "build_causal_lm",
    "generate",
    "load_tokenizer",
    "resolve_device",
    "load_text_config",
    "parameters_on_meta",
    "summarize",
]
