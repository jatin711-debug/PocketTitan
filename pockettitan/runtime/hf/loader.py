"""Build a runnable Hugging Face model whose weights come from a ``.ptitan`` package.

The forward pass is not reimplemented here. ``transformers`` already contains a
correct ``Qwen3_5`` — the Gated DeltaNet recurrence, the sparse full attention,
the interleaved mRoPE, the mask construction and the KV cache — and reproducing
that from the paper is weeks of work with many places to be subtly wrong. This
module keeps that module tree exactly as it is and replaces only the *storage*:
every large parameter becomes a :class:`PackagedLinear` or
:class:`PackagedEmbedding` that decodes itself out of the package when used.

Two consequences worth stating plainly:

* It gives an exact reference. The same class, weights and prompt run through
  ``from_pretrained`` produce logits that ours must match, so quantization error
  becomes measurable instead of a matter of opinion.
* Nothing large is ever resident, so a 27B package runs in a few hundred MB.

The single most important property here is **coverage**: if a parameter name
fails to resolve, the module would silently keep whatever the skeleton was
initialized with and produce fluent-looking nonsense. :func:`build_causal_lm`
therefore refuses to return a model with any unbacked parameter.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from pockettitan.config import QuantMethod
from pockettitan.runtime.hf.weights import (
    PackagedEmbedding,
    PackagedLinear,
    PackageWeights,
)

# Decoding lm_head in row chunks keeps the widest matrix in the model from ever
# being fully resident. 32768 rows of 5120 is ~320 MB at fp16.
DEFAULT_HEAD_CHUNK = 32_768


class LoaderError(RuntimeError):
    """The package and the model architecture do not line up."""


@contextmanager
def parameters_on_meta():
    """Allocate parameters on ``meta`` while leaving buffers real.

    ``torch.device("meta")`` would also fake the buffers, and the rotary
    embedding's ``inv_freq`` is a buffer: a model built that way runs, produces
    numbers, and is wrong. Buffers are tiny, so they are simply built for real.
    """
    original = nn.Module.register_parameter

    def register_parameter(self, name, param):
        if param is not None and param.device.type != "meta":
            param = nn.Parameter(param.data.to("meta"), requires_grad=param.requires_grad)
        original(self, name, param)

    nn.Module.register_parameter = register_parameter
    try:
        yield
    finally:
        nn.Module.register_parameter = original


def load_text_config(package_dir: Union[str, Path]):
    """The language-tower config recorded in the package."""
    from transformers import AutoConfig

    metadata = Path(package_dir) / "metadata"
    config_path = metadata / "config.json"
    if not config_path.exists():
        raise LoaderError(f"package has no metadata/config.json at {config_path}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = AutoConfig.for_model(**raw) if "model_type" not in raw else AutoConfig.from_pretrained(
        str(metadata), trust_remote_code=False
    )
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    # The package is text-only; keeping a vision tower in the config would build
    # modules that no package tensor can fill.
    text_config.tie_word_embeddings = getattr(text_config, "tie_word_embeddings", False)
    return text_config


def _causal_lm_class(text_config):
    model_type = getattr(text_config, "model_type", "")
    if model_type.startswith("qwen3_5"):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5ForCausalLM
    raise LoaderError(
        f"no packaged-weight mapping for model_type={model_type!r}; "
        "add its causal-LM class here once its parameter names are verified"
    )


def _swap_modules(
    model: nn.Module,
    weights: PackageWeights,
    head_chunk: Optional[int],
) -> List[str]:
    """Replace Linear/Embedding leaves with packaged equivalents.

    Returns the parameter names now backed by the package.
    """
    backed: List[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            qualified = f"{module_name}.{child_name}" if module_name else child_name
            weight_name = f"{qualified}.weight"
            if not weights.has(weight_name):
                continue

            if isinstance(child, nn.Linear):
                bias = None
                if child.bias is not None:
                    bias_name = f"{qualified}.bias"
                    if not weights.has(bias_name):
                        raise LoaderError(f"{qualified} has a bias that the package lacks")
                    bias = weights.get(bias_name)
                    backed.append(bias_name)
                chunk = head_chunk if qualified.endswith("lm_head") else None
                setattr(
                    module,
                    child_name,
                    PackagedLinear(
                        weights,
                        weight_name,
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=bias,
                        out_chunk=chunk,
                    ),
                )
                backed.append(weight_name)
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    child_name,
                    PackagedEmbedding(
                        weights,
                        weight_name,
                        num_embeddings=child.num_embeddings,
                        embedding_dim=child.embedding_dim,
                        padding_idx=child.padding_idx,
                    ),
                )
                backed.append(weight_name)
    return backed


def _materialize_remaining(model: nn.Module, weights: PackageWeights) -> Tuple[List[str], List[str]]:
    """Fill the small parameters (norms, recurrence state, conv kernels) in place."""
    filled: List[str] = []
    unbacked: List[str] = []
    for name, param in list(model.named_parameters()):
        if not weights.has(name):
            unbacked.append(name)
            continue
        value = weights.get(name).reshape(param.shape)
        _assign_parameter(model, name, value)
        filled.append(name)
    return filled, unbacked


def _assign_parameter(model: nn.Module, qualified: str, value: torch.Tensor) -> None:
    parent_path, _, leaf = qualified.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    existing = getattr(parent, leaf)
    parent.register_parameter(
        leaf, nn.Parameter(value.detach().clone(), requires_grad=existing.requires_grad)
    )


def build_causal_lm(
    package_dir: Union[str, Path],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    cache_bytes: int = 512 * 1024 * 1024,
    head_chunk: Optional[int] = DEFAULT_HEAD_CHUNK,
    quant_method: QuantMethod = QuantMethod.RTN,
    weights: Optional[PackageWeights] = None,
) -> Tuple[nn.Module, PackageWeights]:
    """Assemble a causal LM backed entirely by ``package_dir``.

    Raises if any parameter is left unbacked, because an unbacked parameter is
    an uninitialized one and the model would generate confident nonsense rather
    than fail.
    """
    weights = weights or PackageWeights(
        package_dir,
        device=device,
        dtype=dtype,
        cache_bytes=cache_bytes,
        quant_method=quant_method,
    )
    text_config = load_text_config(package_dir)
    model_class = _causal_lm_class(text_config)

    with parameters_on_meta():
        model = model_class(text_config)
    model.eval()

    backed = _swap_modules(model, weights, head_chunk)
    filled, unbacked = _materialize_remaining(model, weights)

    if unbacked:
        raise LoaderError(
            f"{len(unbacked)} parameters have no package tensor, so they would run "
            f"uninitialized: {unbacked[:8]}{' ...' if len(unbacked) > 8 else ''}"
        )
    still_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if still_meta:
        raise LoaderError(f"parameters left on meta: {still_meta[:8]}")

    model._pockettitan = {  # noqa: SLF001 - deliberate provenance stamp
        "package": str(package_dir),
        "backed_by_package": len(backed),
        "materialized": len(filled),
        "unused_package_tensors": sorted(
            name
            for name in weights.entries
            if name not in {weights.resolve(p) for p, _ in model.named_parameters()}
            and name not in {weights.resolve(b) for b in backed}
        ),
    }
    return model, weights


def summarize(model: nn.Module) -> Dict[str, Any]:
    """What the loader did, for reporting and tests."""
    info = dict(getattr(model, "_pockettitan", {}))
    info["packaged_linears"] = sum(1 for m in model.modules() if isinstance(m, PackagedLinear))
    info["packaged_embeddings"] = sum(
        1 for m in model.modules() if isinstance(m, PackagedEmbedding)
    )
    info["resident_params"] = sum(p.numel() for p in model.parameters())
    return info
