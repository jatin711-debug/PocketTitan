"""On-demand weight access for a ``.ptitan`` package.

A 27B model dequantized to bf16 is ~55 GB, so nothing that materializes the whole
state dict can run on the target machine. These modules keep the weights on disk
and decode them per use through :func:`decode_record` -- the writer's own inverse
-- so the numbers a layer computes with are exactly the numbers that were packed.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from pockettitan.config import QuantMethod
from pockettitan.package.decode import decode_record, decode_rows
from pockettitan.package.format import DenseTensorEntry, PackageManifest, matrix_dims


class WeightNotFound(KeyError):
    """The package has no tensor under any known alias for this parameter."""


# The checkpoint is multimodal, so its language tower sits one level deeper than
# the text-only module tree HF builds for ``Qwen3_5ForCausalLM``.
NAME_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("model.", "model.language_model."),
    ("", ""),
)


class PackageWeights:
    """Resolves parameter names to package records and decodes them on demand.

    A byte-budgeted LRU keeps small, hot tensors (norms, biases, recurrence
    state) resident while the multi-hundred-megabyte projections are decoded and
    dropped. The budget is a ceiling on decoded bytes, not on file size.
    """

    def __init__(
        self,
        package_dir: Union[str, Path],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        cache_bytes: int = 512 * 1024 * 1024,
        quant_method: QuantMethod = QuantMethod.RTN,
    ):
        self.package_dir = Path(package_dir)
        self.device = device
        self.dtype = dtype
        self.cache_bytes = cache_bytes
        self.quant_method = quant_method

        manifest_path = self.package_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {manifest_path}")
        self.manifest = PackageManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        self.entries: Dict[str, DenseTensorEntry] = {e.name: e for e in self.manifest.dense}

        self._blob = (self.package_dir / "dense" / "blob.bin").open("rb")
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._cache_used = 0
        self.decoded_bytes = 0
        self.decode_calls = 0

    # -- lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        if self._blob is not None and not self._blob.closed:
            self._blob.close()
        self._cache.clear()
        self._cache_used = 0

    def __enter__(self) -> "PackageWeights":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- name resolution --------------------------------------------------- #

    def resolve(self, name: str) -> Optional[str]:
        """The package tensor backing an HF parameter name, or ``None``."""
        for prefix, replacement in NAME_ALIASES:
            if name.startswith(prefix):
                candidate = replacement + name[len(prefix) :]
                if candidate in self.entries:
                    return candidate
        return None

    def entry(self, name: str) -> DenseTensorEntry:
        resolved = self.resolve(name)
        if resolved is None:
            raise WeightNotFound(
                f"{name!r} is not in the package under any alias "
                f"{[alias[1] for alias in NAME_ALIASES]}"
            )
        return self.entries[resolved]

    def has(self, name: str) -> bool:
        return self.resolve(name) is not None

    # -- reads ------------------------------------------------------------- #

    def _payload(self, entry: DenseTensorEntry) -> bytes:
        self._blob.seek(entry.byte_offset)
        raw = self._blob.read(entry.length)
        if len(raw) != entry.length:
            raise EOFError(f"short read for {entry.name}: {len(raw)} of {entry.length} bytes")
        return raw

    def get(self, name: str) -> torch.Tensor:
        """Decode a whole tensor, serving from cache when it is resident."""
        entry = self.entry(name)
        cached = self._cache.get(entry.name)
        if cached is not None:
            self._cache.move_to_end(entry.name)
            return cached

        tensor = decode_record(
            self._payload(entry),
            shape=entry.shape,
            bits=entry.bits,
            group_size=entry.group_size,
            symmetric=entry.symmetric,
            spans=entry.spans,
            method=self.quant_method,
            dtype=self.dtype,
        ).to(self.device)

        self.decode_calls += 1
        self.decoded_bytes += tensor.numel() * tensor.element_size()
        self._admit(entry.name, tensor)
        return tensor

    def get_rows(self, name: str, row_start: int, row_stop: int) -> torch.Tensor:
        """Decode only ``[row_start, row_stop)``; never caches, never reads the rest."""
        entry = self.entry(name)
        tensor = decode_rows(
            self._payload(entry),
            shape=entry.shape,
            bits=entry.bits,
            group_size=entry.group_size,
            symmetric=entry.symmetric,
            row_start=row_start,
            row_stop=row_stop,
            spans=entry.spans,
            method=self.quant_method,
            dtype=self.dtype,
        )
        self.decode_calls += 1
        self.decoded_bytes += tensor.numel() * tensor.element_size()
        return tensor.to(self.device)

    def _admit(self, key: str, tensor: torch.Tensor) -> None:
        size = tensor.numel() * tensor.element_size()
        if size > self.cache_bytes:
            return  # not worth evicting everything else for
        while self._cache and self._cache_used + size > self.cache_bytes:
            _, evicted = self._cache.popitem(last=False)
            self._cache_used -= evicted.numel() * evicted.element_size()
        self._cache[key] = tensor
        self._cache_used += size

    @property
    def resident_bytes(self) -> int:
        return self._cache_used


class PackagedLinear(nn.Module):
    """``nn.Linear`` whose weight lives in the package until the moment it is used.

    ``out_chunk`` splits the output dimension so a very wide head never exists in
    full: ``lm_head`` is 248,320 x 5,120, which is 2.5 GB materialized, and only
    one chunk of its rows needs to be resident at a time.
    """

    def __init__(
        self,
        weights: PackageWeights,
        name: str,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor] = None,
        out_chunk: Optional[int] = None,
    ):
        super().__init__()
        self.weights = weights
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.out_chunk = out_chunk
        self.register_buffer("bias", bias, persistent=False)

    def extra_repr(self) -> str:
        return f"{self.name}, {self.in_features} -> {self.out_features}, packaged"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_chunk is None or self.out_chunk >= self.out_features:
            return F.linear(x, self.weights.get(self.name).to(x.dtype), self.bias)

        parts = []
        for start in range(0, self.out_features, self.out_chunk):
            stop = min(start + self.out_chunk, self.out_features)
            block = self.weights.get_rows(self.name, start, stop).to(x.dtype)
            parts.append(F.linear(x, block))
            del block
        out = torch.cat(parts, dim=-1)
        return out if self.bias is None else out + self.bias


class PackagedEmbedding(nn.Module):
    """Embedding that reads only the rows the batch actually references."""

    def __init__(
        self,
        weights: PackageWeights,
        name: str,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.weights = weights
        self.name = name
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def extra_repr(self) -> str:
        return f"{self.name}, {self.num_embeddings} x {self.embedding_dim}, packaged"

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        unique = sorted(set(flat.tolist()))
        for token_id in unique:
            if not 0 <= token_id < self.num_embeddings:
                raise IndexError(
                    f"token id {token_id} is outside the {self.num_embeddings}-row table"
                )
        # One read per *distinct* id: a prompt repeats tokens far more than it
        # introduces them, and each read is a seek into the blob.
        lookup = {
            token_id: self.weights.get_rows(self.name, token_id, token_id + 1)[0]
            for token_id in unique
        }
        rows = torch.stack([lookup[token_id] for token_id in flat.tolist()])
        out = rows.view(*input_ids.shape, self.embedding_dim)
        if self.padding_idx is not None:
            out = out.masked_fill((input_ids == self.padding_idx).unsqueeze(-1), 0.0)
        return out


def expected_matrix_shape(entry: DenseTensorEntry) -> Tuple[int, int]:
    return matrix_dims(list(entry.shape))
