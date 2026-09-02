"""Public identities and contracts for demand-paged PocketTitan weights."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field

from pockettitan.package.format import ExpertRecordLayout
from pockettitan.package.slicing import SourceSlice


RAW_BF16_CODEC = "pt.raw.bf16.v1"


class ModelRevision(BaseModel):
    """Immutable identity of a checkpoint revision."""

    model_config = ConfigDict(frozen=True)

    repo_id: str
    commit_sha: str


class WeightID(BaseModel):
    """Semantic identity independent of tensor filenames and shard layout."""

    model_config = ConfigDict(frozen=True)

    layer: int
    component: str
    expert_id: Optional[int] = None
    projection: Optional[str] = None


class WeightPageID(BaseModel):
    """Stable identity of one physical cache page."""

    model_config = ConfigDict(frozen=True)

    model_revision: ModelRevision
    page_kind: str
    logical_key: str
    codec: str

    @classmethod
    def expert(
        cls,
        model_revision: ModelRevision,
        layer: int,
        expert: int,
        codec: str = RAW_BF16_CODEC,
    ) -> "WeightPageID":
        return cls(
            model_revision=model_revision,
            page_kind="expert",
            logical_key=f"layers/{layer}/experts/{expert}",
            codec=codec,
        )

    @classmethod
    def tensor(
        cls,
        model_revision: ModelRevision,
        tensor_name: str,
        codec: str = RAW_BF16_CODEC,
    ) -> "WeightPageID":
        if not tensor_name:
            raise ValueError("tensor_name must not be empty")
        return cls(
            model_revision=model_revision,
            page_kind="tensor",
            logical_key=f"tensors/{urllib.parse.quote(tensor_name, safe='')}",
            codec=codec,
        )

    @property
    def cache_key(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def expert_coordinates(self) -> tuple[int, int]:
        parts = self.logical_key.split("/")
        if self.page_kind != "expert" or len(parts) != 4 or parts[0] != "layers":
            raise ValueError(f"Not an expert page key: {self.logical_key!r}")
        if parts[2] != "experts":
            raise ValueError(f"Not an expert page key: {self.logical_key!r}")
        return int(parts[1]), int(parts[3])

    def tensor_name(self) -> str:
        prefix = "tensors/"
        if self.page_kind != "tensor" or not self.logical_key.startswith(prefix):
            raise ValueError(f"Not a tensor page key: {self.logical_key!r}")
        encoded = self.logical_key[len(prefix) :]
        if not encoded:
            raise ValueError(f"Not a tensor page key: {self.logical_key!r}")
        return urllib.parse.unquote(encoded)


class PageDescriptor(BaseModel):
    """Mapping from semantic weights to exact source ranges and page offsets."""

    page_id: WeightPageID
    weight_ids: List[WeightID]
    source_slices: List[SourceSlice]
    output_layout: ExpertRecordLayout
    expected_bytes: int = Field(description="Source payload bytes, excluding alignment padding")


class PageHandle(BaseModel):
    """A verified local page lease returned to a runtime or CLI caller."""

    page_id: WeightPageID
    path: Path
    checksum: str
    size_bytes: int
    cache_hit: bool
    bytes_fetched: int = 0
    bytes_resumed: int = 0
    cache_occupancy_bytes: int = 0
    timings: Dict[str, float] = Field(default_factory=dict)


class StoreStats(BaseModel):
    cache_hits: int = 0
    cache_misses: int = 0
    remote_payload_bytes: int = 0
    resumed_bytes: int = 0
    evictions: int = 0
    corruptions: int = 0
    cache_occupancy_bytes: int = 0
    cached_pages: int = 0


ProgressCallback = Callable[[str, str, int, int], None]


class WeightStore(Protocol):
    """Minimum interface consumed by future HF and native runtimes."""

    def resolve(self, page_id: WeightPageID) -> PageDescriptor: ...

    def materialize(
        self,
        page_id: WeightPageID,
        *,
        progress: Optional[ProgressCallback] = None,
        cancel_event=None,
    ) -> PageHandle: ...

    def prefetch(self, page_ids: List[WeightPageID]): ...

    def release(self, handle: PageHandle) -> None: ...

    def stats(self) -> StoreStats: ...
