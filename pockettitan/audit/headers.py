"""Strict parallel Safetensors header scanner (R0).

Unlike :mod:`pockettitan.metadata.tensor_index`, which tolerates missing shards to
keep interactive ``inspect`` fast, this module is **strict by default**: a single
unreadable shard raises rather than silently yielding an incomplete table. Audit
numbers are used as ground truth for capacity planning, so a partial scan that
looks successful is worse than a hard failure.
"""

import concurrent.futures
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_url
from pydantic import BaseModel, Field

from pockettitan.config import TensorAddress
from pockettitan.metadata.repo import (
    fetch_model_config,
    fetch_model_index,
    list_repository_files,
)
from pockettitan.metadata.safetensors_header import (
    parse_local_safetensors_header,
    parse_remote_safetensors_header,
)


class ShardScanError(RuntimeError):
    """Raised when one or more shards could not be read during a strict scan."""


class ScanDiscrepancy(BaseModel):
    """A single mismatch between the scanned headers and the checkpoint index."""

    kind: str = Field(description="Discrepancy class, e.g. 'missing_tensor' or 'total_size_mismatch'")
    detail: str = Field(description="Human-readable description of the mismatch")


class ShardHeaderScan(BaseModel):
    """Complete tensor inventory recovered from Safetensors headers."""

    model_id: str
    is_local: bool
    config: Dict[str, Any] = Field(default_factory=dict)
    shards: List[str] = Field(default_factory=list)
    tensors: Dict[str, TensorAddress] = Field(default_factory=dict)
    declared_total_bytes: Optional[int] = Field(
        default=None, description="'total_size' from model.safetensors.index.json, if published"
    )
    header_bytes_read: int = Field(default=0, description="Total header bytes transferred")
    elapsed_s: float = Field(default=0.0)
    discrepancies: List[ScanDiscrepancy] = Field(default_factory=list)

    @property
    def total_params(self) -> int:
        return sum(t.num_params for t in self.tensors.values())

    @property
    def total_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors.values())

    @property
    def num_tensors(self) -> int:
        return len(self.tensors)

    def dtype_histogram(self) -> Dict[str, int]:
        """Count tensors by dtype. A mixed-dtype checkpoint invalidates any
        ``total_size / bytes_per_element`` parameter estimate."""
        hist: Dict[str, int] = {}
        for t in self.tensors.values():
            hist[t.dtype] = hist.get(t.dtype, 0) + 1
        return dict(sorted(hist.items(), key=lambda kv: -kv[1]))

    def tensors_in_shard(self, shard: str) -> List[TensorAddress]:
        return [t for t in self.tensors.values() if t.shard == shard]


def _read_header(
    shard: str,
    model_id_or_path: str,
    is_local: bool,
    headers: Optional[Dict[str, str]],
    retries: int,
) -> Tuple[str, Dict[str, Any], int, Optional[str]]:
    """Read one shard header. Returns ``(shard, header, header_bytes, error)``."""
    last_error: Optional[str] = None

    for attempt in range(max(1, retries)):
        try:
            if is_local:
                shard_path = Path(model_id_or_path) / shard
                if not shard_path.exists():
                    return shard, {}, 0, f"shard file not found: {shard_path}"
                header, header_bytes = parse_local_safetensors_header(shard_path)
            else:
                url = hf_hub_url(repo_id=model_id_or_path, filename=shard)
                header, header_bytes = parse_remote_safetensors_header(url, headers=headers)
            return shard, header, header_bytes, None
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the caller
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                time.sleep(0.5 * (2**attempt))

    return shard, {}, 0, last_error


def _discover_shards(
    model_id_or_path: str,
    is_local: bool,
    index: Optional[Dict[str, Any]],
    repo_files: List[str],
) -> List[str]:
    """Resolve the shard list from the index when present, else from the file listing."""
    if index and "weight_map" in index:
        return sorted(set(index["weight_map"].values()))

    shards = sorted(f for f in repo_files if f.endswith(".safetensors"))
    if not shards:
        raise ShardScanError(f"No .safetensors shards found in {model_id_or_path}")
    return shards


def scan_checkpoint(
    model_id_or_path: str,
    token: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    max_workers: int = 16,
    strict: bool = True,
    retries: int = 3,
) -> ShardHeaderScan:
    """Read every shard header and build a complete tensor inventory.

    Args:
        model_id_or_path: Local directory or Hugging Face repository ID.
        token: Optional HF auth token for gated repositories.
        headers: Extra HTTP headers for remote reads.
        max_workers: Parallel header requests. Header reads are latency-bound,
            so oversubscribing cores is intentional.
        strict: Raise :class:`ShardScanError` if any shard fails to parse. Set
            ``False`` only for exploratory use.
        retries: Attempts per shard, with exponential backoff.

    Returns:
        A :class:`ShardHeaderScan` covering 100% of shards.

    Raises:
        ShardScanError: In strict mode, if any shard is unreadable.
    """
    started = time.perf_counter()

    path = Path(model_id_or_path)
    is_local = path.exists() and path.is_dir()

    repo_files = list_repository_files(model_id_or_path, token=token)
    config = fetch_model_config(model_id_or_path, token=token)
    index = fetch_model_index(model_id_or_path, available_files=repo_files, token=token)
    shards = _discover_shards(model_id_or_path, is_local, index, repo_files)

    if token and not headers:
        headers = {"Authorization": f"Bearer {token}"}

    tensors: Dict[str, TensorAddress] = {}
    failures: List[str] = []
    header_bytes_total = 0

    num_workers = min(max_workers, max(1, len(shards)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_read_header, shard, model_id_or_path, is_local, headers, retries)
            for shard in shards
        ]
        for future in concurrent.futures.as_completed(futures):
            shard, header, header_bytes, error = future.result()
            if error is not None:
                failures.append(f"{shard}: {error}")
                continue

            header_bytes_total += header_bytes
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                shape = info.get("shape", [])
                offsets = info.get("data_offsets", [0, 0])
                tensors[name] = TensorAddress(
                    name=name,
                    shard=shard,
                    dtype=info.get("dtype", "F16"),
                    shape=shape,
                    byte_start=header_bytes + offsets[0],
                    byte_end=header_bytes + offsets[1],
                    num_params=math.prod(shape) if shape else 0,
                    size_bytes=offsets[1] - offsets[0],
                )

    if failures and strict:
        raise ShardScanError(
            f"{len(failures)} of {len(shards)} shards failed to parse:\n  "
            + "\n  ".join(failures[:10])
        )

    scan = ShardHeaderScan(
        model_id=model_id_or_path,
        is_local=is_local,
        config=config,
        shards=shards,
        tensors=tensors,
        declared_total_bytes=(index or {}).get("metadata", {}).get("total_size"),
        header_bytes_read=header_bytes_total,
        elapsed_s=time.perf_counter() - started,
    )
    scan.discrepancies = verify_scan(scan, index, failures)
    return scan


def verify_scan(
    scan: ShardHeaderScan,
    index: Optional[Dict[str, Any]],
    failures: Optional[List[str]] = None,
) -> List[ScanDiscrepancy]:
    """Cross-check the scan against the published index.

    Three independent checks, because a checkpoint audit that agrees with itself
    but not with the publisher is not evidence.
    """
    found: List[ScanDiscrepancy] = []

    for failure in failures or []:
        found.append(ScanDiscrepancy(kind="shard_unreadable", detail=failure))

    if index and "weight_map" in index:
        declared = set(index["weight_map"].keys())
        observed = set(scan.tensors.keys())

        for name in sorted(declared - observed)[:20]:
            found.append(
                ScanDiscrepancy(kind="missing_tensor", detail=f"in index but not in headers: {name}")
            )
        for name in sorted(observed - declared)[:20]:
            found.append(
                ScanDiscrepancy(kind="extra_tensor", detail=f"in headers but not in index: {name}")
            )

        for name, shard in index["weight_map"].items():
            addr = scan.tensors.get(name)
            if addr is not None and addr.shard != shard:
                found.append(
                    ScanDiscrepancy(
                        kind="shard_mismatch",
                        detail=f"{name}: index says {shard}, header found in {addr.shard}",
                    )
                )
                break

    if scan.declared_total_bytes:
        delta = scan.total_bytes - int(scan.declared_total_bytes)
        if delta != 0:
            found.append(
                ScanDiscrepancy(
                    kind="total_size_mismatch",
                    detail=(
                        f"summed tensor bytes {scan.total_bytes:,} != index total_size "
                        f"{int(scan.declared_total_bytes):,} (delta {delta:+,})"
                    ),
                )
            )

    return found
