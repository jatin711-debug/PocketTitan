"""Remote checkpoint and durable local page stores for DomainSlice V0."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from huggingface_hub import hf_hub_url

from pockettitan.metadata.safetensors_header import stream_remote_bytes
from pockettitan.metadata.tensor_index import TensorAddressTable, build_tensor_address_table
from pockettitan.package.slicing import layer_index
from pockettitan.package.format import ExpertRecordLayout
from pockettitan.package.slicing import SourceSlice, build_expert_slices

from .types import (
    RAW_BF16_CODEC,
    ModelRevision,
    PageDescriptor,
    PageHandle,
    ProgressCallback,
    StoreStats,
    WeightID,
    WeightPageID,
)


class DomainSliceError(RuntimeError):
    """Base error for remote paging failures."""


class CacheBudgetError(DomainSliceError):
    """A page cannot fit in the configured local cache budget."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_prefix(path: Path, length: int, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        while remaining:
            block = stream.read(min(chunk_size, remaining))
            if not block:
                raise OSError(f"Could not read {length:,} durable bytes from {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class RemoteHuggingFaceStore:
    """Resolve experts to exact checkpoint ranges and stream those ranges."""

    def __init__(
        self,
        model_revision: ModelRevision,
        *,
        token: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        address_table: Optional[TensorAddressTable] = None,
        max_workers: int = 3,
        chunk_size: int = 8 * 1024 * 1024,
        url_resolver: Optional[Callable[[str], str]] = None,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.model_revision = model_revision
        self.token = token
        self.headers = dict(headers or {})
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self._address_table = address_table
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self._url_resolver = url_resolver or (
            lambda shard: hf_hub_url(
                repo_id=model_revision.repo_id,
                filename=shard,
                revision=model_revision.commit_sha,
            )
        )

    @property
    def address_table(self) -> TensorAddressTable:
        if self._address_table is None:
            self._address_table = build_tensor_address_table(
                self.model_revision.repo_id,
                token=self.token,
                headers=self.headers,
                max_workers=self.max_workers,
                revision=self.model_revision.commit_sha,
                strict=True,
            )
        return self._address_table

    def resolve(self, page_id: WeightPageID) -> PageDescriptor:
        if page_id.model_revision != self.model_revision:
            raise DomainSliceError("Page revision does not match the remote store revision")
        if page_id.codec != RAW_BF16_CODEC:
            raise DomainSliceError(
                f"DomainSlice V0 supports only {RAW_BF16_CODEC}, got {page_id.codec}"
            )
        metadata = self.address_table.metadata
        if page_id.page_kind == "tensor":
            tensor_name = page_id.tensor_name()
            try:
                address = self.address_table.get_tensor(tensor_name)
            except KeyError as exc:
                raise DomainSliceError(f"Unknown source tensor {tensor_name!r}") from exc
            if address.dtype != "BF16":
                raise DomainSliceError(
                    f"DomainSlice V0 raw tensor pages require BF16, got {address.dtype} "
                    f"for {tensor_name}"
                )
            source_slice = SourceSlice(
                tensor=address.name,
                shard=address.shard,
                projection="tensor",
                dtype=address.dtype,
                shape=list(address.shape),
                byte_start=address.byte_start,
                byte_end=address.byte_end,
            )
            output_layout = ExpertRecordLayout.build(
                [
                    {
                        "name": "tensor",
                        "shape": address.shape,
                        "bits": 16,
                        "group_size": -1,
                        "symmetric": True,
                        "codec_id": RAW_BF16_CODEC,
                    }
                ]
            )
            tensor_layer = layer_index(address.name)
            return PageDescriptor(
                page_id=page_id,
                weight_ids=[
                    WeightID(
                        layer=tensor_layer if tensor_layer is not None else -1,
                        component="backbone",
                        projection=address.name,
                    )
                ],
                source_slices=[source_slice],
                output_layout=output_layout,
                expected_bytes=address.size_bytes,
            )
        if page_id.page_kind != "expert":
            raise DomainSliceError(f"Unsupported page kind {page_id.page_kind!r}")

        layer, expert = page_id.expert_coordinates()
        if not metadata.is_moe or metadata.num_experts is None:
            raise DomainSliceError(f"{self.model_revision.repo_id} is not a routed MoE model")
        if not 0 <= layer < metadata.num_hidden_layers:
            raise DomainSliceError(
                f"Layer {layer} out of range [0, {metadata.num_hidden_layers})"
            )
        if not 0 <= expert < metadata.num_experts:
            raise DomainSliceError(f"Expert {expert} out of range [0, {metadata.num_experts})")

        resolved = build_expert_slices(
            self.address_table.tensors,
            metadata.num_experts,
            layers=[layer],
            experts=[expert],
        )
        if len(resolved) != 1:
            raise DomainSliceError(f"Could not resolve exactly one expert ({layer}, {expert})")
        source_slices = resolved[0].projections
        if not source_slices:
            raise DomainSliceError(f"Expert ({layer}, {expert}) has no source projections")
        unsupported = sorted({item.dtype for item in source_slices if item.dtype != "BF16"})
        if unsupported:
            raise DomainSliceError(
                "DomainSlice V0 raw pages require BF16 source tensors; found "
                + ", ".join(unsupported)
            )

        output_layout = ExpertRecordLayout.build(
            [
                {
                    "name": item.projection,
                    "shape": item.shape,
                    "bits": 16,
                    "group_size": -1,
                    "symmetric": True,
                    "codec_id": RAW_BF16_CODEC,
                }
                for item in source_slices
            ]
        )
        expected_bytes = sum(item.size_bytes for item in source_slices)
        if output_layout.payload_bytes != expected_bytes:
            raise DomainSliceError(
                f"Raw expert layout reserves {output_layout.payload_bytes:,} bytes but "
                f"source slices contain {expected_bytes:,}"
            )
        return PageDescriptor(
            page_id=page_id,
            weight_ids=[
                WeightID(
                    layer=layer,
                    component="routed_expert",
                    expert_id=expert,
                    projection=item.projection,
                )
                for item in source_slices
            ],
            source_slices=source_slices,
            output_layout=output_layout,
            expected_bytes=expected_bytes,
        )

    def fetch_slice_to_file(
        self,
        source_slice: SourceSlice,
        destination: Path,
        *,
        progress: Optional[ProgressCallback] = None,
        cancel_event=None,
    ) -> tuple[int, int]:
        """Resume one projection fragment and return ``(fetched, reused)`` bytes."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = source_slice.size_bytes
        existing = destination.stat().st_size if destination.exists() else 0
        if existing > expected:
            destination.unlink()
            existing = 0
        if existing == expected:
            return 0, existing

        url = self._url_resolver(source_slice.shard)
        mode = "ab" if existing else "wb"
        with destination.open(mode) as stream:

            def write_block(block: bytes) -> None:
                view = memoryview(block)
                while view:
                    written = stream.write(view)
                    if written is None or written <= 0:
                        raise OSError(f"Could not write partial page fragment {destination}")
                    view = view[written:]

            try:
                fetched = stream_remote_bytes(
                    url,
                    source_slice.byte_start + existing,
                    source_slice.byte_end - 1,
                    write_block,
                    headers=self.headers,
                    chunk_callback=(
                        (
                            lambda count, _total: progress(
                                "download", source_slice.projection, count, expected
                            )
                        )
                        if progress is not None
                        else None
                    ),
                    cancel_event=cancel_event,
                    chunk_size=self.chunk_size,
                )
            finally:
                # A journal entry is written by the composite store only after
                # these bytes are durable.  On a hard crash, unjournaled suffix
                # bytes are discarded rather than trusted on restart.
                stream.flush()
                os.fsync(stream.fileno())
        if destination.stat().st_size != expected:
            raise DomainSliceError(
                f"Fragment {source_slice.projection} is {destination.stat().st_size:,} bytes; "
                f"expected {expected:,}"
            )
        return fetched, existing


class PocketTitanPageStore:
    """Checksum-verified, budgeted, crash-safe local expert page cache."""

    def __init__(self, root: Path | str, max_cache_bytes: int):
        if max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be positive")
        self.root = Path(root)
        self.pages_dir = self.root / "pages"
        self.partial_dir = self.root / "partial"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = int(max_cache_bytes)
        self._lock = threading.RLock()
        self._leases: Dict[str, int] = {}
        self._reservations: Dict[str, int] = {}
        self.evictions = 0
        self.corruptions = 0
        self._db = sqlite3.connect(self.root / "cache.db", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                page_key TEXT PRIMARY KEY,
                page_path TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                last_access_ns INTEGER NOT NULL
            )
            """
        )
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _paths(self, page_id: WeightPageID) -> tuple[Path, Path]:
        key = page_id.cache_key
        return self.pages_dir / f"{key}.ptpage", self.pages_dir / f"{key}.json"

    def _discard(self, page_id: WeightPageID, page: Path, manifest: Path) -> None:
        for path in (page, manifest):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._db.execute("DELETE FROM pages WHERE page_key = ?", (page_id.cache_key,))
        self._db.commit()

    def lookup(self, page_id: WeightPageID, *, acquire: bool = False) -> Optional[PageHandle]:
        page, manifest = self._paths(page_id)
        with self._lock:
            if not page.exists() or not manifest.exists():
                self._db.execute("DELETE FROM pages WHERE page_key = ?", (page_id.cache_key,))
                self._db.commit()
                return None
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                descriptor = PageDescriptor.model_validate(payload["descriptor"])
                checksum = str(payload["sha256"])
                size_bytes = int(payload["size_bytes"])
                if descriptor.page_id != page_id or page.stat().st_size != size_bytes:
                    raise ValueError("page identity or size mismatch")
                if _sha256_file(page) != checksum:
                    raise ValueError("page checksum mismatch")
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                self.corruptions += 1
                self._discard(page_id, page, manifest)
                return None

            now = time.time_ns()
            self._db.execute(
                """
                INSERT OR REPLACE INTO pages
                    (page_key, page_path, manifest_path, size_bytes, checksum, last_access_ns)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (page_id.cache_key, str(page), str(manifest), size_bytes, checksum, now),
            )
            self._db.commit()
            if acquire:
                self._leases[page_id.cache_key] = self._leases.get(page_id.cache_key, 0) + 1
            return PageHandle(
                page_id=page_id,
                path=page,
                checksum=checksum,
                size_bytes=size_bytes,
                cache_hit=True,
                cache_occupancy_bytes=self.occupancy_bytes,
            )

    def resolve(self, page_id: WeightPageID) -> Optional[PageDescriptor]:
        _page, manifest = self._paths(page_id)
        if not manifest.exists():
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            descriptor = PageDescriptor.model_validate(payload["descriptor"])
            return descriptor if descriptor.page_id == page_id else None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    @property
    def occupancy_bytes(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM pages").fetchone()
            return int(row[0])

    @property
    def page_count(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT COUNT(*) FROM pages").fetchone()
            return int(row[0])

    def _ensure_capacity(self, incoming: int, protect_key: str) -> None:
        if incoming > self.max_cache_bytes:
            raise CacheBudgetError(
                f"Page requires {incoming:,} bytes but cache budget is {self.max_cache_bytes:,}"
            )
        reserved_elsewhere = sum(
            size for key, size in self._reservations.items() if key != protect_key
        )
        while self.occupancy_bytes + reserved_elsewhere + incoming > self.max_cache_bytes:
            rows = self._db.execute(
                "SELECT page_key, page_path, manifest_path FROM pages ORDER BY last_access_ns ASC"
            ).fetchall()
            victim = next(
                (
                    row
                    for row in rows
                    if row[0] != protect_key and self._leases.get(str(row[0]), 0) == 0
                ),
                None,
            )
            if victim is None:
                raise CacheBudgetError("Cache is full and every completed page is currently in use")
            key, page_path, manifest_path = victim
            for path in (Path(page_path), Path(manifest_path)):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            self._db.execute("DELETE FROM pages WHERE page_key = ?", (key,))
            self._db.commit()
            self.evictions += 1

    def reserve(self, page_id: WeightPageID, size_bytes: int) -> None:
        """Reserve cache capacity before issuing any remote payload requests."""
        with self._lock:
            if page_id.cache_key in self._reservations:
                return
            self._ensure_capacity(size_bytes, page_id.cache_key)
            self._reservations[page_id.cache_key] = size_bytes

    def cancel_reservation(self, page_id: WeightPageID) -> None:
        with self._lock:
            self._reservations.pop(page_id.cache_key, None)

    def _partial_paths(self, page_id: WeightPageID, count: int) -> tuple[Path, list[Path], Path]:
        directory = self.partial_dir / page_id.cache_key
        directory.mkdir(parents=True, exist_ok=True)
        return directory, [directory / f"{index:03d}.part" for index in range(count)], directory / "journal.json"

    def prepare_fragments(self, descriptor: PageDescriptor) -> tuple[list[Path], int]:
        """Validate reusable partial bytes and discard only invalid fragments."""
        _directory, fragments, journal_path = self._partial_paths(
            descriptor.page_id, len(descriptor.source_slices)
        )
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            journal = {"completed": {}}
        completed = journal.get("completed", {})
        reused = 0
        for index, (source_slice, fragment) in enumerate(
            zip(descriptor.source_slices, fragments)
        ):
            if not fragment.exists():
                continue
            size = fragment.stat().st_size
            if size > source_slice.size_bytes:
                fragment.unlink()
                completed.pop(str(index), None)
                continue
            record = completed.get(str(index))
            if record is not None:
                durable_size = int(record.get("size_bytes", -1))
                if durable_size < 0 or size < durable_size:
                    fragment.unlink()
                    completed.pop(str(index), None)
                    continue
                if _sha256_prefix(fragment, durable_size) != record.get("sha256"):
                    fragment.unlink()
                    completed.pop(str(index), None)
                    continue
                if size > durable_size:
                    with fragment.open("r+b") as stream:
                        stream.truncate(durable_size)
                    size = durable_size
            elif size < source_slice.size_bytes:
                # There is no durable checksum boundary for this prefix.
                fragment.unlink()
                continue

            if size == source_slice.size_bytes:
                checksum = _sha256_file(fragment)
                completed[str(index)] = {
                    "tensor": source_slice.tensor,
                    "size_bytes": size,
                    "sha256": checksum,
                }
            reused += size
        _atomic_json(journal_path, {"version": 1, "completed": completed})
        return fragments, reused

    def record_fragment(
        self,
        descriptor: PageDescriptor,
        index: int,
        fragment: Path,
    ) -> None:
        _directory, _fragments, journal_path = self._partial_paths(
            descriptor.page_id, len(descriptor.source_slices)
        )
        with self._lock:
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                journal = {"version": 1, "completed": {}}
            journal.setdefault("completed", {})[str(index)] = {
                "tensor": descriptor.source_slices[index].tensor,
                "size_bytes": fragment.stat().st_size,
                "sha256": _sha256_file(fragment),
            }
            _atomic_json(journal_path, journal)

    def commit(self, descriptor: PageDescriptor, fragments: list[Path]) -> PageHandle:
        page, manifest = self._paths(descriptor.page_id)
        temp_page = page.with_suffix(".ptpage.tmp")
        with self._lock:
            self._ensure_capacity(descriptor.output_layout.stride, descriptor.page_id.cache_key)
            digest = hashlib.sha256()
            with temp_page.open("wb") as output:
                for source_slice, fragment in zip(descriptor.source_slices, fragments):
                    if fragment.stat().st_size != source_slice.size_bytes:
                        raise DomainSliceError(
                            f"Incomplete fragment for {source_slice.tensor}: "
                            f"{fragment.stat().st_size:,}/{source_slice.size_bytes:,} bytes"
                        )
                    with fragment.open("rb") as source:
                        while True:
                            block = source.read(8 * 1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            digest.update(block)
                padding = descriptor.output_layout.stride - descriptor.expected_bytes
                if padding:
                    zeros = b"\x00" * min(padding, 1024 * 1024)
                    while padding:
                        block = zeros[:padding]
                        output.write(block)
                        digest.update(block)
                        padding -= len(block)
                output.flush()
                os.fsync(output.fileno())
            size_bytes = temp_page.stat().st_size
            if size_bytes != descriptor.output_layout.stride:
                raise DomainSliceError(
                    f"Assembled page is {size_bytes:,} bytes; expected "
                    f"{descriptor.output_layout.stride:,}"
                )
            checksum = digest.hexdigest()
            os.replace(temp_page, page)
            _atomic_json(
                manifest,
                {
                    "version": 1,
                    "descriptor": descriptor.model_dump(mode="json"),
                    "sha256": checksum,
                    "size_bytes": size_bytes,
                },
            )
            now = time.time_ns()
            self._db.execute(
                """
                INSERT OR REPLACE INTO pages
                    (page_key, page_path, manifest_path, size_bytes, checksum, last_access_ns)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    descriptor.page_id.cache_key,
                    str(page),
                    str(manifest),
                    size_bytes,
                    checksum,
                    now,
                ),
            )
            self._db.commit()
            self._reservations.pop(descriptor.page_id.cache_key, None)
            self._leases[descriptor.page_id.cache_key] = (
                self._leases.get(descriptor.page_id.cache_key, 0) + 1
            )
            partial = self.partial_dir / descriptor.page_id.cache_key
            if partial.exists():
                shutil.rmtree(partial)
            return PageHandle(
                page_id=descriptor.page_id,
                path=page,
                checksum=checksum,
                size_bytes=size_bytes,
                cache_hit=False,
                cache_occupancy_bytes=self.occupancy_bytes,
            )

    def release(self, handle: PageHandle) -> None:
        with self._lock:
            key = handle.page_id.cache_key
            count = self._leases.get(key, 0)
            if count <= 1:
                self._leases.pop(key, None)
            else:
                self._leases[key] = count - 1


class _CombinedCancellation:
    def __init__(self, internal: threading.Event, external=None):
        self.internal = internal
        self.external = external

    def is_set(self) -> bool:
        return self.internal.is_set() or (
            self.external is not None and self.external.is_set()
        )

    def wait(self, timeout: Optional[float] = None) -> bool:
        if self.is_set():
            return True
        return self.internal.wait(timeout)


class CompositeWeightStore:
    """Local-first store that faults missing expert pages from Hugging Face."""

    def __init__(
        self,
        local: PocketTitanPageStore,
        remote: RemoteHuggingFaceStore,
        *,
        download_workers: int = 3,
    ):
        if download_workers < 1:
            raise ValueError("download_workers must be at least 1")
        self.local = local
        self.remote = remote
        self.download_workers = download_workers
        self._cache_hits = 0
        self._cache_misses = 0
        self._remote_payload_bytes = 0
        self._resumed_bytes = 0
        self._stats_lock = threading.Lock()
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def close(self) -> None:
        self._prefetch_executor.shutdown(wait=True, cancel_futures=True)
        self.local.close()

    def resolve(self, page_id: WeightPageID) -> PageDescriptor:
        local = self.local.resolve(page_id)
        return local if local is not None else self.remote.resolve(page_id)

    def materialize(
        self,
        page_id: WeightPageID,
        *,
        progress: Optional[ProgressCallback] = None,
        cancel_event=None,
    ) -> PageHandle:
        started = time.perf_counter()
        cached = self.local.lookup(page_id, acquire=True)
        if cached is not None:
            with self._stats_lock:
                self._cache_hits += 1
            cached.timings = {"total_seconds": time.perf_counter() - started}
            return cached

        with self._stats_lock:
            self._cache_misses += 1
        resolve_started = time.perf_counter()
        descriptor = self.remote.resolve(page_id)
        resolve_seconds = time.perf_counter() - resolve_started
        self.local.reserve(page_id, descriptor.output_layout.stride)
        try:
            fragments, resumed = self.local.prepare_fragments(descriptor)
        except BaseException:
            self.local.cancel_reservation(page_id)
            raise
        internal_cancel = threading.Event()
        combined_cancel = _CombinedCancellation(internal_cancel, cancel_event)
        attempt_bytes = 0
        attempt_bytes_lock = threading.Lock()

        def report(stage: str, item: str, count: int, total: int) -> None:
            nonlocal attempt_bytes
            with attempt_bytes_lock:
                attempt_bytes += count
            if progress is not None:
                progress(stage, item, count, total)

        def fetch(index: int) -> int:
            source_slice = descriptor.source_slices[index]
            try:
                fetched, _existing = self.remote.fetch_slice_to_file(
                    source_slice,
                    fragments[index],
                    progress=report,
                    cancel_event=combined_cancel,
                )
                return fetched
            finally:
                if fragments[index].exists() and fragments[index].stat().st_size:
                    self.local.record_fragment(descriptor, index, fragments[index])

        download_started = time.perf_counter()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.download_workers)
        futures = [executor.submit(fetch, index) for index in range(len(fragments))]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            internal_cancel.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            self.local.cancel_reservation(page_id)
            with self._stats_lock:
                self._remote_payload_bytes += attempt_bytes
            raise
        else:
            executor.shutdown(wait=True)
        download_seconds = time.perf_counter() - download_started

        commit_started = time.perf_counter()
        try:
            handle = self.local.commit(descriptor, fragments)
        except BaseException:
            self.local.cancel_reservation(page_id)
            raise
        commit_seconds = time.perf_counter() - commit_started
        handle.bytes_fetched = attempt_bytes
        handle.bytes_resumed = resumed
        handle.timings = {
            "resolve_seconds": resolve_seconds,
            "download_seconds": download_seconds,
            "commit_seconds": commit_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        with self._stats_lock:
            self._remote_payload_bytes += attempt_bytes
            self._resumed_bytes += resumed
        return handle

    def prefetch(self, page_ids: Iterable[WeightPageID]):
        return [self._prefetch_executor.submit(self.materialize, page_id) for page_id in page_ids]

    def release(self, handle: PageHandle) -> None:
        self.local.release(handle)

    def stats(self) -> StoreStats:
        with self._stats_lock:
            return StoreStats(
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                remote_payload_bytes=self._remote_payload_bytes,
                resumed_bytes=self._resumed_bytes,
                evictions=self.local.evictions,
                corruptions=self.local.corruptions,
                cache_occupancy_bytes=self.local.occupancy_bytes,
                cached_pages=self.local.page_count,
            )
