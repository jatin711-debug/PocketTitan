"""Streaming package writer (R1, task T1.6).

Executes a :class:`~pockettitan.package.plan.BuildPlan`: reads each source slice,
quantizes it under a hard VRAM ceiling, and writes the result to its planned byte
address. Three properties make this safe on a 12 GB machine building an 87 GiB
package:

**Every output address is known before the build starts.** Region files are
preallocated to their planned size, so records are written by seek-and-write in
any order. Nothing is buffered waiting for a neighbour.

**Resumable at item granularity.** A journal records completed work items after
each region flush. An interrupted build re-reads only what it had not finished.

**Bounded residency.** One expert (or one tensor tile) is in memory at a time,
and all quantization goes through :class:`MatrixTiler`, which enforces the VRAM
budget.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field

from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod
from pockettitan.package.format import (
    CHECKSUMS_NAME,
    DENSE_DIR,
    EXPERT_BANK_NAME,
    EXPERT_LAYOUT_NAME,
    EXPERTS_DIR,
    INTEGRITY_DIR,
    ItemChecksum,
    MANIFEST_NAME,
    METADATA_DIR,
    PackageChecksums,
    PLE_DIR,
    PLE_INDEX_NAME,
    PLE_TABLE_NAME,
    RegionIntegrity,
    Section,
    SectionSpan,
    section_spans,
)
from pockettitan.package.plan import (
    BuildPlan,
    DenseWorkItem,
    ExpertWorkItem,
    PleShardWorkItem,
    build_ple_index,
)
from pockettitan.package.integrity import crc32c_hex, durable_replace, sha256_file
from pockettitan.package.slicing import SourceSlice
from pockettitan.quantizers import get_quantizer
from pockettitan.quantizers.base import QuantizedResult
from pockettitan.scheduler.tiler import MatrixTiler

DENSE_BLOB_NAME = "blob.bin"
JOURNAL_NAME = "build_journal.json"
DENSE_BATCH_ITEMS = 8
EXPERT_BATCH_ITEMS = 64


class WriteError(RuntimeError):
    """Raised when produced bytes disagree with the planned layout."""


class BuildJournal(BaseModel):
    """Durable resumption state advanced only after a region batch is fsynced."""

    plan_fingerprint: str
    dense_done: List[str] = Field(default_factory=list)
    expert_done: List[int] = Field(
        default_factory=list, description="Completed bank record indices"
    )
    ple_done: List[int] = Field(default_factory=list, description="Completed source shard indices")
    dense_checksums: Dict[str, ItemChecksum] = Field(default_factory=dict)
    expert_checksums: Dict[str, ItemChecksum] = Field(default_factory=dict)
    ple_checksums: Dict[str, ItemChecksum] = Field(default_factory=dict)
    finished: bool = False

    def as_sets(self) -> Tuple[set, set, set]:
        return set(self.dense_done), set(self.expert_done), set(self.ple_done)


class BuildResult(BaseModel):
    """Outcome of a build."""

    output_dir: str
    finished: bool
    items_written: int = 0
    items_skipped: int = 0
    bytes_written: int = 0
    peak_vram_mb: float = 0.0
    elapsed_s: float = 0.0


def plan_fingerprint(plan: BuildPlan) -> str:
    """Stable identity for a plan's *layout*.

    Resuming into a package built from a different layout would corrupt it
    silently, so the journal is keyed on this.
    """
    manifest = plan.manifest
    material = json.dumps(
        {
            "source": manifest.source_model,
            "source_revision": manifest.source_revision,
            "features": manifest.features,
            "precision": manifest.precision_map,
            "profile": manifest.build_profile,
            "complete_model": manifest.complete_model,
            "codecs": {name: spec.model_dump() for name, spec in sorted(manifest.codecs.items())},
            "dense": [(d.name, d.byte_offset, d.packed_bytes) for d in manifest.dense],
            "experts": manifest.expert_layout.model_dump() if manifest.expert_layout else None,
            "ple_rows": plan.ple.total_rows if plan.ple else 0,
            "ple_row": plan.ple.row.model_dump() if plan.ple else None,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _batches(items, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _preallocate(path: Path, size: int) -> None:
    """Create or extend a region file to ``size`` bytes without writing data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+b" if path.exists() else "wb"
    with open(path, mode) as f:
        if f.seek(0, os.SEEK_END) < size:
            f.truncate(size)


def _sections_to_bytes(
    result: QuantizedResult,
    spans: Sequence[SectionSpan],
    label: str,
) -> bytes:
    """Serialize a quantized result in planned section order, validating sizes.

    A silent size mismatch would shift every subsequent record, so this raises
    rather than truncating or padding.
    """
    payload = bytearray()
    for span in spans:
        if span.section is Section.PACKED:
            tensor = result.packed_weights
        elif span.section is Section.SCALES:
            tensor = result.scales
        else:
            tensor = result.zeros

        if tensor is None:
            raise WriteError(f"{label}: quantizer produced no {span.section.value} section")

        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        if len(raw) != span.length:
            raise WriteError(
                f"{label}: {span.section.value} is {len(raw)} bytes, plan reserved {span.length}. "
                "The quantizer's packing does not match the planned layout."
            )
        payload.extend(raw)
    return bytes(payload)


class PackageWriter:
    """Executes a build plan into a ``.ptitan`` directory."""

    def __init__(
        self,
        plan: BuildPlan,
        output_dir: Path,
        reader,
        budget: Optional[MemoryBudgetConfig] = None,
        method: QuantMethod = QuantMethod.RTN,
        device: str = "cuda",
        resume: bool = True,
    ):
        """
        Args:
            plan: Byte-exact layout from :func:`plan_package`.
            output_dir: Destination ``.ptitan`` directory.
            reader: ``LocalTensorReader`` or ``RemoteTensorSliceReader``.
            budget: VRAM ceiling. Defaults to the standard 3.5 GiB cap.
            method: Quantizer backend for every component.
            device: ``cuda`` or ``cpu``; falls back to CPU when CUDA is absent.
            resume: Skip work items already recorded in the journal.
        """
        self.plan = plan
        self.output_dir = Path(output_dir)
        self.reader = reader
        self.budget = budget or MemoryBudgetConfig()
        self.method = method
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.resume = resume
        self.tiler = MatrixTiler(self.budget)
        self._quantizers: Dict[Tuple[float, int, bool], object] = {}
        for codec in self.plan.manifest.codecs.values():
            if codec.method not in {"raw", self.method.value}:
                raise WriteError(
                    f"Plan codec {codec.id} requires method {codec.method}, "
                    f"but writer was configured for {self.method.value}"
                )

    # -- paths ------------------------------------------------------------ #

    @property
    def dense_path(self) -> Path:
        return self.output_dir / DENSE_DIR / DENSE_BLOB_NAME

    @property
    def bank_path(self) -> Path:
        return self.output_dir / EXPERTS_DIR / EXPERT_BANK_NAME

    @property
    def ple_path(self) -> Path:
        return self.output_dir / PLE_DIR / PLE_TABLE_NAME

    @property
    def journal_path(self) -> Path:
        return self.output_dir / JOURNAL_NAME

    # -- helpers ---------------------------------------------------------- #

    def _quantizer(self, bits: float, group_size: int, symmetric: bool):
        key = (bits, group_size, symmetric)
        if key not in self._quantizers:
            method = self.method
            if bits >= 16:
                method = QuantMethod.RTN  # unused; fp16 tensors bypass quantization
            self._quantizers[key] = get_quantizer(
                QuantConfig(
                    method=method,
                    bits=int(bits),
                    group_size=group_size,
                    symmetric=symmetric,
                    device=self.device,
                )
            )
        return self._quantizers[key]

    def _load_journal(self) -> BuildJournal:
        fingerprint = plan_fingerprint(self.plan)
        if self.resume and self.journal_path.exists():
            try:
                existing = BuildJournal.model_validate_json(
                    self.journal_path.read_text(encoding="utf-8")
                )
            except Exception:
                existing = None
            if existing is not None:
                if existing.plan_fingerprint != fingerprint:
                    raise WriteError(
                        f"{self.journal_path} was written for layout "
                        f"{existing.plan_fingerprint}, but this plan is {fingerprint}. "
                        "Resuming would corrupt the package; delete the output directory "
                        "or pass resume=False."
                    )
                missing_checksums = (
                    set(existing.dense_done) - set(existing.dense_checksums)
                    or {str(index) for index in existing.expert_done}
                    - set(existing.expert_checksums)
                    or {str(index) for index in existing.ple_done} - set(existing.ple_checksums)
                )
                if missing_checksums:
                    raise WriteError(
                        "The build journal predates the v1.1 checksum protocol. "
                        "Prototype v1.0 packages cannot be resumed; rebuild the output directory."
                    )
                return existing
        return BuildJournal(plan_fingerprint=fingerprint)

    def _save_journal(self, journal: BuildJournal) -> None:
        durable_replace(
            self.journal_path,
            journal.model_dump_json(indent=2).encode("utf-8"),
        )

    def _preflight_disk_space(self) -> None:
        """Require planned region bytes plus 15% headroom on the output volume."""
        probe = self.output_dir
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        region_paths = (self.dense_path, self.bank_path, self.ple_path)
        existing = sum(path.stat().st_size for path in region_paths if path.exists())
        required = int(self.plan.manifest.totals.total_bytes * 1.15)
        available = usage.free + existing
        if available < required:
            shortfall = required - available
            raise WriteError(
                "Insufficient disk space for a durable package build: "
                f"need {required:,} bytes (planned bytes + 15% headroom), "
                f"have {available:,}; short by {shortfall:,} bytes."
            )

    def _read_slice(self, source: SourceSlice, chunk_cb: Optional[Callable] = None) -> torch.Tensor:
        """Materialize one source slice as a 2-D tensor."""
        from pockettitan.config import TensorAddress

        address = TensorAddress(
            name=source.tensor,
            shard=source.shard,
            dtype=source.dtype,
            shape=list(source.shape),
            byte_start=source.byte_start,
            byte_end=source.byte_end,
            num_params=source.num_params,
            size_bytes=source.size_bytes,
        )
        if hasattr(self.reader, "root_dir"):
            # Local: address the parent tensor and slice it zero-copy.
            return self._read_local_slice(source)
        return self.reader.read_tensor(address, chunk_callback=chunk_cb)

    def _read_local_slice(self, source: SourceSlice) -> torch.Tensor:
        """Zero-copy slice out of a memory-mapped local shard."""
        handle = self.reader._get_handle(source.shard)
        view = handle.get_slice(source.tensor)
        data = view[:] if source.expert_index is None else view[source.expert_index]
        if data.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            data = data.to(torch.float16)
        return data.view(*source.shape)

    def _quantize(self, weight: torch.Tensor, bits: float, group_size: int, symmetric: bool):
        """Quantize under the VRAM budget, or pass fp16 through untouched."""
        if bits >= 16:
            return None, weight.detach().to(torch.float16).cpu(), 0.0
        quantizer = self._quantizer(bits, group_size, symmetric)
        result, peak = self.tiler.quantize_matrix(
            weight, quantizer=quantizer, target_device=self.device
        )
        return result, None, peak

    # -- regions ---------------------------------------------------------- #

    def _write_dense(
        self,
        journal: BuildJournal,
        on_item: Optional[Callable] = None,
        on_start: Optional[Callable] = None,
        on_bytes: Optional[Callable] = None,
    ) -> Tuple[int, int, float]:
        done, _, _ = journal.as_sets()
        pending = [i for i in self.plan.dense if i.address.name not in done]
        if not pending:
            return 0, len(self.plan.dense), 0.0

        total = self.plan.manifest.totals.dense_bytes
        _preallocate(self.dense_path, total)
        written = 0
        peak = 0.0

        chunk_cb = (lambda n, _: on_bytes(n)) if on_bytes else None

        with open(self.dense_path, "r+b") as blob:
            for batch in _batches(pending, DENSE_BATCH_ITEMS):
                committed = []
                for item in batch:
                    if on_start:
                        on_start("dense", item.address.name)
                    tensor = self.reader.read_tensor(self._address_of(item), chunk_callback=chunk_cb)
                    payload, item_peak = self._encode_dense(item, tensor)
                    peak = max(peak, item_peak)
                    blob.seek(item.byte_offset)
                    blob.write(payload)
                    written += len(payload)
                    committed.append(
                        (
                            item,
                            ItemChecksum(
                                value=crc32c_hex(payload),
                                offset=item.byte_offset,
                                length=len(payload),
                            ),
                        )
                    )
                    del tensor, payload
                blob.flush()
                os.fsync(blob.fileno())
                for item, checksum in committed:
                    journal.dense_done.append(item.address.name)
                    journal.dense_checksums[item.address.name] = checksum
                self._save_journal(journal)
                if on_item:
                    for item, _ in committed:
                        on_item("dense", item.address.name)
        return written, len(self.plan.dense) - len(pending), peak

    def _encode_dense(self, item: DenseWorkItem, tensor: torch.Tensor) -> Tuple[bytes, float]:
        if item.bits >= 16:
            raw = (
                tensor.detach()
                .to(torch.float16)
                .cpu()
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            expected = item.spans[0].length
            if len(raw) != expected:
                raise WriteError(
                    f"{item.address.name}: fp16 payload is {len(raw)} bytes, plan reserved {expected}"
                )
            return raw, 0.0

        matrix = tensor if tensor.dim() >= 2 else tensor.view(1, -1)
        result, _, peak = self._quantize(matrix, item.bits, item.group_size, item.symmetric)
        return _sections_to_bytes(result, item.spans, item.address.name), peak

    def _materialize_ple_index(self) -> None:
        """Read exact int64 PLE hash tensors and attach the runtime index.

        These values are structural addresses, not model weights. Quantizing one
        changes which embedding rows are fetched and invalidates the model even
        when every row payload itself round-trips correctly.
        """
        ple = self.plan.ple
        if ple is None:
            return

        expected = {
            "layer_multipliers": None,
            "ngram_heads_offsets": None,
            "ngram_heads_vocab_sizes": None,
        }
        for address in ple.index_tensors:
            key = next((name for name in expected if address.name.endswith(name)), None)
            if key is None:
                continue
            value = self.reader.read_tensor(address)
            if value.dtype != torch.int64:
                raise WriteError(
                    f"PLE structural tensor {address.name} must be int64, got {value.dtype}"
                )
            expected[key] = [int(item) for item in value.view(-1).tolist()]

        missing = [name for name, values in expected.items() if values is None]
        if missing:
            raise WriteError("Missing required PLE structural tensors: " + ", ".join(missing))

        self.plan.manifest.ple_index = build_ple_index(
            ple,
            ple.ngram_size,
            expected["ngram_heads_offsets"],
            expected["ngram_heads_vocab_sizes"],
            expected["layer_multipliers"],
        )

    def _address_of(self, item: DenseWorkItem):
        return item.address

    def _write_experts(
        self,
        journal: BuildJournal,
        on_item: Optional[Callable] = None,
        on_start: Optional[Callable] = None,
        on_bytes: Optional[Callable] = None,
    ) -> Tuple[int, int, float]:
        layout = self.plan.manifest.expert_layout
        if layout is None or not self.plan.experts:
            return 0, 0, 0.0

        _, done, _ = journal.as_sets()
        pending = [i for i in self.plan.experts if i.record_index not in done]
        if not pending:
            return 0, len(self.plan.experts), 0.0

        _preallocate(self.bank_path, layout.total_bytes)
        written = 0
        peak = 0.0
        chunk_cb = (lambda n, _: on_bytes(n)) if on_bytes else None

        with open(self.bank_path, "r+b") as bank:
            for batch in _batches(pending, EXPERT_BATCH_ITEMS):
                committed = []
                for item in batch:
                    if on_start:
                        on_start("expert", f"L{item.layer}E{item.expert}")
                    payload, item_peak = self._encode_expert(item, chunk_cb=chunk_cb)
                    peak = max(peak, item_peak)
                    bank.seek(item.bank_offset)
                    bank.write(payload)
                    written += len(payload)
                    committed.append(
                        (
                            item,
                            ItemChecksum(
                                value=crc32c_hex(payload),
                                offset=item.bank_offset,
                                length=len(payload),
                            ),
                        )
                    )
                    del payload
                bank.flush()
                os.fsync(bank.fileno())
                for item, checksum in committed:
                    journal.expert_done.append(item.record_index)
                    journal.expert_checksums[str(item.record_index)] = checksum
                self._save_journal(journal)
                if on_item:
                    for item, _ in committed:
                        on_item("expert", f"L{item.layer}E{item.expert}")
        return written, len(self.plan.experts) - len(pending), peak

    def _encode_expert(self, item: ExpertWorkItem, chunk_cb: Optional[Callable] = None) -> Tuple[bytes, float]:
        record = self.plan.manifest.expert_layout.record
        payload = bytearray()
        peak = 0.0

        for source in item.expert_slice.projections:
            projection = record.projection(source.projection)
            if projection is None:
                raise WriteError(
                    f"L{item.layer}E{item.expert}: projection '{source.projection}' is not in the record layout"
                )
            weight = self._read_slice(source, chunk_cb=chunk_cb)
            result, _, item_peak = self._quantize(
                weight, projection.bits, projection.group_size, projection.symmetric
            )
            peak = max(peak, item_peak)
            payload.extend(
                _sections_to_bytes(
                    result, projection.spans, f"L{item.layer}E{item.expert}.{source.projection}"
                )
            )
            del weight

        if len(payload) != record.payload_bytes:
            raise WriteError(
                f"L{item.layer}E{item.expert}: record is {len(payload)} bytes, "
                f"plan reserved {record.payload_bytes}"
            )
        return bytes(payload), peak

    def _write_ple(
        self,
        journal: BuildJournal,
        on_item: Optional[Callable] = None,
        on_start: Optional[Callable] = None,
        on_bytes: Optional[Callable] = None,
    ) -> Tuple[int, int, float]:
        ple = self.plan.ple
        if ple is None:
            return 0, 0, 0.0

        _, _, done = journal.as_sets()
        pending = [s for s in ple.shards if s.shard_index not in done]
        if not pending:
            return 0, len(ple.shards), 0.0

        _preallocate(self.ple_path, ple.total_bytes)
        row = ple.row
        row_spans = section_spans([1, row.row_width], row.bits, row.group_size, row.symmetric)
        written = 0
        peak = 0.0
        chunk_cb = (lambda n, _: on_bytes(n)) if on_bytes else None

        with open(self.ple_path, "r+b") as table:
            for batch in _batches(pending, 1):
                committed = []
                for shard in batch:
                    if on_start:
                        on_start("ple", f"shard_{shard.shard_index}")
                    block, shard_peak = self._encode_ple_shard(shard, row_spans, chunk_cb=chunk_cb)
                    peak = max(peak, shard_peak)
                    # Rows are page-packed, so write row by row rather than as one run.
                    for offset_in_shard in range(shard.num_rows):
                        start = offset_in_shard * row.payload_bytes
                        table.seek(row.row_offset(shard.first_row + offset_in_shard))
                        table.write(block[start : start + row.payload_bytes])
                    written += len(block)
                    committed.append(
                        (
                            shard,
                            ItemChecksum(
                                value=crc32c_hex(block),
                                length=len(block),
                                first_row=shard.first_row,
                                num_rows=shard.num_rows,
                            ),
                        )
                    )
                    del block
                table.flush()
                os.fsync(table.fileno())
                for shard, checksum in committed:
                    journal.ple_done.append(shard.shard_index)
                    journal.ple_checksums[str(shard.shard_index)] = checksum
                self._save_journal(journal)
                if on_item:
                    for shard, _ in committed:
                        on_item("ple", f"shard_{shard.shard_index}")
        return written, len(ple.shards) - len(pending), peak

    def _encode_ple_shard(
        self,
        shard: PleShardWorkItem,
        row_spans: Sequence[SectionSpan],
        chunk_cb: Optional[Callable] = None,
    ) -> Tuple[bytes, float]:
        """Quantize one source shard and interleave it into self-contained rows.

        The quantizer emits packed codes, scales, and zeros as three separate
        tensors; on disk each row must carry its own metadata so a single read
        decodes without touching anything else.
        """
        row = self.plan.ple.row
        tensor = self.reader.read_tensor(shard.address, chunk_callback=chunk_cb)
        result, _, peak = self._quantize(tensor, row.bits, row.group_size, row.symmetric)
        del tensor

        sources = {
            Section.PACKED: result.packed_weights,
            Section.SCALES: result.scales,
            Section.ZEROS: result.zeros,
        }
        columns = []
        for span in row_spans:
            source = sources[span.section]
            if source is None:
                raise WriteError(
                    f"ple shard {shard.shard_index}: quantizer produced no {span.section.value} section"
                )
            raw = source.detach().cpu().contiguous().view(torch.uint8).numpy()
            expected = shard.num_rows * span.length
            if raw.size != expected:
                raise WriteError(
                    f"ple shard {shard.shard_index}: {span.section.value} is {raw.size} bytes, "
                    f"plan reserved {expected}"
                )
            columns.append(raw.reshape(shard.num_rows, span.length))

        block = np.concatenate(columns, axis=1).tobytes()
        expected = shard.num_rows * row.payload_bytes
        if len(block) != expected:
            raise WriteError(
                f"ple shard {shard.shard_index}: produced {len(block)} bytes, expected {expected}"
            )
        return block, peak

    # -- entry point ------------------------------------------------------ #

    def build(
        self,
        on_item: Optional[Callable[[str, str], None]] = None,
        on_start: Optional[Callable[[str, str], None]] = None,
        on_bytes: Optional[Callable[[int], None]] = None,
    ) -> BuildResult:
        """Execute the plan. Safe to call repeatedly to resume."""
        started = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._preflight_disk_space()
        journal = self._load_journal()
        self._materialize_ple_index()

        written = skipped = 0
        peak = 0.0
        for region in (self._write_dense, self._write_experts, self._write_ple):
            region_bytes, region_skipped, region_peak = region(journal, on_item, on_start, on_bytes)
            written += region_bytes
            skipped += region_skipped
            peak = max(peak, region_peak)

        self._write_metadata(journal)
        journal.finished = True
        self._save_journal(journal)

        total_items = self.plan.num_work_items
        return BuildResult(
            output_dir=str(self.output_dir),
            finished=True,
            items_written=total_items - skipped,
            items_skipped=skipped,
            bytes_written=written,
            peak_vram_mb=peak,
            elapsed_s=time.perf_counter() - started,
        )

    def _copy_runtime_assets(self) -> None:
        """Copy only text-runtime assets while preserving tokenizer IDs exactly."""
        metadata = self.output_dir / METADATA_DIR
        tokenizer = self.output_dir / "tokenizer"
        metadata.mkdir(parents=True, exist_ok=True)
        tokenizer.mkdir(parents=True, exist_ok=True)

        model_assets = {"config.json": metadata / "config.json"}
        optional_model_assets = {"generation_config.json": metadata / "generation_config.json"}
        tokenizer_assets = (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
        )

        if hasattr(self.reader, "root_dir"):
            root = Path(self.reader.root_dir)

            def read_asset(name: str):
                path = root / name
                return path.read_bytes() if path.is_file() else None

        else:
            try:
                from huggingface_hub.errors import EntryNotFoundError
            except ImportError:
                try:
                    from huggingface_hub.utils import EntryNotFoundError
                except ImportError:
                    try:
                        from huggingface_hub import EntryNotFoundError
                    except ImportError:
                        EntryNotFoundError = Exception
            from huggingface_hub import hf_hub_download

            def read_asset(name: str):
                try:
                    path = hf_hub_download(
                        repo_id=self.reader.model_id,
                        filename=name,
                        revision=self.reader.revision,
                        token=self.reader.token,
                    )
                except (EntryNotFoundError, Exception):
                    return None
                return Path(path).read_bytes()

        for name, destination in model_assets.items():
            payload = read_asset(name)
            if payload is None:
                raise WriteError(f"Required runtime metadata asset is missing: {name}")
            durable_replace(destination, payload)
        for name, destination in optional_model_assets.items():
            payload = read_asset(name)
            if payload is not None:
                durable_replace(destination, payload)
        for name in tokenizer_assets:
            payload = read_asset(name)
            if payload is not None:
                durable_replace(tokenizer / name, payload)

    def _write_metadata(self, journal: BuildJournal) -> None:
        """Emit package descriptors and integrity indexes atomically."""
        manifest = self.plan.manifest
        self._copy_runtime_assets()
        if manifest.expert_layout is not None:
            path = self.output_dir / EXPERTS_DIR / EXPERT_LAYOUT_NAME
            durable_replace(path, manifest.expert_layout.model_dump_json(indent=2).encode("utf-8"))
        if manifest.ple_index is not None:
            path = self.output_dir / PLE_DIR / PLE_INDEX_NAME
            durable_replace(path, manifest.ple_index.model_dump_json(indent=2).encode("utf-8"))
        elif self.plan.ple is not None:
            raise WriteError("PLE table exists but ple/index.json could not be materialized")

        region_paths = {
            "dense": self.dense_path,
            "experts": self.bank_path,
            "ple": self.ple_path,
        }
        manifest.regions = {
            name: RegionIntegrity(
                path=path.relative_to(self.output_dir).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for name, path in region_paths.items()
            if path.exists()
        }
        checksums = PackageChecksums(
            dense=journal.dense_checksums,
            experts=journal.expert_checksums,
            ple=journal.ple_checksums,
        )
        durable_replace(
            self.output_dir / INTEGRITY_DIR / CHECKSUMS_NAME,
            checksums.model_dump_json(indent=2).encode("utf-8"),
        )
        durable_replace(
            self.output_dir / MANIFEST_NAME,
            manifest.model_dump_json(indent=2).encode("utf-8"),
        )
