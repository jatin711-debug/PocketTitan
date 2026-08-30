"""Self-contained validator for the PocketTitan package ABI."""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from pockettitan.package.format import (
    CHECKSUMS_NAME,
    INTEGRITY_DIR,
    MANIFEST_NAME,
    METADATA_DIR,
    PACKAGE_VERSION,
    PLE_INDEX_NAME,
    PackageChecksums,
    PackageManifest,
    Section,
)
from pockettitan.package.integrity import crc32c, crc32c_hex, sha256_file
from pockettitan.package.writer import BuildJournal, JOURNAL_NAME


ValidationMode = Literal["fast", "full"]


class PtitanValidationReport(BaseModel):
    is_valid: bool
    mode: ValidationMode
    items_checked: int = 0
    bytes_checked: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _canonical_json_sha256(payload) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_exact(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(length)
    if len(payload) != length:
        raise ValueError(
            f"short read from {path}: requested [{offset}, {offset + length}), got {len(payload)} bytes"
        )
    return payload


def _check_non_overlapping(spans: list[tuple[int, int, str]], limit: int) -> list[str]:
    errors = []
    previous_end = 0
    for start, end, label in sorted(spans):
        if start < 0 or end < start or end > limit:
            errors.append(f"{label} span [{start}, {end}) is outside [0, {limit})")
        if start < previous_end:
            errors.append(f"{label} overlaps a preceding span at byte {start}")
        previous_end = max(previous_end, end)
    return errors


def _text_config(config: dict) -> dict:
    return config.get("text_config", config)


class PtitanValidator:
    """Validate a package using only its public ABI metadata."""

    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)

    def validate(self, mode: ValidationMode = "fast") -> PtitanValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        checked = 0
        checked_bytes = 0

        if mode not in ("fast", "full"):
            return PtitanValidationReport(
                is_valid=False, mode="fast", errors=[f"unsupported validation mode: {mode}"]
            )
        if not self.package_dir.is_dir():
            return PtitanValidationReport(
                is_valid=False,
                mode=mode,
                errors=[f"package directory does not exist: {self.package_dir}"],
            )

        manifest_path = self.package_dir / MANIFEST_NAME
        checksums_path = self.package_dir / INTEGRITY_DIR / CHECKSUMS_NAME
        journal_path = self.package_dir / JOURNAL_NAME
        config_path = self.package_dir / METADATA_DIR / "config.json"
        required = (manifest_path, checksums_path, journal_path, config_path)
        missing = [
            str(path.relative_to(self.package_dir)) for path in required if not path.is_file()
        ]
        if missing:
            return PtitanValidationReport(
                is_valid=False,
                mode=mode,
                errors=["missing required files: " + ", ".join(missing)],
            )

        try:
            manifest = PackageManifest.model_validate_json(manifest_path.read_text("utf-8"))
            checksums = PackageChecksums.model_validate_json(checksums_path.read_text("utf-8"))
            journal = BuildJournal.model_validate_json(journal_path.read_text("utf-8"))
            source_config = json.loads(config_path.read_text("utf-8"))
        except (OSError, ValueError, ValidationError) as exc:
            return PtitanValidationReport(
                is_valid=False, mode=mode, errors=[f"invalid package metadata: {exc}"]
            )

        if manifest.package_version != PACKAGE_VERSION:
            errors.append(
                f"unsupported package version {manifest.package_version!r}; expected {PACKAGE_VERSION!r}"
            )
        if manifest.source is None or not manifest.source.revision:
            errors.append("manifest has no immutable source provenance")
        elif manifest.source_revision != manifest.source.revision:
            errors.append("source_revision disagrees with source.revision")
        if manifest.source is not None:
            observed_config_hash = _canonical_json_sha256(source_config)
            if observed_config_hash != manifest.source.config_sha256:
                errors.append(
                    "metadata/config.json does not match the pinned source config SHA-256"
                )
        if not journal.finished:
            errors.append("build journal is not complete")

        expected_regions = {
            "dense": manifest.totals.dense_bytes,
            "experts": manifest.totals.expert_bytes,
            "ple": manifest.totals.ple_bytes,
        }
        for name, expected_size in expected_regions.items():
            if expected_size == 0:
                if name in manifest.regions:
                    errors.append(
                        f"zero-sized {name} region is unexpectedly present in the manifest"
                    )
                continue
            region = manifest.regions.get(name)
            if region is None:
                errors.append(f"manifest is missing the {name} region")
                continue
            path = self.package_dir / region.path
            if not path.is_file():
                errors.append(f"region file is missing: {region.path}")
                continue
            observed_size = path.stat().st_size
            if region.size_bytes != expected_size or observed_size != expected_size:
                errors.append(
                    f"{name} size mismatch: totals={expected_size}, manifest={region.size_bytes}, "
                    f"file={observed_size}"
                )
            if not region.sha256:
                errors.append(f"{name} region has no SHA-256")

        errors.extend(self._validate_dense_layout(manifest))
        errors.extend(self._validate_expert_layout(manifest))
        errors.extend(self._validate_ple_layout(manifest, checksums))
        errors.extend(self._validate_codecs(manifest))
        errors.extend(self._validate_vocabulary(manifest, source_config))
        errors.extend(self._validate_journal(manifest, checksums, journal))

        try:
            item_count, item_bytes, item_errors = self._validate_item_checksums(manifest, checksums)
            checked += item_count
            checked_bytes += item_bytes
            errors.extend(item_errors)
        except OSError as exc:
            errors.append(f"failed reading package payloads: {exc}")

        if mode == "full":
            for name, region in manifest.regions.items():
                path = self.package_dir / region.path
                if not path.is_file():
                    continue
                observed = sha256_file(path)
                checked_bytes += path.stat().st_size
                if observed != region.sha256:
                    errors.append(f"{name} region SHA-256 mismatch")
            errors.extend(self._validate_representative_decodes(manifest))

        generation = self.package_dir / METADATA_DIR / "generation_config.json"
        if not generation.exists():
            warnings.append("source repository has no generation_config.json")

        return PtitanValidationReport(
            is_valid=not errors,
            mode=mode,
            items_checked=checked,
            bytes_checked=checked_bytes,
            errors=errors,
            warnings=warnings,
        )

    def _region_path(self, manifest: PackageManifest, name: str) -> Path:
        return self.package_dir / manifest.regions[name].path

    def _validate_dense_layout(self, manifest: PackageManifest) -> list[str]:
        spans = [
            (item.byte_offset, item.byte_offset + item.length, f"dense tensor {item.name}")
            for item in manifest.dense
        ]
        return _check_non_overlapping(spans, manifest.totals.dense_bytes)

    def _validate_expert_layout(self, manifest: PackageManifest) -> list[str]:
        layout = manifest.expert_layout
        if manifest.totals.expert_bytes == 0:
            return [] if layout is None else ["expert layout exists for an empty expert region"]
        if layout is None:
            return ["expert region has no layout"]
        errors = []
        record = layout.record
        if record.alignment <= 0 or record.stride % record.alignment:
            errors.append("expert record stride is not aligned")
        if record.payload_bytes > record.stride:
            errors.append("expert record payload exceeds its stride")
        if layout.total_bytes != manifest.totals.expert_bytes:
            errors.append("expert record count/stride does not match expert region size")
        projection_spans = [
            (projection.offset, projection.offset + projection.length, projection.name)
            for projection in record.projections
        ]
        errors.extend(_check_non_overlapping(projection_spans, record.payload_bytes))
        return errors

    def _validate_ple_layout(
        self, manifest: PackageManifest, checksums: PackageChecksums
    ) -> list[str]:
        index = manifest.ple_index
        if manifest.totals.ple_bytes == 0:
            return [] if index is None else ["PLE index exists for an empty PLE region"]
        if index is None:
            return ["PLE region has no ple/index.json descriptor"]
        errors = []
        index_path = self.package_dir / "ple" / PLE_INDEX_NAME
        if not index_path.is_file():
            errors.append("ple/index.json is missing")
        else:
            try:
                sidecar = type(index).model_validate_json(index_path.read_text("utf-8"))
                if sidecar != index:
                    errors.append("ple/index.json disagrees with manifest.ple_index")
            except (OSError, ValueError, ValidationError) as exc:
                errors.append(f"invalid ple/index.json: {exc}")
        if index.num_heads != len(index.head_offsets) or index.num_heads != len(
            index.head_vocab_sizes
        ):
            errors.append("PLE head dimensions disagree")
        if len(index.layer_multipliers) != index.ngram_size:
            errors.append("PLE multiplier count disagrees with ngram_size")
        if any(size <= 0 for size in index.head_vocab_sizes):
            errors.append("PLE head vocab sizes must be positive")
        if index.head_offsets != sorted(index.head_offsets):
            errors.append("PLE head offsets are not monotonic")
        for head, (offset, size) in enumerate(
            zip(index.head_offsets, index.head_vocab_sizes, strict=False)
        ):
            if offset < 0 or offset + size > index.total_rows:
                errors.append(f"PLE head {head} row range is out of bounds")
        if index.total_bytes != manifest.totals.ple_bytes:
            errors.append("PLE row geometry does not match ple region size")
        row = index.row
        if row.payload_bytes <= 0 or row.rows_per_page <= 0:
            errors.append("PLE row geometry is invalid")
        elif row.rows_per_page * row.payload_bytes > row.page_bytes:
            errors.append("PLE rows overrun their page")

        ranges = []
        for shard, checksum in checksums.ple.items():
            if checksum.first_row is None or checksum.num_rows is None:
                errors.append(f"PLE checksum {shard} has no row range")
                continue
            ranges.append(
                (
                    checksum.first_row,
                    checksum.first_row + checksum.num_rows,
                    f"PLE source shard {shard}",
                )
            )
        errors.extend(_check_non_overlapping(ranges, index.physical_rows))
        if manifest.complete_model and ranges:
            ordered = sorted(ranges)
            if ordered[0][0] != 0 or ordered[-1][1] != index.physical_rows:
                errors.append("complete package PLE checksums do not cover every table row")
            for previous, current in zip(ordered, ordered[1:], strict=False):
                if previous[1] != current[0]:
                    errors.append("complete package PLE checksum ranges contain a gap")
                    break
        logical_ranges = [
            (
                shard.logical_first_row,
                shard.logical_first_row + shard.num_rows,
                f"PLE logical shard {shard.shard_index}",
            )
            for shard in index.shards
        ]
        physical_ranges = [
            (
                shard.physical_first_row,
                shard.physical_first_row + shard.num_rows,
                f"PLE physical shard {shard.shard_index}",
            )
            for shard in index.shards
        ]
        errors.extend(_check_non_overlapping(logical_ranges, index.total_rows))
        errors.extend(_check_non_overlapping(physical_ranges, index.physical_rows))
        if manifest.complete_model and logical_ranges:
            logical_ordered = sorted(logical_ranges)
            if logical_ordered[0][0] != 0 or logical_ordered[-1][1] != index.total_rows:
                errors.append("complete package PLE shard map does not cover every logical row")
        return errors

    def _validate_codecs(self, manifest: PackageManifest) -> list[str]:
        errors = []
        references = [(item.name, item.codec_id, item.bits) for item in manifest.dense]
        if manifest.expert_layout:
            references.extend(
                (f"expert.{projection.name}", projection.codec_id, projection.bits)
                for projection in manifest.expert_layout.record.projections
            )
        if manifest.ple_index:
            references.append(
                ("ple.rows", manifest.ple_index.row.codec_id, manifest.ple_index.row.bits)
            )
        for label, codec_id, bits in references:
            codec = manifest.codecs.get(codec_id)
            if codec is None:
                errors.append(f"{label} references unknown codec {codec_id}")
                continue
            if bits >= 16 and codec.method != "raw":
                errors.append(f"lossless tensor {label} references lossy codec {codec_id}")
            if codec.byte_order != "little":
                errors.append(f"unsupported byte order for codec {codec_id}: {codec.byte_order}")
        return errors

    def _validate_vocabulary(self, manifest: PackageManifest, config: dict) -> list[str]:
        text = _text_config(config)
        vocab_size = text.get("vocab_size")
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            return ["source config has no valid vocab_size"]
        errors = []
        vocabulary_tensors = [
            item
            for item in manifest.dense
            if item.name.endswith("embed_tokens.weight") or item.name.endswith("lm_head.weight")
        ]
        for item in vocabulary_tensors:
            if not item.shape or item.shape[0] != vocab_size:
                errors.append(
                    f"vocabulary tensor {item.name} has {item.shape[0] if item.shape else 0} rows, "
                    f"expected {vocab_size}"
                )
        for key, value in text.items():
            if not key.endswith("token_id"):
                continue
            ids = value if isinstance(value, list) else [value]
            if any(
                isinstance(token_id, int) and not 0 <= token_id < vocab_size for token_id in ids
            ):
                errors.append(f"config token IDs for {key} exceed the unchanged vocabulary")
        return errors

    def _validate_journal(
        self,
        manifest: PackageManifest,
        checksums: PackageChecksums,
        journal: BuildJournal,
    ) -> list[str]:
        errors = []
        dense_names = {item.name for item in manifest.dense}
        if set(journal.dense_done) != dense_names or set(checksums.dense) != dense_names:
            errors.append("dense journal/checksum coverage is incomplete")
        if journal.dense_checksums != checksums.dense:
            errors.append("dense journal checksums disagree with integrity/checksums.json")
        expected_experts = (
            {str(index) for index in range(manifest.expert_layout.num_records)}
            if manifest.expert_layout
            else set()
        )
        if {str(index) for index in journal.expert_done} != expected_experts:
            errors.append("expert journal coverage is incomplete")
        if set(checksums.experts) != expected_experts:
            errors.append("expert checksum coverage is incomplete")
        if journal.expert_checksums != checksums.experts:
            errors.append("expert journal checksums disagree with integrity/checksums.json")
        if set(journal.ple_checksums) != set(checksums.ple):
            errors.append("PLE journal/checksum coverage is incomplete")
        if journal.ple_checksums != checksums.ple:
            errors.append("PLE journal checksums disagree with integrity/checksums.json")
        return errors

    def _validate_item_checksums(
        self, manifest: PackageManifest, checksums: PackageChecksums
    ) -> tuple[int, int, list[str]]:
        errors = []
        items = 0
        byte_count = 0
        if manifest.totals.dense_bytes:
            path = self._region_path(manifest, "dense")
            for name, checksum in checksums.dense.items():
                if checksum.offset is None:
                    errors.append(f"dense checksum {name} has no offset")
                    continue
                payload = _read_exact(path, checksum.offset, checksum.length)
                if crc32c_hex(payload) != checksum.value:
                    errors.append(f"dense tensor checksum mismatch: {name}")
                items += 1
                byte_count += len(payload)
        if manifest.totals.expert_bytes:
            path = self._region_path(manifest, "experts")
            for index, checksum in checksums.experts.items():
                if checksum.offset is None:
                    errors.append(f"expert checksum {index} has no offset")
                    continue
                payload = _read_exact(path, checksum.offset, checksum.length)
                if crc32c_hex(payload) != checksum.value:
                    errors.append(f"expert record checksum mismatch: {index}")
                items += 1
                byte_count += len(payload)
        if manifest.totals.ple_bytes and manifest.ple_index:
            path = self._region_path(manifest, "ple")
            row = manifest.ple_index.row
            with path.open("rb") as stream:
                for shard, checksum in checksums.ple.items():
                    if checksum.first_row is None or checksum.num_rows is None:
                        continue
                    value = 0
                    length = 0
                    for row_id in range(checksum.first_row, checksum.first_row + checksum.num_rows):
                        stream.seek(row.row_offset(row_id))
                        payload = stream.read(row.payload_bytes)
                        if len(payload) != row.payload_bytes:
                            errors.append(f"PLE source shard {shard} contains a short row")
                            break
                        value = crc32c(payload, value)
                        length += len(payload)
                    if f"{value:08x}" != checksum.value or length != checksum.length:
                        errors.append(f"PLE source shard checksum mismatch: {shard}")
                    items += 1
                    byte_count += length
        return items, byte_count, errors

    def _validate_representative_decodes(self, manifest: PackageManifest) -> list[str]:
        """Independently inspect code/scales for one payload per registered codec."""
        errors = []
        seen: set[str] = set()
        dense_path = self._region_path(manifest, "dense") if manifest.totals.dense_bytes else None
        for item in manifest.dense:
            if item.codec_id in seen:
                continue
            payload = _read_exact(dense_path, item.byte_offset, item.length)
            error = self._decode_sanity(payload, item.spans, item.bits, item.codec_id)
            if error:
                errors.append(f"representative decode failed for {item.name}: {error}")
            seen.add(item.codec_id)
        if manifest.expert_layout and manifest.totals.expert_bytes:
            path = self._region_path(manifest, "experts")
            for projection in manifest.expert_layout.record.projections:
                if projection.codec_id in seen:
                    continue
                payload = _read_exact(path, projection.offset, projection.length)
                error = self._decode_sanity(
                    payload, projection.spans, projection.bits, projection.codec_id
                )
                if error:
                    errors.append(
                        f"representative decode failed for expert.{projection.name}: {error}"
                    )
                seen.add(projection.codec_id)
        if manifest.ple_index and manifest.totals.ple_bytes:
            row = manifest.ple_index.row
            if row.codec_id not in seen:
                payload = _read_exact(self._region_path(manifest, "ple"), 0, row.payload_bytes)
                from pockettitan.package.format import section_spans

                spans = section_spans([1, row.row_width], row.bits, row.group_size, row.symmetric)
                error = self._decode_sanity(payload, spans, row.bits, row.codec_id)
                if error:
                    errors.append(f"representative decode failed for PLE row: {error}")
        return errors

    @staticmethod
    def _decode_sanity(payload, spans, bits: float, codec_id: str) -> str | None:
        if codec_id.startswith("raw.f16"):
            values = np.frombuffer(payload[: min(len(payload), 4096)], dtype="<f2")
            return None if values.size and np.isfinite(values).all() else "non-finite fp16 values"
        sections = {
            span.section: payload[span.offset : span.offset + span.length] for span in spans
        }
        packed = sections.get(Section.PACKED, b"")
        scales_raw = sections.get(Section.SCALES, b"")
        if not packed or not scales_raw:
            return "missing packed codes or scales"
        scales = np.frombuffer(scales_raw, dtype="<f2")
        if not scales.size or not np.isfinite(scales).all() or np.any(scales <= 0):
            return "invalid quantization scales"
        integer_bits = int(bits)
        if integer_bits <= 0 or integer_bits > 8:
            return f"unsupported logical bit width {bits}"
        mask = (1 << integer_bits) - 1
        sample = np.frombuffer(packed[: min(len(packed), 4096)], dtype=np.uint8)
        values_per_byte = 8 // integer_bits
        codes = np.concatenate(
            [(sample >> (index * integer_bits)) & mask for index in range(values_per_byte)]
        )
        if not codes.size or int(codes.max()) > mask:
            return "packed codes exceed their logical range"
        zeros_raw = sections.get(Section.ZEROS)
        if zeros_raw:
            zeros = np.frombuffer(zeros_raw, dtype="<f2")
            if not np.isfinite(zeros).all() or np.any(zeros < 0) or np.any(zeros > mask):
                return "invalid zero points"
        bound = float(np.max(scales.astype(np.float32))) * max(mask, 1)
        if not math.isfinite(bound):
            return "non-finite reconstruction bound"
        return None
