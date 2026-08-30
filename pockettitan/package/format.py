"""On-disk format for a PocketTitan package (R1).

Layout::

    model.ptitan/
      manifest.json        provenance, precision map, tier assignment, totals
      dense/blob.bin       VRAM-resident core, offsets in the manifest
      experts/bank.bin     [layer][expert] records, each one contiguous read
      experts/layout.json  (layer, expert) -> byte_offset, length, section offsets
      ple/table.bin        row-aligned n-gram table
      ple/index.json       head offsets, vocab moduli, hash multipliers, row stride
      tokenizer/           copied verbatim from the source repo

Two invariants drive every decision here:

**One expert is one read.** In the source checkpoint an expert is split across
two fused tensors stored ~120 GB apart. The bank interleaves each expert's
projections into a single contiguous record so the runtime issues one ``pread``
per expert instead of two seeks.

**A PLE row never straddles a page.** Rows are page-packed: whole rows only, one
page at a time, so a single row read never costs two pages.
"""

import math
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

PACKAGE_VERSION = "1.1"
PAGE_BYTES = 4096

MANIFEST_NAME = "manifest.json"
DENSE_DIR = "dense"
EXPERTS_DIR = "experts"
PLE_DIR = "ple"
TOKENIZER_DIR = "tokenizer"
METADATA_DIR = "metadata"
INTEGRITY_DIR = "integrity"
EXPERT_BANK_NAME = "bank.bin"
EXPERT_LAYOUT_NAME = "layout.json"
PLE_TABLE_NAME = "table.bin"
PLE_INDEX_NAME = "index.json"
CHECKSUMS_NAME = "checksums.json"


class CodecSpec(BaseModel):
    """Self-contained description of a binary weight codec.

    Runtime code must be able to choose a decoder from this descriptor without
    importing the Python quantizer implementation that produced the package.
    """

    id: str
    version: int = 1
    method: str
    logical_bits: float
    storage_bits: float
    block_size: int = 1
    group_size: int = -1
    symmetric: bool = False
    scale_dtype: Optional[str] = "float16"
    zero_dtype: Optional[str] = "float16"
    byte_order: str = "little"
    packing_order: str = "least-significant-first"


class SourceProvenance(BaseModel):
    """Immutable source identity for a resumable package build."""

    repository: str
    revision: str
    config_sha256: str
    index_sha256: Optional[str] = None


class RegionIntegrity(BaseModel):
    """Expected and observed integrity metadata for one package region."""

    path: str
    size_bytes: int
    sha256: Optional[str] = None


class ItemChecksum(BaseModel):
    """Checksum and address metadata for one independently committed item."""

    algorithm: str = "crc32c"
    value: str
    offset: Optional[int] = None
    length: int
    first_row: Optional[int] = None
    num_rows: Optional[int] = None


class PackageChecksums(BaseModel):
    """Per-item integrity index emitted after all durable batches commit."""

    version: int = 1
    dense: Dict[str, ItemChecksum] = Field(default_factory=dict)
    experts: Dict[str, ItemChecksum] = Field(default_factory=dict)
    ple: Dict[str, ItemChecksum] = Field(default_factory=dict)


class Section(str, Enum):
    """Named byte ranges inside one packed record."""

    PACKED = "packed"
    SCALES = "scales"
    ZEROS = "zeros"


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    if alignment <= 1:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def storage_bits(nominal_bits: float) -> float:
    """Bits actually consumed on disk per weight.

    The sub-byte packer stores ``8 // bits`` values per byte, so only widths that
    divide 8 pack densely. A 3-bit weight occupies 4 bits and a 6-bit weight
    occupies 8. Planning with nominal bits would under-predict the package by
    several GiB, so every byte estimate goes through here.
    """
    bits = int(nominal_bits)
    if bits >= 8 or bits <= 0:
        return float(nominal_bits)
    values_per_byte = 8 // bits
    return 8.0 / values_per_byte


def is_dense_bit_width(nominal_bits: float) -> bool:
    """True when a bit width packs with no wasted bits (1, 2, 4, 8)."""
    return storage_bits(nominal_bits) == float(int(nominal_bits))


def packed_bytes(num_elements: int, bits: float) -> int:
    """Bytes for ``num_elements`` weights, accounting for packing density."""
    return int(math.ceil(num_elements * storage_bits(bits) / 8.0))


def num_groups(num_elements: int, group_size: int) -> int:
    """Group count for group-wise quantization; ``group_size <= 0`` means one group."""
    if group_size <= 0:
        return 1
    return int(math.ceil(num_elements / group_size))


def matrix_dims(shape: List[int]) -> tuple:
    """``(rows, in_features)`` as the quantizers see it: ``weight.view(-1, shape[-1])``."""
    if not shape:
        return 1, 1
    if len(shape) == 1:
        return 1, shape[0]
    return math.prod(shape[:-1]), shape[-1]


def section_spans(
    shape: List[int],
    bits: float,
    group_size: int,
    symmetric: bool,
    scale_bytes: int = 2,
    zero_bytes: int = 2,
) -> List["SectionSpan"]:
    """Byte spans for one quantized matrix: packed codes, then scales, then zeros.

    Group-wise quantizers pad ``in_features`` up to a multiple of ``group_size``
    *before* packing, so a 64-wide row at ``group_size=128`` emits 128 codes.
    Sizing the section from the unpadded element count under-reserves the record
    and shifts everything after it, so the padding is modelled here.
    """
    rows, in_features = matrix_dims(shape)

    if bits >= 16:
        return [SectionSpan(section=Section.PACKED, offset=0, length=rows * in_features * 2)]

    effective_group = group_size if group_size > 0 else in_features
    groups_per_row = int(math.ceil(in_features / effective_group))
    padded_in = groups_per_row * effective_group
    groups = rows * groups_per_row

    spans = [
        SectionSpan(section=Section.PACKED, offset=0, length=packed_bytes(rows * padded_in, bits))
    ]
    offset = spans[0].length
    spans.append(SectionSpan(section=Section.SCALES, offset=offset, length=groups * scale_bytes))
    if not symmetric:
        offset += spans[-1].length
        spans.append(SectionSpan(section=Section.ZEROS, offset=offset, length=groups * zero_bytes))
    return spans


class SectionSpan(BaseModel):
    """Offset and length of one section, relative to its record."""

    section: Section
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


class ProjectionLayout(BaseModel):
    """Byte layout of one projection matrix inside an expert record."""

    name: str = Field(description="Projection name, e.g. 'gate_up_proj'")
    shape: List[int]
    bits: float
    group_size: int
    symmetric: bool
    codec_id: str = "pt.scalar.rtn.v1"
    offset: int = Field(description="Start of this projection within the record")
    spans: List[SectionSpan] = Field(default_factory=list)

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape) if self.shape else 0

    @property
    def length(self) -> int:
        return sum(s.length for s in self.spans)

    def span(self, section: Section) -> Optional[SectionSpan]:
        for s in self.spans:
            if s.section is section:
                return s
        return None


class ExpertRecordLayout(BaseModel):
    """Byte layout shared by every expert record in the bank.

    All experts in a package have identical geometry and precision, so one
    layout describes all 24,576 records. The runtime needs only this plus a base
    offset to address any expert.
    """

    projections: List[ProjectionLayout] = Field(default_factory=list)
    payload_bytes: int = 0
    stride: int = Field(default=0, description="Record pitch including alignment padding")
    alignment: int = PAGE_BYTES

    @property
    def num_params(self) -> int:
        return sum(p.num_elements for p in self.projections)

    @property
    def padding_bytes(self) -> int:
        return self.stride - self.payload_bytes

    def projection(self, name: str) -> Optional[ProjectionLayout]:
        for p in self.projections:
            if p.name == name:
                return p
        return None

    @classmethod
    def build(
        cls,
        projections: List[Dict[str, Any]],
        alignment: int = PAGE_BYTES,
        scale_bytes: int = 2,
        zero_bytes: int = 2,
    ) -> "ExpertRecordLayout":
        """Compute the record layout.

        Args:
            projections: One dict per projection with keys ``name``, ``shape``,
                ``bits``, ``group_size``, ``symmetric``.
            alignment: Record pitch alignment. Page alignment keeps every expert
                read page-aligned, which matters for direct I/O.
            scale_bytes: Bytes per group scale (fp16).
            zero_bytes: Bytes per group zero-point (fp16), omitted when symmetric.
        """
        laid_out: List[ProjectionLayout] = []
        cursor = 0

        for spec in projections:
            shape = list(spec["shape"])
            bits = float(spec["bits"])
            group_size = int(spec["group_size"])
            symmetric = bool(spec.get("symmetric", False))
            codec_id = str(spec.get("codec_id", "pt.scalar.rtn.v1"))
            spans = section_spans(shape, bits, group_size, symmetric, scale_bytes, zero_bytes)
            offset = sum(sp.length for sp in spans)

            laid_out.append(
                ProjectionLayout(
                    name=spec["name"],
                    shape=shape,
                    bits=bits,
                    group_size=group_size,
                    symmetric=symmetric,
                    codec_id=codec_id,
                    offset=cursor,
                    spans=spans,
                )
            )
            cursor += offset

        return cls(
            projections=laid_out,
            payload_bytes=cursor,
            stride=align_up(cursor, alignment),
            alignment=alignment,
        )

    def record_offset(self, record_index: int) -> int:
        """Byte offset of a record within the bank."""
        return record_index * self.stride

    def section_offset(self, record_index: int, projection: str, section: Section) -> int:
        """Absolute bank offset of one section of one projection."""
        proj = self.projection(projection)
        if proj is None:
            raise KeyError(f"Unknown projection '{projection}'")
        span = proj.span(section)
        if span is None:
            raise KeyError(f"Projection '{projection}' has no {section.value} section")
        return self.record_offset(record_index) + proj.offset + span.offset


class ExpertRecordEntry(BaseModel):
    """Explicit logical-to-physical expert mapping used by sparse canary packages."""

    record_index: int
    layer: int
    expert: int
    byte_offset: int


class ExpertLayout(BaseModel):
    """Index over the expert bank: ``(layer, expert) -> record``."""

    num_layers: int
    num_experts: int
    layers: List[int] = Field(
        default_factory=list, description="Source layer indices, in bank order"
    )
    record: ExpertRecordLayout = Field(default_factory=ExpertRecordLayout)
    records: List[ExpertRecordEntry] = Field(
        default_factory=list,
        description="Explicit sparse mapping; empty means the full layer-major Cartesian layout",
    )

    @property
    def num_records(self) -> int:
        return len(self.records) if self.records else len(self.layers) * self.num_experts

    @property
    def total_bytes(self) -> int:
        return self.num_records * self.record.stride

    def record_index(self, layer: int, expert: int) -> int:
        """Bank slot for one expert. Layer-major, so a layer's experts are adjacent."""
        if self.records:
            for entry in self.records:
                if entry.layer == layer and entry.expert == expert:
                    return entry.record_index
            raise KeyError(f"Expert ({layer}, {expert}) is not present in this sparse package")
        try:
            position = self.layers.index(layer)
        except ValueError as exc:
            raise KeyError(f"Layer {layer} has no experts in this package") from exc
        if not 0 <= expert < self.num_experts:
            raise KeyError(f"Expert {expert} out of range [0, {self.num_experts})")
        return position * self.num_experts + expert

    def byte_range(self, layer: int, expert: int) -> tuple:
        """``(offset, length)`` of one expert record — the single read."""
        offset = self.record.record_offset(self.record_index(layer, expert))
        return offset, self.record.payload_bytes


class PleRowLayout(BaseModel):
    """Row geometry for the n-gram table.

    Rows are **page-packed**: as many whole rows as fit go into each 4 KiB page
    and the remainder of the page is left unused. That guarantees a row never
    straddles a page — a straddling row doubles the cost of every lookup — while
    wasting far less than rounding the stride up to a power of two would. For an
    82-byte row, page packing wastes 1.9% where a 128-byte stride wastes 36%.
    """

    row_width: int = Field(description="Elements per row")
    bits: float
    group_size: int
    symmetric: bool = True
    codec_id: str = "pt.ple.rtn.v1"
    payload_bytes: int = 0
    rows_per_page: int = 1
    page_bytes: int = PAGE_BYTES

    @classmethod
    def build(
        cls,
        row_width: int,
        bits: float,
        group_size: int,
        symmetric: bool = True,
        scale_bytes: int = 2,
        zero_bytes: int = 2,
        page_bytes: int = PAGE_BYTES,
        codec_id: str = "pt.ple.rtn.v1",
    ) -> "PleRowLayout":
        # A group must never span rows: each row has to decode independently.
        group_size = min(group_size, row_width) if group_size > 0 else row_width
        payload = sum(
            span.length
            for span in section_spans(
                [1, row_width], bits, group_size, symmetric, scale_bytes, zero_bytes
            )
        )

        if payload > page_bytes:
            # A row larger than a page gets its own aligned run of pages.
            page_bytes = align_up(payload, PAGE_BYTES)
            rows_per_page = 1
        else:
            rows_per_page = max(1, page_bytes // payload)

        return cls(
            row_width=row_width,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            codec_id=codec_id,
            payload_bytes=payload,
            rows_per_page=rows_per_page,
            page_bytes=page_bytes,
        )

    @property
    def waste_fraction(self) -> float:
        """Fraction of each page left unused by row packing."""
        used = self.rows_per_page * self.payload_bytes
        return (self.page_bytes - used) / self.page_bytes

    def row_offset(self, row_id: int) -> int:
        """Byte offset of a row. Never straddles a page by construction."""
        page, slot = divmod(row_id, self.rows_per_page)
        return page * self.page_bytes + slot * self.payload_bytes

    def bytes_for(self, total_rows: int) -> int:
        """Total table size for ``total_rows`` rows, rounded to whole pages."""
        pages = int(math.ceil(total_rows / self.rows_per_page))
        return pages * self.page_bytes


class PleShardMapping(BaseModel):
    """Logical source rows mapped into a compact physical PLE table."""

    shard_index: int
    logical_first_row: int
    physical_first_row: int
    num_rows: int


class PleIndex(BaseModel):
    """Everything needed to resolve a token n-gram to table rows.

    Row ids are a pure function of token ids, which is what makes PLE prefetch
    exact rather than speculative (Plan.md R5).
    """

    ngram_size: int
    num_heads: int
    heads_per_ngram: int = Field(
        description="Number of independently hashed heads for each order from bigram through ngram_size"
    )
    head_offsets: List[int] = Field(description="Cumulative row offset per head")
    head_vocab_sizes: List[int] = Field(description="Per-head hash modulus (distinct primes)")
    layer_multipliers: List[int] = Field(description="Hash constants, one per n-gram position")
    total_rows: int = Field(description="Logical source-table row count")
    physical_rows: int = Field(description="Rows physically included in this package")
    shards: List["PleShardMapping"] = Field(default_factory=list)
    row: PleRowLayout
    source_layer: int = Field(description="Layer index the PLE block belongs to")

    @property
    def total_bytes(self) -> int:
        return self.row.bytes_for(self.physical_rows)

    def physical_row_id(self, logical_row_id: int) -> int:
        """Translate a logical hash row to the compact package row address."""
        for shard in self.shards:
            if shard.logical_first_row <= logical_row_id < shard.logical_first_row + shard.num_rows:
                return shard.physical_first_row + (logical_row_id - shard.logical_first_row)
        raise KeyError(f"logical PLE row {logical_row_id} is not present in this package")

    def row_id(self, head: int, tokens: List[int]) -> int:
        """Resolve one head's row for an n-gram of token ids.

        Mirrors Qwen4ExpTextNGramEmbedding: the first ``heads_per_ngram``
        heads hash bigrams, the next group hashes trigrams, and so on. Hash
        position zero is the current token, followed by previous tokens.
        Arithmetic wraps as signed int64 before ``remainder``.
        """
        if head < 0 or head >= self.num_heads:
            raise IndexError(f"head {head} out of range [0, {self.num_heads})")
        ngram_order = 2 + head // self.heads_per_ngram
        if len(tokens) < ngram_order:
            raise ValueError(f"head {head} requires {ngram_order} tokens, got {len(tokens)}")

        def signed_int64(value: int) -> int:
            value &= 0xFFFFFFFFFFFFFFFF
            return value - (1 << 64) if value >= (1 << 63) else value

        current_first = reversed(tokens[-ngram_order:])
        acc = 0
        for position, token in enumerate(current_first):
            product = signed_int64(token * self.layer_multipliers[position])
            acc = signed_int64(acc ^ product)
        return self.head_offsets[head] + (acc % self.head_vocab_sizes[head])

    def row_ids(self, tokens: List[int]) -> List[int]:
        """All rows for one token position — the unit of prefetch."""
        return [self.row_id(h, tokens) for h in range(self.num_heads)]


class DenseTensorEntry(BaseModel):
    """One tensor in the dense core blob, with its byte address."""

    name: str
    component: str
    shape: List[int]
    bits: float
    group_size: int
    symmetric: bool
    codec_id: str = "pt.scalar.rtn.v1"
    num_params: int
    packed_bytes: int
    byte_offset: int = 0
    spans: List[SectionSpan] = Field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(s.length for s in self.spans) or self.packed_bytes

    def span(self, section: Section) -> Optional[SectionSpan]:
        for s in self.spans:
            if s.section is section:
                return s
        return None


class PackageTotals(BaseModel):
    """Byte and parameter accounting, for validation against the R0 audit."""

    source_params: int = 0
    packaged_params: int = 0
    dropped_params: int = 0
    dense_bytes: int = 0
    expert_bytes: int = 0
    ple_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.dense_bytes + self.expert_bytes + self.ple_bytes

    @property
    def average_bits(self) -> float:
        return (self.total_bytes * 8.0 / self.packaged_params) if self.packaged_params else 0.0


class PackageManifest(BaseModel):
    """Root descriptor written to ``manifest.json``."""

    package_version: str = PACKAGE_VERSION
    pockettitan_version: str = ""
    created_utc: str = ""

    source_model: str
    source_revision: Optional[str] = None
    source: Optional[SourceProvenance] = None
    architecture: str = ""
    build_profile: str = "full"
    complete_model: bool = True

    features: List[str] = Field(default_factory=list)
    precision_map_name: str = ""
    precision_map: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    codecs: Dict[str, CodecSpec] = Field(default_factory=dict)
    regions: Dict[str, RegionIntegrity] = Field(default_factory=dict)

    totals: PackageTotals = Field(default_factory=PackageTotals)
    dense: List[DenseTensorEntry] = Field(default_factory=list)
    expert_layout: Optional[ExpertLayout] = None
    ple_index: Optional[PleIndex] = None

    activated_params_per_token: int = 0
    expert_params_per_token: int = 0
    expert_bytes_per_token: int = 0
    reads_per_token: int = 0
