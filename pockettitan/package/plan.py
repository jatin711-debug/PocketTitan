"""Build planning for a PocketTitan package (R1).

The planner is pure: given an R0 scan, a precision map, and a feature set, it
computes every byte of the output layout without reading a single weight. That
separation is what makes the build resumable and verifiable — the plan can be
diffed against the R0 storage budget before 360 GB moves anywhere.

Responsibilities, mapping to Plan.md R1 tasks:

* **T1.1** capability filtering (``--features text`` drops vision and MTP)
* **T1.3** expert record layout, so one expert becomes one contiguous read
* **T1.4** per-component precision assignment
* **T1.5** PLE row-store geometry and shard ordering
"""

import re
from datetime import datetime, timezone
import math
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from pockettitan.audit.budget import (
    PrecisionMap,
    build_activation_budget,
    infer_expert_geometry,
    infer_ple_geometry,
    text_config,
)
from pockettitan.audit.classify import Capability, Component, classify_all, classify_tensor
from pockettitan.audit.headers import ShardHeaderScan
from pockettitan.config import TensorAddress
from pockettitan.package.format import (
    PAGE_BYTES,
    SectionSpan,
    align_up,
    DenseTensorEntry,
    ExpertLayout,
    ExpertRecordLayout,
    PackageManifest,
    PackageTotals,
    PleIndex,
    PleRowLayout,
    is_dense_bit_width,
    matrix_dims,
    packed_bytes,
    section_spans,
    storage_bits,
)
from pockettitan.package.slicing import (
    ExpertSlice,
    SliceError,
    build_expert_slices,
    layer_index,
    projection_signature,
)

_PLE_SHARD_RE = re.compile(r"shard_(\d+)")

# Dense entries are 64-byte aligned so a quantized matrix starts on a cache line.
DENSE_ENTRY_ALIGNMENT = 64


class PlanError(ValueError):
    """Raised when a package cannot be planned from the given inputs."""


class DenseWorkItem(BaseModel):
    """One dense tensor to quantize and write into the core blob."""

    address: TensorAddress
    component: Component
    bits: float
    group_size: int
    symmetric: bool
    packed_bytes: int
    byte_offset: int = 0
    spans: List[SectionSpan] = Field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(s.length for s in self.spans)


class ExpertWorkItem(BaseModel):
    """One expert to read, quantize, and write as a single bank record."""

    expert_slice: ExpertSlice
    record_index: int
    bank_offset: int

    @property
    def layer(self) -> int:
        return self.expert_slice.layer

    @property
    def expert(self) -> int:
        return self.expert_slice.expert


class PleShardWorkItem(BaseModel):
    """One source n-gram shard to quantize into a row range of ``table.bin``."""

    address: TensorAddress
    shard_index: int
    first_row: int
    num_rows: int
    byte_offset: int

    @property
    def byte_length(self) -> int:
        return self.num_rows


class PlePlan(BaseModel):
    """Row-store geometry plus the source shards that fill it."""

    source_layer: int
    row: PleRowLayout
    total_rows: int
    declared_rows: int = Field(description="Rows addressable by the hash; the remainder is vocab padding")
    shards: List[PleShardWorkItem] = Field(default_factory=list)
    index_tensor_names: List[str] = Field(
        default_factory=list, description="Small int64 tensors the builder must read to emit ple/index.json"
    )

    @property
    def total_bytes(self) -> int:
        return self.row.bytes_for(self.total_rows)


class BuildPlan(BaseModel):
    """Complete, byte-exact description of a package build."""

    manifest: PackageManifest
    dense: List[DenseWorkItem] = Field(default_factory=list)
    experts: List[ExpertWorkItem] = Field(default_factory=list)
    ple: Optional[PlePlan] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def source_read_bytes(self) -> int:
        """Bytes that must be read from the source checkpoint."""
        dense = sum(i.address.size_bytes for i in self.dense)
        experts = sum(i.expert_slice.source_bytes for i in self.experts)
        ple = sum(i.address.size_bytes for i in self.ple.shards) if self.ple else 0
        return dense + experts + ple

    @property
    def num_work_items(self) -> int:
        return len(self.dense) + len(self.experts) + (len(self.ple.shards) if self.ple else 0)


def _quantized_shapes(dense, expert_layout, ple_plan):
    """Every (shape, group_size, label) that will be group-quantized."""
    for item in dense:
        if item.bits < 16:
            yield list(item.address.shape), item.group_size, item.component.value
    if expert_layout is not None:
        for projection in expert_layout.record.projections:
            if projection.bits < 16:
                yield projection.shape, projection.group_size, f"experts_routed.{projection.name}"
    if ple_plan is not None:
        yield [1, ple_plan.row.row_width], ple_plan.row.group_size, "ple_table"


def _ple_shard_index(name: str) -> int:
    match = _PLE_SHARD_RE.search(name)
    if match is None:
        raise PlanError(f"Cannot determine n-gram shard order from tensor name '{name}'")
    return int(match.group(1))


def _plan_ple(
    scan: ShardHeaderScan,
    precision_map: PrecisionMap,
) -> Optional[PlePlan]:
    """Lay out the n-gram row store and order its source shards."""
    geometry = infer_ple_geometry(scan)
    if geometry is None:
        return None

    table_tensors: List[TensorAddress] = []
    index_names: List[str] = []
    source_layer = -1

    for address in scan.tensors.values():
        component = classify_tensor(address.name).component
        if component is Component.PLE_TABLE:
            table_tensors.append(address)
            source_layer = layer_index(address.name) if source_layer < 0 else source_layer
        elif component is Component.PLE_PROJ and address.dtype.upper().startswith("I"):
            index_names.append(address.name)

    if not table_tensors:
        return None

    table_tensors.sort(key=lambda a: _ple_shard_index(a.name))

    entry = precision_map.entry_for(Component.PLE_TABLE)
    row = PleRowLayout.build(
        row_width=geometry.row_width,
        bits=entry.bits,
        group_size=entry.group_size,
        symmetric=entry.symmetric,
    )

    shards: List[PleShardWorkItem] = []
    cursor_rows = 0
    for address in table_tensors:
        rows = address.shape[0]
        shards.append(
            PleShardWorkItem(
                address=address,
                shard_index=_ple_shard_index(address.name),
                first_row=cursor_rows,
                num_rows=rows,
                byte_offset=row.row_offset(cursor_rows),
            )
        )
        cursor_rows += rows

    cfg = text_config(scan.config)
    declared = int(cfg.get("ngram_vocab_size_base", 0) or 0) * geometry.num_heads

    return PlePlan(
        source_layer=source_layer if source_layer >= 0 else 0,
        row=row,
        total_rows=cursor_rows,
        declared_rows=declared or cursor_rows,
        shards=shards,
        index_tensor_names=sorted(index_names),
    )


def _plan_experts(
    scan: ShardHeaderScan,
    precision_map: PrecisionMap,
    alignment: int,
) -> tuple:
    """Build expert slices and the shared record layout.

    Returns ``(items, layout)``, or ``([], None)`` for a dense checkpoint.
    """
    geometry = infer_expert_geometry(scan, classify_all(scan.tensors))
    if geometry is None:
        return [], None

    slices = build_expert_slices(scan.tensors, geometry.num_experts)
    if not slices:
        return [], None

    signature = projection_signature(slices)
    entry = precision_map.entry_for(Component.EXPERTS_ROUTED)
    record = ExpertRecordLayout.build(
        projections=[
            {
                "name": s["name"],
                "shape": s["shape"],
                "bits": entry.bits,
                "group_size": entry.group_size,
                "symmetric": entry.symmetric,
            }
            for s in signature
        ],
        alignment=alignment,
    )

    layers = sorted({s.layer for s in slices})
    layout = ExpertLayout(
        num_layers=len(layers),
        num_experts=geometry.num_experts,
        layers=layers,
        record=record,
    )

    items = [
        ExpertWorkItem(
            expert_slice=expert_slice,
            record_index=(index := layout.record_index(expert_slice.layer, expert_slice.expert)),
            bank_offset=record.record_offset(index),
        )
        for expert_slice in slices
    ]
    return items, layout


def plan_package(
    scan: ShardHeaderScan,
    precision_map: Optional[PrecisionMap] = None,
    features: Sequence[Capability] = (Capability.TEXT,),
    expert_alignment: int = PAGE_BYTES,
    pockettitan_version: str = "",
    source_revision: Optional[str] = None,
) -> BuildPlan:
    """Compute the full package layout without reading any weights.

    Args:
        scan: Completed R0 header scan of the source checkpoint.
        precision_map: Per-component precision. Defaults to ``PT-Q4E``.
        features: Capabilities to keep. Anything else is dropped (T1.1).
        expert_alignment: Record pitch in the expert bank. Page alignment keeps
            every expert read page-aligned for direct I/O.
        pockettitan_version: Recorded in the manifest for provenance.
        source_revision: Source commit sha, recorded for provenance.

    Raises:
        PlanError: If the checkpoint cannot be packaged unambiguously.
    """
    precision_map = precision_map or PrecisionMap.pt_q4e()
    enabled = set(features)
    breakdown = classify_all(scan.tensors)

    try:
        expert_items, expert_layout = _plan_experts(scan, precision_map, expert_alignment)
    except SliceError as exc:
        raise PlanError(str(exc)) from exc

    ple_plan = _plan_ple(scan, precision_map)

    dense: List[DenseWorkItem] = []
    dropped_params = 0
    for address in scan.tensors.values():
        rule = classify_tensor(address.name)
        if rule.capability not in enabled:
            dropped_params += address.num_params
            continue
        if rule.component in (Component.EXPERTS_ROUTED, Component.PLE_TABLE):
            continue

        entry = precision_map.entry_for(rule.component)
        spans = section_spans(list(address.shape), entry.bits, entry.group_size, entry.symmetric)
        dense.append(
            DenseWorkItem(
                address=address,
                component=rule.component,
                bits=entry.bits,
                group_size=entry.group_size,
                symmetric=entry.symmetric,
                packed_bytes=sum(s.length for s in spans),
                spans=spans,
            )
        )

    # Deterministic order, then contiguous 64-byte-aligned addresses in the blob.
    dense.sort(key=lambda i: i.address.name)
    cursor = 0
    for item in dense:
        item.byte_offset = cursor
        cursor = align_up(cursor + item.length, DENSE_ENTRY_ALIGNMENT)
    dense_bytes = cursor

    warnings: List[str] = []

    # Group padding silently inflates storage: a 64-wide row at group_size=128
    # is stored 128 wide, doubling it. Same defect class as the tiler OOM.
    padded: Dict[str, Tuple[int, int, float]] = {}
    for shape, group_size, label in _quantized_shapes(dense, expert_layout, ple_plan):
        _, in_features = matrix_dims(shape)
        if group_size > 0 and in_features % group_size:
            waste = (int(math.ceil(in_features / group_size)) * group_size) / in_features
            if waste > padded.get(label, (0, 0, 0.0))[2]:
                padded[label] = (in_features, group_size, waste)
    for label, (in_features, group_size, waste) in sorted(padded.items()):
        warnings.append(
            f"{label}: in_features={in_features} is not a multiple of group_size={group_size}, "
            f"so it is stored {waste:.2f}x larger. Set group_size to a divisor of {in_features}."
        )

    for component in sorted({i.component for i in dense} | {Component.EXPERTS_ROUTED, Component.PLE_TABLE}):
        bits = precision_map.entry_for(component).bits
        if bits < 16 and not is_dense_bit_width(bits):
            warnings.append(
                f"{component.value}: {bits:g}-bit packs at {storage_bits(bits):g} bits/weight "
                f"({100 * (storage_bits(bits) / bits - 1):.0f}% larger on disk) - "
                f"the packer stores {8 // int(bits)} values per byte"
            )

    activation = build_activation_budget(scan, breakdown, features)
    expert_entry = precision_map.entry_for(Component.EXPERTS_ROUTED)

    totals = PackageTotals(
        source_params=scan.total_params,
        packaged_params=(
            sum(i.address.num_params for i in dense)
            + sum(i.expert_slice.num_params for i in expert_items)
            + (sum(s.address.num_params for s in ple_plan.shards) if ple_plan else 0)
        ),
        dropped_params=dropped_params,
        dense_bytes=dense_bytes,
        expert_bytes=expert_layout.total_bytes if expert_layout else 0,
        ple_bytes=ple_plan.total_bytes if ple_plan else 0,
    )

    manifest = PackageManifest(
        pockettitan_version=pockettitan_version,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_model=scan.model_id,
        source_revision=source_revision,
        architecture=(scan.config.get("architectures") or [""])[0],
        features=[c.value for c in features],
        precision_map_name=precision_map.name,
        precision_map={
            component.value: {
                "bits": precision_map.entry_for(component).bits,
                "group_size": precision_map.entry_for(component).group_size,
                "symmetric": precision_map.entry_for(component).symmetric,
                "effective_bits": precision_map.bits_for(component),
            }
            for component in breakdown.stats
        },
        totals=totals,
        dense=[
            DenseTensorEntry(
                name=i.address.name,
                component=i.component.value,
                shape=list(i.address.shape),
                bits=i.bits,
                group_size=i.group_size,
                symmetric=i.symmetric,
                num_params=i.address.num_params,
                packed_bytes=i.packed_bytes,
                byte_offset=i.byte_offset,
                spans=i.spans,
            )
            for i in dense
        ],
        expert_layout=expert_layout,
        activated_params_per_token=activation.total,
        expert_params_per_token=activation.expert_params,
        expert_bytes_per_token=packed_bytes(activation.expert_params, expert_entry.effective),
        reads_per_token=(
            len(expert_layout.layers) * (infer_expert_geometry(scan, breakdown).top_k)
            if expert_layout
            else 0
        ),
    )

    return BuildPlan(
        manifest=manifest, dense=dense, experts=expert_items, ple=ple_plan, warnings=warnings
    )


def build_ple_index(
    plan: PlePlan,
    ngram_size: int,
    head_offsets: Sequence[int],
    head_vocab_sizes: Sequence[int],
    layer_multipliers: Sequence[int],
) -> PleIndex:
    """Assemble ``ple/index.json`` once the builder has read the int64 index tensors.

    Raises:
        PlanError: If the head table is inconsistent with the planned row store.
    """
    if len(head_offsets) != len(head_vocab_sizes):
        raise PlanError(
            f"head_offsets ({len(head_offsets)}) and head_vocab_sizes "
            f"({len(head_vocab_sizes)}) disagree"
        )
    if len(layer_multipliers) < ngram_size:
        raise PlanError(
            f"need {ngram_size} hash multipliers, got {len(layer_multipliers)}"
        )

    addressable = head_offsets[-1] + head_vocab_sizes[-1]
    if addressable > plan.total_rows:
        raise PlanError(
            f"hash addresses {addressable:,} rows but the table holds {plan.total_rows:,}"
        )

    return PleIndex(
        ngram_size=ngram_size,
        num_heads=len(head_offsets),
        head_offsets=list(head_offsets),
        head_vocab_sizes=list(head_vocab_sizes),
        layer_multipliers=list(layer_multipliers)[:ngram_size],
        total_rows=plan.total_rows,
        row=plan.row,
        source_layer=plan.source_layer,
    )
