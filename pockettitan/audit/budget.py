"""Capacity and throughput budgets derived from a checkpoint scan (R0).

Everything here is derived from measured tensor shapes rather than from
architecture constants, so the numbers stay correct when a config field is
renamed or a layout changes. The single exception is routing topology
(``num_experts`` / ``num_experts_per_tok``), which is not recoverable from
shapes alone and is read from the config.

The four budgets, in the order they constrain the design:

1. :class:`ActivationBudget` — parameters touched per token.
2. :class:`StorageBudget`    — bytes on NVMe and bytes resident per tier.
3. :class:`StateBudget`      — KV, recurrent state, and indexer vs context length.
4. :class:`Roofline`         — SSD bytes/token and the throughput ceiling.
"""

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from pockettitan.audit.classify import (
    ActivationMode,
    Capability,
    Component,
    ComponentBreakdown,
    Tier,
    classify_tensor,
)
from pockettitan.audit.headers import ShardHeaderScan


GIB = float(1 << 30)
MIB = float(1 << 20)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


# --------------------------------------------------------------------------- #
# Precision
# --------------------------------------------------------------------------- #

def effective_bits(
    nominal_bits: float,
    group_size: int = 128,
    symmetric: bool = False,
    scale_bits: int = 16,
    zero_bits: int = 16,
) -> float:
    """Bits per weight **including** group metadata.

    Nominal-bits arithmetic understates real footprint; a 2-bit group of 128 with
    an fp16 scale and fp16 zero costs ``2 + 32/128 = 2.25`` bits/weight. Group
    size ``<= 0`` means per-tensor scaling, where metadata is negligible.
    """
    if nominal_bits >= 16:
        return float(nominal_bits)
    if group_size <= 0:
        return float(nominal_bits)
    metadata = scale_bits if symmetric else (scale_bits + zero_bits)
    return float(nominal_bits) + metadata / float(group_size)


class PrecisionEntry(BaseModel):
    """Quantization settings for one component."""

    bits: float = 4.0
    group_size: int = 128
    symmetric: bool = False

    @property
    def effective(self) -> float:
        return effective_bits(self.bits, self.group_size, self.symmetric)


class PrecisionMap(BaseModel):
    """Per-component precision assignment."""

    name: str = "custom"
    entries: Dict[Component, PrecisionEntry] = Field(default_factory=dict)
    default: PrecisionEntry = Field(default_factory=PrecisionEntry)

    def entry_for(self, component: Component) -> PrecisionEntry:
        return self.entries.get(component, self.default)

    def bits_for(self, component: Component) -> float:
        return self.entry_for(component).effective

    @classmethod
    def uniform(cls, bits: float, group_size: int = 128, name: Optional[str] = None) -> "PrecisionMap":
        """Uniform precision, except routers and norms which stay fp16."""
        return cls(
            name=name or f"uniform-{bits}b",
            default=PrecisionEntry(bits=bits, group_size=group_size),
            entries={
                Component.ROUTER: PrecisionEntry(bits=16, group_size=-1),
                Component.NORM: PrecisionEntry(bits=16, group_size=-1),
            },
        )

    @classmethod
    def pt_q4e(cls) -> "PrecisionMap":
        """``PT-Q4E`` — the first serious configuration (Plan.md R1).

        4-bit experts for a quality-preserving baseline. Routers never quantized.
        """
        return cls(
            name="PT-Q4E",
            default=PrecisionEntry(bits=4, group_size=128),
            entries={
                Component.ROUTER: PrecisionEntry(bits=16, group_size=-1),
                Component.NORM: PrecisionEntry(bits=16, group_size=-1),
                Component.OTHER: PrecisionEntry(bits=16, group_size=-1),
                Component.GDN_ATTN: PrecisionEntry(bits=3, group_size=128),
                Component.FULL_ATTN: PrecisionEntry(bits=4, group_size=128),
                Component.HYPERCONN: PrecisionEntry(bits=4, group_size=128),
                Component.SHARED_EXPERT: PrecisionEntry(bits=4, group_size=128),
                Component.PLE_PROJ: PrecisionEntry(bits=4, group_size=128),
                Component.LM_HEAD: PrecisionEntry(bits=6, group_size=128),
                Component.EMBED: PrecisionEntry(bits=4, group_size=128),
                Component.EXPERTS_ROUTED: PrecisionEntry(bits=4, group_size=128),
                Component.PLE_TABLE: PrecisionEntry(bits=3, group_size=160, symmetric=True),
            },
        )

    @classmethod
    def pt_q2e(cls) -> "PrecisionMap":
        """``PT-Q2E`` — the bandwidth-optimized variant. Experts at 2-bit.

        Halves expert I/O. Gate on structured-output evals (Plan.md E7) before
        shipping: 2-bit experts have been observed breaking JSON/tool-calling on
        a sibling Qwen MoE while perplexity looked healthy.
        """
        pmap = cls.pt_q4e()
        pmap.name = "PT-Q2E"
        pmap.entries[Component.EXPERTS_ROUTED] = PrecisionEntry(bits=2, group_size=128)
        return pmap


PRECISION_PRESETS: Dict[str, Any] = {
    "pt-q4e": PrecisionMap.pt_q4e,
    "pt-q2e": PrecisionMap.pt_q2e,
    "bf16": lambda: PrecisionMap.uniform(16, -1, "bf16"),
    "int8": lambda: PrecisionMap.uniform(8, 128, "int8"),
    "int4": lambda: PrecisionMap.uniform(4, 128, "int4"),
    "int3": lambda: PrecisionMap.uniform(3, 128, "int3"),
    "int2": lambda: PrecisionMap.uniform(2, 128, "int2"),
    "ternary": lambda: PrecisionMap.uniform(1.58, 128, "ternary"),
}


def get_precision_preset(name: str) -> PrecisionMap:
    key = name.strip().lower()
    if key not in PRECISION_PRESETS:
        raise KeyError(f"Unknown precision preset '{name}'. Options: {sorted(PRECISION_PRESETS)}")
    return PRECISION_PRESETS[key]()


# --------------------------------------------------------------------------- #
# Geometry inference
# --------------------------------------------------------------------------- #

def text_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the language-model sub-config, tolerating flat and nested layouts."""
    return config.get("text_config", config)


def _layer_index(name: str) -> Optional[int]:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


class ExpertGeometry(BaseModel):
    """Routing topology and the physical size of one expert."""

    num_experts: int
    top_k: int
    num_expert_layers: int
    params_per_expert: int
    tensors_per_expert: int = Field(description="Reads per expert before repacking; 1 after")

    @property
    def total_slots(self) -> int:
        return self.num_experts * self.num_expert_layers

    @property
    def activations_per_token(self) -> int:
        return self.top_k * self.num_expert_layers

    def record_bytes(self, bits: float) -> int:
        return int(math.ceil(self.params_per_expert * bits / 8.0))


class PleGeometry(BaseModel):
    """Row-lookup geometry for a per-layer-embedding / n-gram table."""

    row_width: int
    total_rows: int
    num_heads: int

    @property
    def rows_per_token(self) -> int:
        return self.num_heads

    @property
    def params_per_token(self) -> int:
        return self.num_heads * self.row_width


def infer_expert_geometry(
    scan: ShardHeaderScan, breakdown: ComponentBreakdown
) -> Optional[ExpertGeometry]:
    """Recover expert geometry from tensor shapes plus routing config.

    Handles both fused ``(num_experts, ...)`` layouts and per-expert tensors.
    """
    cfg = text_config(scan.config)
    num_experts = cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts")
    top_k = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token")
    if not num_experts or not top_k:
        return None

    expert_tensors = [
        t for t in scan.tensors.values()
        if classify_tensor(t.name).component is Component.EXPERTS_ROUTED
    ]
    if not expert_tensors:
        return None

    layers = {idx for t in expert_tensors if (idx := _layer_index(t.name)) is not None}
    num_expert_layers = len(layers) or 1

    # Size one expert from a single layer, so the result is independent of depth.
    reference_layer = min(layers) if layers else None
    in_layer = [
        t for t in expert_tensors
        if reference_layer is None or _layer_index(t.name) == reference_layer
    ]
    layer_params = sum(t.num_params for t in in_layer)
    params_per_expert = layer_params // int(num_experts)

    return ExpertGeometry(
        num_experts=int(num_experts),
        top_k=int(top_k),
        num_expert_layers=num_expert_layers,
        params_per_expert=params_per_expert,
        tensors_per_expert=len(in_layer),
    )


def infer_ple_geometry(scan: ShardHeaderScan) -> Optional[PleGeometry]:
    """Recover n-gram table geometry: row width, row count, and heads per token."""
    table_tensors = [
        t for t in scan.tensors.values()
        if classify_tensor(t.name).component is Component.PLE_TABLE and len(t.shape) == 2
    ]
    if not table_tensors:
        return None

    row_width = table_tensors[0].shape[1]
    total_rows = sum(t.shape[0] for t in table_tensors)

    cfg = text_config(scan.config)
    embed_dim = cfg.get("ple_embed_dim") or cfg.get("hidden_size") or row_width
    num_heads = max(1, int(embed_dim) // int(row_width))

    return PleGeometry(row_width=row_width, total_rows=total_rows, num_heads=num_heads)


# --------------------------------------------------------------------------- #
# Activation budget
# --------------------------------------------------------------------------- #

class ActivationBudget(BaseModel):
    """Parameters read to produce one token, per component."""

    per_component: Dict[Component, int] = Field(default_factory=dict)
    features: List[Capability] = Field(default_factory=list)
    expert_params: int = 0
    notes: List[str] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.per_component.values())

    @property
    def dense_params(self) -> int:
        return self.total - self.expert_params


def build_activation_budget(
    scan: ShardHeaderScan,
    breakdown: ComponentBreakdown,
    features: Sequence[Capability] = (Capability.TEXT,),
    expert_geometry: Optional[ExpertGeometry] = None,
    ple_geometry: Optional[PleGeometry] = None,
) -> ActivationBudget:
    """Compute activated parameters per token.

    Derived entirely from the scanned component breakdown, so it cannot drift
    from the checkpoint the way a hand-written layer formula would.
    """
    enabled = set(features)
    expert_geometry = expert_geometry or infer_expert_geometry(scan, breakdown)
    ple_geometry = ple_geometry or infer_ple_geometry(scan)
    cfg = text_config(scan.config)

    budget = ActivationBudget(features=list(features))

    for component, stat in breakdown.stats.items():
        if stat.capability not in enabled:
            continue

        if stat.activation is ActivationMode.DENSE:
            budget.per_component[component] = stat.params

        elif stat.activation is ActivationMode.ROUTED:
            if expert_geometry is None:
                budget.per_component[component] = stat.params
                budget.notes.append(
                    "Routing topology unavailable; routed experts counted as fully dense."
                )
            else:
                fraction = expert_geometry.top_k / expert_geometry.num_experts
                active = int(round(stat.params * fraction))
                budget.per_component[component] = active
                budget.expert_params = active

        elif stat.activation is ActivationMode.ROW_LOOKUP:
            if component is Component.PLE_TABLE and ple_geometry is not None:
                budget.per_component[component] = ple_geometry.params_per_token
            elif component is Component.EMBED:
                budget.per_component[component] = int(cfg.get("hidden_size", 0))
            else:
                budget.per_component[component] = 0

    return budget


# --------------------------------------------------------------------------- #
# Storage budget
# --------------------------------------------------------------------------- #

class ComponentStorage(BaseModel):
    component: Component
    capability: Capability
    tier: Tier
    params: int
    effective_bits: float
    packed_bytes: int


class StorageBudget(BaseModel):
    """On-disk and per-tier footprint under a given precision map."""

    precision_map_name: str
    entries: List[ComponentStorage] = Field(default_factory=list)
    features: List[Capability] = Field(default_factory=list)

    @property
    def total_packed_bytes(self) -> int:
        return sum(e.packed_bytes for e in self.entries)

    @property
    def total_params(self) -> int:
        return sum(e.params for e in self.entries)

    @property
    def average_bits(self) -> float:
        return (self.total_packed_bytes * 8.0 / self.total_params) if self.total_params else 0.0

    def bytes_in_tier(self, tier: Tier) -> int:
        return sum(e.packed_bytes for e in self.entries if e.tier is tier)


def build_storage_budget(
    breakdown: ComponentBreakdown,
    precision_map: PrecisionMap,
    features: Sequence[Capability] = (Capability.TEXT,),
) -> StorageBudget:
    """Packed size per component, counting only enabled capabilities."""
    enabled = set(features)
    budget = StorageBudget(precision_map_name=precision_map.name, features=list(features))

    for stat in breakdown.ordered():
        if stat.capability not in enabled or stat.params == 0:
            continue
        bits = precision_map.bits_for(stat.component)
        budget.entries.append(
            ComponentStorage(
                component=stat.component,
                capability=stat.capability,
                tier=stat.tier,
                params=stat.params,
                effective_bits=bits,
                packed_bytes=int(math.ceil(stat.params * bits / 8.0)),
            )
        )
    return budget


# --------------------------------------------------------------------------- #
# State budget
# --------------------------------------------------------------------------- #

class StateBudget(BaseModel):
    """KV cache, recurrent state, and sparse-attention indexer footprint."""

    kv_bytes_per_token: int = 0
    indexer_bytes_per_token: int = 0
    recurrent_state_bytes: int = Field(default=0, description="Constant in context length")
    num_full_attn_layers: int = 0
    num_linear_attn_layers: int = 0

    @property
    def bytes_per_token(self) -> int:
        return self.kv_bytes_per_token + self.indexer_bytes_per_token

    def at_context(self, context_length: int) -> int:
        return self.bytes_per_token * context_length + self.recurrent_state_bytes


def build_state_budget(
    scan: ShardHeaderScan,
    kv_bytes_per_element: int = 2,
    state_bytes_per_element: int = 4,
) -> StateBudget:
    """Compute state footprint from layer types and head geometry.

    Layer counts come from tensor names rather than ``layer_types`` so that a
    checkpoint whose config disagrees with its weights is scored on its weights.
    """
    cfg = text_config(scan.config)

    full_layers, linear_layers = set(), set()
    conv_dim = 0
    for tensor in scan.tensors.values():
        if tensor.name.startswith("mtp."):
            continue
        idx = _layer_index(tensor.name)
        if idx is None:
            continue
        if ".self_attn." in tensor.name:
            full_layers.add(idx)
        elif ".linear_attn." in tensor.name:
            linear_layers.add(idx)
            if tensor.name.endswith("conv1d.weight") and tensor.shape:
                conv_dim = max(conv_dim, tensor.shape[0])

    n_full, n_linear = len(full_layers), len(linear_layers)

    kv_heads = int(cfg.get("num_key_value_heads", 0) or 0)
    head_dim = int(cfg.get("head_dim", 0) or 0)
    if not head_dim and cfg.get("hidden_size") and cfg.get("num_attention_heads"):
        head_dim = int(cfg["hidden_size"]) // int(cfg["num_attention_heads"])
    kv_per_token = n_full * kv_heads * head_dim * 2 * kv_bytes_per_element

    idx_heads = int(cfg.get("indexer_kv_heads", 0) or 0)
    idx_dim = int(cfg.get("indexer_head_dim", 0) or 0)
    indexer_per_token = n_full * idx_heads * idx_dim * kv_bytes_per_element

    v_heads = int(cfg.get("linear_num_value_heads", 0) or 0)
    v_dim = int(cfg.get("linear_value_head_dim", 0) or 0)
    k_dim = int(cfg.get("linear_key_head_dim", 0) or 0)
    kernel = int(cfg.get("linear_conv_kernel_dim", 0) or 0)
    recurrent = n_linear * (
        v_heads * v_dim * k_dim * state_bytes_per_element
        + conv_dim * kernel * state_bytes_per_element
    )

    return StateBudget(
        kv_bytes_per_token=kv_per_token,
        indexer_bytes_per_token=indexer_per_token,
        recurrent_state_bytes=recurrent,
        num_full_attn_layers=n_full,
        num_linear_attn_layers=n_linear,
    )


# --------------------------------------------------------------------------- #
# Roofline
# --------------------------------------------------------------------------- #

class RooflineRow(BaseModel):
    hit_rate: float
    ssd_bytes_per_token: int
    tokens_per_second: Dict[float, float] = Field(
        default_factory=dict, description="Keyed by SSD bandwidth in GB/s"
    )


class Roofline(BaseModel):
    """SSD-bound throughput ceiling for expert streaming.

    These are ceilings, not predictions: they count SSD time only and assume
    perfect overlap of everything else. Comparable measured systems land near
    25% of their own ceiling.
    """

    expert_bits: float
    expert_bytes_per_token: int
    expert_record_bytes: int
    reads_per_token: int
    total_expert_slots: int
    cache_slots: int
    cache_capacity_fraction: float
    rows: List[RooflineRow] = Field(default_factory=list)
    efficiency_factor: float = Field(
        default=0.25, description="Observed fraction of ceiling in comparable measured systems"
    )

    def realistic_range(self, hit_rate: float, bandwidth_gbs: float) -> Tuple[float, float]:
        """Ceiling scaled to a plausible band, for planning rather than marketing."""
        for row in self.rows:
            if abs(row.hit_rate - hit_rate) < 1e-9:
                ceiling = row.tokens_per_second.get(bandwidth_gbs, 0.0)
                return ceiling * self.efficiency_factor, ceiling * (self.efficiency_factor * 2)
        return 0.0, 0.0


def build_roofline(
    activation: ActivationBudget,
    expert_geometry: ExpertGeometry,
    precision_map: PrecisionMap,
    ram_budget_bytes: float = 7 * GIB,
    hit_rates: Sequence[float] = (0.0, 0.4, 0.6, 0.75, 0.9),
    bandwidths_gbs: Sequence[float] = (3.0, 5.0, 7.0),
    packed_records: bool = True,
) -> Roofline:
    """SSD bytes/token and throughput ceiling across cache hit rates."""
    bits = precision_map.bits_for(Component.EXPERTS_ROUTED)
    bytes_per_token = int(math.ceil(activation.expert_params * bits / 8.0))
    record_bytes = expert_geometry.record_bytes(bits)
    reads = expert_geometry.activations_per_token * (
        1 if packed_records else expert_geometry.tensors_per_expert
    )

    slots = int(ram_budget_bytes // record_bytes) if record_bytes else 0
    total_slots = expert_geometry.total_slots

    roofline = Roofline(
        expert_bits=bits,
        expert_bytes_per_token=bytes_per_token,
        expert_record_bytes=record_bytes,
        reads_per_token=reads,
        total_expert_slots=total_slots,
        cache_slots=slots,
        cache_capacity_fraction=(slots / total_slots) if total_slots else 0.0,
    )

    for hit in hit_rates:
        miss_bytes = int(bytes_per_token * (1.0 - hit))
        row = RooflineRow(hit_rate=hit, ssd_bytes_per_token=miss_bytes)
        for bw in bandwidths_gbs:
            row.tokens_per_second[bw] = (bw * 1e9 / miss_bytes) if miss_bytes > 0 else float("inf")
        roofline.rows.append(row)

    return roofline


# --------------------------------------------------------------------------- #
# Top-level report object
# --------------------------------------------------------------------------- #

class AuditReport(BaseModel):
    """Everything R0 produces for one checkpoint under one precision map."""

    model_id: str
    num_tensors: int
    num_shards: int
    total_params: int
    total_source_bytes: int
    dtype_histogram: Dict[str, int] = Field(default_factory=dict)
    elapsed_s: float = 0.0
    discrepancies: List[str] = Field(default_factory=list)

    breakdown: ComponentBreakdown
    activation: ActivationBudget
    storage: StorageBudget
    state: StateBudget
    precision_map: PrecisionMap
    expert_geometry: Optional[ExpertGeometry] = None
    ple_geometry: Optional[PleGeometry] = None
    roofline: Optional[Roofline] = None

    @property
    def enabled_params(self) -> int:
        """Parameters retained for the enabled capabilities, tables included."""
        return self.breakdown.params_for_capabilities(self.activation.features)

    @property
    def lm_core_params(self) -> int:
        """Enabled parameters excluding cold row-lookup tables.

        This is the figure conversion tools report as the language model's
        parameter count, because tables like the n-gram/PLE store are emitted as
        a separate artifact rather than as part of the LM weights. Keeping the
        two notions distinct avoids a 51B-parameter accounting error.
        """
        enabled = set(self.activation.features)
        table_params = sum(
            s.params
            for s in self.breakdown.stats.values()
            if s.capability in enabled
            and s.tier is Tier.NVME_COLD
            and s.activation is ActivationMode.ROW_LOOKUP
        )
        return self.enabled_params - table_params

    @property
    def dropped_params(self) -> Dict[Component, int]:
        return self.breakdown.params_dropped_by(self.activation.features)


def build_audit_report(
    scan: ShardHeaderScan,
    precision_map: Optional[PrecisionMap] = None,
    features: Sequence[Capability] = (Capability.TEXT,),
    ram_budget_bytes: float = 7 * GIB,
) -> AuditReport:
    """Run the full R0 analysis over a completed scan."""
    from pockettitan.audit.classify import classify_all

    precision_map = precision_map or PrecisionMap.pt_q4e()
    breakdown = classify_all(scan.tensors)

    expert_geometry = infer_expert_geometry(scan, breakdown)
    ple_geometry = infer_ple_geometry(scan)

    activation = build_activation_budget(
        scan, breakdown, features, expert_geometry, ple_geometry
    )
    storage = build_storage_budget(breakdown, precision_map, features)
    state = build_state_budget(scan)

    roofline = None
    if expert_geometry is not None and activation.expert_params > 0:
        roofline = build_roofline(
            activation, expert_geometry, precision_map, ram_budget_bytes=ram_budget_bytes
        )

    return AuditReport(
        model_id=scan.model_id,
        num_tensors=scan.num_tensors,
        num_shards=len(scan.shards),
        total_params=scan.total_params,
        total_source_bytes=scan.total_bytes,
        dtype_histogram=scan.dtype_histogram(),
        elapsed_s=scan.elapsed_s,
        discrepancies=[f"{d.kind}: {d.detail}" for d in scan.discrepancies],
        breakdown=breakdown,
        activation=activation,
        storage=storage,
        state=state,
        precision_map=precision_map,
        expert_geometry=expert_geometry,
        ple_geometry=ple_geometry,
        roofline=roofline,
    )
