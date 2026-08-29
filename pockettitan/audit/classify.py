"""Component taxonomy for checkpoint tensors (R0).

Classification drives three downstream decisions, so it lives in one place rather
than being re-derived per call site:

* **Capability** — what ``--features text`` may drop (R1 capability filter).
* **Tier** — where a tensor lives at runtime (VRAM / RAM / NVMe).
* **Activation mode** — how often it is read, which is what actually sets the
  per-token byte budget.

Rules are ordered and first-match-wins. Patterns are deliberately anchored on
structural names (``.mlp.experts.``) rather than model families, so the taxonomy
generalizes past Qwen without a per-architecture adapter.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from pockettitan.config import TensorAddress


class Component(str, Enum):
    """Structural role of a tensor within the network."""

    EXPERTS_ROUTED = "experts_routed"
    SHARED_EXPERT = "shared_expert"
    ROUTER = "router"
    PLE_TABLE = "ple_table"
    PLE_PROJ = "ple_proj"
    GDN_ATTN = "gdn_attn"
    FULL_ATTN = "full_attn"
    HYPERCONN = "hyperconn"
    MLP_DENSE = "mlp_dense"
    EMBED = "embed"
    LM_HEAD = "lm_head"
    VISION = "vision"
    MTP = "mtp"
    NORM = "norm"
    OTHER = "other"


class Capability(str, Enum):
    """Model capability a tensor belongs to. Anything not TEXT is droppable."""

    TEXT = "text"
    VISION = "vision"
    MTP = "mtp"


class Tier(str, Enum):
    """Storage tier at inference time."""

    VRAM_HOT = "vram_hot"
    RAM_WARM = "ram_warm"
    NVME_COLD = "nvme_cold"


class ActivationMode(str, Enum):
    """How much of a tensor is read to produce one token.

    DENSE
        Entire tensor, every token.
    ROUTED
        A ``num_experts_per_tok / num_experts`` fraction, every token.
    ROW_LOOKUP
        A handful of rows out of a very large table.
    """

    DENSE = "dense"
    ROUTED = "routed"
    ROW_LOOKUP = "row_lookup"


class ClassificationRule(BaseModel):
    """One ordered pattern rule. First match wins."""

    pattern: str
    component: Component
    capability: Capability = Capability.TEXT
    tier: Tier = Tier.VRAM_HOT
    activation: ActivationMode = ActivationMode.DENSE

    def matches(self, name: str) -> bool:
        return re.search(self.pattern, name) is not None


# Order is load-bearing. Narrow patterns must precede the broad ones they would
# otherwise be swallowed by (e.g. ``.mlp.experts.`` before ``.mlp.``).
DEFAULT_RULES: List[ClassificationRule] = [
    # --- non-text capabilities, matched on prefix before anything else ---
    ClassificationRule(
        pattern=r"(^|\.)visual\.",
        component=Component.VISION,
        capability=Capability.VISION,
        tier=Tier.NVME_COLD,
    ),
    ClassificationRule(
        pattern=r"^vision_(tower|model)\.",
        component=Component.VISION,
        capability=Capability.VISION,
        tier=Tier.NVME_COLD,
    ),
    ClassificationRule(
        pattern=r"^mtp\.",
        component=Component.MTP,
        capability=Capability.MTP,
        tier=Tier.NVME_COLD,
    ),
    # --- per-layer embeddings (PLE / n-gram) ---
    ClassificationRule(
        pattern=r"\.ngram_embedding\.",
        component=Component.PLE_TABLE,
        tier=Tier.NVME_COLD,
        activation=ActivationMode.ROW_LOOKUP,
    ),
    ClassificationRule(pattern=r"\.ple\.", component=Component.PLE_PROJ),
    # --- MoE ---
    ClassificationRule(
        pattern=r"\.mlp\.experts[\._]",
        component=Component.EXPERTS_ROUTED,
        tier=Tier.NVME_COLD,
        activation=ActivationMode.ROUTED,
    ),
    ClassificationRule(
        pattern=r"\.experts\.\d+\.",
        component=Component.EXPERTS_ROUTED,
        tier=Tier.NVME_COLD,
        activation=ActivationMode.ROUTED,
    ),
    # Gates are routers even when named after the shared expert; they must stay
    # in high precision, so they are classified before SHARED_EXPERT.
    ClassificationRule(pattern=r"shared_expert_gate", component=Component.ROUTER),
    ClassificationRule(pattern=r"shared_expert", component=Component.SHARED_EXPERT),
    ClassificationRule(pattern=r"\.mlp\.gate\.weight$", component=Component.ROUTER),
    ClassificationRule(pattern=r"\.(gate|router)\.(weight|bias)$", component=Component.ROUTER),
    ClassificationRule(pattern=r"\.e_score_correction_bias$", component=Component.ROUTER),
    # --- embeddings / output head ---
    ClassificationRule(
        pattern=r"embed_tokens\.weight$",
        component=Component.EMBED,
        tier=Tier.RAM_WARM,
        activation=ActivationMode.ROW_LOOKUP,
    ),
    ClassificationRule(pattern=r"^lm_head\.", component=Component.LM_HEAD),
    # --- attention families ---
    ClassificationRule(pattern=r"hyper_connection", component=Component.HYPERCONN),
    ClassificationRule(pattern=r"\.linear_attn\.", component=Component.GDN_ATTN),
    ClassificationRule(pattern=r"\.self_attn\.", component=Component.FULL_ATTN),
    ClassificationRule(pattern=r"\.attn\.", component=Component.FULL_ATTN),
    # --- dense MLP fallback, then norms ---
    ClassificationRule(pattern=r"\.mlp\.", component=Component.MLP_DENSE),
    ClassificationRule(pattern=r"norm|layernorm|\.ln_", component=Component.NORM),
]


class ClassifiedTensor(BaseModel):
    """A tensor plus its resolved taxonomy."""

    address: TensorAddress
    component: Component
    capability: Capability
    tier: Tier
    activation: ActivationMode


class ComponentStats(BaseModel):
    """Aggregate for one component."""

    component: Component
    capability: Capability
    tier: Tier
    activation: ActivationMode
    num_tensors: int = 0
    params: int = 0
    bytes_source: int = 0

    def share_of(self, total_params: int) -> float:
        return (self.params / total_params) if total_params else 0.0


class ComponentBreakdown(BaseModel):
    """Full component decomposition of a checkpoint."""

    stats: Dict[Component, ComponentStats] = Field(default_factory=dict)
    unclassified: List[str] = Field(
        default_factory=list, description="Tensors that fell through to OTHER; review before trusting a new architecture"
    )

    @property
    def total_params(self) -> int:
        return sum(s.params for s in self.stats.values())

    @property
    def total_bytes(self) -> int:
        return sum(s.bytes_source for s in self.stats.values())

    def params_of(self, component: Component) -> int:
        stat = self.stats.get(component)
        return stat.params if stat else 0

    def params_for_capabilities(self, capabilities: Sequence[Capability]) -> int:
        """Parameters retained when only ``capabilities`` are enabled."""
        enabled = set(capabilities)
        return sum(s.params for s in self.stats.values() if s.capability in enabled)

    def params_dropped_by(self, capabilities: Sequence[Capability]) -> Dict[Component, int]:
        """Parameters removed when only ``capabilities`` are enabled."""
        enabled = set(capabilities)
        return {
            c: s.params for c, s in self.stats.items() if s.capability not in enabled and s.params > 0
        }

    def ordered(self) -> List[ComponentStats]:
        return sorted(self.stats.values(), key=lambda s: -s.params)


def classify_tensor(
    name: str,
    rules: Optional[Sequence[ClassificationRule]] = None,
) -> ClassificationRule:
    """Resolve one tensor name to its rule. Falls back to ``Component.OTHER``."""
    for rule in rules if rules is not None else DEFAULT_RULES:
        if rule.matches(name):
            return rule
    return ClassificationRule(pattern="", component=Component.OTHER)


def classify_all(
    tensors: Dict[str, TensorAddress],
    rules: Optional[Sequence[ClassificationRule]] = None,
) -> ComponentBreakdown:
    """Classify every tensor and aggregate per component."""
    breakdown = ComponentBreakdown()

    for name, address in tensors.items():
        rule = classify_tensor(name, rules)
        if rule.component is Component.OTHER:
            breakdown.unclassified.append(name)

        stat = breakdown.stats.get(rule.component)
        if stat is None:
            stat = ComponentStats(
                component=rule.component,
                capability=rule.capability,
                tier=rule.tier,
                activation=rule.activation,
            )
            breakdown.stats[rule.component] = stat

        stat.num_tensors += 1
        stat.params += address.num_params
        stat.bytes_source += address.size_bytes

    return breakdown


def classified_tensors(
    tensors: Dict[str, TensorAddress],
    rules: Optional[Sequence[ClassificationRule]] = None,
) -> List[ClassifiedTensor]:
    """Per-tensor classification, for callers that need tensor-level detail
    (the R1 capability filter and expert repacker both do)."""
    out: List[ClassifiedTensor] = []
    for address in tensors.values():
        rule = classify_tensor(address.name, rules)
        out.append(
            ClassifiedTensor(
                address=address,
                component=rule.component,
                capability=rule.capability,
                tier=rule.tier,
                activation=rule.activation,
            )
        )
    return out
