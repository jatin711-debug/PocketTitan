"""Checkpoint audit (R0) — freeze architectural ground truth as executable code.

Reads every Safetensors header, classifies each tensor into a component
taxonomy, and derives the four budgets that constrain out-of-core inference:
activated parameters per token, storage per tier, state vs context length, and
the SSD throughput roofline.

Typical use::

    from pockettitan.audit import scan_checkpoint, build_audit_report

    scan = scan_checkpoint("Qwen/Qwen3.8-Flash-Next")
    report = build_audit_report(scan)
    print(report.total_params, report.activation.total)
"""

from pockettitan.audit.budget import (
    AuditReport,
    ActivationBudget,
    ExpertGeometry,
    PleGeometry,
    PrecisionEntry,
    PrecisionMap,
    Roofline,
    StateBudget,
    StorageBudget,
    build_activation_budget,
    build_audit_report,
    build_roofline,
    build_state_budget,
    build_storage_budget,
    effective_bits,
    get_precision_preset,
    infer_expert_geometry,
    infer_ple_geometry,
)
from pockettitan.audit.classify import (
    ActivationMode,
    Capability,
    Component,
    ComponentBreakdown,
    Tier,
    classify_all,
    classify_tensor,
    classified_tensors,
)
from pockettitan.audit.headers import (
    ShardHeaderScan,
    ShardScanError,
    scan_checkpoint,
    verify_scan,
)

__all__ = [
    # headers
    "ShardHeaderScan",
    "ShardScanError",
    "scan_checkpoint",
    "verify_scan",
    # classify
    "ActivationMode",
    "Capability",
    "Component",
    "ComponentBreakdown",
    "Tier",
    "classify_all",
    "classify_tensor",
    "classified_tensors",
    # budget
    "ActivationBudget",
    "AuditReport",
    "ExpertGeometry",
    "PleGeometry",
    "PrecisionEntry",
    "PrecisionMap",
    "Roofline",
    "StateBudget",
    "StorageBudget",
    "build_activation_budget",
    "build_audit_report",
    "build_roofline",
    "build_state_budget",
    "build_storage_budget",
    "effective_bits",
    "get_precision_preset",
    "infer_expert_geometry",
    "infer_ple_geometry",
]
