"""PocketTitan package format and builder (R1).

Turns a source checkpoint into ``model.ptitan/`` — a layout designed for
out-of-core inference rather than for loading:

* the dense core is one blob, sized to stay VRAM-resident;
* every expert is one page-aligned contiguous record, so a routed expert costs
  a single ``pread`` instead of two seeks across a 120 GB span;
* the n-gram table is a row store whose stride divides the page size, so a row
  lookup never touches two pages.

Planning is separate from writing: :func:`plan_package` computes the byte-exact
layout without reading weights, which is what makes the build resumable and lets
the plan be validated against the R0 audit before any data moves.
"""

from pockettitan.package.format import (
    CHECKSUMS_NAME,
    DENSE_DIR,
    EXPERT_BANK_NAME,
    EXPERT_LAYOUT_NAME,
    EXPERTS_DIR,
    INTEGRITY_DIR,
    MANIFEST_NAME,
    METADATA_DIR,
    PACKAGE_VERSION,
    PAGE_BYTES,
    PLE_DIR,
    PLE_INDEX_NAME,
    PLE_TABLE_NAME,
    TOKENIZER_DIR,
    CodecSpec,
    DenseTensorEntry,
    ExpertLayout,
    ExpertRecordEntry,
    ExpertRecordLayout,
    ItemChecksum,
    PackageManifest,
    PackageChecksums,
    PackageTotals,
    PleIndex,
    PleShardMapping,
    PleRowLayout,
    ProjectionLayout,
    RegionIntegrity,
    Section,
    SectionSpan,
    SourceProvenance,
    align_up,
    num_groups,
    matrix_dims,
    packed_bytes,
    resolve_group_size,
)
from pockettitan.package.plan import (
    BuildPlan,
    DenseWorkItem,
    ExpertWorkItem,
    PlanError,
    PlePlan,
    PleShardWorkItem,
    build_ple_index,
    plan_package,
)
from pockettitan.package.writer import (
    DENSE_BLOB_NAME,
    BuildJournal,
    BuildResult,
    PackageWriter,
    WriteError,
    plan_fingerprint,
)
from pockettitan.package.validator import PtitanValidationReport, PtitanValidator
from pockettitan.package.slicing import (
    ExpertSlice,
    SliceError,
    SourceSlice,
    build_expert_slices,
    expert_bank_tensors,
    layer_index,
    projection_signature,
    slice_expert_from_bank,
)

__all__ = [
    # format
    "PACKAGE_VERSION",
    "PAGE_BYTES",
    "MANIFEST_NAME",
    "METADATA_DIR",
    "INTEGRITY_DIR",
    "CHECKSUMS_NAME",
    "DENSE_DIR",
    "EXPERTS_DIR",
    "PLE_DIR",
    "TOKENIZER_DIR",
    "EXPERT_BANK_NAME",
    "EXPERT_LAYOUT_NAME",
    "PLE_TABLE_NAME",
    "PLE_INDEX_NAME",
    "Section",
    "SectionSpan",
    "CodecSpec",
    "SourceProvenance",
    "RegionIntegrity",
    "ItemChecksum",
    "PackageChecksums",
    "ProjectionLayout",
    "ExpertRecordLayout",
    "ExpertLayout",
    "ExpertRecordEntry",
    "PleRowLayout",
    "PleIndex",
    "PleShardMapping",
    "DenseTensorEntry",
    "PackageTotals",
    "PackageManifest",
    "align_up",
    "num_groups",
    "matrix_dims",
    "packed_bytes",
    "resolve_group_size",
    # slicing
    "SourceSlice",
    "ExpertSlice",
    "SliceError",
    "slice_expert_from_bank",
    "build_expert_slices",
    "expert_bank_tensors",
    "projection_signature",
    "layer_index",
    # plan
    "BuildPlan",
    "DenseWorkItem",
    "ExpertWorkItem",
    "PleShardWorkItem",
    "PlePlan",
    "PlanError",
    "plan_package",
    "build_ple_index",
    # writer
    "PackageWriter",
    "BuildJournal",
    "BuildResult",
    "WriteError",
    "plan_fingerprint",
    "DENSE_BLOB_NAME",
    "PtitanValidator",
    "PtitanValidationReport",
]
