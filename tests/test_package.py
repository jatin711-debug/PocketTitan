"""R1 package format, slicing, and build-plan tests.

The golden tests pin the ``PT-Q4E`` layout for ``Qwen/Qwen3.8-Flash-Next`` to
exact byte counts. The most important assertion in this file is
:func:`test_expert_payload_matches_audit_budget`: the packager's explicit
per-record accounting must agree exactly with the R0 audit's amortized
``params × effective_bits`` estimate. Those are two independent derivations, and
if they disagree one of them is wrong.
"""

import pytest

from pockettitan.audit import (
    Capability,
    Component,
    PrecisionMap,
    build_audit_report,
    scan_checkpoint,
)
from pockettitan.config import TensorAddress
from pockettitan.package import (
    PAGE_BYTES,
    ExpertRecordLayout,
    PlanError,
    PleRowLayout,
    Section,
    SliceError,
    align_up,
    build_expert_slices,
    build_ple_index,
    num_groups,
    packed_bytes,
    plan_package,
    projection_signature,
    slice_expert_from_bank,
)

# --- Ground truth for the PT-Q4E package ----------------------------------- #
EXPERT_RECORD_PAYLOAD = 2_611_200        # 2.49 MiB, one contiguous read
EXPERT_RECORD_STRIDE = 2_613_248         # page-aligned
NUM_EXPERT_RECORDS = 24_576
EXPERT_BANK_BYTES = 64_223_182_848
AUDIT_EXPERT_BYTES = 64_172_851_200      # R0's independent estimate
PLE_TOTAL_ROWS = 320_001_536
PLE_ROW_PAYLOAD = 82          # 160 weights at 4 storage bits + one fp16 scale
PLE_ROWS_PER_PAGE = 49
PLE_TABLE_BYTES = 26_749_517_824
DROPPED_PARAMS = 3_056_081_904           # vision + MTP


@pytest.fixture(scope="module")
def qwen_plan(qwen_scan):
    return plan_package(qwen_scan, precision_map=PrecisionMap.pt_q4e())


# --------------------------------------------------------------------------- #
# Expert record layout
# --------------------------------------------------------------------------- #

def test_expert_record_layout(qwen_plan):
    layout = qwen_plan.manifest.expert_layout
    assert layout is not None
    assert layout.num_experts == 512
    assert layout.layers == list(range(48))
    assert layout.num_records == NUM_EXPERT_RECORDS

    record = layout.record
    assert [p.name for p in record.projections] == ["gate_up_proj", "down_proj"]
    assert record.num_params == 4_915_200
    assert record.payload_bytes == EXPERT_RECORD_PAYLOAD
    assert record.stride == EXPERT_RECORD_STRIDE
    assert record.stride % PAGE_BYTES == 0
    assert layout.total_bytes == EXPERT_BANK_BYTES


def test_expert_payload_matches_audit_budget(qwen_scan, qwen_plan):
    """Explicit per-record accounting vs the audit's amortized estimate.

    These are independent derivations of the same quantity; they must agree.
    """
    report = build_audit_report(qwen_scan, precision_map=PrecisionMap.pt_q4e())
    audit_bytes = next(
        e.packed_bytes for e in report.storage.entries if e.component is Component.EXPERTS_ROUTED
    )
    record = qwen_plan.manifest.expert_layout.record
    assert audit_bytes == AUDIT_EXPERT_BYTES
    assert record.payload_bytes * NUM_EXPERT_RECORDS == audit_bytes


def test_alignment_overhead_is_negligible(qwen_plan):
    """Page alignment must not cost meaningful capacity."""
    layout = qwen_plan.manifest.expert_layout
    overhead = layout.total_bytes / (layout.record.payload_bytes * layout.num_records) - 1.0
    assert 0 < overhead < 0.001


def test_record_addressing_is_layer_major(qwen_plan):
    layout = qwen_plan.manifest.expert_layout
    assert layout.record_index(0, 0) == 0
    assert layout.record_index(0, 511) == 511
    assert layout.record_index(1, 0) == 512
    assert layout.record_index(3, 5) == 3 * 512 + 5

    offset, length = layout.byte_range(3, 5)
    assert offset == (3 * 512 + 5) * EXPERT_RECORD_STRIDE
    assert length == EXPERT_RECORD_PAYLOAD
    assert offset % PAGE_BYTES == 0, "expert reads must be page-aligned"


def test_record_addressing_rejects_bad_keys(qwen_plan):
    layout = qwen_plan.manifest.expert_layout
    with pytest.raises(KeyError):
        layout.record_index(99, 0)
    with pytest.raises(KeyError):
        layout.record_index(0, 512)


def test_section_offsets_are_contiguous(qwen_plan):
    record = qwen_plan.manifest.expert_layout.record
    gate_up = record.projection("gate_up_proj")
    down = record.projection("down_proj")

    assert gate_up.offset == 0
    assert gate_up.length == packed_bytes(3_276_800, 4) + num_groups(3_276_800, 128) * 4
    assert down.offset == gate_up.length, "projections must be back-to-back"
    assert gate_up.length + down.length == record.payload_bytes

    for projection in (gate_up, down):
        cursor = 0
        for span in projection.spans:
            assert span.offset == cursor
            cursor += span.length
        assert {s.section for s in projection.spans} == {Section.PACKED, Section.SCALES, Section.ZEROS}


def test_symmetric_projection_has_no_zeros():
    layout = ExpertRecordLayout.build(
        [{"name": "w", "shape": [64, 64], "bits": 4, "group_size": 32, "symmetric": True}]
    )
    spans = {s.section for s in layout.projections[0].spans}
    assert spans == {Section.PACKED, Section.SCALES}


def test_fp16_projection_has_no_quant_metadata():
    layout = ExpertRecordLayout.build(
        [{"name": "w", "shape": [64, 64], "bits": 16, "group_size": -1, "symmetric": False}]
    )
    assert [s.section for s in layout.projections[0].spans] == [Section.PACKED]
    assert layout.payload_bytes == 64 * 64 * 2


# --------------------------------------------------------------------------- #
# PLE row store
# --------------------------------------------------------------------------- #

def test_ple_row_layout(qwen_plan):
    ple = qwen_plan.ple
    assert ple is not None
    assert ple.total_rows == PLE_TOTAL_ROWS
    assert ple.row.row_width == 160
    # 3-bit is requested but the packer stores 2 values/byte, so a row costs 4 bits/weight.
    assert ple.row.payload_bytes == PLE_ROW_PAYLOAD
    assert ple.row.rows_per_page == PLE_ROWS_PER_PAGE
    assert ple.total_bytes == PLE_TABLE_BYTES


def test_ple_rows_never_straddle_a_page(qwen_plan):
    """A row that spans two pages doubles the cost of every lookup."""
    row = qwen_plan.ple.row
    boundaries = (0, 1, PLE_ROWS_PER_PAGE - 1, PLE_ROWS_PER_PAGE, PLE_ROWS_PER_PAGE + 1,
                  2 * PLE_ROWS_PER_PAGE, PLE_TOTAL_ROWS - 1)
    for row_id in boundaries:
        start = row.row_offset(row_id)
        assert start // PAGE_BYTES == (start + row.payload_bytes - 1) // PAGE_BYTES
    # Page packing must beat a power-of-two stride, which is why it exists.
    assert row.waste_fraction < 0.02


@pytest.mark.parametrize("width,bits,expected_payload", [
    (160, 4, 82),      # 80 packed + 2 scale
    (160, 3, 82),      # 3-bit stores as 4-bit
    (160, 2, 42),
    (256, 8, 258),
])
def test_ple_row_packing(width, bits, expected_payload):
    row = PleRowLayout.build(width, bits, group_size=width, symmetric=True)
    assert row.payload_bytes == expected_payload
    assert row.rows_per_page == PAGE_BYTES // expected_payload
    assert row.rows_per_page * row.payload_bytes <= PAGE_BYTES
    # Page packing can never waste more than one row per page, by construction.
    assert row.waste_fraction < row.payload_bytes / PAGE_BYTES


def test_ple_shards_tile_the_table_without_gaps(qwen_plan):
    shards = qwen_plan.ple.shards
    assert len(shards) == 128
    assert [s.shard_index for s in shards] == list(range(128))

    cursor = 0
    for shard in shards:
        assert shard.first_row == cursor, "shard row ranges must be contiguous"
        assert shard.byte_offset == qwen_plan.ple.row.row_offset(cursor)
        cursor += shard.num_rows
    assert cursor == PLE_TOTAL_ROWS


def test_ple_index_hash_and_padding(qwen_plan):
    """Rebuild the index with the real constants and check the hash is sane."""
    head_vocab = [
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
    ]
    head_offsets = [0]
    for size in head_vocab[:-1]:
        head_offsets.append(head_offsets[-1] + size)
    multipliers = [23703573157769, 20109073645365, 8052911324071]

    index = build_ple_index(qwen_plan.ple, 3, head_offsets, head_vocab, multipliers)
    assert index.num_heads == 16
    assert index.total_rows == PLE_TOTAL_ROWS

    rows = index.row_ids([101, 202, 303])
    assert len(rows) == 16
    assert len(set(rows)) == 16, "distinct prime moduli should decorrelate heads"
    for head, row in enumerate(rows):
        assert head_offsets[head] <= row < head_offsets[head] + head_vocab[head]
    # Deterministic: the same n-gram always resolves to the same rows.
    assert index.row_ids([101, 202, 303]) == rows
    # Only the last ngram_size tokens matter.
    assert index.row_ids([7, 7, 101, 202, 303]) == rows

    addressable = head_offsets[-1] + head_vocab[-1]
    assert index.total_rows - addressable == 90, "vocab padding to a multiple of 128"


def test_build_ple_index_rejects_inconsistent_tables(qwen_plan):
    with pytest.raises(PlanError, match="disagree"):
        build_ple_index(qwen_plan.ple, 3, [0, 10], [10], [1, 2, 3])
    with pytest.raises(PlanError, match="multipliers"):
        build_ple_index(qwen_plan.ple, 3, [0], [10], [1])
    with pytest.raises(PlanError, match="table holds"):
        build_ple_index(qwen_plan.ple, 3, [0], [10**12], [1, 2, 3])


# --------------------------------------------------------------------------- #
# Capability filtering and totals
# --------------------------------------------------------------------------- #

def test_capability_filter_drops_vision_and_mtp(qwen_plan):
    names = {item.address.name for item in qwen_plan.dense}
    assert not any(n.startswith("mtp.") for n in names)
    assert not any(".visual." in n for n in names)
    assert qwen_plan.manifest.totals.dropped_params == DROPPED_PARAMS


def test_dense_excludes_cold_tier_components(qwen_plan):
    components = {item.component for item in qwen_plan.dense}
    assert Component.EXPERTS_ROUTED not in components
    assert Component.PLE_TABLE not in components
    assert Component.ROUTER in components
    assert Component.GDN_ATTN in components


def test_routers_stay_fp16_in_the_plan(qwen_plan):
    routers = [i for i in qwen_plan.dense if i.component is Component.ROUTER]
    assert routers
    assert all(i.bits == 16 for i in routers)


def test_plan_accounts_for_every_enabled_parameter(qwen_scan, qwen_plan):
    totals = qwen_plan.manifest.totals
    assert totals.source_params == qwen_scan.total_params
    assert totals.packaged_params + totals.dropped_params == qwen_scan.total_params


def test_plan_totals_match_audit_storage(qwen_scan, qwen_plan):
    report = build_audit_report(qwen_scan, precision_map=PrecisionMap.pt_q4e())
    totals = qwen_plan.manifest.totals
    # The package is larger than the audit estimate for two modelled reasons:
    # record/page alignment, and sub-byte packing density (3-bit stores as 4-bit).
    assert totals.total_bytes >= report.storage.total_packed_bytes
    assert totals.total_bytes / report.storage.total_packed_bytes < 1.10


def test_manifest_carries_runtime_budget(qwen_plan):
    manifest = qwen_plan.manifest
    assert manifest.activated_params_per_token == 6_671_300_515
    assert manifest.expert_params_per_token == 2_359_296_000
    assert manifest.reads_per_token == 480
    assert manifest.expert_bytes_per_token == packed_bytes(2_359_296_000, 4.25)
    assert manifest.precision_map_name == "PT-Q4E"
    assert manifest.features == ["text"]
    assert manifest.source_model == "Qwen/Qwen3.8-Flash-Next"


def test_source_read_bytes_excludes_dropped_capabilities(qwen_scan, qwen_plan):
    """We must not download what we are about to discard."""
    breakdown = build_audit_report(qwen_scan).breakdown
    dropped_bytes = sum(
        stat.bytes_source
        for stat in breakdown.stats.values()
        if stat.capability is not Capability.TEXT
    )
    assert dropped_bytes > 0
    assert qwen_plan.source_read_bytes == qwen_scan.total_bytes - dropped_bytes


def test_plan_with_vision_keeps_vision(qwen_scan):
    plan = plan_package(qwen_scan, features=(Capability.TEXT, Capability.VISION))
    assert plan.manifest.totals.dropped_params == 2_607_150_848  # MTP only
    assert any(".visual." in i.address.name for i in plan.dense)


# --------------------------------------------------------------------------- #
# Expert slicing
# --------------------------------------------------------------------------- #

def test_slices_cover_every_expert(qwen_scan):
    slices = build_expert_slices(qwen_scan.tensors, num_experts=512)
    assert len(slices) == NUM_EXPERT_RECORDS
    assert all(s.source_reads == 2 for s in slices), "fused layout costs 2 source reads"
    assert all(s.num_params == 4_915_200 for s in slices)


def test_slices_are_contiguous_and_disjoint_within_a_bank(qwen_scan):
    """Consecutive experts must tile their bank exactly — no gaps, no overlap."""
    slices = build_expert_slices(qwen_scan.tensors, num_experts=512, layers=[0])
    gate_up = [s.projections[0] for s in slices]
    assert gate_up[0].projection == "gate_up_proj"

    for previous, current in zip(gate_up, gate_up[1:]):
        assert current.byte_start == previous.byte_end

    bank = next(
        t for t in qwen_scan.tensors.values()
        if t.name == "model.language_model.layers.0.mlp.experts.gate_up_proj"
    )
    assert gate_up[0].byte_start == bank.byte_start
    assert gate_up[-1].byte_end == bank.byte_end
    assert gate_up[0].shape == [1280, 2560]


def test_slice_expert_from_bank_rejects_wrong_topology():
    bank = TensorAddress(
        name="model.layers.0.mlp.experts.gate_up_proj",
        shard="s.safetensors", dtype="BF16", shape=[512, 8, 4],
        byte_start=100, byte_end=100 + 512 * 8 * 4 * 2,
        num_params=512 * 8 * 4, size_bytes=512 * 8 * 4 * 2,
    )
    assert slice_expert_from_bank(bank, 0, 512).byte_start == 100
    with pytest.raises(SliceError, match="declares 256 experts"):
        slice_expert_from_bank(bank, 0, 256)
    with pytest.raises(SliceError, match="out of range"):
        slice_expert_from_bank(bank, 512, 512)


def test_slice_expert_from_bank_rejects_non_bank_tensor():
    flat = TensorAddress(
        name="model.layers.0.mlp.experts.gate_up_proj",
        shard="s.safetensors", dtype="BF16", shape=[512],
        byte_start=0, byte_end=1024, num_params=512, size_bytes=1024,
    )
    with pytest.raises(SliceError, match="needs >= 2 dims"):
        slice_expert_from_bank(flat, 0, 512)


def test_projection_signature_rejects_mixed_geometry(qwen_scan):
    slices = build_expert_slices(qwen_scan.tensors, num_experts=512, layers=[0], experts=[0, 1])
    assert projection_signature(slices)
    slices[1].projections[0].shape = [1, 1]
    with pytest.raises(SliceError, match="expected"):
        projection_signature(slices)


# --------------------------------------------------------------------------- #
# Local checkpoint end-to-end planning
# --------------------------------------------------------------------------- #

def test_plan_local_fused_moe(dummy_moe_model):
    scan = scan_checkpoint(str(dummy_moe_model))
    plan = plan_package(scan, precision_map=PrecisionMap.pt_q4e())

    layout = plan.manifest.expert_layout
    assert layout.num_experts == 8
    assert layout.layers == [0, 1]
    assert layout.num_records == 16
    assert [p.name for p in layout.record.projections] == ["gate_up_proj", "down_proj"]
    assert layout.record.num_params == 64 * 64 + 64 * 32

    slices = build_expert_slices(scan.tensors, num_experts=8, layers=[0])
    assert len(slices) == 8
    assert slices[0].projections[0].shape == [64, 64]
    assert slices[0].projections[1].shape == [64, 32]

    assert plan.ple is None, "dense-MLP checkpoint has no n-gram table"
    assert plan.manifest.totals.dropped_params == 0
    assert plan.num_work_items == len(plan.dense) + 16


def test_plan_local_dense_model_has_no_expert_bank(dummy_transformer_model):
    scan = scan_checkpoint(str(dummy_transformer_model))
    plan = plan_package(scan)
    assert plan.manifest.expert_layout is None
    assert plan.experts == []
    assert plan.ple is None
    assert plan.manifest.reads_per_token == 0


def test_planning_reads_no_weights(qwen_scan, qwen_plan):
    """The plan is computed from headers alone; nothing is fetched."""
    assert qwen_plan.source_read_bytes > 0
    assert qwen_scan.is_local is False


# --------------------------------------------------------------------------- #
# Layout arithmetic
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value,alignment,expected", [
    (0, 4096, 0), (1, 4096, 4096), (4096, 4096, 4096), (4097, 4096, 8192), (100, 1, 100),
])
def test_align_up(value, alignment, expected):
    assert align_up(value, alignment) == expected


@pytest.mark.parametrize("elements,bits,expected", [
    (8, 1, 1), (8, 2, 2), (8, 4, 4), (8, 16, 16), (3, 4, 2),
])
def test_packed_bytes(elements, bits, expected):
    assert packed_bytes(elements, bits) == expected


@pytest.mark.parametrize("elements,group,expected", [
    (128, 128, 1), (129, 128, 2), (100, 128, 1), (100, -1, 1), (256, 64, 4),
])
def test_num_groups(elements, group, expected):
    assert num_groups(elements, group) == expected


# --------------------------------------------------------------------------- #
# Live verification against the source checkpoint
# --------------------------------------------------------------------------- #

@pytest.mark.network
def test_expert_slices_decode_from_live_checkpoint(qwen_scan):
    """Fetch real expert bytes using our computed offsets and decode them.

    This is the ground-truth check on the slicing arithmetic. bf16 is unforgiving:
    an offset wrong by a single byte reinterprets every subsequent pair, producing
    denormals and infinities. Well-conditioned weights therefore prove alignment.
    """
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_url

    from pockettitan.metadata.safetensors_header import fetch_remote_bytes

    slices = build_expert_slices(qwen_scan.tensors, num_experts=512, layers=[0], experts=[0, 511])
    seen = []

    for expert_slice in slices:
        for projection in expert_slice.projections:
            url = hf_hub_url(repo_id=qwen_scan.model_id, filename=projection.shard)
            raw = fetch_remote_bytes(url, projection.byte_start, projection.byte_end - 1)
            assert len(raw) == projection.size_bytes

            tensor = (
                torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy())
                .view(torch.bfloat16)
                .view(*projection.shape)
                .float()
            )
            assert torch.isfinite(tensor).all(), "misaligned reads decode to inf/nan"
            assert 1e-4 < tensor.std().item() < 1.0, "not a plausible trained weight matrix"
            assert (tensor == 0).float().mean().item() < 0.01
            seen.append(tensor)

    assert seen[0].shape == (1280, 2560)
    assert seen[1].shape == (2560, 640)
    # Distinct experts must not resolve to the same bytes.
    assert not torch.equal(seen[0], seen[2])


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _render(plan) -> str:
    import io

    from rich.console import Console

    from pockettitan.package.report import render_plan

    buffer = io.StringIO()
    render_plan(Console(file=buffer, width=160, legacy_windows=False), plan)
    return buffer.getvalue()


def test_render_plan_covers_every_region(qwen_plan):
    """Exercise all three region branches.

    The PLE branch is only reachable on a plan that has an n-gram table, which is
    how a stale field reference in it survived the rest of the suite.
    """
    output = _render(qwen_plan)
    assert "dense/" in output
    assert "experts/" in output
    assert "ple/" in output
    assert "Expert Record Layout" in output
    assert "176,943,899,555" in output


def test_render_plan_without_moe_or_ple(dummy_transformer_model, tmp_path):
    from pockettitan.audit import scan_checkpoint

    plan = plan_package(scan_checkpoint(str(dummy_transformer_model)))
    output = _render(plan)
    assert "dense/" in output
    assert "experts/" not in output
    assert "ple/" not in output


def test_render_survives_legacy_codepage(qwen_plan):
    import io

    from rich.console import Console

    from pockettitan.package.report import render_plan

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    render_plan(Console(file=stream, width=160, legacy_windows=False), qwen_plan)
    stream.flush()
    assert raw.getvalue()
