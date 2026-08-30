"""R1 T1.6 package writer tests.

The load-bearing tests here are the round-trips: weights are written into the
package at planned offsets, then read back out *using only the manifest* and
dequantized. That is the property the runtime depends on — if an offset or a
section length is wrong, the reconstruction is garbage rather than merely
imprecise, so an approximate comparison is a sharp test.
"""

import json

import pytest
import safetensors.torch
import torch

from pockettitan.audit import PrecisionMap, scan_checkpoint
from pockettitan.config import MemoryBudgetConfig, QuantConfig, QuantMethod
from pockettitan.package import (
    BuildJournal,
    PackageWriter,
    Section,
    WriteError,
    plan_fingerprint,
    plan_package,
)
from pockettitan.quantizers import get_quantizer
from pockettitan.quantizers.base import QuantizedResult
from pockettitan.streaming.reader import LocalTensorReader

TEST_PRECISION = PrecisionMap.uniform(4, 32, "test-int4")


def build_package(model_dir, tmp_path, precision=TEST_PRECISION, **kwargs):
    """Plan and write a package from a local checkpoint."""
    scan = scan_checkpoint(str(model_dir))
    plan = plan_package(scan, precision_map=precision)
    out = tmp_path / "pkg.ptitan"
    writer = PackageWriter(
        plan,
        out,
        LocalTensorReader(model_dir),
        budget=MemoryBudgetConfig(max_vram_mb=512.0),
        method=QuantMethod.RTN,
        device="cpu",
        **kwargs,
    )
    return plan, writer, out


def dequantize_from_bytes(payload, spans, shape, bits, group_size, symmetric):
    """Reconstruct a weight matrix from raw package bytes and a span list."""
    sections = {}
    for span in spans:
        sections[span.section] = payload[span.offset : span.offset + span.length]

    packed = torch.frombuffer(bytearray(sections[Section.PACKED]), dtype=torch.uint8)
    scales = torch.frombuffer(bytearray(sections[Section.SCALES]), dtype=torch.float16)
    zeros = (
        torch.frombuffer(bytearray(sections[Section.ZEROS]), dtype=torch.float16)
        if Section.ZEROS in sections
        else None
    )

    config = QuantConfig(
        method=QuantMethod.RTN,
        bits=int(bits),
        group_size=group_size,
        symmetric=symmetric,
        device="cpu",
    )
    result = QuantizedResult(
        packed_weights=packed,
        scales=scales,
        zeros=zeros,
        codebook=None,
        quant_config=config,
        original_shape=tuple(shape),
        original_dtype=torch.float16,
        bit_width=float(bits),
        device="cpu",
    )
    return get_quantizer(config).dequantize(result)


def source_tensors(model_dir):
    shard = next(model_dir.glob("*.safetensors"))
    return safetensors.torch.load_file(str(shard))


# --------------------------------------------------------------------------- #
# Layout fidelity
# --------------------------------------------------------------------------- #


def test_build_matches_planned_layout(dummy_moe_model, tmp_path):
    plan, writer, out = build_package(dummy_moe_model, tmp_path)
    result = writer.build()

    assert result.finished
    assert result.items_written == plan.num_work_items
    assert result.items_skipped == 0

    assert writer.dense_path.stat().st_size == plan.manifest.totals.dense_bytes
    assert writer.bank_path.stat().st_size == plan.manifest.totals.expert_bytes
    assert (out / "manifest.json").exists()
    assert (out / "experts" / "layout.json").exists()


def test_manifest_is_self_describing(dummy_moe_model, tmp_path):
    """A consumer must be able to address the package from the manifest alone."""
    plan, writer, out = build_package(dummy_moe_model, tmp_path)
    writer.build()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_model"].endswith("dummy_moe")
    assert manifest["expert_layout"]["num_experts"] == 8
    assert manifest["dense"], "dense entries must carry byte offsets"
    assert all("byte_offset" in entry for entry in manifest["dense"])
    assert manifest["totals"]["packaged_params"] > 0


def test_dense_entries_do_not_overlap(dummy_moe_model, tmp_path):
    plan, _, _ = build_package(dummy_moe_model, tmp_path)
    entries = sorted(plan.dense, key=lambda i: i.byte_offset)
    for previous, current in zip(entries, entries[1:]):
        assert previous.byte_offset + previous.length <= current.byte_offset
    last = entries[-1]
    assert last.byte_offset + last.length <= plan.manifest.totals.dense_bytes


# --------------------------------------------------------------------------- #
# Round-trips
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("layer,expert", [(0, 0), (0, 7), (1, 3)])
def test_expert_roundtrip_from_bank(dummy_moe_model, tmp_path, layer, expert):
    """Read one expert back using only the manifest, and compare to source."""
    plan, writer, out = build_package(dummy_moe_model, tmp_path)
    writer.build()

    layout = plan.manifest.expert_layout
    offset, length = layout.byte_range(layer, expert)
    with open(writer.bank_path, "rb") as bank:
        bank.seek(offset)
        record = bank.read(length)
    assert len(record) == layout.record.payload_bytes

    originals = source_tensors(dummy_moe_model)
    for projection in layout.record.projections:
        payload = record[projection.offset : projection.offset + projection.length]
        recovered = dequantize_from_bytes(
            payload,
            projection.spans,
            projection.shape,
            projection.bits,
            projection.group_size,
            projection.symmetric,
        )
        expected = originals[f"model.layers.{layer}.mlp.experts.{projection.name}"][expert]

        assert recovered.shape == expected.shape
        error = (recovered.float() - expected.float()).abs().max().item()
        scale = expected.float().abs().max().item()
        assert error < 0.15 * scale, f"{projection.name} reconstruction is not 4-bit accurate"


def test_expert_records_are_distinct(dummy_moe_model, tmp_path):
    """Guards against every record being written to the same offset."""
    plan, writer, _ = build_package(dummy_moe_model, tmp_path)
    writer.build()
    layout = plan.manifest.expert_layout

    records = []
    with open(writer.bank_path, "rb") as bank:
        for expert in range(layout.num_experts):
            offset, length = layout.byte_range(0, expert)
            bank.seek(offset)
            records.append(bank.read(length))
    assert len(set(records)) == layout.num_experts


def test_dense_roundtrip_from_blob(dummy_moe_model, tmp_path):
    plan, writer, _ = build_package(dummy_moe_model, tmp_path)
    writer.build()
    originals = source_tensors(dummy_moe_model)

    target = next(i for i in plan.dense if i.address.name.endswith("self_attn.q_proj.weight"))
    with open(writer.dense_path, "rb") as blob:
        blob.seek(target.byte_offset)
        payload = blob.read(target.length)

    recovered = dequantize_from_bytes(
        payload,
        target.spans,
        target.address.shape,
        target.bits,
        target.group_size,
        target.symmetric,
    )
    expected = originals[target.address.name]
    error = (recovered.float() - expected.float()).abs().max().item()
    assert error < 0.15 * expected.float().abs().max().item()


def test_fp16_components_are_stored_verbatim(dummy_moe_model, tmp_path):
    """Routers must survive packaging bit-exact — a wrong top-k is unrecoverable."""
    plan, writer, _ = build_package(dummy_moe_model, tmp_path, precision=PrecisionMap.pt_q4e())
    writer.build()
    originals = source_tensors(dummy_moe_model)

    router = next(i for i in plan.dense if i.address.name.endswith("mlp.gate.weight"))
    assert router.bits == 16

    with open(writer.dense_path, "rb") as blob:
        blob.seek(router.byte_offset)
        payload = blob.read(router.length)
    recovered = torch.frombuffer(bytearray(payload), dtype=torch.float16).view(
        *router.address.shape
    )
    assert torch.equal(recovered, originals[router.address.name])


# --------------------------------------------------------------------------- #
# PLE row store
# --------------------------------------------------------------------------- #


def test_ple_table_roundtrip(dummy_ple_model, tmp_path):
    """Every row must be independently addressable and decodable."""
    plan, writer, _ = build_package(dummy_ple_model, tmp_path)
    writer.build()

    ple = plan.ple
    assert ple is not None
    assert plan.manifest.ple_index is not None
    assert (writer.output_dir / "ple" / "index.json").is_file()
    assert plan.manifest.ple_index.layer_multipliers == [11, 17, 23]
    assert plan.manifest.ple_index.head_offsets == [0, 32, 64, 96]
    assert plan.manifest.ple_index.head_vocab_sizes == [32, 32, 32, 32]
    assert writer.ple_path.stat().st_size == ple.total_bytes

    originals = source_tensors(dummy_ple_model)
    row = ple.row
    from pockettitan.package.format import section_spans

    spans = section_spans([1, row.row_width], row.bits, row.group_size, row.symmetric)
    with open(writer.ple_path, "rb") as table:
        for shard in ple.shards:
            source = originals[shard.address.name]
            for local_row in (0, shard.num_rows // 2, shard.num_rows - 1):
                table.seek(row.row_offset(shard.first_row + local_row))
                payload = table.read(row.payload_bytes)
                recovered = dequantize_from_bytes(
                    payload, spans, [1, row.row_width], row.bits, row.group_size, row.symmetric
                )
                expected = source[local_row]
                error = (recovered.view(-1).float() - expected.float()).abs().max().item()
                assert error < 0.2 * expected.float().abs().max().item()


def test_ple_rows_land_on_planned_offsets(dummy_ple_model, tmp_path):
    plan, _, _ = build_package(dummy_ple_model, tmp_path)
    row = plan.ple.row
    for shard in plan.ple.shards:
        assert shard.byte_offset == row.row_offset(shard.first_row)
    # No row crosses a page boundary.
    for row_id in range(0, plan.ple.total_rows, max(1, plan.ple.total_rows // 17)):
        start = row.row_offset(row_id)
        assert start // row.page_bytes == (start + row.payload_bytes - 1) // row.page_bytes


# --------------------------------------------------------------------------- #
# Resumption
# --------------------------------------------------------------------------- #


def test_resume_skips_completed_work(dummy_moe_model, tmp_path):
    plan, writer, out = build_package(dummy_moe_model, tmp_path)
    first = writer.build()
    assert first.items_written == plan.num_work_items

    _, resumed_writer, _ = build_package(dummy_moe_model, tmp_path)
    second = resumed_writer.build()
    assert second.items_skipped == plan.num_work_items
    assert second.items_written == 0
    assert second.bytes_written == 0


def test_resume_completes_a_partial_build(dummy_moe_model, tmp_path):
    """Simulate a crash: drop half the expert journal, then rebuild."""
    plan, writer, out = build_package(dummy_moe_model, tmp_path)
    writer.build()

    journal = BuildJournal.model_validate_json(writer.journal_path.read_text(encoding="utf-8"))
    survivors = journal.expert_done[: len(journal.expert_done) // 2]
    dropped = set(journal.expert_done) - set(survivors)
    journal.expert_done = survivors
    journal.finished = False
    writer.journal_path.write_text(journal.model_dump_json(), encoding="utf-8")

    _, resumed_writer, _ = build_package(dummy_moe_model, tmp_path)
    result = resumed_writer.build()
    assert result.items_written == len(dropped)

    final = BuildJournal.model_validate_json(writer.journal_path.read_text(encoding="utf-8"))
    assert set(final.expert_done) == set(range(plan.manifest.expert_layout.num_records))
    assert final.finished


def test_resume_rejects_a_different_layout(dummy_moe_model, tmp_path):
    """Resuming into a package built from another plan would corrupt it."""
    _, writer, _ = build_package(dummy_moe_model, tmp_path)
    writer.build()

    _, other_writer, _ = build_package(
        dummy_moe_model, tmp_path, precision=PrecisionMap.uniform(2, 32, "different")
    )
    with pytest.raises(WriteError, match="Resuming would corrupt"):
        other_writer.build()


def test_resume_rejects_source_revision_drift(dummy_moe_model, tmp_path):
    _, writer, _ = build_package(dummy_moe_model, tmp_path)
    writer.build()

    scan = scan_checkpoint(str(dummy_moe_model))
    changed = plan_package(
        scan,
        precision_map=TEST_PRECISION,
        source_revision="different-source-revision",
    )
    resumed = PackageWriter(
        changed,
        writer.output_dir,
        LocalTensorReader(dummy_moe_model),
        budget=MemoryBudgetConfig(max_vram_mb=512),
        method=QuantMethod.RTN,
        device="cpu",
    )
    with pytest.raises(WriteError, match="Resuming would corrupt"):
        resumed.build()


def test_uncommitted_batch_is_repeated_after_crash(dummy_moe_model, tmp_path, monkeypatch):
    """Bytes may reach disk before the journal; those bytes must be rewritten."""
    plan, writer, _ = build_package(dummy_moe_model, tmp_path)
    original_save = writer._save_journal
    failed = False

    def fail_first_commit(journal):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected termination before journal commit")
        original_save(journal)

    monkeypatch.setattr(writer, "_save_journal", fail_first_commit)
    with pytest.raises(RuntimeError, match="injected termination"):
        writer.build()
    assert not writer.journal_path.exists()

    _, resumed, _ = build_package(dummy_moe_model, tmp_path)
    result = resumed.build()
    assert result.items_written == plan.num_work_items
    assert BuildJournal.model_validate_json(
        resumed.journal_path.read_text(encoding="utf-8")
    ).finished


def test_peak_vram_includes_dense_and_ple(dummy_ple_model, tmp_path, monkeypatch):
    _, writer, _ = build_package(dummy_ple_model, tmp_path)
    dense_encoder = writer._encode_dense
    ple_encoder = writer._encode_ple_shard

    def dense_with_peak(item, tensor):
        payload, _ = dense_encoder(item, tensor)
        return payload, 11.0

    def ple_with_peak(item, spans):
        payload, _ = ple_encoder(item, spans)
        return payload, 33.0

    monkeypatch.setattr(writer, "_encode_dense", dense_with_peak)
    monkeypatch.setattr(writer, "_encode_ple_shard", ple_with_peak)
    assert writer.build().peak_vram_mb == 33.0


def test_peak_vram_includes_experts(dummy_moe_model, tmp_path, monkeypatch):
    _, writer, _ = build_package(dummy_moe_model, tmp_path)
    dense_encoder = writer._encode_dense
    expert_encoder = writer._encode_expert

    def dense_with_peak(item, tensor):
        payload, _ = dense_encoder(item, tensor)
        return payload, 11.0

    def expert_with_peak(item):
        payload, _ = expert_encoder(item)
        return payload, 22.0

    monkeypatch.setattr(writer, "_encode_dense", dense_with_peak)
    monkeypatch.setattr(writer, "_encode_expert", expert_with_peak)
    assert writer.build().peak_vram_mb == 22.0


def test_text_runtime_assets_are_copied_without_vision_processor(dummy_ple_model, tmp_path):
    _, writer, output = build_package(dummy_ple_model, tmp_path)
    writer.build()

    assert (output / "metadata" / "config.json").is_file()
    assert (output / "metadata" / "generation_config.json").is_file()
    assert (output / "tokenizer" / "tokenizer.json").is_file()
    assert (output / "tokenizer" / "tokenizer_config.json").is_file()
    assert not (output / "tokenizer" / "preprocessor_config.json").exists()


def test_resume_false_rebuilds_everything(dummy_moe_model, tmp_path):
    plan, writer, _ = build_package(dummy_moe_model, tmp_path)
    writer.build()

    _, fresh, _ = build_package(dummy_moe_model, tmp_path, resume=False)
    result = fresh.build()
    assert result.items_written == plan.num_work_items


def test_plan_fingerprint_tracks_layout_not_identity(dummy_moe_model, tmp_path):
    plan_a, _, _ = build_package(dummy_moe_model, tmp_path)
    plan_b, _, _ = build_package(dummy_moe_model, tmp_path)
    assert plan_fingerprint(plan_a) == plan_fingerprint(plan_b)

    plan_c, _, _ = build_package(
        dummy_moe_model, tmp_path, precision=PrecisionMap.uniform(2, 32, "other")
    )
    assert plan_fingerprint(plan_c) != plan_fingerprint(plan_a)


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


def test_section_size_mismatch_is_fatal(dummy_moe_model, tmp_path):
    """A short write would shift every later record, so it must raise."""
    plan, writer, _ = build_package(dummy_moe_model, tmp_path)
    plan.manifest.expert_layout.record.projections[0].spans[0].length += 8
    with pytest.raises(WriteError, match="plan reserved"):
        writer.build()


def test_dense_only_model_builds(dummy_transformer_model, tmp_path):
    plan, writer, out = build_package(dummy_transformer_model, tmp_path)
    result = writer.build()
    assert result.finished
    assert plan.manifest.expert_layout is None
    assert not writer.bank_path.exists()
    assert writer.dense_path.stat().st_size == plan.manifest.totals.dense_bytes


@pytest.mark.gpu
def test_build_respects_vram_budget(dummy_moe_model, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    budget = MemoryBudgetConfig(max_vram_mb=3584.0)
    scan = scan_checkpoint(str(dummy_moe_model))
    plan = plan_package(scan, precision_map=TEST_PRECISION)
    writer = PackageWriter(
        plan,
        tmp_path / "gpu.ptitan",
        LocalTensorReader(dummy_moe_model),
        budget=budget,
        method=QuantMethod.RTN,
        device="cuda",
    )
    result = writer.build()
    assert result.finished
    assert result.peak_vram_mb < budget.usable_vram_mb
