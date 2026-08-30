"""Adversarial tests for the .ptitan v1.1 integrity contract."""

import json
from collections import namedtuple

import pytest

from pockettitan.audit import PrecisionMap, scan_checkpoint
from pockettitan.config import MemoryBudgetConfig, QuantMethod
from pockettitan.package import PackageWriter, PtitanValidator, WriteError, plan_package
from pockettitan.package.integrity import _CRC32C_TABLE, crc32c, crc32c_hex
from pockettitan.streaming.reader import LocalTensorReader


def build_package(model_dir, tmp_path, profile="full"):
    scan = scan_checkpoint(str(model_dir))
    plan = plan_package(
        scan,
        precision_map=PrecisionMap.uniform(4, 32, "validator-test"),
        quant_method=QuantMethod.RTN,
        build_profile=profile,
    )
    output = tmp_path / "model.ptitan"
    writer = PackageWriter(
        plan,
        output,
        LocalTensorReader(model_dir),
        budget=MemoryBudgetConfig(max_vram_mb=512, runtime_reserve_mb=32, safety_margin_mb=32),
        method=QuantMethod.RTN,
        device="cpu",
    )
    writer.build()
    return plan, writer, output


def flip_byte(path, offset):
    with path.open("r+b") as stream:
        stream.seek(offset)
        value = stream.read(1)
        stream.seek(offset)
        stream.write(bytes([value[0] ^ 0x40]))


def _reference_crc32c(data: bytes, crc: int = 0) -> int:
    """Textbook Castagnoli CRC-32C, independent of whichever backend is installed."""
    value = crc ^ 0xFFFFFFFF
    for byte in data:
        value = _CRC32C_TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


def test_crc32c_matches_castagnoli_golden_vector():
    assert crc32c_hex(b"123456789") == "e3069283"


def test_seeded_crc32c_continues_a_running_checksum():
    """PLE shard checksums accumulate row by row, so the seeded call must chain.

    The accelerated backend is optional, so a package written on a machine that
    has it must validate on a machine that does not. Chaining and one-shot must
    therefore agree, and both must match the reference implementation.
    """
    head = b"first PLE row payload"
    tail = b"second PLE row payload"

    chained = crc32c(tail, crc32c(head))

    assert chained == crc32c(head + tail)
    assert chained == _reference_crc32c(head + tail)
    assert chained == _reference_crc32c(tail, _reference_crc32c(head))


def test_fast_and_full_validation_pass(dummy_ple_model, tmp_path):
    _, _, output = build_package(dummy_ple_model, tmp_path)

    fast = PtitanValidator(output).validate("fast")
    full = PtitanValidator(output).validate("full")

    assert fast.is_valid, fast.errors
    assert full.is_valid, full.errors
    assert fast.items_checked > 0
    assert full.bytes_checked > fast.bytes_checked


def test_compact_canary_passes_both_validation_modes(dummy_ple_model, tmp_path):
    plan, _, output = build_package(dummy_ple_model, tmp_path, profile="canary")

    assert not plan.manifest.complete_model
    assert [item.shard_index for item in plan.ple.shards] == [0, 3]
    assert plan.ple.total_rows == 64
    assert plan.ple.logical_total_rows == 128
    assert PtitanValidator(output).validate("fast").is_valid
    assert PtitanValidator(output).validate("full").is_valid


def test_dense_corruption_is_detected(dummy_ple_model, tmp_path):
    plan, writer, output = build_package(dummy_ple_model, tmp_path)
    target = plan.dense[0]
    flip_byte(writer.dense_path, target.byte_offset)

    report = PtitanValidator(output).validate("fast")

    assert not report.is_valid
    assert any("dense tensor checksum mismatch" in error for error in report.errors)


def test_expert_corruption_is_detected(dummy_moe_model, tmp_path):
    plan, writer, output = build_package(dummy_moe_model, tmp_path)
    flip_byte(writer.bank_path, plan.experts[3].bank_offset + 7)

    report = PtitanValidator(output).validate("fast")

    assert not report.is_valid
    assert any("expert record checksum mismatch" in error for error in report.errors)


def test_ple_corruption_is_detected(dummy_ple_model, tmp_path):
    plan, writer, output = build_package(dummy_ple_model, tmp_path)
    flip_byte(writer.ple_path, plan.ple.row.row_offset(7))

    report = PtitanValidator(output).validate("fast")

    assert not report.is_valid
    assert any("PLE source shard checksum mismatch" in error for error in report.errors)


def test_truncated_region_is_detected(dummy_moe_model, tmp_path):
    _, writer, output = build_package(dummy_moe_model, tmp_path)
    with writer.bank_path.open("r+b") as stream:
        stream.truncate(writer.bank_path.stat().st_size - 1)

    report = PtitanValidator(output).validate("fast")

    assert not report.is_valid
    assert any("size mismatch" in error for error in report.errors)


def test_v1_prototype_is_explicitly_rejected(dummy_ple_model, tmp_path):
    _, _, output = build_package(dummy_ple_model, tmp_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["package_version"] = "1.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = PtitanValidator(output).validate("fast")

    assert not report.is_valid
    assert any("unsupported package version" in error for error in report.errors)


def test_uncommitted_journal_is_rejected(dummy_ple_model, tmp_path):
    _, writer, output = build_package(dummy_ple_model, tmp_path)
    journal = json.loads(writer.journal_path.read_text("utf-8"))
    journal["finished"] = False
    writer.journal_path.write_text(json.dumps(journal), encoding="utf-8")

    report = PtitanValidator(output).validate("fast")

    assert not report.is_valid
    assert "build journal is not complete" in report.errors


def test_unknown_validation_mode_is_rejected(dummy_ple_model, tmp_path):
    _, _, output = build_package(dummy_ple_model, tmp_path)
    report = PtitanValidator(output).validate("impossible")
    assert not report.is_valid


def test_disk_preflight_requires_fifteen_percent_headroom(dummy_ple_model, tmp_path, monkeypatch):
    scan = scan_checkpoint(str(dummy_ple_model))
    plan = plan_package(scan, precision_map=PrecisionMap.uniform(4, 32, "disk-test"))
    writer = PackageWriter(
        plan,
        tmp_path / "too-large.ptitan",
        LocalTensorReader(dummy_ple_model),
        method=QuantMethod.RTN,
        device="cpu",
    )
    DiskUsage = namedtuple("DiskUsage", "total used free")
    monkeypatch.setattr(
        "pockettitan.package.writer.shutil.disk_usage",
        lambda _: DiskUsage(1_000_000, 999_999, 1),
    )

    with pytest.raises(WriteError, match="15% headroom"):
        writer.build()
