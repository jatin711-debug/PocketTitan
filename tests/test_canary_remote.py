"""Pinned remote canary verification test for Qwen3.8-Flash-Next (R1 / T1.10)."""

import pytest
from pockettitan.audit import PrecisionMap, scan_checkpoint
from pockettitan.config import MemoryBudgetConfig, QuantMethod
from pockettitan.package import PackageWriter, PtitanValidator, plan_package
from pockettitan.streaming.reader import RemoteTensorSliceReader


@pytest.mark.network
def test_remote_canary_live_build_and_validation(tmp_path):
    """Build and validate a sparse canary package directly over remote HTTP range requests."""
    model_id = "Qwen/Qwen3.8-Flash-Next"
    
    # 1. Scan remote headers
    scan = scan_checkpoint(model_id, max_workers=8)
    assert scan.num_tensors == 1658
    
    # 2. Plan a compact canary package
    plan = plan_package(
        scan,
        precision_map=PrecisionMap.pt_q4e(),
        quant_method=QuantMethod.RTN,
        build_profile="canary",
    )
    assert plan.manifest.build_profile == "canary"
    
    # 3. Stream and build canary package
    output_dir = tmp_path / "canary.ptitan"
    reader = RemoteTensorSliceReader(model_id=model_id)
    writer = PackageWriter(
        plan=plan,
        output_dir=output_dir,
        reader=reader,
        budget=MemoryBudgetConfig(max_vram_mb=3000),
        method=QuantMethod.RTN,
        device="cpu",
    )
    writer.build()
    
    # 4. Validate integrity of built canary
    validator = PtitanValidator(output_dir)
    fast_report = validator.validate("fast")
    assert fast_report.is_valid, fast_report.errors
    
    full_report = validator.validate("full")
    assert full_report.is_valid, full_report.errors
