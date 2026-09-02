"""Opt-in live DomainSlice canary against the pinned public OLMoE revision."""

import os

import pytest
import torch
from typer.testing import CliRunner

from pockettitan.cli import app
from pockettitan.domainslice import (
    CompositeWeightStore,
    ModelRevision,
    PocketTitanPageStore,
    RemoteHuggingFaceStore,
    WeightPageID,
)
from pockettitan.domainslice.hypothesis import (
    run_olmoe_block_hypothesis,
    run_olmoe_full_model_hypothesis,
    run_olmoe_layer_hypothesis,
)

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"


@pytest.mark.network
def test_live_olmoe_layer9_expert7_exact_range_and_cache(tmp_path):
    revision = ModelRevision(repo_id=MODEL_ID, commit_sha=REVISION)
    remote = RemoteHuggingFaceStore(
        revision,
        token=os.environ.get("HF_TOKEN"),
        max_workers=3,
    )
    local = PocketTitanPageStore(tmp_path / "cache", max_cache_bytes=50 * 1024**3)
    store = CompositeWeightStore(local, remote, download_workers=3)
    page_id = WeightPageID.expert(revision, 9, 7)
    try:
        descriptor = store.resolve(page_id)
        assert descriptor.expected_bytes == 12_582_912
        assert len(descriptor.source_slices) == 3

        first = store.materialize(page_id)
        assert first.bytes_fetched == 12_582_912
        store.release(first)
        before = store.stats().remote_payload_bytes

        second = store.materialize(page_id)
        assert second.cache_hit is True
        assert store.stats().remote_payload_bytes == before
        store.release(second)
    finally:
        store.close()


@pytest.mark.network
def test_live_domainslice_cli_reports_miss_then_hit(tmp_path):
    runner = CliRunner()
    args = [
        "domainslice",
        "fetch-expert",
        MODEL_ID,
        "--layer",
        "9",
        "--expert",
        "7",
        "--cache-dir",
        str(tmp_path / "cli-cache"),
        "--revision",
        REVISION,
        "--download-workers",
        "3",
        "--max-cache",
        "50GB",
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.stdout
    assert "MISS" in first.stdout

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.stdout
    assert "HIT" in second.stdout
    assert "0.00 B" in second.stdout


@pytest.mark.network
def test_live_olmoe_paged_block_matches_transformers(tmp_path):
    revision = ModelRevision(repo_id=MODEL_ID, commit_sha=REVISION)
    remote = RemoteHuggingFaceStore(
        revision,
        token=os.environ.get("HF_TOKEN"),
        max_workers=3,
    )
    local = PocketTitanPageStore(tmp_path / "block-cache", max_cache_bytes=2 * 1024**3)
    store = CompositeWeightStore(local, remote, download_workers=3)
    try:
        result = run_olmoe_block_hypothesis(
            store,
            remote,
            layer=9,
            tokens=1,
            seed=42,
            execution_device="cpu",
        )
        assert result.passed is True
        assert len(result.selected_experts) == 8
        assert result.cold.page_faults == 8
        assert result.cold.remote_bytes == 8 * 12_582_912
        assert result.warm.page_hits == 8
        assert result.warm.remote_bytes == 0
        assert result.max_abs_error == 0.0
    finally:
        store.close()


@pytest.mark.network
def test_live_complete_olmoe_layer_is_warm_deterministic(tmp_path):
    revision = ModelRevision(repo_id=MODEL_ID, commit_sha=REVISION)
    remote = RemoteHuggingFaceStore(
        revision,
        token=os.environ.get("HF_TOKEN"),
        max_workers=3,
    )
    local = PocketTitanPageStore(tmp_path / "layer-cache", max_cache_bytes=2 * 1024**3)
    store = CompositeWeightStore(local, remote, download_workers=3)
    try:
        result = run_olmoe_layer_hypothesis(
            store,
            remote,
            layer=9,
            seed=42,
            execution_device="cpu",
        )
        assert result.passed is True
        assert result.backbone.tensors == 9
        assert result.backbone.remote_bytes == 33_832_960
        assert len(result.selected_experts) == 8
        assert result.first.remote_bytes == 8 * 12_582_912
        assert result.warm.remote_bytes == 0
        assert result.warm_max_abs_delta == 0.0
    finally:
        store.close()


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("POCKETTITAN_FULL_MODEL_CANARY") != "1",
    reason="set POCKETTITAN_FULL_MODEL_CANARY=1 for the multi-gigabyte live run",
)
def test_live_full_olmoe_token_has_exact_warm_logits(tmp_path):
    revision = ModelRevision(repo_id=MODEL_ID, commit_sha=REVISION)
    remote = RemoteHuggingFaceStore(
        revision,
        token=os.environ.get("HF_TOKEN"),
        max_workers=3,
    )
    local = PocketTitanPageStore(tmp_path / "model-cache", max_cache_bytes=4 * 1024**3)
    store = CompositeWeightStore(local, remote, download_workers=3)
    try:
        result = run_olmoe_full_model_hypothesis(
            store,
            remote,
            input_token_id=1,
            execution_device="cuda" if torch.cuda.is_available() else "cpu",
            max_vram_bytes=3584 * 1024**2,
            max_ram_bytes=12 * 1024**3,
        )
        assert result.passed is True
        assert result.num_layers == 16
        assert result.first.experts_executed == 128
        assert result.warm.total_remote_bytes == 0
        assert result.logits_max_abs_delta == 0.0
    finally:
        store.close()
