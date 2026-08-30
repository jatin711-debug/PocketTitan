"""R0 audit tests.

The golden tests pin PocketTitan's architectural ground truth for
``Qwen/Qwen3.8-Flash-Next`` (Plan.md §2) to exact integers, offline, from a
checked-in header fixture. If one of them fails, either the taxonomy regressed
or the checkpoint changed — both require updating Plan.md §2 in the same commit,
per the reporting rule in Plan.md §9.
"""

import json

import pytest

from pockettitan.audit import (
    Capability,
    Component,
    PrecisionMap,
    ShardScanError,
    Tier,
    build_audit_report,
    classify_all,
    classify_tensor,
    effective_bits,
    get_precision_preset,
    scan_checkpoint,
)

# --- Ground truth: Qwen/Qwen3.8-Flash-Next @ de4b8e4d ----------------------- #
TOTAL_PARAMS = 179_999_981_459
TOTAL_BYTES = 359_999_963_128
TEXT_ONLY_CORE = 125_743_653_795  # matches llama.cpp's reported 125.74 B
ACTIVATED_PER_TOKEN = 6_671_300_515
EXPERT_PARAMS_PER_TOKEN = 2_359_296_000

COMPONENT_PARAMS = {
    Component.EXPERTS_ROUTED: 120_795_955_200,
    Component.PLE_TABLE: 51_200_245_760,
    Component.MTP: 2_607_150_848,
    Component.GDN_ATTN: 2_086_510_464,
    Component.HYPERCONN: 640_624_640,
    Component.EMBED: 635_699_200,
    Component.LM_HEAD: 635_699_200,
    Component.FULL_ATTN: 617_358_336,
    Component.VISION: 448_931_056,
    Component.SHARED_EXPERT: 235_929_600,
    Component.ROUTER: 63_037_440,
    Component.PLE_PROJ: 32_839_715,
}


@pytest.fixture(scope="module")
def qwen_report(qwen_scan):
    return build_audit_report(qwen_scan, precision_map=PrecisionMap.pt_q4e())


# --------------------------------------------------------------------------- #
# Golden: the R0 gate
# --------------------------------------------------------------------------- #


def test_fixture_shape(qwen_scan):
    assert qwen_scan.num_tensors == 1658
    assert len(qwen_scan.shards) == 131


def test_total_params_exact(qwen_scan):
    """The R0 gate: reproduce the published total exactly."""
    assert qwen_scan.total_params == TOTAL_PARAMS


def test_summed_bytes_match_published_index(qwen_scan):
    """Independent cross-check: our sum must equal the index's total_size."""
    assert qwen_scan.total_bytes == TOTAL_BYTES
    assert qwen_scan.total_bytes == qwen_scan.declared_total_bytes


def test_component_breakdown_exact(qwen_scan):
    breakdown = classify_all(qwen_scan.tensors)
    for component, expected in COMPONENT_PARAMS.items():
        assert breakdown.params_of(component) == expected, f"{component.value} drifted"
    assert breakdown.total_params == TOTAL_PARAMS


def test_every_tensor_is_classified(qwen_scan):
    """An unclassified tensor means the taxonomy does not understand this
    architecture, which invalidates every downstream budget."""
    breakdown = classify_all(qwen_scan.tensors)
    assert breakdown.unclassified == []


def test_text_only_core_matches_llama_cpp(qwen_report):
    """The second R0 gate: text-only core must equal llama.cpp's 125.74 B."""
    assert qwen_report.lm_core_params == TEXT_ONLY_CORE
    assert round(qwen_report.lm_core_params / 1e9, 2) == 125.74
    # The table is still text capability; it is merely stored separately.
    assert (
        qwen_report.enabled_params - qwen_report.lm_core_params
        == COMPONENT_PARAMS[Component.PLE_TABLE]
    )


def test_capability_stripping_totals(qwen_report):
    dropped = qwen_report.dropped_params
    assert dropped[Component.VISION] == COMPONENT_PARAMS[Component.VISION]
    assert dropped[Component.MTP] == COMPONENT_PARAMS[Component.MTP]
    # Plan.md §2.4: capability stripping is a rounding error, not a strategy.
    saved_fraction = sum(dropped.values()) / TOTAL_PARAMS
    assert saved_fraction < 0.02


def test_activated_params_per_token_exact(qwen_report):
    assert qwen_report.activation.total == ACTIVATED_PER_TOKEN
    assert qwen_report.activation.expert_params == EXPERT_PARAMS_PER_TOKEN
    assert qwen_report.activation.dense_params == ACTIVATED_PER_TOKEN - EXPERT_PARAMS_PER_TOKEN
    # The whole thesis in one assertion.
    assert qwen_report.activation.expert_params / TOTAL_PARAMS < 0.014


def test_row_lookup_components_are_nearly_free(qwen_report):
    """PLE is 28% of the model but must contribute ~nothing per token."""
    per_component = qwen_report.activation.per_component
    assert per_component[Component.PLE_TABLE] == 16 * 160
    assert per_component[Component.EMBED] == 2560


def test_expert_geometry(qwen_report):
    geom = qwen_report.expert_geometry
    assert geom is not None
    assert geom.num_experts == 512
    assert geom.top_k == 10
    assert geom.num_expert_layers == 48
    assert geom.params_per_expert == 4_915_200
    assert geom.total_slots == 24_576
    assert geom.activations_per_token == 480
    assert geom.tensors_per_expert == 2  # fused gate_up + down, i.e. 2 reads before repacking


def test_ple_geometry(qwen_report):
    geom = qwen_report.ple_geometry
    assert geom is not None
    assert geom.row_width == 160
    assert geom.num_heads == 16
    assert geom.total_rows == 320_001_536
    assert geom.params_per_token == 2560


def test_state_budget(qwen_report):
    state = qwen_report.state
    assert state.num_full_attn_layers == 12
    assert state.num_linear_attn_layers == 36
    assert state.kv_bytes_per_token == 24 * 1024
    assert state.indexer_bytes_per_token == 3 * 1024
    assert state.recurrent_state_bytes == 119_144_448
    # Recurrent state must not grow with context.
    assert state.at_context(8192) - state.at_context(4096) == state.bytes_per_token * 4096


def test_storage_budget_pt_q4e(qwen_report):
    storage = qwen_report.storage
    by_component = {e.component: e for e in storage.entries}

    assert by_component[Component.ROUTER].effective_bits == 16.0, "routers must never be quantized"
    assert by_component[Component.EXPERTS_ROUTED].effective_bits == pytest.approx(4.25)
    assert by_component[Component.PLE_TABLE].effective_bits == pytest.approx(3.10)

    assert by_component[Component.EXPERTS_ROUTED].packed_bytes == 64_172_851_200
    assert by_component[Component.PLE_TABLE].packed_bytes == 19_840_095_232

    # Dense core must fit a 4 GB card alongside state and staging.
    vram = storage.bytes_in_tier(Tier.VRAM_HOT)
    assert 2.0 * (1 << 30) < vram < 2.3 * (1 << 30)

    # Vision and MTP excluded from a text-only package.
    assert Component.VISION not in by_component
    assert Component.MTP not in by_component


def test_roofline_expert_traffic(qwen_report):
    roofline = qwen_report.roofline
    assert roofline is not None
    assert roofline.reads_per_token == 480
    assert roofline.total_expert_slots == 24_576
    # 4-bit expert record and per-token traffic (Plan.md §2.2).
    assert roofline.expert_record_bytes == pytest.approx(2.49 * (1 << 20), rel=0.01)
    assert roofline.expert_bytes_per_token == pytest.approx(1195 * (1 << 20), rel=0.01)
    # The open risk: cache holds ~12% of experts at 4-bit.
    assert 0.10 < roofline.cache_capacity_fraction < 0.13


def test_roofline_2bit_halves_traffic(qwen_scan):
    q4 = build_audit_report(qwen_scan, precision_map=PrecisionMap.pt_q4e()).roofline
    q2 = build_audit_report(qwen_scan, precision_map=PrecisionMap.pt_q2e()).roofline
    ratio = q4.expert_bytes_per_token / q2.expert_bytes_per_token
    assert 1.85 < ratio < 1.95  # 4.25 / 2.25


# --------------------------------------------------------------------------- #
# Unit: precision arithmetic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bits,group,symmetric,expected",
    [
        (4, 128, False, 4.25),
        (2, 128, False, 2.25),
        (3, 128, False, 3.25),
        (2, 64, False, 2.5),
        (3, 160, True, 3.10),
        (4, -1, False, 4.0),
        (16, 128, False, 16.0),
    ],
)
def test_effective_bits(bits, group, symmetric, expected):
    assert effective_bits(bits, group, symmetric) == pytest.approx(expected)


def test_nominal_bits_understate_footprint():
    """Guards the Plan.md §6 claim that nominal-bits arithmetic is wrong."""
    assert effective_bits(2, 128) > 2.0
    assert effective_bits(1, 128) > 1.0


def test_precision_presets_resolve():
    for name in ("pt-q4e", "pt-q2e", "bf16", "int8", "int4", "int3", "int2", "ternary"):
        assert get_precision_preset(name) is not None
    with pytest.raises(KeyError):
        get_precision_preset("does-not-exist")


def test_uniform_preset_still_protects_routers():
    pmap = get_precision_preset("int2")
    assert pmap.bits_for(Component.ROUTER) == 16.0
    assert pmap.bits_for(Component.EXPERTS_ROUTED) == pytest.approx(2.25)


# --------------------------------------------------------------------------- #
# Unit: taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.language_model.layers.7.mlp.experts.gate_up_proj", Component.EXPERTS_ROUTED),
        ("model.layers.3.mlp.experts.11.down_proj.weight", Component.EXPERTS_ROUTED),
        (
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_9.weight",
            Component.PLE_TABLE,
        ),
        ("model.language_model.layers.1.ple.key_proj.weight", Component.PLE_PROJ),
        ("model.language_model.layers.7.mlp.gate.weight", Component.ROUTER),
        ("model.language_model.layers.7.mlp.shared_expert_gate.weight", Component.ROUTER),
        ("model.language_model.layers.7.mlp.shared_expert.up_proj.weight", Component.SHARED_EXPERT),
        ("model.visual.blocks.3.attn.qkv.weight", Component.VISION),
        ("mtp.layers.0.self_attn.q_proj.weight", Component.MTP),
        ("model.language_model.layers.2.linear_attn.in_proj_qkv.weight", Component.GDN_ATTN),
        ("model.language_model.layers.3.self_attn.q_proj.weight", Component.FULL_ATTN),
        ("model.language_model.layers.5.attn_hyper_connection.hc_norm.weight", Component.HYPERCONN),
        ("model.language_model.embed_tokens.weight", Component.EMBED),
        ("lm_head.weight", Component.LM_HEAD),
        ("model.layers.0.mlp.gate_proj.weight", Component.MLP_DENSE),
        ("model.layers.0.input_layernorm.weight", Component.NORM),
    ],
)
def test_classification_rules(name, expected):
    assert classify_tensor(name).component is expected


def test_vision_and_mtp_are_droppable_capabilities():
    assert classify_tensor("model.visual.patch_embed.proj.weight").capability is Capability.VISION
    assert classify_tensor("mtp.fc_hidden.weight").capability is Capability.MTP
    assert classify_tensor("lm_head.weight").capability is Capability.TEXT


def test_cold_tier_assignment():
    """Only experts and the n-gram table belong on NVMe."""
    assert classify_tensor("model.layers.0.mlp.experts.gate_up_proj").tier is Tier.NVME_COLD
    assert (
        classify_tensor("m.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight").tier
        is Tier.NVME_COLD
    )
    assert classify_tensor("model.layers.3.self_attn.q_proj.weight").tier is Tier.VRAM_HOT
    assert classify_tensor("model.embed_tokens.weight").tier is Tier.RAM_WARM


# --------------------------------------------------------------------------- #
# Unit: scanner behaviour on real files
# --------------------------------------------------------------------------- #


def test_scan_local_checkpoint(dummy_transformer_model):
    scan = scan_checkpoint(str(dummy_transformer_model))
    assert scan.is_local
    assert len(scan.shards) == 2
    assert scan.num_tensors == 21
    assert scan.total_params == sum(t.num_params for t in scan.tensors.values())
    assert scan.dtype_histogram() == {"F16": 21}


def test_scan_local_report_is_coherent(dummy_transformer_model):
    scan = scan_checkpoint(str(dummy_transformer_model))
    report = build_audit_report(scan)
    # Dense model: no MoE, so no roofline and everything is activated per token.
    assert report.expert_geometry is None
    assert report.roofline is None
    assert report.activation.total > 0


def test_strict_scan_raises_on_missing_shard(dummy_transformer_model, tmp_path):
    """A partial scan that looks successful is worse than a hard failure."""
    broken = tmp_path / "broken_model"
    broken.mkdir()
    (broken / "config.json").write_text(
        (dummy_transformer_model / "config.json").read_text(), encoding="utf-8"
    )
    index = json.loads((dummy_transformer_model / "model.safetensors.index.json").read_text())
    (broken / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    # Copy only the first shard; the index still references both.
    shard = "model-00001-of-00002.safetensors"
    (broken / shard).write_bytes((dummy_transformer_model / shard).read_bytes())

    with pytest.raises(ShardScanError, match="failed to parse"):
        scan_checkpoint(str(broken), retries=1)

    lenient = scan_checkpoint(str(broken), strict=False, retries=1)
    assert any(d.kind == "shard_unreadable" for d in lenient.discrepancies)
    assert any(d.kind == "missing_tensor" for d in lenient.discrepancies)


def test_total_size_mismatch_is_reported(dummy_transformer_model):
    """conftest's fixture publishes a deliberately wrong total_size (1000000)."""
    scan = scan_checkpoint(str(dummy_transformer_model))
    assert any(d.kind == "total_size_mismatch" for d in scan.discrepancies)


# --------------------------------------------------------------------------- #
# Unit: rendering
# --------------------------------------------------------------------------- #


def test_render_report_smoke(qwen_report):
    """Exercise every table in the render path; a KeyError here would only ever
    surface in production output."""
    import io

    from rich.console import Console

    from pockettitan.audit.report import render_report

    buffer = io.StringIO()
    render_report(Console(file=buffer, width=140, legacy_windows=False), qwen_report)
    output = buffer.getvalue()

    assert "179,999,981,459" in output
    assert "Component Decomposition" in output
    assert "Capability Stripping" in output
    assert "Storage Budget" in output
    assert "State Budget" in output
    assert "SSD Roofline" in output


def test_render_survives_legacy_codepage(qwen_report):
    """Windows consoles use cp1252; rendering must not raise UnicodeEncodeError."""
    import io

    from rich.console import Console

    from pockettitan.audit.report import render_report

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    render_report(Console(file=stream, width=140, legacy_windows=False), qwen_report)
    stream.flush()
    assert raw.getvalue()
