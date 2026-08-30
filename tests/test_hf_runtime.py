"""End-to-end: a real Qwen3_5 checkpoint -> .ptitan package -> logits.

This is the test the project did not have. Everything before it verified that
bytes round-tripped; this one verifies that a *model* round-trips: the same
architecture, weights and prompt run two ways must agree.

The fixture is a genuine (tiny) ``Qwen3_5ForCausalLM`` with the real hybrid
layer pattern -- Gated DeltaNet layers plus full-attention layers -- so the
loader is exercised against the same module tree the 27B uses, only smaller.
"""

import json

import pytest
import safetensors.torch
import torch

from pockettitan.audit import PrecisionMap, scan_checkpoint
from pockettitan.config import MemoryBudgetConfig, QuantMethod
from pockettitan.package import PackageWriter, plan_package
from pockettitan.runtime.hf import (
    LoaderError,
    PackagedEmbedding,
    PackagedLinear,
    PackageWeights,
    build_causal_lm,
    summarize,
)
from pockettitan.streaming.reader import LocalTensorReader

transformers = pytest.importorskip("transformers")

HIDDEN = 64
LAYERS = 4
VOCAB = 128


def _text_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        # The real 27B is 48 linear-attention + 16 full-attention at interval 4.
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_value_head_dim=16,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
    )


@pytest.fixture(scope="module")
def qwen3_5_checkpoint(tmp_path_factory):
    """A real, randomly-initialized Qwen3_5 saved the way the 27B is stored."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(11)
    config = _text_config()
    model = Qwen3_5ForCausalLM(config).to(torch.float32).eval()

    directory = tmp_path_factory.mktemp("qwen3_5_src")
    # Mirror the published layout: the language tower is nested under
    # `model.language_model.` and `lm_head` sits at the top level.
    tensors = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("model."):
            name = "model.language_model." + name[len("model.") :]
        tensors[name] = tensor.detach().to(torch.float16).contiguous()
    safetensors.torch.save_file(tensors, str(directory / "model.safetensors"))

    payload = config.to_dict()
    payload.setdefault("architectures", ["Qwen3_5ForConditionalGeneration"])
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    return directory, model, config


def _build_package(source_dir, out_dir, precision):
    scan = scan_checkpoint(str(source_dir))
    plan = plan_package(scan, precision_map=precision)
    PackageWriter(
        plan,
        out_dir,
        LocalTensorReader(source_dir),
        budget=MemoryBudgetConfig(max_vram_mb=512.0),
        method=QuantMethod.RTN,
        device="cpu",
    ).build()
    return plan


@pytest.fixture(scope="module")
def fp16_package(qwen3_5_checkpoint, tmp_path_factory):
    """A 16-bit package: no quantization error, so any logit gap is a loader bug."""
    source, _, _ = qwen3_5_checkpoint
    out = tmp_path_factory.mktemp("pkg_fp16") / "model.ptitan"
    _build_package(source, out, PrecisionMap.uniform(16, -1, "fp16"))
    return out


# --------------------------------------------------------------------------- #
# The load contract
# --------------------------------------------------------------------------- #


def test_every_parameter_is_backed_by_the_package(fp16_package):
    """An unbacked parameter runs uninitialized and produces fluent nonsense."""
    model, weights = build_causal_lm(fp16_package, dtype=torch.float32)
    try:
        info = summarize(model)
        assert info["packaged_linears"] > 0
        assert info["packaged_embeddings"] == 1
        assert not [n for n, p in model.named_parameters() if p.device.type == "meta"]
        # Only the small tensors stay resident: norms, A_log, dt_bias, conv1d.
        for name, _ in model.named_parameters():
            assert any(
                key in name
                for key in ("norm", "A_log", "dt_bias", "conv1d")
            ), f"{name} should have been swapped for a packaged module"
    finally:
        weights.close()


def test_missing_tensor_is_refused_not_silently_ignored(fp16_package, tmp_path):
    """Dropping a tensor from the manifest must fail the load."""
    import shutil

    broken = tmp_path / "broken.ptitan"
    shutil.copytree(fp16_package, broken)
    manifest = json.loads((broken / "manifest.json").read_text(encoding="utf-8"))
    before = len(manifest["dense"])
    manifest["dense"] = [
        entry for entry in manifest["dense"] if not entry["name"].endswith("layers.1.mlp.up_proj.weight")
    ]
    assert len(manifest["dense"]) == before - 1
    (broken / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(LoaderError, match="no package tensor"):
        build_causal_lm(broken, dtype=torch.float32)


def test_name_aliasing_handles_the_nested_language_tower(fp16_package):
    weights = PackageWeights(fp16_package)
    try:
        assert weights.resolve("model.layers.0.mlp.gate_proj.weight") == (
            "model.language_model.layers.0.mlp.gate_proj.weight"
        )
        assert weights.resolve("lm_head.weight") == "lm_head.weight"
        assert weights.resolve("model.layers.0.nonexistent.weight") is None
    finally:
        weights.close()


# --------------------------------------------------------------------------- #
# The property that actually matters
# --------------------------------------------------------------------------- #


def test_fp16_package_reproduces_the_source_models_logits(qwen3_5_checkpoint, fp16_package):
    """A 16-bit package is lossless up to fp16 rounding, so the logits must match.

    This is the end-to-end statement the project needs: the same architecture and
    the same weights, reached through the package, compute the same thing. Any
    gap here is a defect in packing, addressing, or decoding -- not quantization.
    """
    _, reference, _ = qwen3_5_checkpoint
    ids = torch.tensor([[3, 17, 42, 5, 17, 99]])

    with torch.no_grad():
        expected = reference(input_ids=ids, use_cache=False).logits

    model, weights = build_causal_lm(fp16_package, dtype=torch.float32)
    try:
        with torch.no_grad():
            got = model(input_ids=ids, use_cache=False).logits
    finally:
        weights.close()

    assert got.shape == expected.shape
    gap = (got - expected).abs().max().item()
    scale = expected.abs().max().item()
    assert gap < 5e-3 * scale, f"max logit gap {gap:.6f} against a scale of {scale:.3f}"
    assert torch.equal(got.argmax(-1), expected.argmax(-1)), "argmax token disagrees"


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_package_tracks_the_source_model(qwen3_5_checkpoint, tmp_path_factory, bits):
    """Quantization degrades logits; it must not decorrelate them.

    A correct pipeline at 8 bits stays very close and at 4 bits stays clearly
    correlated. A *broken* pipeline lands near zero correlation, which is what a
    wrong offset or a mis-addressed record actually looks like.
    """
    source, reference, _ = qwen3_5_checkpoint
    out = tmp_path_factory.mktemp(f"pkg_{bits}") / "model.ptitan"
    _build_package(source, out, PrecisionMap.uniform(bits, 32, f"int{bits}"))

    ids = torch.tensor([[3, 17, 42, 5, 17, 99]])
    with torch.no_grad():
        expected = reference(input_ids=ids, use_cache=False).logits

    model, weights = build_causal_lm(out, dtype=torch.float32)
    try:
        with torch.no_grad():
            got = model(input_ids=ids, use_cache=False).logits
    finally:
        weights.close()

    correlation = torch.corrcoef(
        torch.stack([got.flatten().float(), expected.flatten().float()])
    )[0, 1].item()
    floor = 0.99 if bits == 8 else 0.85
    assert correlation > floor, f"{bits}-bit logits correlate {correlation:.4f} (floor {floor})"


# --------------------------------------------------------------------------- #
# Bounded residency
# --------------------------------------------------------------------------- #


def test_large_weights_are_never_resident(fp16_package):
    """The point of the exercise: a package larger than RAM still runs."""
    model, weights = build_causal_lm(fp16_package, dtype=torch.float32, cache_bytes=0)
    try:
        resident = sum(p.numel() for p in model.parameters())
        packaged = sum(
            m.in_features * m.out_features
            for m in model.modules()
            if isinstance(m, PackagedLinear)
        )
        assert weights.resident_bytes == 0
        assert resident < packaged / 10, (
            f"{resident:,} parameters are held in memory against {packaged:,} packaged"
        )
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    finally:
        weights.close()


def test_generation_runs_and_is_deterministic_under_greedy_decoding(fp16_package):
    model, weights = build_causal_lm(fp16_package, dtype=torch.float32)
    try:
        ids = torch.tensor([[7, 11, 13]])
        with torch.no_grad():
            first = model.generate(ids, max_new_tokens=4, do_sample=False)
            second = model.generate(ids, max_new_tokens=4, do_sample=False)
        assert first.shape[1] == ids.shape[1] + 4
        assert torch.equal(first, second)
    finally:
        weights.close()


def test_embedding_reads_only_the_rows_it_needs(fp16_package):
    """A 248,320-row table must not be materialized to look up six tokens."""
    model, weights = build_causal_lm(fp16_package, dtype=torch.float32, cache_bytes=0)
    try:
        embedding = next(m for m in model.modules() if isinstance(m, PackagedEmbedding))
        weights.decoded_bytes = 0
        ids = torch.tensor([[3, 17, 3, 17]])
        out = embedding(ids)
        assert out.shape == (1, 4, HIDDEN)
        # Two distinct ids, one row each -- not the whole table.
        assert weights.decoded_bytes <= 2 * HIDDEN * 4 * 2
    finally:
        weights.close()


# --------------------------------------------------------------------------- #
# The user-facing path: package + tokenizer -> text
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    """A real fast tokenizer whose vocabulary fits the fixture model."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<s>", "</s>"] + [f"w{i}" for i in range(VOCAB - 3)]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()

    directory = tmp_path_factory.mktemp("tok")
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
    )
    fast.save_pretrained(str(directory))
    return directory


def test_generate_produces_text_and_reports_its_io(fp16_package, tiny_tokenizer, tmp_path):
    """The whole user-facing path, including the tokenizer copied into the package."""
    import shutil

    from pockettitan.runtime.hf import generate

    package = tmp_path / "runnable.ptitan"
    shutil.copytree(fp16_package, package)
    if (package / "tokenizer").exists():
        shutil.rmtree(package / "tokenizer")
    shutil.copytree(tiny_tokenizer, package / "tokenizer")

    result = generate(
        package, prompt="w5 w9 w5", max_new_tokens=4, chat=False, dtype="float32",
    )

    assert result.generated_tokens == 4
    assert result.prompt_tokens == 3
    assert isinstance(result.text, str)
    # The accounting has to be real, not a placeholder.
    assert result.decode_calls > 0
    assert result.decoded_bytes > 0
    assert result.tokens_per_second > 0
    assert len(result.summary_lines()) == 3


def test_generate_reports_a_missing_tokenizer_clearly(fp16_package, tmp_path):
    import shutil

    from pockettitan.runtime.hf import generate

    package = tmp_path / "no_tokenizer.ptitan"
    shutil.copytree(fp16_package, package)
    if (package / "tokenizer").exists():
        shutil.rmtree(package / "tokenizer")

    with pytest.raises(FileNotFoundError, match="no tokenizer"):
        generate(package, prompt="hi", max_new_tokens=1)
