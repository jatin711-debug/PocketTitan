"""Unit tests for DomainSlice prompt-driven text generation."""

import types
import torch

from pockettitan.domainslice import (
    ModelRevision,
)
from pockettitan.domainslice.generate import (
    DomainSliceGenerateResult,
    _sample_next_token,
    generate_olmoe_text,
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from test_olmoe_paged import _StaticPageStore, _add_tensor_page, _build_pages


class _MockTokenizer:
    """Deterministic mock tokenizer for testing generation loops."""

    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.chat_template = True

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = [max(1, (ord(c) % (self.vocab_size - 1)) + 1) for c in text if not c.isspace()]
        return tokens[:1] or [1]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return " ".join(f"t{tid}" for tid in token_ids)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return " ".join(msg["content"] for msg in messages)


def test_sample_next_token_greedy():
    logits = torch.tensor([0.1, 2.5, -1.0, 5.2, 0.3])
    sampled = _sample_next_token(logits, temperature=0.0)
    assert sampled == 3


def test_sample_next_token_temperature():
    torch.manual_seed(42)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    sampled = _sample_next_token(logits, temperature=0.8, top_p=0.9)
    assert 0 <= sampled < 4


def test_generate_olmoe_text_with_static_fixture(tmp_path):
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    torch.manual_seed(42)
    revision = ModelRevision(repo_id="fixture/olmoe", commit_sha="e" * 40)
    config = OlmoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=4,
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    reference = OlmoeForCausalLM(config).to(dtype=torch.bfloat16).eval()
    pages = {}
    for layer_idx, layer in enumerate(reference.model.layers):
        gate_up = layer.mlp.experts.gate_up_proj.detach().clone()
        pages.update(
            _build_pages(
                tmp_path,
                revision,
                gate_up[:, : config.intermediate_size],
                gate_up[:, config.intermediate_size :],
                layer.mlp.experts.down_proj.detach().clone(),
                layer=layer_idx,
            )
        )
        for name, parameter in layer.named_parameters():
            if name.startswith("mlp.experts."):
                continue
            _add_tensor_page(
                tmp_path,
                pages,
                revision,
                f"model.layers.{layer_idx}.{name}",
                parameter.detach().clone(),
            )
    _add_tensor_page(
        tmp_path,
        pages,
        revision,
        "model.embed_tokens.weight",
        reference.model.embed_tokens.weight.detach().clone(),
    )
    _add_tensor_page(
        tmp_path,
        pages,
        revision,
        "model.norm.weight",
        reference.model.norm.weight.detach().clone(),
    )
    _add_tensor_page(
        tmp_path,
        pages,
        revision,
        "lm_head.weight",
        reference.lm_head.weight.detach().clone(),
    )

    store = _StaticPageStore(pages)
    remote = types.SimpleNamespace(model_revision=revision, token=None)
    mock_tokenizer = _MockTokenizer(vocab_size=16)

    streamed_tokens = []

    def on_stream(chunk: str):
        streamed_tokens.append(chunk)

    result = generate_olmoe_text(
        store,
        remote,
        prompt="hello",
        tokenizer=mock_tokenizer,
        max_new_tokens=2,
        temperature=0.0,
        execution_device="cpu",
        head_chunk_bytes=64,
        stream_callback=on_stream,
        config=config,
    )

    assert isinstance(result, DomainSliceGenerateResult)
    assert len(result.prompt_tokens) == 1
    assert len(result.generated_tokens) == 2
    assert len(streamed_tokens) == 2
    assert result.prefill_elapsed_s > 0
    assert result.decode_elapsed_s > 0
    assert result.total_elapsed_s > 0
    assert result.final_kv_cache_bytes > 0
    assert result.peak_rss_bytes > 0

    summary = result.summary_lines()
    assert len(summary) >= 5
    assert any("Prompt length: 1 tokens" in line for line in summary)
    assert any("Generated: 2 tokens" in line for line in summary)
