"""Unit tests for S2-MoE Self-Speculative Decoding Engine."""

from types import SimpleNamespace
import torch
from transformers.cache_utils import DynamicCache

from pockettitan.domainslice.speculative import SpeculativeMoEDecoder


class _MockLayer:
    def __init__(self, top_k=4):
        self.mlp = SimpleNamespace(gate=SimpleNamespace(top_k=top_k))


class _MockRunner:
    def __init__(self, vocab_size=16, num_experts_per_tok=4):
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            num_experts_per_tok=num_experts_per_tok,
        )
        self._resident_layers = [_MockLayer(num_experts_per_tok) for _ in range(2)]
        self.call_history = []

    def run(self, token_id, position_id=0, past_key_values=None, use_cache=True):
        self.call_history.append((token_id, position_id, self._resident_layers[0].mlp.gate.top_k))
        # Update dummy KV cache
        if past_key_values is not None:
            # Simulate adding keys/values to dynamic cache
            k = torch.zeros(1, 1, 1, 8)
            v = torch.zeros(1, 1, 1, 8)
            if len(past_key_values) == 0:
                past_key_values.update(k, v, 0)
            else:
                past_key_values.update(k, v, 0)

        # Deterministic dummy logits: next token is (token_id + 1) % vocab_size
        logits = torch.zeros(1, 1, self.config.vocab_size)
        next_tok = (token_id + 1) % self.config.vocab_size
        logits[0, 0, next_tok] = 10.0
        return logits, None


def test_speculative_decoder_all_accepted():
    runner = _MockRunner(vocab_size=16, num_experts_per_tok=4)
    decoder = SpeculativeMoEDecoder(runner, spec_k=2, draft_top_k=1)

    cache = DynamicCache()
    res = decoder.generate_step(current_token_id=2, position_id=0, past_key_values=cache)

    assert res.num_drafted == 2
    assert res.draft_tokens == [3, 4]
    # In deterministic mock, target matches draft, so both draft tokens + bonus token accepted!
    assert res.accepted_tokens == [3, 4, 5]
    assert res.num_accepted == 3
    assert decoder.total_drafted == 2
    assert decoder.total_accepted == 3


def test_speculative_decoder_partial_rejection():
    runner = _MockRunner(vocab_size=16, num_experts_per_tok=4)

    # Override run to diverge when top_k == 4 on token 3
    orig_run = runner.run

    def divergent_run(token_id, position_id=0, past_key_values=None, use_cache=True):
        if token_id == 2 and runner._resident_layers[0].mlp.gate.top_k == 4:
            # Target disagrees with draft! Predicts 9 instead of 3
            logits = torch.zeros(1, 1, 16)
            logits[0, 0, 9] = 10.0
            return logits, None
        return orig_run(token_id, position_id, past_key_values, use_cache)

    runner.run = divergent_run
    decoder = SpeculativeMoEDecoder(runner, spec_k=2, draft_top_k=1)

    cache = DynamicCache()
    res = decoder.generate_step(current_token_id=2, position_id=0, past_key_values=cache)

    assert res.num_drafted == 2
    assert res.draft_tokens == [3, 4]
    # First draft token rejected, target's alternative (9) accepted immediately
    assert res.accepted_tokens == [9]
    assert res.num_accepted == 1
