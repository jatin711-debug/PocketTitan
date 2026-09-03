"""S2-MoE Self-Speculative Decoding Engine for DomainSlice.

Implements Top-1 self-speculative drafting with target verification and KV-cache
management to accelerate MoE inference without auxiliary draft models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List
import torch

from pockettitan.runtime.hf.olmoe_model import PagedOlmoeOneTokenRunner


@dataclass
class SpeculativeStepResult:
    accepted_tokens: List[int]
    draft_tokens: List[int]
    num_drafted: int
    num_accepted: int
    step_duration: float


class SpeculativeMoEDecoder:
    """Self-speculative decoding orchestrator for PagedOlmoeOneTokenRunner.

    Uses fast Top-1 expert drafting followed by full Top-k target verification.
    """

    def __init__(
        self,
        runner: PagedOlmoeOneTokenRunner,
        spec_k: int = 3,
        draft_top_k: int = 1,
    ):
        self.runner = runner
        self.spec_k = max(1, int(spec_k))
        self.draft_top_k = max(1, int(draft_top_k))
        self.orig_top_k = int(runner.config.num_experts_per_tok)

        self.total_drafted = 0
        self.total_accepted = 0

    def _set_model_top_k(self, k: int) -> None:
        for layer in self.runner._resident_layers:
            if hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "top_k"):
                layer.mlp.gate.top_k = k

    def generate_step(
        self,
        current_token_id: int,
        position_id: int,
        past_key_values,
    ) -> SpeculativeStepResult:
        """Run one speculative round: draft K tokens, verify with target model."""
        step_start = time.perf_counter()
        pos_start = position_id

        # 1. Draft Phase: Top-1 routing
        self._set_model_top_k(self.draft_top_k)
        draft_tokens: List[int] = []
        curr_id = current_token_id

        try:
            for step in range(self.spec_k):
                pos = pos_start + step
                logits, _ = self.runner.run(
                    curr_id,
                    position_id=pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_draft_id = int(torch.argmax(logits[0, -1]).item())
                draft_tokens.append(next_draft_id)
                curr_id = next_draft_id
        finally:
            # Crop KV cache back to original starting position
            if hasattr(past_key_values, "crop") and len(draft_tokens) > 0:
                try:
                    past_key_values.crop(-len(draft_tokens))
                except Exception:
                    try:
                        past_key_values.crop(pos_start)
                    except Exception:
                        pass
            # Restore full target routing
            self._set_model_top_k(self.orig_top_k)

        # 2. Verification Phase: Full Top-k routing
        accepted_tokens: List[int] = []
        curr_verify_id = current_token_id

        for i, draft_id in enumerate(draft_tokens):
            pos = pos_start + i
            logits, _ = self.runner.run(
                curr_verify_id,
                position_id=pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            target_next_id = int(torch.argmax(logits[0, -1]).item())

            if target_next_id == draft_id:
                # Draft token verified and accepted
                accepted_tokens.append(draft_id)
                curr_verify_id = draft_id
                # If this was the last draft token, also sample the continuation token
                if i == len(draft_tokens) - 1:
                    pos_bonus = pos_start + i + 1
                    bonus_logits, _ = self.runner.run(
                        curr_verify_id,
                        position_id=pos_bonus,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    bonus_token = int(torch.argmax(bonus_logits[0, -1]).item())
                    accepted_tokens.append(bonus_token)
            else:
                # Draft token rejected, accept target's correction and stop
                accepted_tokens.append(target_next_id)
                break

        num_drafted = len(draft_tokens)
        num_accepted = len(accepted_tokens)
        self.total_drafted += num_drafted
        self.total_accepted += num_accepted

        return SpeculativeStepResult(
            accepted_tokens=accepted_tokens,
            draft_tokens=draft_tokens,
            num_drafted=num_drafted,
            num_accepted=num_accepted,
            step_duration=time.perf_counter() - step_start,
        )
