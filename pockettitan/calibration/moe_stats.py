"""MoE router calibration and per-expert token dispatch statistics."""

from typing import Dict
import torch

from pockettitan.calibration.hessian import HessianAccumulator


class MoERouterCalibrator:
    """Dispatches calibration tokens through MoE router gate to per-expert Hessian accumulators."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
        device: str = "cpu",
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.device = device

        # Per-expert Hessian accumulators
        self.expert_hessians: Dict[int, HessianAccumulator] = {
            e: HessianAccumulator(hidden_dim, device=device) for e in range(num_experts)
        }
        self.expert_token_counts: Dict[int, int] = {e: 0 for e in range(num_experts)}

    def dispatch_batch(self, x: torch.Tensor, router_weights: torch.Tensor) -> None:
        """Route token batch through router logits and update expert Hessians."""
        # x: [num_tokens, hidden_dim]
        x_2d = x.view(-1, self.hidden_dim).float()
        w_gate = router_weights.float()

        # Compute router logits: [num_tokens, num_experts]
        logits = torch.matmul(x_2d, w_gate.t())

        # Top-k selection
        topk_weights, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)

        for e in range(self.num_experts):
            # Find tokens assigned to expert e
            mask = (topk_indices == e).any(dim=-1)
            if mask.any():
                expert_tokens = x_2d[mask]
                self.expert_hessians[e].add_batch(expert_tokens)
                self.expert_token_counts[e] += expert_tokens.shape[0]

    def get_expert_hessian(self, expert_idx: int) -> torch.Tensor:
        """Get regularized Hessian for a specific expert."""
        return self.expert_hessians[expert_idx].get_normalized_hessian()
