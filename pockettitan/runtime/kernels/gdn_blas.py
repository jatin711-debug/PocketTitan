"""BLAS-accelerated Gated Delta Net (GDN) linear state recurrence (Phase R9)."""

import torch


class GDNRecurrenceBLAS:
    """Standardized 48-head x 128x128 FP32 GDN state-space linear recurrence using tuned BLAS.
    
    Architecture (Plan.md §5 / R9):
    - Maintains state S_t = beta_t * S_{t-1} + (v_t outer k_t).
    - Computes output o_t = S_t @ q_t.
    - Utilizes hardware BLAS (torch.bmm / sgemv / sger) without hand-rolled kernel overhead.
    """

    def __init__(
        self,
        num_heads: int = 48,
        head_dim: int = 128,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Recurrent state tensor: [num_heads, head_dim, head_dim]
        self.state = torch.zeros(
            (num_heads, head_dim, head_dim),
            dtype=dtype,
            device=device,
        )

    def reset_state(self) -> None:
        self.state.zero_()

    def step(
        self,
        q: torch.Tensor,     # [num_heads, head_dim]
        k: torch.Tensor,     # [num_heads, head_dim]
        v: torch.Tensor,     # [num_heads, head_dim]
        beta: torch.Tensor,  # [num_heads, 1, 1] decay factor
    ) -> torch.Tensor:
        """Advance the linear recurrence by 1 token step."""
        q_fp32 = q.to(device=self.device, dtype=self.dtype)
        k_fp32 = k.to(device=self.device, dtype=self.dtype)
        v_fp32 = v.to(device=self.device, dtype=self.dtype)
        b_fp32 = beta.to(device=self.device, dtype=self.dtype)

        # 1. Decay previous state: S = beta * S_{t-1}
        self.state = self.state * b_fp32

        # 2. Outer product update: delta_S = v_t @ k_t.T  ([num_heads, head_dim, 1] @ [num_heads, 1, head_dim])
        outer = torch.bmm(v_fp32.unsqueeze(-1), k_fp32.unsqueeze(1))
        self.state = self.state + outer

        # 3. Compute output: o_t = S_t @ q_t  ([num_heads, head_dim, head_dim] @ [num_heads, head_dim, 1])
        out = torch.bmm(self.state, q_fp32.unsqueeze(-1)).squeeze(-1)

        return out.to(dtype=q.dtype)
