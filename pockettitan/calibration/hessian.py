"""Online Hessian accumulation and activation outlier detector."""

import torch


class HessianAccumulator:
    """Online second-order Hessian matrix accumulator: H = (1/N) * sum(X^T * X)."""

    def __init__(self, in_features: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.in_features = in_features
        self.device = device
        self.dtype = dtype
        self.num_samples = 0
        self.hessian = torch.zeros((in_features, in_features), dtype=dtype, device=device)

    def add_batch(self, x: torch.Tensor) -> None:
        """Accumulate activation matrix X of shape [batch, seq_len, in_features] or [N, in_features]."""
        x_2d = x.view(-1, self.in_features).to(self.device, dtype=self.dtype)
        batch_count = x_2d.shape[0]

        # In-place symmetric accumulation: H += X^T @ X
        self.hessian.addmm_(x_2d.t(), x_2d)
        self.num_samples += batch_count
        del x_2d

    def get_normalized_hessian(self, dampening_lambda: float = 0.01) -> torch.Tensor:
        """Return (1/N)*H with ridge dampening on the diagonal for numerical stability."""
        if self.num_samples == 0:
            return torch.eye(self.in_features, dtype=self.dtype, device=self.device)

        h_norm = self.hessian / max(1, self.num_samples)

        # Add diagonal dampening lambda * mean(diag(H)) * I
        diag_mean = torch.mean(torch.diag(h_norm)).item()
        damp = dampening_lambda * max(1e-5, diag_mean)
        h_norm.diagonal().add_(damp)
        return h_norm

    def get_outlier_indices(self, outlier_ratio: float = 0.01) -> torch.Tensor:
        """Find the top 1% highest energy activation channels from diagonal of H."""
        diag = torch.diag(self.hessian)
        num_outliers = max(1, int(self.in_features * outlier_ratio))
        _, indices = torch.topk(diag, k=num_outliers, largest=True)
        return indices
