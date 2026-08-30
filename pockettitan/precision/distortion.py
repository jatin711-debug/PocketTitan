"""Distortion, SNR, Cosine Similarity, and activation error evaluation."""

import math
from typing import Optional
from pydantic import BaseModel
import torch


class DistortionReport(BaseModel):
    weight_distortion: float  # Relative squared Frobenius norm error
    snr_db: float  # Signal to noise ratio in decibels
    cosine_similarity: float  # Cosine similarity between original and reconstructed weights
    activation_distortion: Optional[float] = None  # Activation-weighted output error


def compute_weight_distortion(w_orig: torch.Tensor, w_deq: torch.Tensor) -> float:
    """Compute relative Frobenius weight distortion: ||W - Q(W)||_F^2 / ||W||_F^2."""
    orig_f = w_orig.float()
    deq_f = w_deq.float()

    diff_norm_sq = torch.sum((orig_f - deq_f) ** 2).item()
    orig_norm_sq = torch.sum(orig_f**2).item()
    return diff_norm_sq / max(1e-9, orig_norm_sq)


def compute_snr_db(w_orig: torch.Tensor, w_deq: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in dB."""
    dist = compute_weight_distortion(w_orig, w_deq)
    if dist <= 1e-9:
        return 100.0  # Perfect reconstruction
    return -10.0 * math.log10(dist)


def compute_cosine_similarity(w_orig: torch.Tensor, w_deq: torch.Tensor) -> float:
    """Compute Cosine Similarity between original and reconstructed weight tensors."""
    orig_flat = w_orig.float().view(-1)
    deq_flat = w_deq.float().view(-1)

    dot = torch.dot(orig_flat, deq_flat).item()
    norm_orig = torch.norm(orig_flat).item()
    norm_deq = torch.norm(deq_flat).item()

    denom = max(1e-9, norm_orig * norm_deq)
    return max(-1.0, min(1.0, dot / denom))


def compute_activation_distortion(
    w_orig: torch.Tensor,
    w_deq: torch.Tensor,
    x: torch.Tensor,
) -> float:
    """Compute activation-weighted output error: ||XW - XQ(W)||_F^2 / ||XW||_F^2."""
    # Assuming PyTorch Linear layout: Y = X @ W.T where W is [out_features, in_features]
    orig_f = w_orig.float()
    deq_f = w_deq.float()
    x_f = x.float()

    y_orig = torch.matmul(x_f, orig_f.t())
    y_deq = torch.matmul(x_f, deq_f.t())

    diff_norm_sq = torch.sum((y_orig - y_deq) ** 2).item()
    orig_norm_sq = torch.sum(y_orig**2).item()
    return diff_norm_sq / max(1e-9, orig_norm_sq)


def evaluate_quantization_quality(
    w_orig: torch.Tensor,
    w_deq: torch.Tensor,
    x: Optional[torch.Tensor] = None,
) -> DistortionReport:
    """Generate comprehensive quality and distortion metrics."""
    w_dist = compute_weight_distortion(w_orig, w_deq)
    snr = compute_snr_db(w_orig, w_deq)
    cos_sim = compute_cosine_similarity(w_orig, w_deq)
    act_dist = compute_activation_distortion(w_orig, w_deq, x) if x is not None else None

    return DistortionReport(
        weight_distortion=round(w_dist, 6),
        snr_db=round(snr, 2),
        cosine_similarity=round(cos_sim, 6),
        activation_distortion=round(act_dist, 6) if act_dist is not None else None,
    )
