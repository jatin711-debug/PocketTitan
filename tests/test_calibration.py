"""Unit tests for calibration spool, Hessian accumulation, and MoE router calibration."""

import pytest
import torch

from pockettitan.calibration.dataset import load_calibration_dataset
from pockettitan.calibration.hessian import HessianAccumulator
from pockettitan.calibration.moe_stats import MoERouterCalibrator
from pockettitan.calibration.spool import ActivationSpool


def test_activation_spool(tmp_path):
    spool = ActivationSpool(max_in_memory_mb=0.01, spool_dir=tmp_path / "spool")
    
    # Add batches that exceed 0.01MB to trigger disk spooling
    for _ in range(5):
        batch = torch.randn(10, 128, dtype=torch.float32)
        spool.add_activation_batch(batch)
        
    assert len(spool._disk_spool_files) > 0
    
    collected = list(spool.get_batches())
    assert len(collected) == 5
    assert collected[0].shape == (10, 128)
    
    spool.clear()
    assert len(spool._disk_spool_files) == 0


def test_moe_router_calibrator():
    num_experts = 4
    top_k = 2
    hidden_dim = 64
    
    calibrator = MoERouterCalibrator(num_experts=num_experts, top_k=top_k, hidden_dim=hidden_dim)
    
    x = torch.randn(32, hidden_dim)
    router_w = torch.randn(num_experts, hidden_dim)
    
    calibrator.dispatch_batch(x, router_w)
    
    total_dispatched = sum(calibrator.expert_token_counts.values())
    assert total_dispatched == 32 * top_k
    
    h0 = calibrator.get_expert_hessian(0)
    assert h0.shape == (hidden_dim, hidden_dim)
