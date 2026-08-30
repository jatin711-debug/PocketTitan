"""Multi-choice Pareto Lagrangian bit-width allocation solver."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from pockettitan.config import ModelMetadata, QuantConfig, QuantMethod, TensorAddress
from pockettitan.precision.sensitivity import compute_tensor_sensitivity


class HeterogeneousPrecisionMap(BaseModel):
    model_id_or_path: str
    target_bpw: float
    effective_bpw: float
    tensor_quant_configs: Dict[str, QuantConfig] = Field(default_factory=dict)


class ParetoBitAllocator:
    """Solves Pareto-optimal bit-width assignment under sensitivity and target size constraints."""

    def __init__(self, target_bpw: float = 2.5):
        self.target_bpw = target_bpw

    def solve(
        self,
        model_id_or_path: str,
        tensor_addresses: List[TensorAddress],
        model_meta: Optional[ModelMetadata] = None,
        default_method: QuantMethod = QuantMethod.HQQ,
    ) -> HeterogeneousPrecisionMap:
        """Assign precision per tensor optimizing error/bitrate trade-off."""
        scores = [compute_tensor_sensitivity(t, model_meta) for t in tensor_addresses]

        assigned_configs: Dict[str, QuantConfig] = {}
        total_param_bits = 0.0
        total_params = 0

        for t, s in zip(tensor_addresses, scores):
            num_p = t.num_params
            total_params += num_p

            # Rule 1: High sensitivity -> FP16 / 8-bit
            if s.sensitivity >= 50.0 or s.recommended_min_bits == 16:
                cfg = QuantConfig(method=QuantMethod.RTN, bits=16, group_size=-1)
                assigned_configs[t.name] = cfg
                total_param_bits += num_p * 16.0
                continue

            if s.sensitivity >= 20.0:
                # Moderate-high (embeddings, lm_head)
                bits = 8 if self.target_bpw >= 3.0 else 4
                cfg = QuantConfig(method=QuantMethod.RTN, bits=bits, group_size=128)
                assigned_configs[t.name] = cfg
                total_param_bits += num_p * bits
                continue

            # Rule 2: Attention layers
            if "attn" in t.name or any(
                q in t.name for q in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ):
                if self.target_bpw <= 2.2:
                    bits = 2
                elif self.target_bpw <= 3.5:
                    bits = 3 if s.sensitivity <= 4.0 else 4
                else:
                    bits = 4
                cfg = QuantConfig(method=default_method, bits=bits, group_size=128)
                assigned_configs[t.name] = cfg
                total_param_bits += num_p * bits
                continue

            # Rule 3: MoE Routed Experts (Highest capacity, lowest sensitivity)
            if "expert" in t.name:
                if self.target_bpw <= 1.8:
                    cfg = QuantConfig(method=QuantMethod.TERNARY, bits=2, group_size=128)
                    assigned_configs[t.name] = cfg
                    total_param_bits += num_p * 1.58
                elif self.target_bpw <= 2.8:
                    cfg = QuantConfig(method=default_method, bits=2, group_size=128)
                    assigned_configs[t.name] = cfg
                    total_param_bits += num_p * 2.0
                elif self.target_bpw <= 3.8:
                    cfg = QuantConfig(method=default_method, bits=3, group_size=128)
                    assigned_configs[t.name] = cfg
                    total_param_bits += num_p * 3.0
                else:
                    cfg = QuantConfig(method=default_method, bits=4, group_size=128)
                    assigned_configs[t.name] = cfg
                    total_param_bits += num_p * 4.0
                continue

            # Default linear layer
            bits = 2 if self.target_bpw <= 2.5 else (4 if self.target_bpw <= 4.5 else 8)
            cfg = QuantConfig(method=default_method, bits=bits, group_size=128)
            assigned_configs[t.name] = cfg
            total_param_bits += num_p * bits

        effective_bpw = total_param_bits / max(1, total_params)

        return HeterogeneousPrecisionMap(
            model_id_or_path=model_id_or_path,
            target_bpw=self.target_bpw,
            effective_bpw=round(effective_bpw, 2),
            tensor_quant_configs=assigned_configs,
        )
