"""Layer, attention, and MoE expert sensitivity analysis for heterogeneous precision allocation."""

from typing import Optional
from pydantic import BaseModel

from pockettitan.config import ModelMetadata, TensorAddress


class TensorSensitivityScore(BaseModel):
    name: str
    num_params: int
    sensitivity: float
    recommended_min_bits: int
    recommended_max_bits: int


def compute_tensor_sensitivity(
    tensor_addr: TensorAddress, model_meta: Optional[ModelMetadata] = None
) -> TensorSensitivityScore:
    """Compute architectural sensitivity score and allowable bit-width range for a tensor."""
    name_lower = tensor_addr.name.lower()

    # 1. Critical components: RMSNorm, biases, router gates, embeddings, lm_head
    if any(k in name_lower for k in ["norm", "ln_", "bias"]):
        # Always retain in full precision / FP16
        return TensorSensitivityScore(
            name=tensor_addr.name,
            num_params=tensor_addr.num_params,
            sensitivity=100.0,
            recommended_min_bits=16,
            recommended_max_bits=16,
        )

    if any(k in name_lower for k in ["router", "gate.weight", "gate_logits"]):
        # Router logits determine expert selection -> high sensitivity
        return TensorSensitivityScore(
            name=tensor_addr.name,
            num_params=tensor_addr.num_params,
            sensitivity=50.0,
            recommended_min_bits=8,
            recommended_max_bits=16,
        )

    if any(k in name_lower for k in ["embed_tokens", "wte", "lm_head"]):
        return TensorSensitivityScore(
            name=tensor_addr.name,
            num_params=tensor_addr.num_params,
            sensitivity=30.0,
            recommended_min_bits=8,
            recommended_max_bits=16,
        )

    # 2. Self Attention projections (Q, K, V, O)
    if any(k in name_lower for k in ["q_proj", "k_proj", "v_proj", "o_proj", "attn"]):
        # First 2 layers and last 2 layers are more sensitive
        layer_num = 0
        import re

        match = re.search(r"layers?\.(\d+)", name_lower)
        if match:
            layer_num = int(match.group(1))

        is_boundary_layer = (layer_num <= 1) or (
            model_meta and layer_num >= model_meta.num_hidden_layers - 2
        )
        base_sens = 8.0 if is_boundary_layer else 4.0
        return TensorSensitivityScore(
            name=tensor_addr.name,
            num_params=tensor_addr.num_params,
            sensitivity=base_sens,
            recommended_min_bits=2,
            recommended_max_bits=8,
        )

    # 3. MoE routed experts & Dense MLP projections (high capacity redundancy)
    if "expert" in name_lower:
        return TensorSensitivityScore(
            name=tensor_addr.name,
            num_params=tensor_addr.num_params,
            sensitivity=1.0,
            recommended_min_bits=2,
            recommended_max_bits=4,
        )

    # Default linear projection
    return TensorSensitivityScore(
        name=tensor_addr.name,
        num_params=tensor_addr.num_params,
        sensitivity=2.0,
        recommended_min_bits=2,
        recommended_max_bits=8,
    )
