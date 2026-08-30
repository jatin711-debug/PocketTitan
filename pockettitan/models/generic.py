"""Generic Transformer layer structural analyzer and tensor classifier."""

from typing import List, Optional
from pydantic import BaseModel

from pockettitan.config import TensorAddress


class AttentionWeights(BaseModel):
    q_proj: Optional[TensorAddress] = None
    k_proj: Optional[TensorAddress] = None
    v_proj: Optional[TensorAddress] = None
    o_proj: Optional[TensorAddress] = None
    qkv_fused: Optional[TensorAddress] = None


class DenseMLPWeights(BaseModel):
    gate_proj: Optional[TensorAddress] = None
    up_proj: Optional[TensorAddress] = None
    down_proj: Optional[TensorAddress] = None
    fused_gate_up: Optional[TensorAddress] = None


class TransformerLayerStructure(BaseModel):
    layer_idx: int
    input_layernorm: Optional[TensorAddress] = None
    attention: AttentionWeights = AttentionWeights()
    post_attention_layernorm: Optional[TensorAddress] = None
    mlp: Optional[DenseMLPWeights] = None
    other_tensors: List[TensorAddress] = []


def parse_transformer_layer_structure(
    layer_idx: int,
    layer_tensors: List[TensorAddress],
) -> TransformerLayerStructure:
    """Classify all tensors belonging to a transformer layer into standard roles."""
    struct = TransformerLayerStructure(layer_idx=layer_idx)

    for t in layer_tensors:
        name_lower = t.name.lower()

        # Norms
        if (
            "input_layernorm" in name_lower
            or "attn_norm" in name_lower
            or "ln_1" in name_lower
            or "norm1" in name_lower
        ):
            struct.input_layernorm = t
        elif (
            "post_attention_layernorm" in name_lower
            or "ffn_norm" in name_lower
            or "ln_2" in name_lower
            or "norm2" in name_lower
        ):
            struct.post_attention_layernorm = t

        # Attention
        elif "q_proj" in name_lower or "query" in name_lower or "wq" in name_lower:
            struct.attention.q_proj = t
        elif "k_proj" in name_lower or "key" in name_lower or "wk" in name_lower:
            struct.attention.k_proj = t
        elif "v_proj" in name_lower or "value" in name_lower or "wv" in name_lower:
            struct.attention.v_proj = t
        elif (
            "o_proj" in name_lower
            or "out_proj" in name_lower
            or "wo" in name_lower
            or "dense" in name_lower
            and "attn" in name_lower
        ):
            struct.attention.o_proj = t
        elif "qkv" in name_lower or "w_qkv" in name_lower:
            struct.attention.qkv_fused = t

        # Dense MLP (non-expert)
        elif not any(exp in name_lower for exp in ["expert", "experts"]):
            if struct.mlp is None:
                struct.mlp = DenseMLPWeights()
            if "gate_proj" in name_lower or "w_gate" in name_lower or "w1" in name_lower:
                struct.mlp.gate_proj = t
            elif "up_proj" in name_lower or "w_up" in name_lower or "w3" in name_lower:
                struct.mlp.up_proj = t
            elif "down_proj" in name_lower or "w_down" in name_lower or "w2" in name_lower:
                struct.mlp.down_proj = t
            elif "gate_up_proj" in name_lower or "w13" in name_lower:
                struct.mlp.fused_gate_up = t
            else:
                struct.other_tensors.append(t)
        else:
            struct.other_tensors.append(t)

    return struct
