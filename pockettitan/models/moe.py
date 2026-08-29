"""Mixture-of-Experts (MoE) layer and expert structural decomposition."""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from pockettitan.config import TensorAddress
from pockettitan.models.generic import AttentionWeights, TransformerLayerStructure


class ExpertWeights(BaseModel):
    expert_idx: int
    gate_proj: Optional[TensorAddress] = None
    up_proj: Optional[TensorAddress] = None
    down_proj: Optional[TensorAddress] = None
    fused_gate_up: Optional[TensorAddress] = None
    other_tensors: List[TensorAddress] = []

    def get_all_tensors(self) -> List[TensorAddress]:
        tensors = []
        if self.gate_proj:
            tensors.append(self.gate_proj)
        if self.up_proj:
            tensors.append(self.up_proj)
        if self.down_proj:
            tensors.append(self.down_proj)
        if self.fused_gate_up:
            tensors.append(self.fused_gate_up)
        tensors.extend(self.other_tensors)
        return tensors


class MoELayerStructure(BaseModel):
    layer_idx: int
    input_layernorm: Optional[TensorAddress] = None
    attention: AttentionWeights = AttentionWeights()
    post_attention_layernorm: Optional[TensorAddress] = None
    router_gate: Optional[TensorAddress] = None
    shared_experts: Optional[ExpertWeights] = None
    routed_experts: Dict[int, ExpertWeights] = Field(default_factory=dict)
    other_tensors: List[TensorAddress] = []

    @property
    def num_routed_experts(self) -> int:
        return len(self.routed_experts)


def parse_moe_layer_structure(
    layer_idx: int,
    layer_tensors: List[TensorAddress],
) -> MoELayerStructure:
    """Classify all tensors in an MoE layer into router, shared experts, and routed experts."""
    struct = MoELayerStructure(layer_idx=layer_idx)
    
    # Regex pattern to identify expert indices: e.g. .experts.12. or .expert_12. or .experts_12.
    expert_pattern = re.compile(r"experts?[._](\d+)")
    
    for t in layer_tensors:
        name_lower = t.name.lower()
        
        # Norms
        if "input_layernorm" in name_lower or "attn_norm" in name_lower or "ln_1" in name_lower:
            struct.input_layernorm = t
        elif "post_attention_layernorm" in name_lower or "ffn_norm" in name_lower or "ln_2" in name_lower:
            struct.post_attention_layernorm = t
            
        # Attention
        elif "q_proj" in name_lower or "query" in name_lower or "wq" in name_lower:
            struct.attention.q_proj = t
        elif "k_proj" in name_lower or "key" in name_lower or "wk" in name_lower:
            struct.attention.k_proj = t
        elif "v_proj" in name_lower or "value" in name_lower or "wv" in name_lower:
            struct.attention.v_proj = t
        elif "o_proj" in name_lower or "out_proj" in name_lower or "wo" in name_lower:
            struct.attention.o_proj = t
        elif "qkv" in name_lower:
            struct.attention.qkv_fused = t
            
        # Router Gate Logits
        elif any(g in name_lower for g in ["gate.weight", "router.weight", "gate_logits", "router.classifier"]):
            struct.router_gate = t
            
        # Shared Experts (DeepSeek, GLM, Qwen-MoE)
        elif "shared_expert" in name_lower or "shared_experts" in name_lower:
            if struct.shared_experts is None:
                struct.shared_experts = ExpertWeights(expert_idx=-1)
            if "gate_proj" in name_lower or "w1" in name_lower:
                struct.shared_experts.gate_proj = t
            elif "up_proj" in name_lower or "w3" in name_lower:
                struct.shared_experts.up_proj = t
            elif "down_proj" in name_lower or "w2" in name_lower:
                struct.shared_experts.down_proj = t
            else:
                struct.shared_experts.other_tensors.append(t)
                
        # Routed Experts
        elif "expert" in name_lower:
            match = expert_pattern.search(name_lower)
            if match:
                e_idx = int(match.group(1))
                if e_idx not in struct.routed_experts:
                    struct.routed_experts[e_idx] = ExpertWeights(expert_idx=e_idx)
                    
                exp = struct.routed_experts[e_idx]
                if "gate_proj" in name_lower or "w1" in name_lower or "w_gate" in name_lower:
                    exp.gate_proj = t
                elif "up_proj" in name_lower or "w3" in name_lower or "w_up" in name_lower:
                    exp.up_proj = t
                elif "down_proj" in name_lower or "w2" in name_lower or "w_down" in name_lower:
                    exp.down_proj = t
                elif "gate_up_proj" in name_lower or "w13" in name_lower:
                    exp.fused_gate_up = t
                else:
                    exp.other_tensors.append(t)
            else:
                struct.other_tensors.append(t)
        else:
            struct.other_tensors.append(t)
            
    return struct
