"""Model architecture adapters and layer/expert decomposition."""

from pockettitan.models.generic import TransformerLayerStructure, parse_transformer_layer_structure
from pockettitan.models.moe import MoELayerStructure, parse_moe_layer_structure

__all__ = [
    "TransformerLayerStructure",
    "parse_transformer_layer_structure",
    "MoELayerStructure",
    "parse_moe_layer_structure",
]
