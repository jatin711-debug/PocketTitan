"""Model Architecture Adapters for extracting structural topologies across diverse LLM families."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseModelAdapter(ABC):
    """Abstract model adapter defining the interface for topology extraction."""

    def __init__(self, raw_config: Dict[str, Any]):
        self.raw_config = raw_config
        # Allow nested text_config if present
        self.config = raw_config.get("text_config", raw_config)

    @abstractmethod
    def extract_architecture_name(self) -> str:
        """Extract canonical architecture name."""
        pass

    @abstractmethod
    def extract_dimensions(self) -> Dict[str, Any]:
        """Extract hidden_size, num_layers, heads, intermediate_size, vocab_size."""
        pass

    @abstractmethod
    def extract_moe_topology(self) -> Dict[str, Any]:
        """Extract MoE routing parameters, expert dimensions, and dense layers."""
        pass

    @abstractmethod
    def is_moe_architecture(self) -> bool:
        """Return True if model uses Mixture-of-Experts routing."""
        pass

    def extract_source_dtype(self) -> Tuple[str, bool]:
        """Extract source dtype string and whether it is FP8 format.

        Returns:
            (source_dtype_str, is_fp8)
        """
        dtype = self.config.get("torch_dtype", self.raw_config.get("torch_dtype", "bfloat16"))
        dtype_str = str(dtype).replace("torch.", "")
        is_fp8 = dtype_str.upper() in [
            "FP8",
            "FLOAT8",
            "F8_E4M3",
            "F8_E5M2",
            "FLOAT8_E4M3FN",
            "FLOAT8_E5M2",
        ]
        return dtype_str, is_fp8


class GenericAdapter(BaseModelAdapter):
    """Adapter for standard dense models (Llama, Mistral, Gemma, Falcon, Qwen dense)."""

    def extract_architecture_name(self) -> str:
        archs = self.raw_config.get("architectures", ["DenseCausalLM"])
        return archs[0] if isinstance(archs, list) and archs else "DenseCausalLM"

    def extract_dimensions(self) -> Dict[str, Any]:
        hidden_size = self.config.get("hidden_size", self.config.get("d_model", 4096))
        num_layers = self.config.get(
            "num_hidden_layers", self.config.get("n_layer", self.config.get("num_layers", 32))
        )
        num_heads = self.config.get("num_attention_heads", self.config.get("n_head", 32))
        num_kv_heads = self.config.get("num_key_value_heads", num_heads)
        intermediate_size = self.config.get("intermediate_size", self.config.get("n_inner", None))
        vocab_size = self.config.get("vocab_size", 32000)
        return {
            "hidden_size": int(hidden_size),
            "num_hidden_layers": int(num_layers),
            "num_attention_heads": int(num_heads),
            "num_key_value_heads": int(num_kv_heads),
            "intermediate_size": int(intermediate_size) if intermediate_size else None,
            "vocab_size": int(vocab_size),
        }

    def is_moe_architecture(self) -> bool:
        return False

    def extract_moe_topology(self) -> Dict[str, Any]:
        return {
            "is_moe": False,
            "num_experts": None,
            "num_experts_per_tok": None,
            "expert_intermediate_size": None,
            "shared_expert_intermediate_size": None,
            "first_k_dense_replace": None,
        }


class GLM5NextAdapter(BaseModelAdapter):
    """Adapter for GLM-5.3-Flash, ChatGLM, and GLM-4/5 MoE architectures."""

    def extract_architecture_name(self) -> str:
        archs = self.raw_config.get("architectures", ["GLM5ForCausalLM"])
        return archs[0] if isinstance(archs, list) and archs else "GLM5ForCausalLM"

    def extract_dimensions(self) -> Dict[str, Any]:
        # Resolve from text_config if nested
        cfg = self.config
        hidden_size = cfg.get("hidden_size", cfg.get("d_model", 4096))
        num_layers = cfg.get("num_hidden_layers", cfg.get("num_layers", 48))
        num_heads = cfg.get("num_attention_heads", cfg.get("n_head", 32))
        num_kv_heads = cfg.get("num_key_value_heads", cfg.get("multi_query_group_num", num_heads))
        intermediate_size = cfg.get("intermediate_size", None)
        vocab_size = cfg.get("vocab_size", 151552)
        return {
            "hidden_size": int(hidden_size),
            "num_hidden_layers": int(num_layers),
            "num_attention_heads": int(num_heads),
            "num_key_value_heads": int(num_kv_heads),
            "intermediate_size": int(intermediate_size) if intermediate_size else None,
            "vocab_size": int(vocab_size),
        }

    def is_moe_architecture(self) -> bool:
        cfg = self.config
        return (
            cfg.get("n_routed_experts") is not None
            or cfg.get("num_experts") is not None
            or cfg.get("moe_intermediate_size") is not None
            or cfg.get("num_local_experts") is not None
        )

    def extract_moe_topology(self) -> Dict[str, Any]:
        cfg = self.config
        is_moe = self.is_moe_architecture()
        num_experts = cfg.get(
            "n_routed_experts", cfg.get("num_experts", cfg.get("num_local_experts", None))
        )
        num_experts_per_tok = cfg.get(
            "num_experts_per_tok", cfg.get("top_k", cfg.get("moe_top_k", None))
        )
        expert_intermediate_size = cfg.get(
            "moe_intermediate_size", cfg.get("expert_intermediate_size", None)
        )
        shared_expert_intermediate_size = cfg.get("shared_expert_intermediate_size", None)
        first_k_dense_replace = cfg.get("first_k_dense_replace", 0)

        # In GLM architectures, shared experts can be defined by num_shared_experts
        num_shared = cfg.get("n_shared_experts", cfg.get("num_shared_experts", 0))
        if (
            num_shared > 0
            and shared_expert_intermediate_size is None
            and expert_intermediate_size is not None
        ):
            shared_expert_intermediate_size = expert_intermediate_size * num_shared

        return {
            "is_moe": is_moe,
            "num_experts": int(num_experts) if num_experts is not None else None,
            "num_experts_per_tok": int(num_experts_per_tok)
            if num_experts_per_tok is not None
            else None,
            "expert_intermediate_size": int(expert_intermediate_size)
            if expert_intermediate_size is not None
            else None,
            "shared_expert_intermediate_size": int(shared_expert_intermediate_size)
            if shared_expert_intermediate_size is not None
            else None,
            "first_k_dense_replace": int(first_k_dense_replace)
            if first_k_dense_replace is not None
            else None,
        }


class DeepSeekAdapter(BaseModelAdapter):
    """Adapter for DeepSeek-V2, DeepSeek-V3, and DeepSeek-R1 architectures."""

    def extract_architecture_name(self) -> str:
        archs = self.raw_config.get("architectures", ["DeepseekV3ForCausalLM"])
        return archs[0] if isinstance(archs, list) and archs else "DeepseekV3ForCausalLM"

    def extract_dimensions(self) -> Dict[str, Any]:
        cfg = self.config
        hidden_size = cfg.get("hidden_size", 7168)
        num_layers = cfg.get("num_hidden_layers", 61)
        num_heads = cfg.get("num_attention_heads", 128)
        num_kv_heads = cfg.get("num_key_value_heads", num_heads)
        intermediate_size = cfg.get("intermediate_size", 18432)
        vocab_size = cfg.get("vocab_size", 129280)
        return {
            "hidden_size": int(hidden_size),
            "num_hidden_layers": int(num_layers),
            "num_attention_heads": int(num_heads),
            "num_key_value_heads": int(num_kv_heads),
            "intermediate_size": int(intermediate_size),
            "vocab_size": int(vocab_size),
        }

    def is_moe_architecture(self) -> bool:
        return (
            self.config.get("n_routed_experts") is not None
            or self.config.get("n_shared_experts") is not None
        )

    def extract_moe_topology(self) -> Dict[str, Any]:
        cfg = self.config
        num_experts = cfg.get("n_routed_experts", 256)
        num_experts_per_tok = cfg.get("num_experts_per_tok", 8)
        expert_intermediate_size = cfg.get("moe_intermediate_size", 2048)

        num_shared = cfg.get("n_shared_experts", 1)
        shared_size = cfg.get(
            "shared_expert_intermediate_size",
            expert_intermediate_size * num_shared if expert_intermediate_size else None,
        )
        first_k_dense_replace = cfg.get("first_k_dense_replace", 3)

        return {
            "is_moe": True,
            "num_experts": int(num_experts),
            "num_experts_per_tok": int(num_experts_per_tok),
            "expert_intermediate_size": int(expert_intermediate_size),
            "shared_expert_intermediate_size": int(shared_size) if shared_size else None,
            "first_k_dense_replace": int(first_k_dense_replace),
        }


class QwenMoEAdapter(BaseModelAdapter):
    """Adapter for Qwen-MoE architectures (Qwen1.5-MoE, Qwen2-MoE, Qwen3.6 MoE)."""

    def extract_architecture_name(self) -> str:
        archs = self.raw_config.get("architectures", ["Qwen2MoeForCausalLM"])
        return archs[0] if isinstance(archs, list) and archs else "Qwen2MoeForCausalLM"

    def extract_dimensions(self) -> Dict[str, Any]:
        cfg = self.config
        hidden_size = cfg.get("hidden_size", 2048)
        num_layers = cfg.get("num_hidden_layers", 24)
        num_heads = cfg.get("num_attention_heads", 16)
        num_kv_heads = cfg.get("num_key_value_heads", num_heads)
        intermediate_size = cfg.get("intermediate_size", None)
        vocab_size = cfg.get("vocab_size", 151936)
        return {
            "hidden_size": int(hidden_size),
            "num_hidden_layers": int(num_layers),
            "num_attention_heads": int(num_heads),
            "num_key_value_heads": int(num_kv_heads),
            "intermediate_size": int(intermediate_size) if intermediate_size else None,
            "vocab_size": int(vocab_size),
        }

    def is_moe_architecture(self) -> bool:
        return self.config.get("num_experts", self.config.get("moe_num_experts")) is not None

    def extract_moe_topology(self) -> Dict[str, Any]:
        cfg = self.config
        num_experts = cfg.get("num_experts", cfg.get("moe_num_experts", 64))
        num_experts_per_tok = cfg.get("num_experts_per_tok", cfg.get("moe_num_experts_per_tok", 4))
        expert_intermediate_size = cfg.get("moe_intermediate_size", 1408)
        shared_size = cfg.get("shared_expert_intermediate_size", None)

        return {
            "is_moe": True,
            "num_experts": int(num_experts),
            "num_experts_per_tok": int(num_experts_per_tok),
            "expert_intermediate_size": int(expert_intermediate_size),
            "shared_expert_intermediate_size": int(shared_size) if shared_size else None,
            "first_k_dense_replace": cfg.get("first_k_dense_replace", None),
        }


class MixtralAdapter(BaseModelAdapter):
    """Adapter for Mixtral 8x7B / 8x22B architectures."""

    def extract_architecture_name(self) -> str:
        archs = self.raw_config.get("architectures", ["MixtralForCausalLM"])
        return archs[0] if isinstance(archs, list) and archs else "MixtralForCausalLM"

    def extract_dimensions(self) -> Dict[str, Any]:
        cfg = self.config
        hidden_size = cfg.get("hidden_size", 4096)
        num_layers = cfg.get("num_hidden_layers", 32)
        num_heads = cfg.get("num_attention_heads", 32)
        num_kv_heads = cfg.get("num_key_value_heads", 8)
        intermediate_size = cfg.get("intermediate_size", 14336)
        vocab_size = cfg.get("vocab_size", 32000)
        return {
            "hidden_size": int(hidden_size),
            "num_hidden_layers": int(num_layers),
            "num_attention_heads": int(num_heads),
            "num_key_value_heads": int(num_kv_heads),
            "intermediate_size": int(intermediate_size),
            "vocab_size": int(vocab_size),
        }

    def is_moe_architecture(self) -> bool:
        return (
            self.config.get("num_local_experts") is not None
            or self.config.get("num_experts") is not None
        )

    def extract_moe_topology(self) -> Dict[str, Any]:
        cfg = self.config
        num_experts = cfg.get("num_local_experts", cfg.get("num_experts", 8))
        num_experts_per_tok = cfg.get("num_experts_per_tok", 2)
        intermediate_size = cfg.get("intermediate_size", 14336)
        return {
            "is_moe": True,
            "num_experts": int(num_experts),
            "num_experts_per_tok": int(num_experts_per_tok),
            "expert_intermediate_size": int(intermediate_size),
            "shared_expert_intermediate_size": None,
            "first_k_dense_replace": None,
        }


def get_model_adapter(config: Dict[str, Any]) -> BaseModelAdapter:
    """Factory function returning the specialized ModelAdapter for a given config payload."""
    arch_str = str(config.get("architectures", [""])).lower()
    model_type = str(config.get("model_type", "")).lower()

    # 1. GLM check
    if "glm" in arch_str or "glm" in model_type or "text_config" in config:
        return GLM5NextAdapter(config)

    # 2. DeepSeek check
    if (
        "deepseek" in arch_str
        or "deepseek" in model_type
        or "n_routed_experts" in config
        or "n_shared_experts" in config
    ):
        return DeepSeekAdapter(config)

    # 3. Qwen MoE check
    if (
        "qwen2moe" in arch_str
        or "qwen2_moe" in model_type
        or ("qwen" in model_type and ("num_experts" in config or "moe_num_experts" in config))
    ):
        return QwenMoEAdapter(config)

    # 4. Mixtral check
    if "mixtral" in arch_str or "mixtral" in model_type or "num_local_experts" in config:
        return MixtralAdapter(config)

    # 5. Generic MoE check
    if any(k in config for k in ["num_experts", "moe_intermediate_size", "num_routed_experts"]):
        return DeepSeekAdapter(config)

    # Fallback to GenericAdapter
    return GenericAdapter(config)
