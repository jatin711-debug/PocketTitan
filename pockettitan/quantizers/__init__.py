"""Pluggable Quantization Backends."""

from pockettitan.config import QuantConfig, QuantMethod
from pockettitan.quantizers.base import BaseQuantizer, QuantizerCapabilities, QuantizedResult
from pockettitan.quantizers.rtn import RTNQuantizer
from pockettitan.quantizers.ternary import TernaryQuantizer
from pockettitan.quantizers.intx import INTxQuantizer
from pockettitan.quantizers.hqq import HQQQuantizer
from pockettitan.quantizers.gptq import GPTQQuantizer
from pockettitan.quantizers.awq import AWQQuantizer
from pockettitan.quantizers.autoround import AutoRoundQuantizer


def get_quantizer(config: QuantConfig) -> BaseQuantizer:
    """Instantiate appropriate quantizer backend from configuration."""
    method = config.method
    if method == QuantMethod.RTN:
        return RTNQuantizer(config)
    elif method == QuantMethod.TERNARY:
        return TernaryQuantizer(config)
    elif method == QuantMethod.INTX:
        return INTxQuantizer(config)
    elif method == QuantMethod.HQQ:
        return HQQQuantizer(config)
    elif method == QuantMethod.GPTQ:
        return GPTQQuantizer(config)
    elif method == QuantMethod.AWQ:
        return AWQQuantizer(config)
    elif method == QuantMethod.AUTOROUND:
        return AutoRoundQuantizer(config)
    elif method == QuantMethod.AUTO:
        if config.bits == 1:
            return TernaryQuantizer(config)
        return HQQQuantizer(config)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")


__all__ = [
    "BaseQuantizer",
    "QuantizerCapabilities",
    "QuantizedResult",
    "RTNQuantizer",
    "TernaryQuantizer",
    "INTxQuantizer",
    "HQQQuantizer",
    "GPTQQuantizer",
    "AWQQuantizer",
    "AutoRoundQuantizer",
    "get_quantizer",
]
