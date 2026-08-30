"""Precision, distortion measurement, sensitivity analysis, and Pareto bit allocation."""

from pockettitan.precision.distortion import (
    compute_weight_distortion,
    compute_snr_db,
    compute_cosine_similarity,
    compute_activation_distortion,
    evaluate_quantization_quality,
    DistortionReport,
)
from pockettitan.precision.sensitivity import (
    TensorSensitivityScore,
    compute_tensor_sensitivity,
)
from pockettitan.precision.allocator import (
    HeterogeneousPrecisionMap,
    ParetoBitAllocator,
)
from pockettitan.precision.two_population import (
    TwoPopulationAllocator,
    TwoPopulationPlan,
)

__all__ = [
    "compute_weight_distortion",
    "compute_snr_db",
    "compute_cosine_similarity",
    "compute_activation_distortion",
    "evaluate_quantization_quality",
    "DistortionReport",
    "TensorSensitivityScore",
    "compute_tensor_sensitivity",
    "HeterogeneousPrecisionMap",
    "ParetoBitAllocator",
    "TwoPopulationAllocator",
    "TwoPopulationPlan",
]
