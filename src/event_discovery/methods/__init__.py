"""Event discovery methods."""

from .baseline_dense import DenseVLMMethod, VLMScoringError
from .geometric_outlier import GeometricOutlierMethod
from .hierarchical_energy import EnergyConfig, HierarchicalEnergyMethod
from .optimization_sparse import OptimizationConfig, PureOptimizationMethod

__all__ = [
    "HierarchicalEnergyMethod",
    "EnergyConfig",
    "GeometricOutlierMethod",
    "PureOptimizationMethod",
    "OptimizationConfig",
    "DenseVLMMethod",
    "VLMScoringError",
]
