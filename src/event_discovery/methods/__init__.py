"""Event discovery methods."""

from .hierarchical_energy import HierarchicalEnergyMethod, EnergyConfig
from .geometric_outlier import GeometricOutlierMethod
from .optimization_sparse import PureOptimizationMethod, OptimizationConfig
from .baseline_dense import DenseVLMMethod

__all__ = [
    "HierarchicalEnergyMethod",
    "EnergyConfig",
    "GeometricOutlierMethod",
    "PureOptimizationMethod",
    "OptimizationConfig",
    "DenseVLMMethod",
]
