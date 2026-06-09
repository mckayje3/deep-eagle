"""Utility functions"""

from .checkpoint import safe_torch_load
from .deprecation import (
    DeprecatedClass,
    deprecate,
    deprecated,
    deprecated_argument,
    warn_on_import,
)
from .feature_importance import (
    FeatureImportance,
    SHAPExplainer,
    calculate_all_importances,
    plot_feature_importance,
)
from .helpers import count_parameters, get_device, set_seed
from .metrics import mae, mape, mse, r2_score, rmse

__all__ = [
    "mse",
    "rmse",
    "mae",
    "mape",
    "r2_score",
    "set_seed",
    "get_device",
    "count_parameters",
    "safe_torch_load",
    "deprecate",
    "deprecated",
    "deprecated_argument",
    "DeprecatedClass",
    "warn_on_import",
    "FeatureImportance",
    "calculate_all_importances",
    "plot_feature_importance",
    "SHAPExplainer",
]
