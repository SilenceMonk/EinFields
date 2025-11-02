"""
Gaussian Einstein Fields v3.0

A physically-consistent spacetime representation via 3+1 ADM decomposition.

This module implements a novel approach to representing spacetime metrics using
4D Gaussian basis functions and the ADM (Arnowitt-Deser-Misner) formalism.

Key components:
    - gaussian_primitives: 4D Gaussian basis functions
    - adm_decomposition: 3+1 spacetime decomposition utilities
    - gef_model: Main GEF v3.0 model with Lapse, Shift, and Spatial Metric fields
    - losses: Training loss functions for representer mode

Example:
    >>> from gaussian_einstein_fields import GaussianEinsteinFields
    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> # Create model
    >>> model = GaussianEinsteinFields(num_gaussians_lapse=100,
    ...                                num_gaussians_shift=100,
    ...                                num_gaussians_spatial=100)
    >>>
    >>> # Initialize and evaluate
    >>> key = jax.random.PRNGKey(0)
    >>> p = jnp.array([0.0, 5.0, jnp.pi/2, 0.0])
    >>> params = model.init(key, p)
    >>> g = model.apply(params, p)  # 4D metric tensor
"""

from .gaussian_primitives import (
    Gaussian4DField,
    GaussianParams,
    gaussian_4d,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    build_covariance_4d
)

from .adm_decomposition import (
    ADMVariables,
    extract_adm_from_4d_metric,
    reconstruct_4d_metric_from_adm,
    check_adm_constraints,
    verify_reconstruction
)

from .gef_model import (
    LapseField,
    ShiftField,
    SpatialMetricField,
    GaussianEinsteinFields
)

from .losses import (
    GEFLoss,
    create_loss_function,
    matrix_log_safe,
    frobenius_norm_squared
)

__version__ = '3.0.0'

__all__ = [
    # Gaussian primitives
    'Gaussian4DField',
    'GaussianParams',
    'gaussian_4d',
    'normalize_quaternion',
    'quaternion_to_rotation_matrix',
    'build_covariance_4d',

    # ADM decomposition
    'ADMVariables',
    'extract_adm_from_4d_metric',
    'reconstruct_4d_metric_from_adm',
    'check_adm_constraints',
    'verify_reconstruction',

    # GEF model
    'LapseField',
    'ShiftField',
    'SpatialMetricField',
    'GaussianEinsteinFields',

    # Losses
    'GEFLoss',
    'create_loss_function',
    'matrix_log_safe',
    'frobenius_norm_squared',
]
