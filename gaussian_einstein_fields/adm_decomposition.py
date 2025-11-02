""" MIT License
#
# Copyright (c) 2025 GEF v3.0 Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ADMVariables:
    """
    Container for ADM 3+1 decomposition variables.

    In the 3+1 formalism, the 4D spacetime metric is decomposed as:
        ds² = -N² dt² + γ_ij (dx^i + β^i dt)(dx^j + β^j dt)

    Attributes:
        lapse: Lapse function N (scalar, N > 0)
        shift: Shift vector β^i (3D vector)
        spatial_metric: Spatial 3-metric γ_ij (3x3 symmetric positive definite matrix)
    """
    lapse: jax.Array  # Shape: () or (N,) - scalar field
    shift: jax.Array  # Shape: (3,) or (N, 3) - 3D vector field
    spatial_metric: jax.Array  # Shape: (3, 3) or (N, 3, 3) - 3D metric


def extract_adm_from_4d_metric(g: jax.Array) -> ADMVariables:
    """
    Extract ADM variables (lapse, shift, spatial metric) from 4D metric tensor.

    The 4D metric in ADM form is:
        g_μν = [[-N² + β_i β^i,  β_j        ],
                [β_i,             γ_ij      ]]

    where β^i = γ^ij β_j (raised with spatial metric).

    Args:
        g: 4D metric tensor of shape (4, 4) or (N, 4, 4)

    Returns:
        ADMVariables containing lapse, shift, and spatial metric

    Notes:
        The spatial metric γ_ij is simply the 3x3 spatial block of g_μν.
        The shift vector β_i are the off-diagonal time-space components.
        The lapse is computed from N² = β^i β_i - g_tt.
    """
    if g.ndim == 2:
        # Single metric tensor
        # Extract spatial metric (3x3 block)
        gamma_ij = g[1:, 1:]  # (3, 3)

        # Extract shift (time-space components)
        beta_lower = g[0, 1:]  # (3,) - β_i (lowered)

        # Compute lapse: N² = β^i β_i - g_tt
        # First raise the shift: β^i = γ^ij β_j
        gamma_inv = jnp.linalg.inv(gamma_ij + jnp.eye(3) * 1e-10)
        beta_upper = jnp.dot(gamma_inv, beta_lower)  # β^i

        # Compute lapse squared
        beta_squared = jnp.dot(beta_upper, beta_lower)  # β^i β_i
        N_squared = beta_squared - g[0, 0]  # N² = β^i β_i - g_tt

        # Ensure N² is positive (it must be for physical spacetime)
        N_squared = jnp.maximum(N_squared, 1e-10)
        N = jnp.sqrt(N_squared)

        return ADMVariables(lapse=N, shift=beta_lower, spatial_metric=gamma_ij)

    elif g.ndim == 3:
        # Batch of metric tensors
        batch_size = g.shape[0]

        # Extract spatial metrics
        gamma_ij = g[:, 1:, 1:]  # (N, 3, 3)

        # Extract shifts
        beta_lower = g[:, 0, 1:]  # (N, 3) - β_i

        # Compute lapse for each metric in batch
        gamma_inv = jnp.linalg.inv(gamma_ij + jnp.eye(3)[None, :, :] * 1e-10)
        beta_upper = jnp.einsum('nij,nj->ni', gamma_inv, beta_lower)  # (N, 3)

        beta_squared = jnp.sum(beta_upper * beta_lower, axis=-1)  # (N,)
        N_squared = beta_squared - g[:, 0, 0]  # (N,)

        N_squared = jnp.maximum(N_squared, 1e-10)
        N = jnp.sqrt(N_squared)

        return ADMVariables(lapse=N, shift=beta_lower, spatial_metric=gamma_ij)

    else:
        raise ValueError(f"Input metric must be 2D or 3D, got shape {g.shape}")


def reconstruct_4d_metric_from_adm(
    lapse: jax.Array,
    shift: jax.Array,
    spatial_metric: jax.Array
) -> jax.Array:
    """
    Reconstruct 4D metric tensor from ADM variables.

    The 4D line element is:
        ds² = -N² dt² + γ_ij (dx^i + β^i dt)(dx^j + β^j dt)

    The 4D metric components are:
        g_tt = -N² + β_i β^i = -N² + γ_ij β^i β^j
        g_ti = β_i
        g_it = β_i
        g_ij = γ_ij

    Args:
        lapse: Lapse function N, shape () or (batch,)
        shift: Shift vector β^i, shape (3,) or (batch, 3)
        spatial_metric: Spatial metric γ_ij, shape (3, 3) or (batch, 3, 3)

    Returns:
        4D metric tensor g_μν, shape (4, 4) or (batch, 4, 4)
    """
    # Handle single vs batch inputs
    is_batched = spatial_metric.ndim == 3

    if not is_batched:
        # Single case
        g = jnp.zeros((4, 4))

        # g_ij = γ_ij (spatial block)
        g = g.at[1:, 1:].set(spatial_metric)

        # g_ti = g_it = β_i
        g = g.at[0, 1:].set(shift)
        g = g.at[1:, 0].set(shift)

        # g_tt = -N² + β_i β^i
        beta_squared = jnp.dot(shift, jnp.dot(spatial_metric, shift))
        g_tt = -lapse**2 + beta_squared
        g = g.at[0, 0].set(g_tt)

        return g

    else:
        # Batched case
        batch_size = spatial_metric.shape[0]
        g = jnp.zeros((batch_size, 4, 4))

        # g_ij = γ_ij
        g = g.at[:, 1:, 1:].set(spatial_metric)

        # g_ti = g_it = β_i
        g = g.at[:, 0, 1:].set(shift)
        g = g.at[:, 1:, 0].set(shift)

        # g_tt = -N² + β_i β^i
        beta_squared = jnp.einsum('ni,nij,nj->n', shift, spatial_metric, shift)
        g_tt = -lapse**2 + beta_squared
        g = g.at[:, 0, 0].set(g_tt)

        return g


def check_adm_constraints(adm: ADMVariables, verbose: bool = True) -> Dict[str, bool]:
    """
    Check if ADM variables satisfy physical constraints.

    Constraints:
        1. Lapse must be positive: N > 0
        2. Spatial metric must be symmetric
        3. Spatial metric must be positive definite (all eigenvalues > 0)

    Args:
        adm: ADMVariables to check
        verbose: If True, print constraint violations

    Returns:
        Dictionary with constraint check results
    """
    results = {}

    # Check 1: Lapse positivity
    if adm.lapse.ndim == 0:
        lapse_positive = adm.lapse > 0
    else:
        lapse_positive = jnp.all(adm.lapse > 0)
    results['lapse_positive'] = bool(lapse_positive)

    if verbose and not lapse_positive:
        print(f"WARNING: Lapse must be positive! Min value: {jnp.min(adm.lapse)}")

    # Check 2: Spatial metric symmetry
    if adm.spatial_metric.ndim == 2:
        gamma = adm.spatial_metric
        symmetry_error = jnp.max(jnp.abs(gamma - gamma.T))
        results['spatial_metric_symmetric'] = bool(symmetry_error < 1e-10)
    else:
        gamma = adm.spatial_metric
        symmetry_error = jnp.max(jnp.abs(gamma - jnp.transpose(gamma, (0, 2, 1))))
        results['spatial_metric_symmetric'] = bool(symmetry_error < 1e-10)

    if verbose and not results['spatial_metric_symmetric']:
        print(f"WARNING: Spatial metric not symmetric! Max error: {symmetry_error}")

    # Check 3: Spatial metric positive definiteness
    if adm.spatial_metric.ndim == 2:
        eigenvalues = jnp.linalg.eigvalsh(adm.spatial_metric)
        min_eigenvalue = jnp.min(eigenvalues)
        results['spatial_metric_positive_definite'] = bool(min_eigenvalue > 0)

        if verbose and not results['spatial_metric_positive_definite']:
            print(f"WARNING: Spatial metric not positive definite! Min eigenvalue: {min_eigenvalue}")
    else:
        eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(adm.spatial_metric)
        min_eigenvalue = jnp.min(eigenvalues)
        results['spatial_metric_positive_definite'] = bool(min_eigenvalue > 0)

        if verbose and not results['spatial_metric_positive_definite']:
            print(f"WARNING: Spatial metric not positive definite! Min eigenvalue: {min_eigenvalue}")

    return results


def verify_reconstruction(
    g_original: jax.Array,
    g_reconstructed: jax.Array,
    atol: float = 1e-10
) -> bool:
    """
    Verify that reconstructed metric matches original.

    Args:
        g_original: Original 4D metric
        g_reconstructed: Reconstructed 4D metric
        atol: Absolute tolerance for comparison

    Returns:
        True if metrics match within tolerance
    """
    max_error = jnp.max(jnp.abs(g_original - g_reconstructed))
    matches = bool(max_error < atol)

    if not matches:
        print(f"Reconstruction error: {max_error}")

    return matches


if __name__ == '__main__':
    print("Testing ADM 3+1 Decomposition...")

    # Test 1: Extract ADM from Minkowski metric
    print("\n--- Test 1: Minkowski Metric ---")
    g_minkowski = jnp.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    adm_minkowski = extract_adm_from_4d_metric(g_minkowski)
    print(f"Lapse: {adm_minkowski.lapse}")
    print(f"Shift: {adm_minkowski.shift}")
    print(f"Spatial metric:\n{adm_minkowski.spatial_metric}")

    # Check constraints
    constraints = check_adm_constraints(adm_minkowski, verbose=True)
    print(f"Constraints satisfied: {all(constraints.values())}")

    # Reconstruct and verify
    g_recon = reconstruct_4d_metric_from_adm(
        adm_minkowski.lapse,
        adm_minkowski.shift,
        adm_minkowski.spatial_metric
    )
    print(f"Reconstruction matches: {verify_reconstruction(g_minkowski, g_recon)}")

    # Test 2: Schwarzschild metric in spherical coordinates
    print("\n--- Test 2: Schwarzschild Metric (Spherical) ---")
    M = 1.0
    r = 5.0
    theta = jnp.pi / 2

    g_schwarzschild = jnp.array([
        [-(1.0 - 2*M/r), 0.0, 0.0, 0.0],
        [0.0, 1.0/(1.0 - 2*M/r), 0.0, 0.0],
        [0.0, 0.0, r**2, 0.0],
        [0.0, 0.0, 0.0, r**2 * jnp.sin(theta)**2]
    ])

    adm_schw = extract_adm_from_4d_metric(g_schwarzschild)
    print(f"Lapse: {adm_schw.lapse}")
    print(f"Shift: {adm_schw.shift}")
    print(f"Spatial metric:\n{adm_schw.spatial_metric}")

    constraints_schw = check_adm_constraints(adm_schw, verbose=True)
    print(f"Constraints satisfied: {all(constraints_schw.values())}")

    g_recon_schw = reconstruct_4d_metric_from_adm(
        adm_schw.lapse,
        adm_schw.shift,
        adm_schw.spatial_metric
    )
    print(f"Reconstruction matches: {verify_reconstruction(g_schwarzschild, g_recon_schw)}")

    # Test 3: Batch processing
    print("\n--- Test 3: Batch Processing ---")
    batch_metrics = jnp.stack([g_minkowski, g_schwarzschild])
    adm_batch = extract_adm_from_4d_metric(batch_metrics)
    print(f"Batch lapse shape: {adm_batch.lapse.shape}")
    print(f"Batch shift shape: {adm_batch.shift.shape}")
    print(f"Batch spatial metric shape: {adm_batch.spatial_metric.shape}")

    g_recon_batch = reconstruct_4d_metric_from_adm(
        adm_batch.lapse,
        adm_batch.shift,
        adm_batch.spatial_metric
    )
    print(f"Batch reconstruction matches: {verify_reconstruction(batch_metrics, g_recon_batch)}")
