""" MIT License
#
# Copyright (c) 2025 GEF v3.0 Contributors
#
# Gaussian Einstein Fields v3.0: A Physically-Consistent Spacetime Representation
# via 3+1 Decomposition
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
import flax.linen as nn
from typing import Tuple, Optional, Callable
from .gaussian_primitives import Gaussian4DField, gaussian_4d, build_covariance_4d, normalize_quaternion
from .adm_decomposition import ADMVariables, reconstruct_4d_metric_from_adm


class LapseField(nn.Module):
    """
    Lapse function field N(p) represented as Gaussian basis expansion.

    Since N > 0 is required, we model log(N) as an unconstrained field:
        log N(p) = Σᵢ [αᵢᴺ * G(p; μᵢ, Σᵢ)]
        N(p) = exp(log N(p))

    Attributes:
        num_gaussians: Number of Gaussian basis functions
        init_scale: Initial scale for Gaussian centers
        init_alpha_mean: Mean value for initial alpha (controls baseline lapse)
        init_alpha_std: Standard deviation for initial alpha
    """
    num_gaussians: int
    init_scale: float = 1.0
    init_alpha_mean: float = 0.0  # log(1.0) = 0 for N ≈ 1
    init_alpha_std: float = 0.1

    def setup(self):
        # Initialize Gaussian centers
        self.mu = self.param('mu',
                            nn.initializers.normal(stddev=self.init_scale),
                            (self.num_gaussians, 4))

        # Initialize quaternions
        self.q_left = self.param('q_left',
                                lambda key, shape: jnp.concatenate([
                                    jnp.ones((shape[0], 1)),
                                    jax.random.normal(key, (shape[0], 3)) * 0.1
                                ], axis=-1),
                                (self.num_gaussians, 4))

        self.q_right = self.param('q_right',
                                 lambda key, shape: jnp.concatenate([
                                     jnp.ones((shape[0], 1)),
                                     jax.random.normal(key, (shape[0], 3)) * 0.1
                                 ], axis=-1),
                                 (self.num_gaussians, 4))

        # Initialize scales
        self.scales = self.param('scales',
                                nn.initializers.constant(1.0),
                                (self.num_gaussians, 4))

        # Initialize amplitude for log(N)
        self.alpha = self.param('alpha',
                               lambda key, shape: jax.random.normal(key, shape) * self.init_alpha_std + self.init_alpha_mean,
                               (self.num_gaussians,))

    def get_covariances(self) -> jax.Array:
        """Compute covariance matrices for all Gaussians."""
        q_left_norm = jax.vmap(normalize_quaternion)(self.q_left)
        q_right_norm = jax.vmap(normalize_quaternion)(self.q_right)
        scales_pos = jnp.abs(self.scales) + 1e-6
        return build_covariance_4d(q_left_norm, q_right_norm, scales_pos)

    def __call__(self, p: jax.Array) -> jax.Array:
        """
        Evaluate lapse function N(p).

        Args:
            p: Spacetime point(s) of shape (4,) or (M, 4)

        Returns:
            Lapse value(s) N(p), shape () or (M,)
        """
        # Evaluate Gaussian basis functions
        Sigma = self.get_covariances()
        G = gaussian_4d(p, self.mu, Sigma)  # (num_gaussians,) or (M, num_gaussians)

        # Compute log(N) as linear combination
        if G.ndim == 1:
            # Single point
            log_N = jnp.sum(self.alpha * G)
        else:
            # Multiple points
            log_N = jnp.dot(G, self.alpha)  # (M,)

        # Return N = exp(log(N))
        return jnp.exp(log_N)


class ShiftField(nn.Module):
    """
    Shift vector field β^i(p) represented as Gaussian basis expansion.

    The shift is a 3D vector field modeled directly as:
        β^i(p) = Σᵢ [vᵢᵝ * G(p; μᵢ, Σᵢ)]

    Attributes:
        num_gaussians: Number of Gaussian basis functions
        init_scale: Initial scale for Gaussian centers
    """
    num_gaussians: int
    init_scale: float = 1.0

    def setup(self):
        # Initialize Gaussian centers
        self.mu = self.param('mu',
                            nn.initializers.normal(stddev=self.init_scale),
                            (self.num_gaussians, 4))

        # Initialize quaternions
        self.q_left = self.param('q_left',
                                lambda key, shape: jnp.concatenate([
                                    jnp.ones((shape[0], 1)),
                                    jax.random.normal(key, (shape[0], 3)) * 0.1
                                ], axis=-1),
                                (self.num_gaussians, 4))

        self.q_right = self.param('q_right',
                                 lambda key, shape: jnp.concatenate([
                                     jnp.ones((shape[0], 1)),
                                     jax.random.normal(key, (shape[0], 3)) * 0.1
                                 ], axis=-1),
                                 (self.num_gaussians, 4))

        # Initialize scales
        self.scales = self.param('scales',
                                nn.initializers.constant(1.0),
                                (self.num_gaussians, 4))

        # Initialize 3D vector template for each Gaussian
        self.v_beta = self.param('v_beta',
                                nn.initializers.normal(stddev=0.01),
                                (self.num_gaussians, 3))

    def get_covariances(self) -> jax.Array:
        """Compute covariance matrices for all Gaussians."""
        q_left_norm = jax.vmap(normalize_quaternion)(self.q_left)
        q_right_norm = jax.vmap(normalize_quaternion)(self.q_right)
        scales_pos = jnp.abs(self.scales) + 1e-6
        return build_covariance_4d(q_left_norm, q_right_norm, scales_pos)

    def __call__(self, p: jax.Array) -> jax.Array:
        """
        Evaluate shift vector β(p).

        Args:
            p: Spacetime point(s) of shape (4,) or (M, 4)

        Returns:
            Shift vector(s), shape (3,) or (M, 3)
        """
        # Evaluate Gaussian basis functions
        Sigma = self.get_covariances()
        G = gaussian_4d(p, self.mu, Sigma)  # (num_gaussians,) or (M, num_gaussians)

        # Compute shift as weighted sum of vector templates
        if G.ndim == 1:
            # Single point: β^i = Σ_n [v_n^β * G_n(p)]
            beta = jnp.sum(G[:, None] * self.v_beta, axis=0)  # (3,)
        else:
            # Multiple points
            beta = jnp.dot(G, self.v_beta)  # (M, 3)

        return beta


class SpatialMetricField(nn.Module):
    """
    Spatial 3-metric field γ_ij(p) represented via matrix exponential.

    To ensure γ_ij is symmetric positive definite (SPD), we use the log-domain:
        h_γ(p) = Σᵢ [αᵢᵞ * (vᵢᵞ ⊗ vᵢᵞ) * G(p; μᵢ, Σᵢ)]
        γ(p) = expm(h_γ(p))

    Since h_γ is symmetric, expm(h_γ) is guaranteed to be SPD.

    Attributes:
        num_gaussians: Number of Gaussian basis functions
        init_scale: Initial scale for Gaussian centers
    """
    num_gaussians: int
    init_scale: float = 1.0

    def setup(self):
        # Initialize Gaussian centers
        self.mu = self.param('mu',
                            nn.initializers.normal(stddev=self.init_scale),
                            (self.num_gaussians, 4))

        # Initialize quaternions
        self.q_left = self.param('q_left',
                                lambda key, shape: jnp.concatenate([
                                    jnp.ones((shape[0], 1)),
                                    jax.random.normal(key, (shape[0], 3)) * 0.1
                                ], axis=-1),
                                (self.num_gaussians, 4))

        self.q_right = self.param('q_right',
                                 lambda key, shape: jnp.concatenate([
                                     jnp.ones((shape[0], 1)),
                                     jax.random.normal(key, (shape[0], 3)) * 0.1
                                 ], axis=-1),
                                 (self.num_gaussians, 4))

        # Initialize scales
        self.scales = self.param('scales',
                                nn.initializers.constant(1.0),
                                (self.num_gaussians, 4))

        # Initialize 3D vector template for outer product
        self.v_gamma = self.param('v_gamma',
                                 nn.initializers.normal(stddev=0.1),
                                 (self.num_gaussians, 3))

        # Initialize amplitude
        self.alpha_gamma = self.param('alpha_gamma',
                                     nn.initializers.normal(stddev=0.01),
                                     (self.num_gaussians,))

    def get_covariances(self) -> jax.Array:
        """Compute covariance matrices for all Gaussians."""
        q_left_norm = jax.vmap(normalize_quaternion)(self.q_left)
        q_right_norm = jax.vmap(normalize_quaternion)(self.q_right)
        scales_pos = jnp.abs(self.scales) + 1e-6
        return build_covariance_4d(q_left_norm, q_right_norm, scales_pos)

    def compute_log_metric(self, p: jax.Array) -> jax.Array:
        """
        Compute the log-domain metric h_γ(p).

        Args:
            p: Spacetime point(s) of shape (4,) or (M, 4)

        Returns:
            Log-metric h_γ, shape (3, 3) or (M, 3, 3)
        """
        # Evaluate Gaussian basis functions
        Sigma = self.get_covariances()
        G = gaussian_4d(p, self.mu, Sigma)  # (num_gaussians,) or (M, num_gaussians)

        if G.ndim == 1:
            # Single point
            # h_γ = Σ_n [α_n * (v_n ⊗ v_n) * G_n(p)]
            h_gamma = jnp.zeros((3, 3))
            for i in range(self.num_gaussians):
                outer = jnp.outer(self.v_gamma[i], self.v_gamma[i])
                h_gamma += self.alpha_gamma[i] * outer * G[i]
        else:
            # Multiple points
            # Vectorized computation
            h_gamma = jnp.zeros((p.shape[0], 3, 3))
            for i in range(self.num_gaussians):
                outer = jnp.outer(self.v_gamma[i], self.v_gamma[i])
                h_gamma += self.alpha_gamma[i] * outer[None, :, :] * G[:, i, None, None]

        return h_gamma

    def __call__(self, p: jax.Array) -> jax.Array:
        """
        Evaluate spatial metric γ(p).

        Args:
            p: Spacetime point(s) of shape (4,) or (M, 4)

        Returns:
            Spatial metric γ, shape (3, 3) or (M, 3, 3)
        """
        h_gamma = self.compute_log_metric(p)

        # Apply matrix exponential
        if h_gamma.ndim == 2:
            # Single point
            gamma = jax.scipy.linalg.expm(h_gamma)
        else:
            # Multiple points - vectorize expm
            gamma = jax.vmap(jax.scipy.linalg.expm)(h_gamma)

        return gamma


class GaussianEinsteinFields(nn.Module):
    """
    Gaussian Einstein Fields v3.0: Complete 4D spacetime representation
    via 3+1 decomposition.

    This model represents the 4D metric tensor g_μν through its ADM decomposition:
        ds² = -N² dt² + γ_ij (dx^i + β^i dt)(dx^j + β^j dt)

    The three components are modeled as independent Gaussian fields:
        - Lapse N(p): via log-domain to ensure N > 0
        - Shift β^i(p): direct 3D vector field
        - Spatial metric γ_ij(p): via matrix exponential to ensure SPD

    Attributes:
        num_gaussians_lapse: Number of Gaussians for lapse field
        num_gaussians_shift: Number of Gaussians for shift field
        num_gaussians_spatial: Number of Gaussians for spatial metric field
        init_scale: Initial scale for all Gaussian centers
    """
    num_gaussians_lapse: int = 100
    num_gaussians_shift: int = 100
    num_gaussians_spatial: int = 100
    init_scale: float = 1.0

    def setup(self):
        self.lapse_field = LapseField(
            num_gaussians=self.num_gaussians_lapse,
            init_scale=self.init_scale
        )
        self.shift_field = ShiftField(
            num_gaussians=self.num_gaussians_shift,
            init_scale=self.init_scale
        )
        self.spatial_metric_field = SpatialMetricField(
            num_gaussians=self.num_gaussians_spatial,
            init_scale=self.init_scale
        )

    def get_adm_variables(self, p: jax.Array) -> ADMVariables:
        """
        Evaluate ADM variables at spacetime point(s) p.

        Args:
            p: Spacetime point(s) of shape (4,) or (M, 4)

        Returns:
            ADMVariables containing lapse, shift, and spatial metric
        """
        lapse = self.lapse_field(p)
        shift = self.shift_field(p)
        spatial_metric = self.spatial_metric_field(p)

        return ADMVariables(
            lapse=lapse,
            shift=shift,
            spatial_metric=spatial_metric
        )

    def __call__(self, p: jax.Array) -> jax.Array:
        """
        Evaluate 4D metric tensor g_μν at spacetime point(s) p.

        Args:
            p: Spacetime point(s) of shape (4,) or (M, 4)

        Returns:
            4D metric tensor, shape (4, 4) or (M, 4, 4)
        """
        # Get ADM variables
        adm = self.get_adm_variables(p)

        # Reconstruct 4D metric
        g = reconstruct_4d_metric_from_adm(
            adm.lapse,
            adm.shift,
            adm.spatial_metric
        )

        return g


if __name__ == '__main__':
    print("Testing Gaussian Einstein Fields v3.0...")

    # Initialize model
    key = jax.random.PRNGKey(42)
    model = GaussianEinsteinFields(
        num_gaussians_lapse=50,
        num_gaussians_shift=50,
        num_gaussians_spatial=50,
        init_scale=2.0
    )

    # Test single point evaluation
    print("\n--- Single Point Test ---")
    p_test = jnp.array([0.0, 5.0, jnp.pi/2, 0.0])  # (t, r, θ, φ)
    params = model.init(key, p_test)

    # Get metric
    g = model.apply(params, p_test)
    print(f"Metric shape: {g.shape}")
    print(f"Metric:\n{g}")

    # Get ADM variables
    adm = model.apply(params, p_test, method=model.get_adm_variables)
    print(f"\nLapse: {adm.lapse}")
    print(f"Shift: {adm.shift}")
    print(f"Spatial metric:\n{adm.spatial_metric}")

    # Verify Lorentzian signature
    eigenvalues = jnp.linalg.eigvalsh(g)
    print(f"\nMetric eigenvalues: {eigenvalues}")
    print(f"Expected signature (-,+,+,+): {jnp.sum(eigenvalues < 0)} negative eigenvalues")

    # Test batch evaluation
    print("\n--- Batch Test ---")
    p_batch = jax.random.normal(key, (10, 4))
    g_batch = model.apply(params, p_batch)
    print(f"Batch metric shape: {g_batch.shape}")

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"\nTotal parameters: {param_count}")
