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
import flax.linen as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GaussianParams:
    """
    Parameters for 4D Gaussian primitives.

    Attributes:
        mu: Center position in 4D spacetime (N, 4)
        q_left: Left quaternion for rotation (N, 4)
        q_right: Right quaternion for rotation (N, 4)
        scales: Scaling factors for each axis (N, 4)
    """
    mu: jax.Array  # (N, 4) - center positions
    q_left: jax.Array  # (N, 4) - left quaternion
    q_right: jax.Array  # (N, 4) - right quaternion
    scales: jax.Array  # (N, 4) - scaling factors


def normalize_quaternion(q: jax.Array) -> jax.Array:
    """
    Normalize a quaternion to unit length.

    Args:
        q: Quaternion array of shape (..., 4)

    Returns:
        Normalized quaternion of same shape
    """
    norm = jnp.sqrt(jnp.sum(q**2, axis=-1, keepdims=True))
    return q / (norm + 1e-8)


def quaternion_to_rotation_matrix(q: jax.Array) -> jax.Array:
    """
    Convert quaternion to 4D rotation matrix using SO(4) double quaternion representation.

    In 4D, rotations are represented by pairs of unit quaternions (q_L, q_R),
    where a point p is rotated as: p' = q_L * p * q_R^*

    Args:
        q: Quaternion of shape (..., 4) with components [w, x, y, z]

    Returns:
        Rotation matrix of shape (..., 4, 4)
    """
    q = normalize_quaternion(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Create 4x4 rotation matrix from quaternion
    # This is the left multiplication matrix for quaternions
    R = jnp.stack([
        jnp.stack([w, -x, -y, -z], axis=-1),
        jnp.stack([x, w, -z, y], axis=-1),
        jnp.stack([y, z, w, -x], axis=-1),
        jnp.stack([z, -y, x, w], axis=-1)
    ], axis=-2)

    return R


def build_covariance_4d(
    q_left: jax.Array,
    q_right: jax.Array,
    scales: jax.Array
) -> jax.Array:
    """
    Build 4D covariance matrix from quaternion rotations and scales.

    The covariance is constructed as:
        Σ = R_L * S * S^T * R_L^T
    where R_L is from q_left and S is a diagonal matrix of scales.

    Args:
        q_left: Left quaternion (N, 4)
        q_right: Right quaternion (N, 4) [can be used for more complex rotations]
        scales: Scaling factors (N, 4)

    Returns:
        Covariance matrices (N, 4, 4)
    """
    # Convert quaternions to rotation matrices
    R_left = quaternion_to_rotation_matrix(q_left)  # (N, 4, 4)

    # Create diagonal scale matrix
    S = jnp.diag(scales)  # (N, 4, 4) if scales is (N, 4)
    if scales.ndim == 1:
        S = jnp.diag(scales)
    else:  # (N, 4)
        S = jax.vmap(jnp.diag)(scales)

    # Compute covariance: Σ = R * S * S^T * R^T
    RS = jnp.einsum('...ij,...jk->...ik', R_left, S)
    Sigma = jnp.einsum('...ij,...kj->...ik', RS, RS)

    return Sigma


def gaussian_4d(
    p: jax.Array,
    mu: jax.Array,
    Sigma: jax.Array
) -> jax.Array:
    """
    Evaluate 4D Gaussian at point p.

    G(p) = exp(-0.5 * (p - μ)^T * Σ^{-1} * (p - μ))

    Args:
        p: Query point (4,) or (M, 4)
        mu: Gaussian center (4,) or (N, 4)
        Sigma: Covariance matrix (4, 4) or (N, 4, 4)

    Returns:
        Gaussian value(s), shape depends on inputs
    """
    # Handle different input shapes
    if p.ndim == 1 and mu.ndim == 1:
        # Single point, single Gaussian
        diff = p - mu
        Sigma_inv = jnp.linalg.inv(Sigma + jnp.eye(4) * 1e-6)
        exponent = -0.5 * jnp.dot(diff, jnp.dot(Sigma_inv, diff))
        return jnp.exp(exponent)
    elif p.ndim == 1 and mu.ndim == 2:
        # Single point, multiple Gaussians
        diff = p[None, :] - mu  # (N, 4)
        Sigma_inv = jnp.linalg.inv(Sigma + jnp.eye(4)[None, :, :] * 1e-6)  # (N, 4, 4)
        # Compute (p - mu)^T * Sigma^{-1} * (p - mu) for each Gaussian
        temp = jnp.einsum('ni,nij->nj', diff, Sigma_inv)  # (N, 4)
        exponent = -0.5 * jnp.sum(diff * temp, axis=-1)  # (N,)
        return jnp.exp(exponent)
    elif p.ndim == 2 and mu.ndim == 2:
        # Multiple points, multiple Gaussians
        # p: (M, 4), mu: (N, 4), Sigma: (N, 4, 4)
        # Result: (M, N)
        diff = p[:, None, :] - mu[None, :, :]  # (M, N, 4)
        Sigma_inv = jnp.linalg.inv(Sigma + jnp.eye(4)[None, :, :] * 1e-6)  # (N, 4, 4)
        # Compute for each point and each Gaussian
        temp = jnp.einsum('mni,nij->mnj', diff, Sigma_inv)  # (M, N, 4)
        exponent = -0.5 * jnp.sum(diff * temp, axis=-1)  # (M, N)
        return jnp.exp(exponent)
    else:
        raise ValueError(f"Unsupported input shapes: p {p.shape}, mu {mu.shape}")


class Gaussian4DField(nn.Module):
    """
    Base class for 4D Gaussian field representation.

    This represents a scalar, vector, or tensor field as a linear combination
    of 4D Gaussian basis functions.
    """
    num_gaussians: int
    init_scale: float = 1.0

    def setup(self):
        # Initialize Gaussian centers randomly in spacetime
        self.mu = self.param('mu',
                            nn.initializers.normal(stddev=self.init_scale),
                            (self.num_gaussians, 4))

        # Initialize quaternions (w, x, y, z) with w=1, others=0 for identity
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

    def get_covariances(self) -> jax.Array:
        """
        Compute covariance matrices for all Gaussians.

        Returns:
            Covariance matrices (num_gaussians, 4, 4)
        """
        # Normalize quaternions
        q_left_norm = jax.vmap(normalize_quaternion)(self.q_left)
        q_right_norm = jax.vmap(normalize_quaternion)(self.q_right)

        # Ensure positive scales
        scales_pos = jnp.abs(self.scales) + 1e-6

        # Build covariances
        return build_covariance_4d(q_left_norm, q_right_norm, scales_pos)

    def evaluate_gaussians(self, p: jax.Array) -> jax.Array:
        """
        Evaluate all Gaussian basis functions at point(s) p.

        Args:
            p: Query point(s) of shape (4,) or (M, 4)

        Returns:
            Gaussian values of shape (num_gaussians,) or (M, num_gaussians)
        """
        Sigma = self.get_covariances()
        return gaussian_4d(p, self.mu, Sigma)


if __name__ == '__main__':
    # Test the Gaussian primitives
    import numpy as np

    print("Testing 4D Gaussian Primitives...")

    # Test quaternion normalization
    q = jnp.array([1.0, 2.0, 3.0, 4.0])
    q_norm = normalize_quaternion(q)
    print(f"Normalized quaternion: {q_norm}")
    print(f"Norm: {jnp.sqrt(jnp.sum(q_norm**2))}")

    # Test rotation matrix
    R = quaternion_to_rotation_matrix(q_norm)
    print(f"Rotation matrix shape: {R.shape}")
    print(f"Rotation matrix:\n{R}")

    # Test covariance building
    q_left = jnp.array([1.0, 0.0, 0.0, 0.0])
    q_right = jnp.array([1.0, 0.0, 0.0, 0.0])
    scales = jnp.array([1.0, 1.0, 1.0, 1.0])
    Sigma = build_covariance_4d(q_left[None, :], q_right[None, :], scales[None, :])
    print(f"Covariance matrix shape: {Sigma.shape}")
    print(f"Covariance matrix:\n{Sigma[0]}")

    # Test Gaussian evaluation
    p = jnp.array([0.0, 0.0, 0.0, 0.0])
    mu = jnp.array([0.0, 0.0, 0.0, 0.0])
    g_val = gaussian_4d(p, mu, Sigma[0])
    print(f"Gaussian value at center: {g_val}")

    # Test with multiple Gaussians
    key = jax.random.PRNGKey(0)
    model = Gaussian4DField(num_gaussians=10)
    params = model.init(key, jnp.zeros(4))

    p_test = jnp.array([1.0, 2.0, 3.0, 4.0])
    g_values = model.apply(params, p_test, method=model.evaluate_gaussians)
    print(f"Gaussian values shape: {g_values.shape}")
    print(f"Gaussian values: {g_values}")
