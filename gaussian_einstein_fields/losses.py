""" MIT License
#
# Copyright (c) 2025 GEF v3.0 Contributors
#
# Training losses for Gaussian Einstein Fields v3.0
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
from typing import Dict, Callable, Tuple
from .adm_decomposition import ADMVariables, extract_adm_from_4d_metric


def matrix_log_safe(A: jax.Array, epsilon: float = 1e-8) -> jax.Array:
    """
    Compute matrix logarithm with numerical stabilization.

    Args:
        A: Input matrix, shape (..., n, n)
        epsilon: Small value added to eigenvalues for stability

    Returns:
        log(A), same shape as input
    """
    if A.ndim == 2:
        # Add small regularization to ensure positive definiteness
        A_reg = A + jnp.eye(A.shape[0]) * epsilon
        return jax.scipy.linalg.logm(A_reg)
    else:
        # Batch case
        A_reg = A + jnp.eye(A.shape[-1])[None, :, :] * epsilon
        return jax.vmap(jax.scipy.linalg.logm)(A_reg)


def frobenius_norm_squared(A: jax.Array) -> jax.Array:
    """
    Compute squared Frobenius norm of matrix/matrices.

    Args:
        A: Matrix of shape (..., m, n)

    Returns:
        ||A||_F^2 = sum_{i,j} |A_{ij}|^2
    """
    return jnp.sum(A * A, axis=(-2, -1))


def compute_jacobian(func: Callable, x: jax.Array) -> jax.Array:
    """
    Compute Jacobian of a function at point x.

    Args:
        func: Function to differentiate (x -> y)
        x: Input point, shape (n,)

    Returns:
        Jacobian matrix, shape (m, n) where m is output dimension
    """
    return jax.jacfwd(func)(x)


def compute_hessian(func: Callable, x: jax.Array) -> jax.Array:
    """
    Compute Hessian of a scalar function at point x.

    Args:
        func: Scalar function (x -> scalar)
        x: Input point, shape (n,)

    Returns:
        Hessian matrix, shape (n, n)
    """
    return jax.hessian(func)(x)


class GEFLoss:
    """
    Loss functions for training Gaussian Einstein Fields v3.0.

    This class implements Sobolev losses in the natural domains for each
    ADM component:
        - Lapse: in log domain
        - Shift: direct domain
        - Spatial metric: in log domain

    The total loss is:
        L = L_lapse + L_shift + L_spatial

    Each component loss can include:
        - L0: MSE of the field values
        - L1: MSE of first derivatives (Jacobian)
        - L2: MSE of second derivatives (Hessian)
    """

    def __init__(
        self,
        lambda_lapse_0: float = 1.0,
        lambda_lapse_1: float = 0.1,
        lambda_lapse_2: float = 0.01,
        lambda_shift_0: float = 1.0,
        lambda_shift_1: float = 0.1,
        lambda_shift_2: float = 0.01,
        lambda_spatial_0: float = 1.0,
        lambda_spatial_1: float = 0.1,
        lambda_spatial_2: float = 0.01,
    ):
        """
        Initialize loss function with weights.

        Args:
            lambda_*_0: Weight for 0th order (value) loss
            lambda_*_1: Weight for 1st order (gradient) loss
            lambda_*_2: Weight for 2nd order (Hessian) loss
        """
        self.lambda_lapse_0 = lambda_lapse_0
        self.lambda_lapse_1 = lambda_lapse_1
        self.lambda_lapse_2 = lambda_lapse_2

        self.lambda_shift_0 = lambda_shift_0
        self.lambda_shift_1 = lambda_shift_1
        self.lambda_shift_2 = lambda_shift_2

        self.lambda_spatial_0 = lambda_spatial_0
        self.lambda_spatial_1 = lambda_spatial_1
        self.lambda_spatial_2 = lambda_spatial_2

    def lapse_loss(
        self,
        log_N_pred: jax.Array,
        log_N_gt: jax.Array,
        include_derivatives: bool = False,
        log_N_jac_pred: jax.Array = None,
        log_N_jac_gt: jax.Array = None,
        log_N_hess_pred: jax.Array = None,
        log_N_hess_gt: jax.Array = None,
    ) -> Dict[str, jax.Array]:
        """
        Compute lapse field loss in log domain.

        Args:
            log_N_pred: Predicted log(N)
            log_N_gt: Ground truth log(N)
            include_derivatives: Whether to include derivative losses
            log_N_jac_pred: Jacobian of log(N) predicted
            log_N_jac_gt: Jacobian of log(N) ground truth
            log_N_hess_pred: Hessian of log(N) predicted
            log_N_hess_gt: Hessian of log(N) ground truth

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # L0: Value loss
        l0 = jnp.mean((log_N_pred - log_N_gt) ** 2)
        losses['lapse_l0'] = l0
        losses['lapse_total'] = self.lambda_lapse_0 * l0

        if include_derivatives:
            # L1: Gradient loss
            if log_N_jac_pred is not None and log_N_jac_gt is not None:
                l1 = jnp.mean((log_N_jac_pred - log_N_jac_gt) ** 2)
                losses['lapse_l1'] = l1
                losses['lapse_total'] += self.lambda_lapse_1 * l1

            # L2: Hessian loss
            if log_N_hess_pred is not None and log_N_hess_gt is not None:
                l2 = jnp.mean((log_N_hess_pred - log_N_hess_gt) ** 2)
                losses['lapse_l2'] = l2
                losses['lapse_total'] += self.lambda_lapse_2 * l2

        return losses

    def shift_loss(
        self,
        beta_pred: jax.Array,
        beta_gt: jax.Array,
        include_derivatives: bool = False,
        beta_jac_pred: jax.Array = None,
        beta_jac_gt: jax.Array = None,
    ) -> Dict[str, jax.Array]:
        """
        Compute shift field loss.

        Args:
            beta_pred: Predicted shift β^i, shape (..., 3)
            beta_gt: Ground truth shift β^i, shape (..., 3)
            include_derivatives: Whether to include derivative losses
            beta_jac_pred: Jacobian of β^i, shape (..., 3, 4)
            beta_jac_gt: Jacobian of β^i, shape (..., 3, 4)

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # L0: Value loss
        l0 = jnp.mean((beta_pred - beta_gt) ** 2)
        losses['shift_l0'] = l0
        losses['shift_total'] = self.lambda_shift_0 * l0

        if include_derivatives:
            # L1: Gradient loss
            if beta_jac_pred is not None and beta_jac_gt is not None:
                l1 = jnp.mean((beta_jac_pred - beta_jac_gt) ** 2)
                losses['shift_l1'] = l1
                losses['shift_total'] += self.lambda_shift_1 * l1

        return losses

    def spatial_metric_loss(
        self,
        h_gamma_pred: jax.Array,
        h_gamma_gt: jax.Array,
        include_derivatives: bool = False,
        h_gamma_jac_pred: jax.Array = None,
        h_gamma_jac_gt: jax.Array = None,
    ) -> Dict[str, jax.Array]:
        """
        Compute spatial metric loss in log domain.

        Args:
            h_gamma_pred: Predicted log(γ), shape (..., 3, 3)
            h_gamma_gt: Ground truth log(γ), shape (..., 3, 3)
            include_derivatives: Whether to include derivative losses
            h_gamma_jac_pred: Jacobian of h_γ
            h_gamma_jac_gt: Jacobian of h_γ

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # L0: Value loss (Frobenius norm)
        diff = h_gamma_pred - h_gamma_gt
        l0 = jnp.mean(frobenius_norm_squared(diff))
        losses['spatial_l0'] = l0
        losses['spatial_total'] = self.lambda_spatial_0 * l0

        if include_derivatives:
            # L1: Gradient loss
            if h_gamma_jac_pred is not None and h_gamma_jac_gt is not None:
                jac_diff = h_gamma_jac_pred - h_gamma_jac_gt
                l1 = jnp.mean(jac_diff ** 2)
                losses['spatial_l1'] = l1
                losses['spatial_total'] += self.lambda_spatial_1 * l1

        return losses

    def total_loss(
        self,
        adm_pred: ADMVariables,
        adm_gt: ADMVariables,
        include_derivatives: bool = False,
    ) -> Dict[str, jax.Array]:
        """
        Compute total GEF loss from ADM variables.

        Args:
            adm_pred: Predicted ADM variables
            adm_gt: Ground truth ADM variables
            include_derivatives: Whether to include derivative losses

        Returns:
            Dictionary with all loss components
        """
        # Convert to log domains
        log_N_pred = jnp.log(adm_pred.lapse + 1e-10)
        log_N_gt = jnp.log(adm_gt.lapse + 1e-10)

        h_gamma_pred = matrix_log_safe(adm_pred.spatial_metric)
        h_gamma_gt = matrix_log_safe(adm_gt.spatial_metric)

        # Compute individual losses
        lapse_losses = self.lapse_loss(log_N_pred, log_N_gt, include_derivatives=False)
        shift_losses = self.shift_loss(adm_pred.shift, adm_gt.shift, include_derivatives=False)
        spatial_losses = self.spatial_metric_loss(h_gamma_pred, h_gamma_gt, include_derivatives=False)

        # Combine all losses
        total = (
            lapse_losses['lapse_total'] +
            shift_losses['shift_total'] +
            spatial_losses['spatial_total']
        )

        return {
            **lapse_losses,
            **shift_losses,
            **spatial_losses,
            'total': total
        }


def create_loss_function(
    model,
    metric_gt_fn: Callable[[jax.Array], jax.Array],
    loss_config: Dict = None
) -> Callable:
    """
    Create a loss function for training GEF model.

    Args:
        model: GEF model instance
        metric_gt_fn: Function that returns ground truth metric at a point
        loss_config: Dictionary with loss hyperparameters

    Returns:
        Loss function with signature (params, batch) -> (loss, metrics)
    """
    if loss_config is None:
        loss_config = {}

    gef_loss = GEFLoss(**loss_config)

    def loss_fn(params, batch):
        """
        Compute loss for a batch of points.

        Args:
            params: Model parameters
            batch: Dictionary with 'coords' key containing points (N, 4)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        coords = batch['coords']  # (N, 4)

        # Get predicted ADM variables
        adm_pred = model.apply(params, coords, method=model.get_adm_variables)

        # Get ground truth metrics and extract ADM
        g_gt = jax.vmap(metric_gt_fn)(coords)  # (N, 4, 4)
        adm_gt = extract_adm_from_4d_metric(g_gt)

        # Compute losses
        losses = gef_loss.total_loss(adm_pred, adm_gt, include_derivatives=False)

        return losses['total'], losses

    return loss_fn


if __name__ == '__main__':
    print("Testing GEF Loss Functions...")

    # Test with simple data
    print("\n--- Test 1: Basic Loss Computation ---")

    # Create dummy ADM variables
    N_pred = jnp.array([1.0, 0.9, 1.1])
    N_gt = jnp.array([1.0, 1.0, 1.0])

    beta_pred = jnp.zeros((3, 3))
    beta_gt = jnp.zeros((3, 3))

    gamma_pred = jnp.stack([jnp.eye(3)] * 3)
    gamma_gt = jnp.stack([jnp.eye(3)] * 3)

    adm_pred = ADMVariables(lapse=N_pred, shift=beta_pred, spatial_metric=gamma_pred)
    adm_gt = ADMVariables(lapse=N_gt, shift=beta_gt, spatial_metric=gamma_gt)

    # Compute loss
    gef_loss = GEFLoss()
    losses = gef_loss.total_loss(adm_pred, adm_gt)

    print("Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value:.6f}")

    # Test with Schwarzschild metric
    print("\n--- Test 2: Schwarzschild Metric Loss ---")
    from general_relativity.metrics.schwarzschild import schwarzschild_metric_spherical

    M = 1.0
    r = 5.0
    coords = jnp.array([
        [0.0, 5.0, jnp.pi/2, 0.0],
        [0.0, 6.0, jnp.pi/2, 0.0],
        [0.0, 7.0, jnp.pi/2, 0.0],
    ])

    # Ground truth
    g_gt = jax.vmap(lambda c: schwarzschild_metric_spherical(c, M))(coords)
    adm_gt_schw = extract_adm_from_4d_metric(g_gt)

    # Perturbed prediction (add noise)
    key = jax.random.PRNGKey(0)
    noise_scale = 0.01

    N_pred_schw = adm_gt_schw.lapse + jax.random.normal(key, adm_gt_schw.lapse.shape) * noise_scale
    beta_pred_schw = adm_gt_schw.shift + jax.random.normal(jax.random.PRNGKey(1), adm_gt_schw.shift.shape) * noise_scale
    gamma_pred_schw = adm_gt_schw.spatial_metric + jax.random.normal(jax.random.PRNGKey(2), adm_gt_schw.spatial_metric.shape) * noise_scale

    adm_pred_schw = ADMVariables(lapse=N_pred_schw, shift=beta_pred_schw, spatial_metric=gamma_pred_schw)

    losses_schw = gef_loss.total_loss(adm_pred_schw, adm_gt_schw)

    print("Schwarzschild loss components:")
    for key, value in losses_schw.items():
        print(f"  {key}: {value:.6f}")
