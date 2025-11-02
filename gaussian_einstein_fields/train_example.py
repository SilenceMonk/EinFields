"""
Example training script for Gaussian Einstein Fields v3.0

This script demonstrates how to train a GEF model to represent a known
spacetime solution (e.g., Schwarzschild black hole).

The training is performed in "representer" mode, where the goal is to
compress and accurately represent a known analytical solution.
"""

import jax
import jax.numpy as jnp
import optax
from tqdm import trange
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaussian_einstein_fields import (
    GaussianEinsteinFields,
    create_loss_function,
    extract_adm_from_4d_metric,
    GEFLoss
)
from general_relativity.metrics.schwarzschild import schwarzschild_metric_spherical


def create_training_data(
    metric_fn,
    num_points: int,
    domain_bounds: Dict[str, Tuple[float, float]],
    key: jax.Array
) -> Dict[str, jax.Array]:
    """
    Create training dataset by sampling spacetime points.

    Args:
        metric_fn: Function that returns metric tensor at a point
        num_points: Number of training points
        domain_bounds: Dictionary with bounds for each coordinate
            e.g., {'t': (0, 10), 'r': (3, 20), 'theta': (0, pi), 'phi': (0, 2pi)}
        key: JAX random key

    Returns:
        Dictionary with 'coords' and 'metrics'
    """
    keys = jax.random.split(key, 4)

    # Sample coordinates uniformly in bounds
    t = jax.random.uniform(keys[0], (num_points,),
                          minval=domain_bounds['t'][0],
                          maxval=domain_bounds['t'][1])
    r = jax.random.uniform(keys[1], (num_points,),
                          minval=domain_bounds['r'][0],
                          maxval=domain_bounds['r'][1])
    theta = jax.random.uniform(keys[2], (num_points,),
                              minval=domain_bounds['theta'][0],
                              maxval=domain_bounds['theta'][1])
    phi = jax.random.uniform(keys[3], (num_points,),
                            minval=domain_bounds['phi'][0],
                            maxval=domain_bounds['phi'][1])

    coords = jnp.stack([t, r, theta, phi], axis=-1)  # (num_points, 4)

    # Compute ground truth metrics
    metrics = jax.vmap(metric_fn)(coords)  # (num_points, 4, 4)

    return {
        'coords': coords,
        'metrics': metrics
    }


def train_step(params, opt_state, batch, model, optimizer, loss_fn):
    """Single training step."""
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, metrics


def train_gef_model(
    model,
    metric_fn,
    num_epochs: int = 1000,
    num_train_points: int = 1000,
    num_val_points: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    domain_bounds: Dict = None,
    loss_config: Dict = None,
    seed: int = 42
):
    """
    Train GEF model to represent a known spacetime metric.

    Args:
        model: GaussianEinsteinFields model instance
        metric_fn: Function returning metric tensor at a point
        num_epochs: Number of training epochs
        num_train_points: Number of training points
        num_val_points: Number of validation points
        batch_size: Batch size for training
        learning_rate: Learning rate
        domain_bounds: Bounds for coordinate sampling
        loss_config: Loss function configuration
        seed: Random seed

    Returns:
        Trained parameters and training history
    """
    # Set default domain bounds if not provided
    if domain_bounds is None:
        domain_bounds = {
            't': (0.0, 1.0),
            'r': (3.0, 20.0),
            'theta': (0.1, jnp.pi - 0.1),
            'phi': (0.0, 2 * jnp.pi)
        }

    # Initialize
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    # Create datasets
    print("Creating training dataset...")
    train_data = create_training_data(metric_fn, num_train_points, domain_bounds, subkey)

    key, subkey = jax.random.split(key)
    val_data = create_training_data(metric_fn, num_val_points, domain_bounds, subkey)

    # Initialize model
    print("Initializing model...")
    key, subkey = jax.random.split(key)
    sample_point = train_data['coords'][0]
    params = model.init(subkey, sample_point)

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {param_count:,}")

    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Setup loss function
    loss_fn = create_loss_function(model, metric_fn, loss_config)

    # JIT compile training step
    train_step_jit = jax.jit(lambda p, o, b: train_step(p, o, b, model, optimizer, loss_fn))

    # Training loop
    print("Starting training...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'lapse_loss': [],
        'shift_loss': [],
        'spatial_loss': []
    }

    for epoch in trange(num_epochs, desc="Training"):
        # Shuffle training data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, num_train_points)
        train_coords_shuffled = train_data['coords'][perm]

        # Mini-batch training
        epoch_losses = []
        num_batches = num_train_points // batch_size

        for i in range(num_batches):
            batch_coords = train_coords_shuffled[i*batch_size:(i+1)*batch_size]
            batch = {'coords': batch_coords}

            params, opt_state, loss, metrics = train_step_jit(params, opt_state, batch)
            epoch_losses.append(loss)

        # Record metrics
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        history['train_loss'].append(float(avg_train_loss))

        # Validation
        if epoch % 10 == 0:
            val_batch = {'coords': val_data['coords']}
            val_loss, val_metrics = loss_fn(params, val_batch)
            history['val_loss'].append(float(val_loss))
            history['lapse_loss'].append(float(val_metrics.get('lapse_total', 0.0)))
            history['shift_loss'].append(float(val_metrics.get('shift_total', 0.0)))
            history['spatial_loss'].append(float(val_metrics.get('spatial_total', 0.0)))

            if epoch % 100 == 0:
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {avg_train_loss:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print(f"  Lapse: {val_metrics.get('lapse_total', 0.0):.6f}")
                print(f"  Shift: {val_metrics.get('shift_total', 0.0):.6f}")
                print(f"  Spatial: {val_metrics.get('spatial_total', 0.0):.6f}")

    return params, history


def plot_training_history(history: Dict):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot total loss
    axes[0].plot(history['train_loss'], label='Train', alpha=0.7)
    if len(history['val_loss']) > 0:
        val_epochs = jnp.arange(0, len(history['train_loss']), 10)[:len(history['val_loss'])]
        axes[0].plot(val_epochs, history['val_loss'], label='Val', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Plot component losses
    if len(history['lapse_loss']) > 0:
        val_epochs = jnp.arange(0, len(history['train_loss']), 10)[:len(history['lapse_loss'])]
        axes[1].plot(val_epochs, history['lapse_loss'], label='Lapse', marker='o')
        axes[1].plot(val_epochs, history['shift_loss'], label='Shift', marker='s')
        axes[1].plot(val_epochs, history['spatial_loss'], label='Spatial', marker='^')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Component Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("=" * 60)
    print("Gaussian Einstein Fields v3.0 - Training Example")
    print("=" * 60)

    # Configuration
    M = 1.0  # Black hole mass

    # Create metric function
    def metric_fn(coords):
        return schwarzschild_metric_spherical(coords, M)

    # Create model
    print("\nCreating GEF model...")
    model = GaussianEinsteinFields(
        num_gaussians_lapse=50,
        num_gaussians_shift=50,
        num_gaussians_spatial=50,
        init_scale=2.0
    )

    # Domain bounds (avoiding horizon at r=2M)
    domain_bounds = {
        't': (0.0, 1.0),
        'r': (2.5, 15.0),  # Outside Schwarzschild radius r=2M
        'theta': (0.2, jnp.pi - 0.2),
        'phi': (0.0, 2 * jnp.pi)
    }

    # Loss configuration
    loss_config = {
        'lambda_lapse_0': 1.0,
        'lambda_lapse_1': 0.0,  # Start without derivatives
        'lambda_lapse_2': 0.0,
        'lambda_shift_0': 1.0,
        'lambda_shift_1': 0.0,
        'lambda_shift_2': 0.0,
        'lambda_spatial_0': 1.0,
        'lambda_spatial_1': 0.0,
        'lambda_spatial_2': 0.0,
    }

    # Train model
    params, history = train_gef_model(
        model=model,
        metric_fn=metric_fn,
        num_epochs=500,
        num_train_points=2000,
        num_val_points=500,
        batch_size=128,
        learning_rate=1e-3,
        domain_bounds=domain_bounds,
        loss_config=loss_config,
        seed=42
    )

    # Plot results
    print("\nPlotting training history...")
    fig = plot_training_history(history)
    plt.savefig('/home/user/EinFields/gef_training_history.png', dpi=150, bbox_inches='tight')
    print("Saved training history plot to: gef_training_history.png")

    # Test on a few points
    print("\n" + "=" * 60)
    print("Testing trained model...")
    print("=" * 60)

    test_coords = jnp.array([
        [0.0, 5.0, jnp.pi/2, 0.0],
        [0.0, 10.0, jnp.pi/2, 0.0],
        [0.5, 7.0, jnp.pi/4, jnp.pi/2],
    ])

    for i, coord in enumerate(test_coords):
        print(f"\nTest point {i+1}: t={coord[0]:.1f}, r={coord[1]:.1f}, θ={coord[2]:.2f}, φ={coord[3]:.2f}")

        # Ground truth
        g_gt = metric_fn(coord)
        adm_gt = extract_adm_from_4d_metric(g_gt)

        # Prediction
        g_pred = model.apply(params, coord)
        adm_pred = model.apply(params, coord, method=model.get_adm_variables)

        # Compare
        print(f"  Lapse    - GT: {adm_gt.lapse:.6f}, Pred: {adm_pred.lapse:.6f}, Error: {abs(adm_gt.lapse - adm_pred.lapse):.6e}")
        print(f"  Shift    - GT: {jnp.linalg.norm(adm_gt.shift):.6f}, Pred: {jnp.linalg.norm(adm_pred.shift):.6f}")
        print(f"  Metric   - Frobenius error: {jnp.linalg.norm(g_gt - g_pred, 'fro'):.6e}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
