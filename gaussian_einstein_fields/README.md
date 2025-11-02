# Gaussian Einstein Fields v3.0

## A Physically-Consistent Spacetime Representation via 3+1 Decomposition

---

## Overview

**Gaussian Einstein Fields (GEF) v3.0** is a novel neural field approach for representing 4D spacetime metrics using **4D Gaussian basis functions** and the **ADM (Arnowitt-Deser-Misner) 3+1 formalism**.

Inspired by recent advances in 4D Gaussian Splatting for dynamic scenes, GEF v3.0 leverages explicit primitives (4D Gaussians) to represent spacetime in a way that **analytically guarantees physical consistency**.

### Key Innovation: 3+1 Decomposition

Instead of directly representing the 4D metric tensor `g_μν`, we decompose it into three physically meaningful components:

**ds² = -N² dt² + γ_ij (dx^i + β^i dt)(dx^j + β^j dt)**

Where:
- **N(t, x^k)**: **Lapse function** (scalar field, N > 0) - describes proper time flow
- **β^i(t, x^k)**: **Shift vector** (3D vector field) - describes coordinate dragging
- **γ_ij(t, x^k)**: **Spatial 3-metric** (3×3 SPD matrix field) - describes spatial geometry

### Why This Approach?

1. **Analytic Physical Guarantees**:
   - Lapse positivity (N > 0) enforced via log-domain representation
   - Spatial metric SPD property enforced via matrix exponential
   - Lorentzian signature (-,+,+,+) guaranteed by construction

2. **Problem Decomposition**:
   - Breaks complex 4D tensor field into simpler components
   - Each field has clear physical interpretation
   - Enables component-specific loss functions and regularization

3. **Compatibility with Numerical Relativity**:
   - Aligns with standard ADM/BSSN formalism
   - Direct path to evolution equations for solver mode
   - Easy comparison with traditional methods

---

## Architecture

### Three Independent Gaussian Fields

Each ADM component is represented by an independent field of 4D Gaussian primitives:

#### 1. **Lapse Field** N(p)

Since N > 0 is required, we model log(N):

```
log N(p) = Σᵢ [αᵢᴺ * G(p; μᵢ, Σᵢ)]
N(p) = exp(log N(p))
```

- **Parameters**: Gaussian centers μᵢ, quaternions (qₗ, qᵣ), scales sᵢ, scalar amplitudes αᵢᴺ
- **Guarantee**: N > 0 always satisfied

#### 2. **Shift Field** β^i(p)

Direct representation as 3D vector field:

```
β^i(p) = Σᵢ [vᵢᵝ * G(p; μᵢ, Σᵢ)]
```

- **Parameters**: Gaussian centers μᵢ, quaternions (qₗ, qᵣ), scales sᵢ, 3D vector templates vᵢᵝ
- **Output**: 3D vector at each point

#### 3. **Spatial Metric Field** γ_ij(p)

Matrix exponential ensures SPD:

```
h_γ(p) = Σᵢ [αᵢᵞ * (vᵢᵞ ⊗ vᵢᵞ) * G(p; μᵢ, Σᵢ)]
γ(p) = expm(h_γ(p))
```

- **Parameters**: Gaussian centers μᵢ, quaternions (qₗ, qᵣ), scales sᵢ, 3D vectors vᵢᵞ, scalar amplitudes αᵢᵞ
- **Guarantee**: γ is always symmetric positive definite

### 4D Gaussian Primitives

Each Gaussian G(p; μ, Σ) is parameterized by:
- **μ**: Center in 4D spacetime (4 values)
- **qₗ, qᵣ**: Pair of quaternions for 4D rotations (8 values)
- **s**: Scaling factors (4 values)

Covariance: **Σ = R_L · S · S^T · R_L^T**

---

## Training Paradigm: Representer Mode

GEF v3.0 first focuses on **learning known solutions** (compression/representation task).

### Loss Function

Sobolev losses in natural domains for each component:

```
L = L_lapse + L_shift + L_spatial
```

**Lapse loss** (in log domain):
```
L_N = 𝔼_p [ || log N(p) - log N_gt(p) ||² + λ₁ || ∂(log N) - ∂(log N_gt) ||² ]
```

**Shift loss**:
```
L_β = 𝔼_p [ || β(p) - β_gt(p) ||² + λ₁ || ∂β - ∂β_gt ||² ]
```

**Spatial metric loss** (in log domain):
```
L_γ = 𝔼_p [ || h_γ(p) - logm(γ_gt(p)) ||²_F + λ₁ || ∂h_γ - ∂(logm(γ_gt)) ||²_F ]
```

### Adaptive Refinement

- **Densification**: Add Gaussians in high-error regions
- **Pruning**: Remove low-contribution Gaussians
- **Component-aware**: Independent adaptation for each field

---

## Installation

```bash
# Already included in EinFields repository
cd /path/to/EinFields
export PYTHONPATH=/path/to/EinFields:$PYTHONPATH
```

### Dependencies

- JAX (≥0.4.0)
- Flax (≥0.7.0)
- Optax (≥0.1.0)
- NumPy
- Matplotlib (for visualization)

---

## Quick Start

### Basic Usage

```python
import jax
import jax.numpy as jnp
from gaussian_einstein_fields import GaussianEinsteinFields

# Create model
model = GaussianEinsteinFields(
    num_gaussians_lapse=100,
    num_gaussians_shift=100,
    num_gaussians_spatial=100,
    init_scale=2.0
)

# Initialize
key = jax.random.PRNGKey(0)
p = jnp.array([0.0, 5.0, jnp.pi/2, 0.0])  # (t, r, θ, φ)
params = model.init(key, p)

# Evaluate metric
g = model.apply(params, p)  # Shape: (4, 4)
print(f"4D metric:\n{g}")

# Get ADM components
adm = model.apply(params, p, method=model.get_adm_variables)
print(f"Lapse: {adm.lapse}")
print(f"Shift: {adm.shift}")
print(f"Spatial metric:\n{adm.spatial_metric}")
```

### Training on Schwarzschild Metric

```python
from gaussian_einstein_fields import create_loss_function
from general_relativity.metrics.schwarzschild import schwarzschild_metric_spherical

# Define ground truth
M = 1.0
metric_fn = lambda coords: schwarzschild_metric_spherical(coords, M)

# Create loss function
loss_fn = create_loss_function(model, metric_fn)

# Training loop (simplified)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

for epoch in range(1000):
    # Sample batch
    coords = sample_spacetime_points(batch_size=256)
    batch = {'coords': coords}

    # Compute loss and gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)

    # Update
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
```

See `train_example.py` for complete training script.

---

## Module Structure

```
gaussian_einstein_fields/
├── __init__.py                 # Module exports
├── gaussian_primitives.py      # 4D Gaussian basis functions
├── adm_decomposition.py        # 3+1 ADM utilities
├── gef_model.py               # Main GEF v3.0 model
├── losses.py                  # Training loss functions
├── train_example.py           # Example training script
└── README.md                  # This file
```

### Key Classes

- **`GaussianEinsteinFields`**: Main model combining all three fields
- **`LapseField`**: Lapse function N(p) representation
- **`ShiftField`**: Shift vector β^i(p) representation
- **`SpatialMetricField`**: Spatial metric γ_ij(p) representation
- **`ADMVariables`**: Container for ADM decomposition
- **`GEFLoss`**: Loss function for training

### Key Functions

- **`extract_adm_from_4d_metric(g)`**: Extract ADM from metric tensor
- **`reconstruct_4d_metric_from_adm(N, β, γ)`**: Rebuild 4D metric
- **`create_loss_function(model, metric_fn)`**: Create training loss
- **`check_adm_constraints(adm)`**: Verify physical constraints

---

## Examples

### Example 1: Testing ADM Decomposition

```python
from gaussian_einstein_fields import (
    extract_adm_from_4d_metric,
    reconstruct_4d_metric_from_adm,
    verify_reconstruction
)

# Minkowski metric
g_minkowski = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))

# Extract ADM
adm = extract_adm_from_4d_metric(g_minkowski)
print(f"Lapse: {adm.lapse}")  # Should be 1.0
print(f"Shift: {adm.shift}")  # Should be [0, 0, 0]

# Reconstruct
g_recon = reconstruct_4d_metric_from_adm(adm.lapse, adm.shift, adm.spatial_metric)
print(f"Reconstruction matches: {verify_reconstruction(g_minkowski, g_recon)}")
```

### Example 2: Single Field Testing

```python
from gaussian_einstein_fields import LapseField

# Create lapse field
lapse_field = LapseField(num_gaussians=50)

# Initialize
key = jax.random.PRNGKey(0)
p = jnp.array([0.0, 5.0, jnp.pi/2, 0.0])
params = lapse_field.init(key, p)

# Evaluate
N = lapse_field.apply(params, p)
print(f"Lapse value: {N}")  # Always > 0

# Batch evaluation
p_batch = jax.random.normal(key, (100, 4))
N_batch = lapse_field.apply(params, p_batch)
print(f"Batch lapse shape: {N_batch.shape}")  # (100,)
print(f"All positive: {jnp.all(N_batch > 0)}")  # True
```

---

## Testing

Run tests for each module:

```bash
# Test Gaussian primitives
python gaussian_primitives.py

# Test ADM decomposition
python adm_decomposition.py

# Test main model
python gef_model.py

# Test losses
python losses.py

# Run full training example
python train_example.py
```

---

## Future Directions

### Solver Mode (Ongoing)

Extend to solve Einstein's equations from initial conditions:

- **Constraint equations**: Hamiltonian and momentum constraints
- **Evolution equations**: Time evolution of γ_ij and K_ij (extrinsic curvature)
- **Physics-informed loss**: Enforce Einstein field equations

### Advanced Features

- [ ] Adaptive Gaussian refinement (densification/pruning)
- [ ] Support for matter fields (Tμν)
- [ ] Multiple coordinate systems
- [ ] Checkpoint saving/loading
- [ ] Visualization tools for ADM variables
- [ ] Integration with existing EinFields training pipeline

---

## Citation

If you use this code, please cite:

```bibtex
@misc{gef_v3_2025,
  title={Gaussian Einstein Fields v3.0: A Physically-Consistent Spacetime Representation via 3+1 Decomposition},
  author={GEF Contributors},
  year={2025},
  howpublished={https://github.com/SilenceMonk/EinFields}
}
```

And the original EinFields paper:

```bibtex
@misc{cranganore2025einsteinfieldsneuralperspective,
  title={Einstein Fields: A Neural Perspective To Computational General Relativity},
  author={Sandeep Suresh Cranganore and Andrei Bodnar and Arturs Berzins and Johannes Brandstetter},
  year={2025},
  eprint={2507.11589},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.11589}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## Acknowledgments

This work builds on:
- **EinFields**: Neural field framework for GR
- **4D Gaussian Splatting**: Inspiration for explicit 4D primitives
- **ADM Formalism**: Standard 3+1 decomposition in numerical relativity
- **JAX/Flax**: Automatic differentiation and neural networks

---

**GEF v3.0** - Bridging neural fields and numerical relativity through physically-consistent representations.
