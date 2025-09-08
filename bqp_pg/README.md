# bqp_pg: Projected Gradient Method for Bound-Constrained Quadratic Optimization

This folder contains a MATLAB implementation of a **projected gradient method** for solving bound-constrained quadratic optimization problems of the form:

$$
\min_{x \in \mathbb{R}^n} \quad q(x) = c^T x + \frac{1}{2} x^T H x \quad \text{subject to} \quad \ell \leq x \leq u
$$

where $H$ is symmetric positive definite, and $\ell, u$ are the lower and upper bounds.

---

## Features

- Implements an **inexact Cauchy step** strategy with a backtracking line search.
- Supports optional **subspace acceleration** (based on active-set identification).
- Returns dual variables, KKT residuals, objective values, and convergence status.
- Designed for use in continuous optimization and machine learning projects.

---

## Usage

```matlab
[x, info] = bqp_pg(c, H, l, u, x0, params);

```

### Inputs:

- `c` — Coefficient vector ($n \times 1$)
- `H` — Symmetric positive definite matrix ($n \times n$)
- `l` — Lower bound vector ($n \times 1$)
- `u` — Upper bound vector ($n \times 1$)
- `x0` — Initial guess ($n \times 1$)
- `params` — Structure with fields:
  - `maxit`: Maximum iterations
  - `tol`: Convergence tolerance
  - `ssm`: (0 or 1) Enable subspace acceleration
  - `printlevel`: (0 or 1) Display iteration info

### Outputs:

- `x` — Solution vector
- `info` — Structure containing:
  - `obj`: Objective value at solution
  - `iter`: Number of iterations
  - `res`: KKT residual vector
  - `zl`, `zu`: Dual variables for lower and upper bounds
  - `status`: 0 if converged, 1 if max iterations reached

---

## Example

See the `test_bqp_pg.m` script for a usage example with random problem data.

---

## License

This code is licensed under the [MIT License](../LICENSE). 
