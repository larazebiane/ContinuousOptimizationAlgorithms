# uncMIN: Unconstrained Optimization Solver with Steepest Descent, Modified Newton, and Truncated CG

This folder contains a MATLAB implementation of an **unconstrained optimization solver** that supports three direction strategies:

- Steepest Descent  
- Modified Newton  
- Truncated Newton Conjugate Gradient (CG)

It solves smooth unconstrained problems of the form:

$$
\min_{x \in \mathbb{R}^n} \quad f(x)
$$

where $ f: \mathbb{R}^n \to \mathbb{R} $ is a twice-differentiable objective function.

---

## 🚀 Features

- Supports **multiple search directions**, including:
  - Classic steepest descent
  - Modified Newton with eigenvalue regularization
  - Truncated Newton-CG using Hessian-vector products
- Uses **line search** satisfying Wolfe conditions
- Outputs detailed information about the optimization process
- Modular design for easy testing of custom objective functions

---

## 📂 Files

- `uncMIN.m`: Main unconstrained minimization solver supporting different direction strategies.
- `cg_ls.m`: Conjugate gradient method for solving linear systems (used in Newton-CG).
- `ModNewton.m`: Modified Newton matrix constructor to ensure positive definiteness.
- `README.md`: This file explaining the folder contents.

---

## 📌 Usage

```matlab
[x, info] = uncMIN(fun_hands, x0, params);
```

### 📥 Inputs

- `fun_hands` — Structure of function handles:
  - `f_hand(x)` — Objective function $ f(x) $
  - `g_hand(x)` — Gradient $ \nabla f(x) $
  - `H_hand(x)` — Hessian $ \nabla^2 f(x) $
  - `Hv_hand(x, v)` — Hessian-vector product $ \nabla^2 f(x) \cdot v $

- `x0` — Initial guess vector $(n \times 1)$

- `params` — Structure with fields:
  - `dir_type` — Search direction type (`'SteepestDescent'`, `'ModifiedNewton'`, `'NewtonCG'`)
  - `maxit` — Maximum number of iterations
  - `tol` — Convergence tolerance
  - `printlevel` — 0 (silent) or 1 (print iteration info)

---

### 📤 Outputs

- `x` — Final solution vector

- `info` — Structure with the following fields:
  - `f` — Final objective value
  - `g` — Final gradient vector
  - `norm_g` — Norm of the final gradient
  - `x` — Final iterate (same as `x` output)
  - `f_evals` — Number of function evaluations
  - `g_evals` — Number of gradient evaluations
  - `H_evals` — Number of Hessian evaluations
  - `Hv_evals` — Number of Hessian-vector evaluations
  - `iter` — Number of iterations performed
  - `status` — `0` if converged, `1` if max iterations reached
  - `sta` — Status flag from the CG solver (if used)
  - `f_values` — Objective function values per iteration

---

## Example

See the `test_uncMIN.m` script for a usage example with random problem data.

---

## License

This code is licensed under the [MIT License](../LICENSE). 
