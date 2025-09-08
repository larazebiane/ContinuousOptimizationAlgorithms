# uncMIN_TR: Trust-Region Method for Unconstrained Optimization

This folder contains a MATLAB implementation of a **trust-region optimization method** for solving unconstrained problems of the form:

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

where $ f $ is twice continuously differentiable. The algorithm supports both:
- **Cauchy step**: a simple projected steepest descent direction
- **Steihaug Conjugate Gradient method**: an iterative approach that solves the trust-region subproblem without forming the Hessian

---

## 🚀 Features

- Supports two step types:
  - `'CauchyStep'`: simple first-order model-based step
  - `'NewtonCG'`: second-order model via Steihaug CG (using `cg_tr.m`)
- No explicit Hessian needed — works with Hessian-vector products
- Tracks iterations, gradient norm, function calls, and termination status
- Flexible, modular code for use in numerical optimization

---

## 📂 Files

- `uncMIN_TR.m`: Main trust-region solver implementing various step types.
- `cg_tr.m`: Conjugate gradient solver for trust-region subproblems (Steihaug CG method).
- `README.md`: This file explaining the folder contents.

---

## 📌 Usage

```matlab
[x, info] = uncMIN_TR(fun_hands, x0, params);
```

### 📥 Inputs:

- `fun_hands` — A structure with the following function handles:
  - `f_hand(x)`: Computes the objective function \( f(x) \)
  - `g_hand(x)`: Computes the gradient \( \nabla f(x) \)
  - `H_hand(x)`: Computes the Hessian \( \nabla^2 f(x) \)
  - `Hv_hand(x, v)`: Computes the Hessian-vector product \( \nabla^2 f(x) \cdot v \)
- `x0` — Initial guess vector (\( n \times 1 \))
- `params` — A structure with fields:
  - `step_type`: `'CauchyStep'` or `'NewtonCG'` (determines trial step type)
  - `maxit`: Maximum number of iterations
  - `tol`: Convergence tolerance
  - `printlevel`: Verbosity level (0: silent, 1: print iteration info)

### 📤 Outputs:

- `x` — Final solution vector
- `info` — A structure containing:
  - `f`: Final objective value
  - `g`: Gradient at final solution
  - `H`: Hessian at final solution (empty if not used)
  - `x`: Final iterate
  - `f_evals`: Number of objective function evaluations
  - `g_evals`: Number of gradient evaluations
  - `H_evals`: Number of Hessian evaluations
  - `Hv_evals`: Number of Hessian-vector products
  - `iter`: Number of iterations
  - `status`: 
    - `0` if converged (tolerance met)
    - `1` if maximum iterations reached
  - `f_values`: Objective value at each iteration

---

## Example

See the `test_uncMIN_TR.m` script for a usage example on test problems from class.

---

## License

This code is licensed under the [MIT License](../LICENSE).



