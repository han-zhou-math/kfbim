# IIM Defect Correction for the 2D Interface Poisson Problem

## Problem Statement

Solve $-\Delta u = f$ on $\Omega = [0,1]^2$ with homogeneous Dirichlet BC, where
$u$ and $f$ are piecewise smooth across an interface $\Gamma$.

### Jump Convention

Define the jump as **interior minus exterior**:
$$
\boxed{ [u]_\Gamma = u_{int} - u_{ext} }
$$
where $u_{int}$ is the smooth branch on the interior (label 1) and $u_{ext}$ is the
smooth branch on the exterior (label 0). Similarly,
$$
[\partial_n u]_\Gamma = \partial_n u_{int} - \partial_n u_{ext}
$$

The solution is $u = u_{int}$ inside $\Gamma$ and $u = u_{ext}$ outside.

## Regular vs Irregular Nodes

- **Regular node**: all four face-adjacent neighbors are on the same side of $\Gamma$.  
  Standard FDM is $O(h^2)$ accurate.

- **Irregular node**: at least one neighbor is on the opposite side.  
  The stencil picks up a value from the wrong branch, introducing an $O(1/h^2)$ error that destroys convergence.

## Correction Function

Define the **correction function** (smooth everywhere):

$$
C(\mathbf{x}) = u_{int}(\mathbf{x}) - u_{ext}(\mathbf{x})
$$

At every grid node, regardless of which side contains the node,

$$
C_n = C(\mathbf{x}_n)
    = u_{int}(\mathbf{x}_n) - u_{ext}(\mathbf{x}_n).
$$

## Defect Correction Derivation

Let the domain labels be $\ell_n \in \{0,1\}$ (0 = exterior, 1 = interior).

The corrected right-hand side $F_n$ for $-\Delta_h u = F_n$ is:

$$
\boxed{
  F_n = f_n + \sum_{\substack{n_b \sim n \\ \ell_{n_b} \neq \ell_n}}
        \frac{\ell_n - \ell_{n_b}}{h^2}\, C(\mathbf{x}_{n_b})
}
$$

### Sign Verification

1.  **Node $n$ is exterior ($\ell_n=0$), neighbor $n_b$ is interior ($\ell_{n_b}=1$):**
    *   Stencil at $n$ needs $u_{ext}(x_{n_b})$.
    *   Actual value is $u(x_{n_b}) = u_{int}(x_{n_b}) = u_{ext}(x_{n_b}) + C(x_{n_b})$.
    *   The stencil picks up an extra $+C(x_{n_b})/h^2$ term (note the $-1/h^2$ weight for neighbors in the Laplacian).
    *   To correct $-\Delta u = f$, we must subtract $C/h^2$ from the RHS.
    *   Formula gives: $(\ell_n - \ell_{n_b}) = (0 - 1) = -1 \Rightarrow -C/h^2$. Correct.

2.  **Node $n$ is interior ($\ell_n=1$), neighbor $n_b$ is exterior ($\ell_{n_b}=0$):**
    *   Stencil at $n$ needs $u_{int}(x_{n_b})$.
    *   Actual value is $u(x_{n_b}) = u_{ext}(x_{n_b}) = u_{int}(x_{n_b}) - C(x_{n_b})$.
    *   The stencil is short by $C(x_{n_b})/h^2$.
    *   To correct, we must add $C/h^2$ to the RHS.
    *   Formula gives: $(\ell_n - \ell_{n_b}) = (1 - 0) = +1 \Rightarrow +C/h^2$. Correct.

## Restrict (Interpolation)

To recover the interior trace $u_{int}$ at an interface point $x_q$ from bulk values $u$:
*   If a stencil node $x_{n}$ is interior ($\ell_n=1$), use $u(x_n)$ directly.
*   If a stencil node $x_{n}$ is exterior ($\ell_n=0$), use $u_{int}(x_n) = u(x_n) + C(x_n)$.
*   Then fit the polynomial to these "un-jumped" values.

## Manufactured Solution (Test Case)

Box $[0,1]^2$, star interface.

| Region  | Exact solution | $-\Delta u$ |
|---------|---------------|-------------|
| Interior (label 1) | $u_{int} = \sin(2\pi x)\sin(2\pi y)$ | $f_{int} = 8\pi^2 u_{int}$ |
| Exterior (label 0) | $u_{ext} = \sin(\pi x)\sin(\pi y)$ | $f_{ext} = 2\pi^2 u_{ext}$ |

Jump: $[u] = u_{int} - u_{ext} = \sin(2\pi x)\sin(2\pi y) - \sin(\pi x)\sin(\pi y)$.
Note: the exterior branch vanishes on the box boundary.
