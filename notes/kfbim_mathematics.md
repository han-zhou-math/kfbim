# Mathematical Foundation of the Kernel-Free Boundary Integral Method (KFBIM)

This note documents the mathematical framework for the elliptic PDE solvers in this repository.

## 1. Geometry Representation

The domain $\Omega \subset \mathbb{R}^d$ is partitioned by an interface $\Gamma$ into an interior region $\Omega_{int}$ (domain label 1) and an exterior region $\Omega_{ext}$ (domain label 0).

*   **Normals $\mathbf{n}$**: Unit outward normal pointing from $\Omega_{int}$ to $\Omega_{ext}$.
*   **Jump Convention**: The jump of a function $u$ across $\Gamma$ is defined as **interior minus exterior**:
    $$ [u]_\Gamma = u_{int} - u_{ext} $$
    $$ [\partial_n u]_\Gamma = \partial_n u_{int} - \partial_n u_{ext} $$

## 2. The Constant Interface Problem (Poisson)

Consider the Poisson equation with piecewise smooth components:
$$ -\Delta u = f \quad \text{in } \Omega_{int} \cup \Omega_{ext} $$
subject to jump conditions $[u] = \mu$ and $[\partial_n u] = \sigma$.

### Representation Formula
The solution $u$ is given by:
$$ u(\mathbf{x}) = \mathcal{S} \sigma(\mathbf{x}) - \mathcal{D} \mu(\mathbf{x}) + \mathcal{V} f(\mathbf{x}) $$
where $\mathcal{S}$ is the single-layer potential, $\mathcal{D}$ the double-layer potential, and $\mathcal{V}$ the volume potential.

### Boundary Traces
The traces on the interior ($u_{int}$) and exterior ($u_{ext}$) are:
$$ u_{int}|_\Gamma = S\sigma - (K - \tfrac{1}{2}I)\mu + Vf $$
$$ u_{ext}|_\Gamma = S\sigma - (K + \tfrac{1}{2}I)\mu + Vf $$
where $S$ is the single-layer operator and $K$ is the double-layer operator.

## 3. KFBIM "Interface Solver" Mapping

The `LaplaceInterfaceSolver2D::solve(jumps, f)` function takes jumps $\mu$ (`u_jump`) and $\sigma$ (`un_jump`) and returns the **averaged interface values** $u_{avg}$ and $(\partial_n u)_{avg}$.

$$ u_{avg} = \frac{u_{int} + u_{ext}}{2} = u_{int} - \frac{\mu}{2} $$
$$ (\partial_n u)_{avg} = \frac{\partial_n u_{int} + \partial_n u_{ext}}{2} = \partial_n u_{int} - \frac{\sigma}{2} $$

$$ \text{Solve}(\mu, \sigma, f) \to (u_{avg}, (\partial_n u)_{avg}) $$

From the representation formula, we have the following operational mappings:

| Input Jumps $(\mu, \sigma)$ | Bulk Forcing $f$ | Output ($u_{avg}$) | Mathematical Operator |
| :--- | :--- | :--- | :--- |
| $(0, \phi)$ | $0$ | $u_{avg}$ | $S\phi$ (Single Layer) |
| $(\phi, 0)$ | $0$ | $u_{avg}$ | $-K\phi$ (Double Layer average) |
| $(0, 0)$ | $f$ | $u_{avg}$ | $Vf$ (Volume Potential average) |

## 4. Boundary Integral Equations (BIEs)

BVPs are solved by finding a density $\phi$ such that the representation formula satisfies the boundary condition.

### Interior Dirichlet BVP ($u_{int} = g$ on $\Gamma$)
*   **Representation**: $u = \mathcal{D}\phi - \mathcal{V}f$.
*   **BIE**: $(\tfrac{1}{2}I - K)\phi = g + Vf|_\Gamma$.
    *   `LaplaceKFBIMode::Dirichlet` in code.
    *   Solve via GMRES: `op.apply(phi)` adds the $\tfrac{1}{2}I$ term to the solver's $-K\phi$ output to reconstruct $u_{int}$.

### Interior Neumann BVP ($\partial_n u_{int} = g$ on $\Gamma$)
*   **Representation**: $u = \mathcal{S}\phi - \mathcal{V}f$.
*   **BIE**: $(\tfrac{1}{2}I + K')\phi = g + \partial_n Vf|_\Gamma$.
    *   `LaplaceKFBIMode::Neumann` in code.
    *   Solve via GMRES: `op.apply(phi)` adds the $\tfrac{1}{2}I$ term to the solver's $K'\phi$ output to reconstruct $\partial_n u_{int}$.

## 5. Summary of KFBIM Operator Modes

| Mode | Ansatz (Jump) | Output Calculation | Effective BIE Operator |
| :--- | :--- | :--- | :--- |
| `Dirichlet` | $\mu = \phi, \sigma = 0$ | $u_{avg} + \mu/2$ | $\tfrac{1}{2}I - K$ |
| `Neumann` | $\sigma = \phi, \mu = 0$ | $(\partial_n u)_{avg} + \sigma/2$ | $\tfrac{1}{2}I + K'$ |
