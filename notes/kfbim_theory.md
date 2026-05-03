# Mathematical Foundation of the Kernel-Free Boundary Integral Method (KFBIM)

This note documents the mathematical framework for the elliptic PDE solvers in this repository.

## Table of Contents
- [1. Geometry Representation](#1-geometry-representation)
- [2. The Constant Interface Problem (Elliptic)](#2-the-constant-interface-problem-elliptic)
- [3. Boundary Integral Operators](#3-boundary-integral-operators)
- [4. Interface Solver Outputs (KFBIM Operator Evaluation)](#4-interface-solver-outputs-kfbim-operator-evaluation)
- [5. Boundary Value Problems (BVPs) to BIEs](#5-boundary-value-problems-bvps-to-bies)
- [6. Generic Interface Problems (Variable Coefficients)](#6-generic-interface-problems-variable-coefficients)
  - [7. BIE for Generic Interface Problems](#7-bie-for-generic-interface-problems)

## 1. Geometry Representation

The domain $\Omega \subset \mathbb{R}^d$ is partitioned by an interface $\Gamma$ into an interior region $\Omega_{int}$ and an exterior region $\Omega_{ext}$.

*   **Interface $\Gamma$**: Represented by a collection of panels.

### 2D: Curve Panels

In 2D, $\Gamma$ is a closed curve represented by isoparametric polynomial panel
segments. Each panel maps the reference interval $[-1,1]$ to $\mathbb{R}^2$. The
panel nodes serve simultaneously as geometry definition and DOF/quadrature
locations. New 2D Laplace work uses Chebyshev-Lobatto panel nodes
$s=\{-1,0,1\}$ by default. The local correction path generates four expansion
centers per panel at $s=\{-0.75,-0.25,0.25,0.75\}$. The older 3-point
Gauss-Legendre panel layout remains available only as an explicit legacy
regression path.

### 3D: Curved Triangle Patches

In 3D, $\Gamma$ is partitioned into curved triangle patches. The design mirrors
the 2D Lobatto-panel pattern: each patch is defined by Lobatto interpolation
nodes on a reference triangle, and local Cauchy corrections are computed at a
set of uniform expansion centers in the patch interior.

**Reference triangle.** Let $\mathcal{T} \subset \mathbb{R}^2$ be the reference
triangle in barycentric coordinates:

$$\mathcal{T} = \{(\lambda_1, \lambda_2, \lambda_3) \mid \lambda_1 + \lambda_2 + \lambda_3 = 1,\; \lambda_i \ge 0\}.$$

Equivalently, in Cartesian coordinates on the reference plane:
$\mathcal{T} = \{(u, v) \mid u \ge 0,\; v \ge 0,\; u+v \le 1\}$ with
$\lambda_1 = 1-u-v$, $\lambda_2 = u$, $\lambda_3 = v$.

**Lobatto geometry nodes.** The Chebyshev-Lobatto distribution on the interval
does not have a unique generalization to the triangle. We use a **warped/blended
Chebyshev-Lobatto** construction that directly parallels the 2D case:

*   **Edge nodes.** On each edge of the reference triangle, place the standard
    1D Chebyshev-Lobatto points (degree $p$): $p+1$ points including both
    endpoints, clustering near vertices as $\cos(k\pi/p)$.
*   **Interior nodes.** Blend the three edge distributions into the triangle
    interior using a Gordon-Hall (transfinite) warp. For each candidate point
    with barycentric coordinates $(\lambda_1, \lambda_2, \lambda_3)$, the warp
    corrects the naive bilinear blend to match the Chebyshev-Lobatto edge data
    exactly.

For the **quadratic** default $p=2$, the 1D Chebyshev-Lobatto nodes are
$\{-1, 0, 1\}$ on each edge, yielding the same 6 nodes as the equidistant set:
3 vertices $(1,0,0)$, $(0,1,0)$, $(0,0,1)$ and 3 edge midpoints
$(\tfrac12,\tfrac12,0)$, $(\tfrac12,0,\tfrac12)$,
$(0,\tfrac12,\tfrac12)$. The warp is nontrivial only for $p \ge 3$, where
the Chebyshev-Lobatto edge clustering differs from the equidistant distribution.

Total nodes per patch: $(p+1)(p+2)/2$.

**Curved patch mapping.** Patch $m$ is a mapping
$X_m : \mathcal{T} \to \mathbb{R}^3$ given by Lagrange interpolation through its
$N_L = (p+1)(p+2)/2$ Lobatto control points
$\{\mathbf{P}_{m,\ell}\}_{\ell=1}^{N_L} \subset \mathbb{R}^3$:

$$X_m(\boldsymbol{\lambda}) = \sum_{\ell=1}^{N_L} \mathbf{P}_{m,\ell}\,
L_\ell(\boldsymbol{\lambda}),$$

where $L_\ell$ is the bivariate Lagrange basis polynomial equal to 1 at Lobatto
node $\ell$ and 0 at all others. The control points $\mathbf{P}_{m,\ell}$ at the
patch vertices and edge midpoints are shared with neighboring patches to ensure
$C^0$ continuity of $\Gamma$.

**Uniform expansion centers.** Following the 2D pattern (4 uniform centers per
panel), the reference triangle $\mathcal{T}$ is first uniformly subdivided and
the centroids of the subtriangles are taken as expansion centers on the curved
patch surface.

**Uniform subdivision.** Partition $\mathcal{T}$ at level $L \ge 2$ by dividing
each edge into $L$ equal segments and connecting the subdivision points with
lines parallel to the edges. This yields $L^2$ small congruent triangles, each
similar to $\mathcal{T}$ with side length $1/L$.

**Expansion centers as subtriangle centroids.** The $N_c = L^2$ expansion
centers per patch are the centroids (in reference coordinates) of these $L^2$
subtriangles. In barycentric coordinates, each centroid is the arithmetic mean
of its three subtriangle vertices (which lie on the reference grid with spacing
$1/L$). Every centroid has strictly positive barycentric coordinates (all
$\lambda_i > 0$), so expansion centers lie strictly in the interior of
$\mathcal{T}$, away from the Lobatto boundary nodes — exactly as in 2D where
centers occupy $s \in \{-0.75, -0.25, 0.25, 0.75\}$ while Chebyshev-Lobatto
nodes are at $s \in \{-1, 0, 1\}$.

The default subdivision level $L = 2$ gives $N_c = 4$ expansion centers per
patch, directly matching the 2D panel. Finer levels ($L = 3, 4, \dots$) give
$9, 16, \dots$ centers for higher-order local corrections.

The Cartesian image in $\mathbb{R}^3$ follows from the patch mapping:
$\mathbf{c}_{m,q} = X_m(\boldsymbol{\lambda}_q)$ for each centroid
$\boldsymbol{\lambda}_q$. The outward unit normal $\mathbf{n}_{m,q}$ at each
center is obtained from the parametric tangent vectors:

$$\mathbf{n} = \frac{\partial_u X_m \times \partial_v X_m}
{\|\partial_u X_m \times \partial_v X_m\|}.$$

**Panel-major storage.** Expansion centers are stored in patch-major order.
Patch $m$ occupies global indices $[m N_c,\,(m+1) N_c)$. The global index of
expansion center $q$ on patch $m$ is

$$i(m, q) = m N_c + q, \qquad m = 0,\dots,N_p-1,\;\; q = 0,\dots,N_c-1.$$

The jump DOFs $( \mu, \sigma )$ are defined at the expansion centers. To
evaluate jump data at an arbitrary point on the patch (e.g., for spread
operations or for the Lobatto node interpolation), polynomial interpolation from
the expansion centers to the target point is used.

**Multi-component surfaces.** Disconnected parts of $\Gamma$ (e.g., multiple
immersed bodies) are supported by assigning each panel a component ID
$c_p \in \{0,\dots,C-1\}$ where $C$ is the number of components. Interior
domain labels increment per component: $\ell = 1$ inside component 0,
$\ell = 2$ inside component 1, etc.

**Parametric surface factories.** Standard surfaces are built from angular
parameterizations:
*   **UV sphere**: latitude rings ($M$) and longitude segments ($N$), with
    triangle caps at the poles. $N_p = 2MN$.
*   **Ellipsoid**: same connectivity as the UV sphere, with semi-axes
    $(a,b,c)$.
*   **Torus**: major-radius circles ($N_{\text{tor}}$) and minor-radius circles
    ($N_{\text{pol}}$), giving a genus-1 surface.

*   **Normals $\mathbf{n}$**: Unit outward normal pointing from $\Omega_{int}$ to $\Omega_{ext}$.
*   **Domain Labels $\ell$**: 
    *   $\ell(\mathbf{x}) = 1$ if $\mathbf{x} \in \Omega_{int}$.
    *   $\ell(\mathbf{x}) = 0$ if $\mathbf{x} \in \Omega_{ext}$.

## 2. The Constant Interface Problem (Elliptic)

Consider the elliptic equation on a rectangular box $B = [0, L_x] \times [0, L_y]$ containing the interface $\Gamma$:

$$
-\Delta u + \lambda^2 u = f \quad \text{in } B \setminus \Gamma
$$

subject to:
- **Homogeneous boundary conditions on the box** $\partial B$: typically $u = 0$ (Dirichlet); other homogeneous types (Neumann $\partial_n u = 0$, periodic) are also supported and the choice determines the eigenfunction expansion of the box Green's function.
- **Jump conditions on the interface** $\Gamma$:

$$
[u]_\Gamma = u_{int} - u_{ext} = \mu, \qquad [\partial_n u]_\Gamma = \partial_n u_{int} - \partial_n u_{ext} = \sigma
$$

where $u_{int}$ and $u_{ext}$ are the smooth extensions of the solution from either side.

### Relation with Layer Potentials
The solution $u$ to this problem is given by the classic representation formula:
$$
u(\mathbf{x}) = \mathcal{S} \sigma(\mathbf{x}) - \mathcal{D} \mu(\mathbf{x}) + \mathcal{V} f(\mathbf{x})
$$
where:
*   $\mathcal{S}$ is the **Single-Layer Potential**: $(\mathcal{S}\sigma)(\mathbf{x}) = \int_\Gamma G(\mathbf{x}, \mathbf{y}) \sigma(\mathbf{y}) ds_y$
*   $\mathcal{D}$ is the **Double-Layer Potential**: $(\mathcal{D}\mu)(\mathbf{x}) = \int_\Gamma \frac{\partial G}{\partial n_y}(\mathbf{x}, \mathbf{y}) \mu(\mathbf{y}) ds_y$
*   $\mathcal{V}$ is the **Volume Potential**: $(\mathcal{V}f)(\mathbf{x}) = \int_\Omega G(\mathbf{x}, \mathbf{y}) f(\mathbf{y}) dy$
*   $G$ is the Green's function of the rectangular box domain $B$ with homogeneous boundary conditions, satisfying:
    $$(-\Delta_{\mathbf{x}} + \lambda^2) G(\mathbf{x}, \mathbf{y}) = \delta(\mathbf{x} - \mathbf{y}) \quad \text{in } B, \qquad G(\mathbf{x}, \mathbf{y}) = 0 \quad \text{on } \partial B$$
    For the unit box $B = [0,1]^2$ with homogeneous Dirichlet BC, the eigenfunction expansion is:
    $$G(x,y; \xi,\eta) = 4\sum_{m=1}^{\infty}\sum_{n=1}^{\infty} \frac{\sin(m\pi x)\sin(n\pi y)\sin(m\pi \xi)\sin(n\pi \eta)}{\pi^2(m^2 + n^2)}$$

### Interface Problems Satisfied by Each Potential

Each potential is itself the solution to a specific interface problem on the box $B$:

| Potential | PDE in $B\setminus\Gamma$ | BC on $\partial B$ | $[u]$ on $\Gamma$ | $[\partial_n u]$ on $\Gamma$ |
|:---------|:--------------------------|:-------------------|:------------------|:----------------------------|
| $\mathcal{S}\sigma$ | $-\Delta u + \lambda^2 u = 0$ | $u = 0$ | $0$ | $\sigma$ |
| $\mathcal{D}\mu$    | $-\Delta u + \lambda^2 u = 0$ | $u = 0$ | $-\mu$ | $0$ |
| $\mathcal{V}f$      | $-\Delta u + \lambda^2 u = f$ | $u = 0$ | $0$ | $0$ |

The double-layer sign $[u] = -\mu$ follows from the jump relation of the normal derivative of $G$:
$$\lim_{\mathbf{x}\to\Gamma^{\pm}} \int_\Gamma \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,\mu(\mathbf{y})\,ds_y = \mp\frac{1}{2}\mu + \int_\Gamma \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,\mu(\mathbf{y})\,ds_y$$
so that $[\mathcal{D}\mu] = \mathcal{D}\mu|_{\Gamma^+} - \mathcal{D}\mu|_{\Gamma^-} = -\mu$ (with normal $\mathbf{n}$ pointing from $\Omega^+$ to $\Omega^-$).

By linearity, the full representation $u = \mathcal{S}\sigma - \mathcal{D}\mu + \mathcal{V}f$ satisfies:
$$-\Delta u + \lambda^2 u = f \quad \text{in } B\setminus\Gamma, \qquad u = 0 \quad \text{on } \partial B, \qquad [u] = \mu, \qquad [\partial_n u] = \sigma$$
which recovers the constant interface problem of Section 2.

### KFBIM Approach
Instead of evaluating the layer-potential integrals (or the eigenfunction expansion) directly, KFBIM solves the interface problem on a uniform Cartesian grid using the **Immersed Interface Method (IIM)**. The box Green's function $G$ is never explicitly formed; applying it is equivalent to solving $-\Delta u + \lambda^2 u = f$ with homogeneous BCs on $B$, which is done efficiently via FFT.
1.  **Spread**: Convert jumps $(\mu, \sigma)$ into a defect correction for the bulk RHS $f$.
2.  **Bulk Solve**: Solve the corrected elliptic system $-\Delta_h u = F$ using a fast solver (FFT/ZFFT).
3.  **Restrict**: Interpolate the bulk solution $u$ back to $\Gamma$ using jump-aware interpolation to recover the boundary traces.

## 3. Boundary Integral Operators

The traces of the potentials on $\Gamma$ define the boundary integral operators:
*   **Single-layer operator $S$**: $S\sigma = (\mathcal{S}\sigma)|_\Gamma$
*   **Double-layer operator $K$**: $K\mu = \text{avg}(\mathcal{D}\mu)|_\Gamma$

Using the jump relations of layer potentials:
$$
u_{int}|_\Gamma = S\sigma - (K - \tfrac{1}{2}I)\mu + V f
$$
$$
u_{ext}|_\Gamma = S\sigma - (K + \tfrac{1}{2}I)\mu + V f
$$
The KFBIM "Interface Solver" effectively acts as a black-box evaluator for these operators. For example, by setting $\mu=0, f=0$, the solver returns $u_{int}|_\Gamma = S\sigma$.

## 4. Interface Solver Outputs (KFBIM Operator Evaluation)

`LaplaceInterfaceSolver2D` takes jump data $(\mu, \sigma)$ and bulk forcing $f$
and returns the volume solution $u_{bulk}$ together with averaged interface
quantities:

$$
u_{avg} = \frac{u^+ + u^-}{2}, \qquad
(\partial_n u)_{avg} = \frac{\partial_n u^+ + \partial_n u^-}{2}.
$$

Interior and exterior traces are recovered by the jump relations:

$$
u^+ = u_{avg} + \frac{\mu}{2}, \qquad
u^- = u_{avg} - \frac{\mu}{2},
$$

$$
\partial_n u^+ = (\partial_n u)_{avg} + \frac{\sigma}{2}, \qquad
\partial_n u^- = (\partial_n u)_{avg} - \frac{\sigma}{2}.
$$

`LaplacePotentialEval2D` wraps the same pipeline and exposes reusable averaged
operators:

| Primitive | $(\mu, \sigma)$ | RHS data | Returned outputs |
|:----------|:---------------|:---------|:-----------------|
| Double-layer jump primitive | $(\phi, 0)$ | $0$ | $K\phi = u_{avg}$, $H\phi = (\partial_n u)_{avg}$ |
| Single-layer primitive | $(0, \psi)$ | $0$ | $S\psi = u_{avg}$, $K'\psi = (\partial_n u)_{avg}$ |
| Interface forcing primitive | $(0, 0)$ | $[f]=q$ | $Nq = u_{avg}$, $\partial_n Nq = (\partial_n u)_{avg}$ |

The double-layer primitive follows the implementation convention $[u]=\phi$.
Relative to the classical representation $-\mathcal{D}\phi$, this is the sign
used by the current second-kind BVP operator code.

## 5. Boundary Value Problems (BVPs) to BIEs

BVPs are solved by formulating them as Boundary Integral Equations (BIEs) on $\Gamma$.

**Zero-extension of the bulk forcing.** In a BVP, the bulk forcing $f$ is physically defined only on the problem domain ($\Omega_{int}$ for interior BVPs, $\Omega_{ext}$ for exterior BVPs). However, the representation formula and the interface solver both operate on the full box domain $B$. We therefore extend $f$ by zero to $B$:

$$
\tilde{f}(\mathbf{x}) = \begin{cases}
f(\mathbf{x}) & \mathbf{x} \in \Omega_{int} \text{ (or } \Omega_{ext}\text{)} \\[2pt]
0 & \text{otherwise in } B
\end{cases}
$$

The volume potential $\mathcal{V}f$ that appears in the BIE right-hand sides is then understood as $\mathcal{V}\tilde{f}$ — integrating $\tilde{f}$ over $B$. In the KFBIM interface solver, this corresponds to passing $f_{bulk} = \tilde{f}$ (the zero-extended $f$ on the full box).

### Interior Dirichlet BVP

Find $u$ such that $-\Delta u + \lambda^2 u = f$ in $\Omega_{int}$ with $u|_\Gamma = g$.

**Representation.** Write $u$ using the general representation formula $u = \mathcal{S}\sigma - \mathcal{D}\mu + \mathcal{V}f$ with $\sigma = 0$ and an unknown double-layer density $\mu = \phi$:

$$
u(\mathbf{x}) = -\mathcal{D}\phi(\mathbf{x}) + \mathcal{V}f(\mathbf{x}), \qquad \mathbf{x} \in \Omega_{int}
$$

**Derivation of the BIE.** Take the limit $\mathbf{x} \to \Gamma$ from the interior ($\Gamma^+$). Using the jump relation $\mathcal{D}\phi|_{\Gamma^+} = K\phi - \tfrac{1}{2}\phi$:

$$
\begin{aligned}
g = u|_{\Gamma^+} &= -\mathcal{D}\phi|_{\Gamma^+} + \mathcal{V}f|_\Gamma \\
&= -(K\phi - \tfrac{1}{2}\phi) + \mathcal{V}f|_\Gamma \\
&= (\tfrac{1}{2}I - K)\phi + \mathcal{V}f|_\Gamma
\end{aligned}
$$

Rearranging gives the **boundary integral equation**:

$$
\boxed{(\tfrac{1}{2}I - K)\phi = g - \mathcal{V}f|_\Gamma}
$$

**KFBIM evaluation.** In each GMRES iteration, the left-hand operator $(\tfrac{1}{2}I - K)$ applied to a density $\phi$ is evaluated by calling the interface solver with jumps $(\mu = \phi,\; \sigma = 0)$ and $f_{bulk} = 0$. The solver returns the averaged trace; the interior trace is recovered as $u^+ = u_{avg} + \tfrac{1}{2}\phi$ and equals $(\tfrac{1}{2}I - K)\phi$:

$$
(\tfrac{1}{2}I - K)\phi = -(\mathcal{D}\phi)|_{\Gamma^+} = u_{int} \quad \text{(from interface solver with } \mu=\phi,\; \sigma=0,\; f=0\text{)}
$$

The right-hand side $g - \mathcal{V}f|_\Gamma$ is assembled by computing $\mathcal{V}f|_\Gamma$ once via the interface solver with jumps $(0, 0)$ and $f_{bulk} = f$.

### Interior Neumann BVP

Find $u$ such that $-\Delta u + \lambda^2 u = f$ in $\Omega_{int}$ with $\partial_n u|_\Gamma = g$.

**Representation.** Use the single-layer ansatz ($\sigma = \psi$, $\mu = 0$):

$$
u(\mathbf{x}) = \mathcal{S}\psi(\mathbf{x}) + \mathcal{V}f(\mathbf{x}), \qquad \mathbf{x} \in \Omega_{int}
$$

**Derivation of the BIE.** Take the interior normal derivative $\mathbf{x} \to \Gamma^+$. Using the jump relation $\partial_n (\mathcal{S}\psi)|_{\Gamma^+} = K'\psi + \tfrac{1}{2}\psi$ where $K'$ is the adjoint double-layer operator:

$$
\begin{aligned}
g = \partial_n u|_{\Gamma^+} &= \partial_n(\mathcal{S}\psi)|_{\Gamma^+} + \partial_n(\mathcal{V}f)|_\Gamma \\
&= (K'\psi + \tfrac{1}{2}\psi) + \partial_n(\mathcal{V}f)|_\Gamma
\end{aligned}
$$

Rearranging:

$$
\boxed{(\tfrac{1}{2}I + K')\psi = g - \partial_n(\mathcal{V}f)|_\Gamma}
$$

**Nullspace ($\kappa = 0$).** For the Laplace equation, the operator $(\tfrac{1}{2}I + K')$ has a one-dimensional nullspace (the constant density). This reflects the fact that the interior Neumann problem determines $u$ only up to an additive constant. A solution exists only when the solvability condition holds:

$$
\int_\Gamma g\,ds + \int_{\Omega_{int}} f\,d\mathbf{x} = 0
$$

In practice, the nullspace is handled by fixing the density at one point or by projecting out the constant mode during GMRES.

**KFBIM evaluation.** $(\tfrac{1}{2}I + K')\psi$ is the interior normal derivative of $\mathcal{S}\psi$. Call the interface solver with jumps $(\mu = 0,\; \sigma = \psi)$ and $f_{bulk} = 0$; recover $un^+ = un_{avg} + \tfrac{1}{2}\psi$ to get $(\tfrac{1}{2}I + K')\psi$. The volume contribution $\partial_n(\mathcal{V}f)|_\Gamma$ is obtained via the interface solver with jumps $(0, 0)$ and $f_{bulk} = f$.

### Exterior Dirichlet BVP

Find $u$ such that $-\Delta u + \lambda^2 u = f$ in $\Omega_{ext}$ with $u|_\Gamma = g$.

**Representation.** Same double-layer ansatz as the interior case ($\sigma = 0$, $\mu = \phi$):

$$
u(\mathbf{x}) = -\mathcal{D}\phi(\mathbf{x}) + \mathcal{V}f(\mathbf{x}), \qquad \mathbf{x} \in \Omega_{ext}
$$

**Derivation of the BIE.** Take the exterior trace $\mathbf{x} \to \Gamma^-$. Using the jump relation $\mathcal{D}\phi|_{\Gamma^-} = K\phi + \tfrac{1}{2}\phi$:

$$
\begin{aligned}
g = u|_{\Gamma^-} &= -\mathcal{D}\phi|_{\Gamma^-} + \mathcal{V}f|_\Gamma \\
&= -(K\phi + \tfrac{1}{2}\phi) + \mathcal{V}f|_\Gamma
\end{aligned}
$$

Rearranging:

$$
\boxed{(K + \tfrac{1}{2}I)\phi = \mathcal{V}f|_\Gamma - g}
$$

**KFBIM evaluation.** Call the interface solver with jumps $(\mu = \phi,\; \sigma = 0)$ and $f_{bulk} = 0$. Recover the exterior trace $u^- = u_{avg} - \tfrac{1}{2}\phi$ of $-\mathcal{D}\phi$. It equals $-(K + \tfrac{1}{2}I)\phi$, so $(K + \tfrac{1}{2}I)\phi = -u^-$.

### Exterior Neumann BVP

Find $u$ such that $-\Delta u + \lambda^2 u = f$ in $\Omega_{ext}$ with $\partial_n u|_\Gamma = g$.

**Representation.** Single-layer ansatz ($\sigma = \psi$, $\mu = 0$):

$$
u(\mathbf{x}) = \mathcal{S}\psi(\mathbf{x}) + \mathcal{V}f(\mathbf{x}), \qquad \mathbf{x} \in \Omega_{ext}
$$

**Derivation of the BIE.** Take the exterior normal derivative $\mathbf{x} \to \Gamma^-$. Using the jump relation $\partial_n(\mathcal{S}\psi)|_{\Gamma^-} = K'\psi - \tfrac{1}{2}\psi$:

$$
\begin{aligned}
g = \partial_n u|_{\Gamma^-} &= \partial_n(\mathcal{S}\psi)|_{\Gamma^-} + \partial_n(\mathcal{V}f)|_\Gamma \\
&= (K'\psi - \tfrac{1}{2}\psi) + \partial_n(\mathcal{V}f)|_\Gamma
\end{aligned}
$$

Rearranging:

$$
\boxed{(K' - \tfrac{1}{2}I)\psi = g - \partial_n(\mathcal{V}f)|_\Gamma}
$$

**Nullspace ($\kappa = 0$).** For the Laplace equation, the operator $(K' - \tfrac{1}{2}I)$ also has a one-dimensional nullspace in 2D, corresponding to the constant density. This is handled by the same projection technique as the interior Neumann case.

**KFBIM evaluation.** Call the interface solver with jumps $(\mu = 0,\; \sigma = \psi)$ and $f_{bulk} = 0$; recover the exterior normal derivative $un^- = un_{avg} - \tfrac{1}{2}\psi$, which gives $(K' - \tfrac{1}{2}I)\psi$.

### BIE Summary

| BVP | Representation | BIE |
|:----|:---------------|:----|
| Interior Dirichlet | $u = -\mathcal{D}\phi + \mathcal{V}f$ | $(\tfrac{1}{2}I - K)\phi = g - \mathcal{V}f\|_\Gamma$ |
| Interior Neumann   | $u = \mathcal{S}\psi + \mathcal{V}f$  | $(\tfrac{1}{2}I + K')\psi = g - \partial_n(\mathcal{V}f)\|_\Gamma$ |
| Exterior Dirichlet | $u = -\mathcal{D}\phi + \mathcal{V}f$ | $(K + \tfrac{1}{2}I)\phi = \mathcal{V}f\|_\Gamma - g$ |
| Exterior Neumann   | $u = \mathcal{S}\psi + \mathcal{V}f$  | $(K' - \tfrac{1}{2}I)\psi = g - \partial_n(\mathcal{V}f)\|_\Gamma$ |

## 6. Generic Interface Problems (Variable Coefficients)

The generic second-order elliptic interface problem is:

$$
\boxed{-\nabla \cdot (\beta(\mathbf{x}) \nabla u) + \kappa^2(\mathbf{x})\, u = f(\mathbf{x}) \quad \text{in } \Omega_{int} \cup \Omega_{ext}}
$$

where both coefficients are piecewise constant with a jump across $\Gamma$:

$$
\beta(\mathbf{x}) = \begin{cases}
\beta_{int} & \mathbf{x} \in \Omega_{int} \\[2pt]
\beta_{ext} & \mathbf{x} \in \Omega_{ext}
\end{cases},
\qquad
\kappa^2(\mathbf{x}) = \begin{cases}
\kappa^2_{int} & \mathbf{x} \in \Omega_{int} \\[2pt]
\kappa^2_{ext} & \mathbf{x} \in \Omega_{ext}
\end{cases}
$$

The jump conditions on $\Gamma$ become:

$$
[u]_\Gamma = \mu, \qquad [\beta\,\partial_n u]_\Gamma = \sigma
$$

This reduces to the constant-coefficient interface problem (Section 2) when $\beta_{int} = \beta_{ext} = 1$ and $\kappa_{int} = \kappa_{ext} = 0$.

### 7. BIE for Generic Interface Problems

We now consider the generic interface problem from Section 6:
$$
-\nabla \cdot (\beta(\mathbf{x}) \nabla u) + \kappa^2(\mathbf{x})\, u = f(\mathbf{x}) \quad \text{in } \Omega_{int} \cup \Omega_{ext}
$$
with jump conditions $[u] = \mu$ and $[\beta \partial_n u] = \sigma$.

#### Constant $\beta/\kappa^2$ Ratio

First, we consider the special case where the ratio of $\kappa^2$ to $\beta$ is the same on both sides of the interface. Let this constant ratio be $\lambda^2$:
$$
\frac{\kappa^2_{int}}{\beta_{int}} = \frac{\kappa^2_{ext}}{\beta_{ext}} \equiv \lambda^2
$$

Since $\beta$ is piecewise constant, we can divide the PDE in each subdomain by its respective $\beta$ value. This yields the modified Helmholtz equation on the entire domain $B \setminus \Gamma$:
$$
-\Delta u + \lambda^2 u = \frac{f}{\beta}
$$

**Representation.** We can express the solution using the representation formula for the operator $-\Delta + \lambda^2$:
$$
u(\mathbf{x}) = \mathcal{S}\psi(\mathbf{x}) - \mathcal{D}\mu(\mathbf{x}) + \mathcal{V}(f/\beta)(\mathbf{x})
$$
where $\psi$ is an unknown single-layer density. By construction, the double-layer density is chosen to be exactly $\mu$ to satisfy the Dirichlet jump condition:
$$
[u] = [\mathcal{S}\psi] - [\mathcal{D}\mu] + [\mathcal{V}(f/\beta)] = 0 - (-\mu) + 0 = \mu
$$

**Derivation of the BIE.** To find the unknown density $\psi$, we use the flux jump condition $[\beta \partial_n u] = \sigma$. First, we take the normal derivative of the representation formula from both the interior and exterior:
$$
\partial_n u_{int} = (K' + \tfrac{1}{2}I)\psi - \partial_n(\mathcal{D}\mu)|_\Gamma + \partial_n \mathcal{V}(f/\beta)|_\Gamma
$$
$$
\partial_n u_{ext} = (K' - \tfrac{1}{2}I)\psi - \partial_n(\mathcal{D}\mu)|_\Gamma + \partial_n \mathcal{V}(f/\beta)|_\Gamma
$$
Note that the normal derivatives of the double-layer potential and the volume potential are continuous across the interface.

Substituting these into the flux jump condition $\beta_{int} \partial_n u_{int} - \beta_{ext} \partial_n u_{ext} = \sigma$:
$$
\beta_{int} \left( (K' + \tfrac{1}{2}I)\psi - \partial_n(\mathcal{D}\mu) + \partial_n \mathcal{V}(f/\beta) \right) - \beta_{ext} \left( (K' - \tfrac{1}{2}I)\psi - \partial_n(\mathcal{D}\mu) + \partial_n \mathcal{V}(f/\beta) \right) = \sigma
$$

Grouping the terms with $\psi$:
$$
\beta_{int}(K' + \tfrac{1}{2}I)\psi - \beta_{ext}(K' - \tfrac{1}{2}I)\psi = \frac{\beta_{int} + \beta_{ext}}{2}\psi + (\beta_{int} - \beta_{ext})K'\psi
$$

For the continuous potential terms, we get a factor of $(\beta_{int} - \beta_{ext})$:
$$
-(\beta_{int} - \beta_{ext})\partial_n(\mathcal{D}\mu)|_\Gamma + (\beta_{int} - \beta_{ext})\partial_n \mathcal{V}(f/\beta)|_\Gamma
$$

Equating the sum to $\sigma$ and rearranging yields the Boundary Integral Equation for $\psi$:
$$
\frac{\beta_{int} + \beta_{ext}}{2}\psi + (\beta_{int} - \beta_{ext})K'\psi = \sigma + (\beta_{int} - \beta_{ext}) \Big( \partial_n(\mathcal{D}\mu)|_\Gamma - \partial_n \mathcal{V}(f/\beta)|_\Gamma \Big)
$$

Dividing by the average $\bar{\beta} = \frac{\beta_{int} + \beta_{ext}}{2}$, this can be written as a standard Fredholm integral equation of the second kind:
$$
\boxed{ (I + 2\frac{\beta_{int} - \beta_{ext}}{\beta_{int} + \beta_{ext}} K')\psi = \frac{2}{\beta_{int} + \beta_{ext}}\sigma + 2\frac{\beta_{int} - \beta_{ext}}{\beta_{int} + \beta_{ext}} \Big( \partial_n(\mathcal{D}\mu)|_\Gamma - \partial_n \mathcal{V}(f/\beta)|_\Gamma \Big) }
$$

This BIE can be solved for $\psi$, after which the full solution $u$ can be evaluated using the representation formula.
