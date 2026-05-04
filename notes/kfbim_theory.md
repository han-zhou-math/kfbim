# Mathematical Foundation of the Kernel-Free Boundary Integral Method (KFBIM)

This note documents the mathematical framework for the elliptic PDE solvers in this repository.

## Table of Contents
- [1. Geometry Representation](#1-geometry-representation)
- [2. The Constant Interface Problem (Elliptic)](#2-the-constant-interface-problem-elliptic)
- [3. Boundary Integral Operators](#3-boundary-integral-operators)
- [4. Interface Solver Outputs (KFBIM Operator Evaluation)](#4-interface-solver-outputs-kfbim-operator-evaluation)
- [5. Boundary Value Problems (BVPs) to BIEs](#5-boundary-value-problems-bvps-to-bies)
- [6. Generic Interface Problems (Variable Coefficients)](#6-generic-interface-problems-variable-coefficients)
  - [6.1 Common Ratio](#61-common-ratio)
  - [6.2 Different Ratios](#62-different-ratios)

**Notation conventions.** Throughout this note, $\Omega_{int}$ denotes the
interior domain, $\Omega_{ext}$ denotes the exterior domain, and the unit normal
$n$ points from $\Omega_{int}$ to $\Omega_{ext}$. For a piecewise smooth
quantity $u$, write

$$
\gamma_{int}u=u_{int}|_\Gamma,\qquad
\gamma_{ext}u=u_{ext}|_\Gamma,
$$

and

$$
\gamma_{n,int}u=(\partial_n u_{int})|_\Gamma,\qquad
\gamma_{n,ext}u=(\partial_n u_{ext})|_\Gamma.
$$

Jumps are always interior minus exterior:

$$
[u]=\gamma_{int}u-\gamma_{ext}u,\qquad
[\partial_n u]=\gamma_{n,int}u-\gamma_{n,ext}u.
$$

The adjoint double-layer operator is denoted throughout by $K^*$. The
hypersingular operator, namely the averaged normal derivative of a double-layer
potential, is denoted by $H$ (or $H_s$ for phase-dependent kernels in Section 6.2).

## 1. Geometry Representation

The domain $\Omega \subset \mathbb{R}^d$ is partitioned by an interface $\Gamma$ into an interior region $\Omega_{int}$ and an exterior region $\Omega_{ext}$.

*   **Interface $\Gamma$**: Represented by a collection of curved panels in 2D
    and curved triangle patches in 3D.

### 2D: Curve Panels

In 2D, $\Gamma$ is a closed curve represented by polynomial panels. Panel $m$
is a map

$$
X_m:[-1,1]\to\mathbb{R}^2
$$

defined by polynomial interpolation from Chebyshev-Lobatto nodes on the
reference interval. For degree $p$, let

$$
-1=s_0<s_1<\cdots<s_p=1
$$

be the Chebyshev-Lobatto points, and let $L_j$ be the corresponding Lagrange
basis polynomials. If $\mathbf{P}_{m,j}\in\mathbb{R}^2$ are the physical panel
nodes, then

$$
X_m(s)=\sum_{j=0}^{p}\mathbf{P}_{m,j}L_j(s).
$$

The current 2D Laplace path uses the quadratic Chebyshev-Lobatto panel
($p=2$), with nodes

$$
s=\{-1,0,1\}.
$$

The two endpoints of adjacent panels represent the same physical interface
point and are allocated as one global DOF. Thus a closed curve with $N_p$
quadratic panels has $2N_p$ interface DOFs: $N_p$ shared endpoints and $N_p$
panel midpoints.

Expansion centers for the local correction functions are not additional
geometry DOFs. They are sample points on the already-curved panel,

$$
\mathbf{c}_{m,q}=X_m(\xi_q),
\qquad
\xi_q\in\{-0.75,-0.25,0.25,0.75\},
$$

so the current 2D correction path uses four expansion centers per panel. The
normal at each center is computed from the panel tangent $X_m'(\xi_q)$ and
oriented from $\Omega_{int}$ to $\Omega_{ext}$.

### 3D: Curved Triangle Patches

In 3D, $\Gamma$ is represented similarly, but panels are replaced by curved
triangle patches. Patch $m$ is a polynomial map

$$
X_m:\mathcal{T}\to\mathbb{R}^3,
$$

where $\mathcal{T}$ is the reference triangle in barycentric coordinates:

$$\mathcal{T} = \{(\lambda_1, \lambda_2, \lambda_3) \mid \lambda_1 + \lambda_2 + \lambda_3 = 1,\; \lambda_i \ge 0\}.$$

Equivalently, in Cartesian coordinates on the reference plane:
$\mathcal{T} = \{(u, v) \mid u \ge 0,\; v \ge 0,\; u+v \le 1\}$ with
$\lambda_1 = 1-u-v$, $\lambda_2 = u$, $\lambda_3 = v$.

The patch map is defined by polynomial interpolation from Lobatto-type nodes on
$\mathcal{T}$. If the interpolation nodes are
$\{\boldsymbol{\lambda}_\ell\}_{\ell=1}^{N_L}$, the physical patch nodes are
$\mathbf{P}_{m,\ell}\in\mathbb{R}^3$, and $L_\ell$ are the triangular Lagrange
basis polynomials, then

$$X_m(\boldsymbol{\lambda}) = \sum_{\ell=1}^{N_L} \mathbf{P}_{m,\ell}\,
L_\ell(\boldsymbol{\lambda}),$$

with $L_\ell(\boldsymbol{\lambda}_r)=\delta_{\ell r}$. Patch nodes on shared
vertices and shared edges are allocated as one global interface DOF, exactly as
shared endpoints are allocated once for neighboring 2D panels.

Expansion centers for 3D correction functions are sampled from the curved patch.
One natural choice, parallel to the four 2D centers per panel, is to subdivide
the reference triangle at level $L=2$, take the four subtriangle centroids
$\boldsymbol{\xi}_q$, and map them to the physical surface:

$$
\mathbf{c}_{m,q}=X_m(\boldsymbol{\xi}_q),
\qquad q=1,\dots,4.
$$

Finer subdivisions give $L^2$ centers per patch. The outward unit normal at an
expansion center is computed from the patch tangent vectors:

$$\mathbf{n} = \frac{\partial_u X_m \times \partial_v X_m}
{\|\partial_u X_m \times \partial_v X_m\|}.$$

Expansion centers are stored in patch-major order. If each patch has $N_c$
centers, patch $m$ occupies global indices $[mN_c,(m+1)N_c)$, and the global
index of expansion center $q$ on patch $m$ is

$$i(m, q) = m N_c + q, \qquad m = 0,\dots,N_p-1,\;\; q = 0,\dots,N_c-1.$$

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
[u]_\Gamma = \gamma_{int}u-\gamma_{ext}u = \mu,
\qquad
[\partial_n u]_\Gamma
=\gamma_{n,int}u-\gamma_{n,ext}u = \sigma
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
*   $\mathcal{V}$ is the **Volume Potential**: $(\mathcal{V}f)(\mathbf{x}) = \int_B G(\mathbf{x}, \mathbf{y}) f(\mathbf{y}) dy$
*   $G$ is the Green's function of the rectangular box domain $B$ with homogeneous boundary conditions, satisfying:
    $$(-\Delta_{\mathbf{x}} + \lambda^2) G(\mathbf{x}, \mathbf{y}) = \delta(\mathbf{x} - \mathbf{y}) \quad \text{in } B, \qquad G(\mathbf{x}, \mathbf{y}) = 0 \quad \text{on } \partial B$$
    For the unit box $B = [0,1]^2$ with homogeneous Dirichlet BC, the eigenfunction expansion is:
    $$G(x,y; \xi,\eta) = 4\sum_{m=1}^{\infty}\sum_{n=1}^{\infty} \frac{\sin(m\pi x)\sin(n\pi y)\sin(m\pi \xi)\sin(n\pi \eta)}{\pi^2(m^2 + n^2)+\lambda^2}$$

### Interface Problems Satisfied by Each Potential

Each potential is itself the solution to a specific interface problem on the box $B$:

| Potential | PDE in $B\setminus\Gamma$ | BC on $\partial B$ | $[u]$ on $\Gamma$ | $[\partial_n u]$ on $\Gamma$ |
|:---------|:--------------------------|:-------------------|:------------------|:----------------------------|
| $\mathcal{S}\sigma$ | $-\Delta u + \lambda^2 u = 0$ | $u = 0$ | $0$ | $\sigma$ |
| $\mathcal{D}\mu$    | $-\Delta u + \lambda^2 u = 0$ | $u = 0$ | $-\mu$ | $0$ |
| $\mathcal{V}f$      | $-\Delta u + \lambda^2 u = f$ | $u = 0$ | $0$ | $0$ |

The double-layer sign $[u] = -\mu$ follows from the trace relations

$$
\gamma_{int}\mathcal{D}\mu=K\mu-\frac12\mu,\qquad
\gamma_{ext}\mathcal{D}\mu=K\mu+\frac12\mu,
$$

so $[\mathcal{D}\mu]=\gamma_{int}\mathcal{D}\mu-\gamma_{ext}\mathcal{D}\mu=-\mu$.

By linearity, the full representation $u = \mathcal{S}\sigma - \mathcal{D}\mu + \mathcal{V}f$ satisfies:
$$-\Delta u + \lambda^2 u = f \quad \text{in } B\setminus\Gamma, \qquad u = 0 \quad \text{on } \partial B, \qquad [u] = \mu, \qquad [\partial_n u] = \sigma$$
which recovers the constant interface problem of Section 2.

### KFBIM Approach
Instead of evaluating the layer-potential integrals (or the eigenfunction expansion) directly, KFBIM solves the interface problem on a uniform Cartesian grid using the **Immersed Interface Method (IIM)**. The box Green's function $G$ is never explicitly formed; applying it is equivalent to solving $-\Delta u + \lambda^2 u = f$ with homogeneous BCs on $B$, which is done efficiently via FFT.
1.  **Spread**: Convert jumps $(\mu, \sigma)$ into a defect correction for the bulk RHS $f$.
2.  **Bulk Solve**: Solve the corrected elliptic system $(-\Delta_h+\lambda^2 I)u = F$ using a fast solver (FFT/ZFFT).
3.  **Restrict**: Interpolate the bulk solution $u$ back to $\Gamma$ using jump-aware interpolation to recover the boundary traces.

## 3. Boundary Integral Operators

The traces of the potentials on $\Gamma$ define the boundary integral operators:
*   **Single-layer operator $S$**: $S\sigma = (\mathcal{S}\sigma)|_\Gamma$
*   **Double-layer operator $K$**: $K\mu = \text{avg}(\mathcal{D}\mu)|_\Gamma$
*   **Adjoint double-layer operator $K^*$**:
    $K^*\sigma=\text{avg}(\partial_n\mathcal{S}\sigma)|_\Gamma$
*   **Hypersingular operator $H$**:
    $H\mu=\text{avg}(\partial_n\mathcal{D}\mu)|_\Gamma$
*   **Volume trace operators**:
    $V_\Gamma f=\gamma_{int}\mathcal{V}f=\gamma_{ext}\mathcal{V}f$ and
    $V_{n,\Gamma}f=\gamma_{n,int}\mathcal{V}f=\gamma_{n,ext}\mathcal{V}f$

Using the jump relations of layer potentials:
$$
\gamma_{int}u = S\sigma - (K - \tfrac{1}{2}I)\mu + V_\Gamma f
$$
$$
\gamma_{ext}u = S\sigma - (K + \tfrac{1}{2}I)\mu + V_\Gamma f
$$
The KFBIM "Interface Solver" effectively acts as a black-box evaluator for
these operators. For example, by setting $\mu=0$ and $f=0$, the solver returns
$\gamma_{int}u=S\sigma$.

## 4. Interface Solver Outputs (KFBIM Operator Evaluation)

`LaplaceInterfaceSolver2D` takes jump data $(\mu, \sigma)$ and bulk forcing $f$
and returns the volume solution $u_{bulk}$ together with averaged interface
quantities:

$$
u_{avg} = \frac{\gamma_{int}u + \gamma_{ext}u}{2}, \qquad
u_{n,avg} = \frac{\gamma_{n,int}u + \gamma_{n,ext}u}{2}.
$$

Interior and exterior traces are recovered by the jump relations:

$$
\gamma_{int}u = u_{avg} + \frac{\mu}{2}, \qquad
\gamma_{ext}u = u_{avg} - \frac{\mu}{2},
$$

$$
\gamma_{n,int}u = u_{n,avg} + \frac{\sigma}{2}, \qquad
\gamma_{n,ext}u = u_{n,avg} - \frac{\sigma}{2}.
$$

`LaplacePotentialEval2D` wraps the same pipeline and exposes reusable averaged
operators:

| Primitive | $(\mu, \sigma)$ | RHS data | Returned outputs |
|:----------|:---------------|:---------|:-----------------|
| Double-layer jump primitive | $(\phi, 0)$ | $0$ | $K\phi = u_{avg}$, $H\phi = u_{n,avg}$ |
| Single-layer primitive | $(0, \psi)$ | $0$ | $S\psi = u_{avg}$, $K^*\psi = u_{n,avg}$ |
| Volume-forcing primitive | $(0, 0)$ | $q$ | $V_\Gamma q = u_{avg}$, $V_{n,\Gamma}q = u_{n,avg}$ |

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

The volume-potential traces $V_\Gamma f$ and $V_{n,\Gamma}f$ that appear in
the BIE right-hand sides are then understood as traces of
$\mathcal{V}\tilde{f}$, integrating $\tilde{f}$ over $B$. In the KFBIM
interface solver, this corresponds to passing $f_{bulk} = \tilde{f}$ (the
zero-extended $f$ on the full box).

### Interior Dirichlet BVP

Find $u$ such that $-\Delta u + \lambda^2 u = f$ in $\Omega_{int}$ with $\gamma_{int}u = g$.

**Representation.** Write $u$ using the general representation formula $u = \mathcal{S}\sigma - \mathcal{D}\mu + \mathcal{V}f$ with $\sigma = 0$ and an unknown double-layer density $\mu = \phi$:

$$
u(\mathbf{x}) = -\mathcal{D}\phi(\mathbf{x}) + \mathcal{V}f(\mathbf{x}), \qquad \mathbf{x} \in \Omega_{int}
$$

**Derivation of the BIE.** Take the interior trace. Using the jump relation
$\gamma_{int}\mathcal{D}\phi = K\phi - \tfrac{1}{2}\phi$:

$$
\begin{aligned}
g = \gamma_{int}u &= -\gamma_{int}\mathcal{D}\phi + V_\Gamma f \\
&= -(K\phi - \tfrac{1}{2}\phi) + V_\Gamma f \\
&= (\tfrac{1}{2}I - K)\phi + V_\Gamma f
\end{aligned}
$$

Rearranging gives the **boundary integral equation**:

$$
\boxed{(\tfrac{1}{2}I - K)\phi = g - V_\Gamma f}
$$

**KFBIM evaluation.** In each GMRES iteration, the left-hand operator $(\tfrac{1}{2}I - K)$ applied to a density $\phi$ is evaluated by calling the interface solver with jumps $(\mu = \phi,\; \sigma = 0)$ and $f_{bulk} = 0$. The solver returns the averaged trace; the interior trace is recovered as $\gamma_{int}u = u_{avg} + \tfrac{1}{2}\phi$ and equals $(\tfrac{1}{2}I - K)\phi$:

$$
(\tfrac{1}{2}I - K)\phi = -\gamma_{int}\mathcal{D}\phi
\quad \text{(from interface solver with } \mu=\phi,\; \sigma=0,\; f=0\text{)}
$$

The right-hand side $g - V_\Gamma f$ is assembled by computing $V_\Gamma f$
once via the interface solver with jumps $(0, 0)$ and $f_{bulk} = f$.

### Interior Neumann BVP

Find $u$ such that $-\Delta u + \lambda^2 u = f$ in $\Omega_{int}$ with $\gamma_{n,int}u = g$.

**Representation.** Use the single-layer ansatz ($\sigma = \psi$, $\mu = 0$):

$$
u(\mathbf{x}) = \mathcal{S}\psi(\mathbf{x}) + \mathcal{V}f(\mathbf{x}), \qquad \mathbf{x} \in \Omega_{int}
$$

**Derivation of the BIE.** Take the interior normal trace. Using the jump
relation $\gamma_{n,int}\mathcal{S}\psi = K^*\psi + \tfrac{1}{2}\psi$:

$$
\begin{aligned}
g = \gamma_{n,int}u &= \gamma_{n,int}\mathcal{S}\psi + V_{n,\Gamma}f \\
&= (K^*\psi + \tfrac{1}{2}\psi) + V_{n,\Gamma}f
\end{aligned}
$$

Rearranging:

$$
\boxed{(\tfrac{1}{2}I + K^*)\psi = g - V_{n,\Gamma}f}
$$

**Nullspace ($\kappa = 0$).** For the Laplace equation, the operator $(\tfrac{1}{2}I + K^*)$ has a one-dimensional nullspace (the constant density). This reflects the fact that the interior Neumann problem determines $u$ only up to an additive constant. A solution exists only when the solvability condition holds:

$$
\int_\Gamma g\,ds + \int_{\Omega_{int}} f\,d\mathbf{x} = 0
$$

In practice, the nullspace is handled by fixing the density at one point or by projecting out the constant mode during GMRES.

**KFBIM evaluation.** $(\tfrac{1}{2}I + K^*)\psi$ is the interior normal derivative of $\mathcal{S}\psi$. Call the interface solver with jumps $(\mu = 0,\; \sigma = \psi)$ and $f_{bulk} = 0$; recover $\gamma_{n,int}u = u_{n,avg} + \tfrac{1}{2}\psi$ to get $(\tfrac{1}{2}I + K^*)\psi$. The volume contribution $V_{n,\Gamma}f$ is obtained via the interface solver with jumps $(0, 0)$ and $f_{bulk} = f$.

### Exterior Dirichlet BVP

Find $u$ such that $-\Delta u + \lambda^2 u = f$ in $\Omega_{ext}$ with $\gamma_{ext}u = g$.

**Representation.** Same double-layer ansatz as the interior case ($\sigma = 0$, $\mu = \phi$):

$$
u(\mathbf{x}) = -\mathcal{D}\phi(\mathbf{x}) + \mathcal{V}f(\mathbf{x}), \qquad \mathbf{x} \in \Omega_{ext}
$$

**Derivation of the BIE.** Take the exterior trace. Using the jump relation
$\gamma_{ext}\mathcal{D}\phi = K\phi + \tfrac{1}{2}\phi$:

$$
\begin{aligned}
g = \gamma_{ext}u &= -\gamma_{ext}\mathcal{D}\phi + V_\Gamma f \\
&= -(K\phi + \tfrac{1}{2}\phi) + V_\Gamma f
\end{aligned}
$$

Rearranging:

$$
\boxed{(K + \tfrac{1}{2}I)\phi = V_\Gamma f - g}
$$

**KFBIM evaluation.** Call the interface solver with jumps $(\mu = \phi,\; \sigma = 0)$ and $f_{bulk} = 0$. Recover the exterior trace $\gamma_{ext}u = u_{avg} - \tfrac{1}{2}\phi$ of $-\mathcal{D}\phi$. It equals $-(K + \tfrac{1}{2}I)\phi$, so $(K + \tfrac{1}{2}I)\phi = -\gamma_{ext}u$.

### Exterior Neumann BVP

Find $u$ such that $-\Delta u + \lambda^2 u = f$ in $\Omega_{ext}$ with $\gamma_{n,ext}u = g$.

**Representation.** Single-layer ansatz ($\sigma = \psi$, $\mu = 0$):

$$
u(\mathbf{x}) = \mathcal{S}\psi(\mathbf{x}) + \mathcal{V}f(\mathbf{x}), \qquad \mathbf{x} \in \Omega_{ext}
$$

**Derivation of the BIE.** Take the exterior normal trace. Using the jump
relation $\gamma_{n,ext}\mathcal{S}\psi = K^*\psi - \tfrac{1}{2}\psi$:

$$
\begin{aligned}
g = \gamma_{n,ext}u &= \gamma_{n,ext}\mathcal{S}\psi + V_{n,\Gamma}f \\
&= (K^*\psi - \tfrac{1}{2}\psi) + V_{n,\Gamma}f
\end{aligned}
$$

Rearranging:

$$
\boxed{(K^* - \tfrac{1}{2}I)\psi = g - V_{n,\Gamma}f}
$$

**Nullspace ($\kappa = 0$).** For the Laplace equation, the operator $(K^* - \tfrac{1}{2}I)$ also has a one-dimensional nullspace in 2D, corresponding to the constant density. This is handled by the same projection technique as the interior Neumann case.

**KFBIM evaluation.** Call the interface solver with jumps $(\mu = 0,\; \sigma = \psi)$ and $f_{bulk} = 0$; recover the exterior normal trace $\gamma_{n,ext}u = u_{n,avg} - \tfrac{1}{2}\psi$, which gives $(K^* - \tfrac{1}{2}I)\psi$.

### BIE Summary

| BVP | Representation | BIE |
|:----|:---------------|:----|
| Interior Dirichlet | $u = -\mathcal{D}\phi + \mathcal{V}f$ | $(\tfrac{1}{2}I - K)\phi = g - V_\Gamma f$ |
| Interior Neumann   | $u = \mathcal{S}\psi + \mathcal{V}f$  | $(\tfrac{1}{2}I + K^*)\psi = g - V_{n,\Gamma}f$ |
| Exterior Dirichlet | $u = -\mathcal{D}\phi + \mathcal{V}f$ | $(K + \tfrac{1}{2}I)\phi = V_\Gamma f - g$ |
| Exterior Neumann   | $u = \mathcal{S}\psi + \mathcal{V}f$  | $(K^* - \tfrac{1}{2}I)\psi = g - V_{n,\Gamma}f$ |

## 6. Generic Interface Problems (Variable Coefficients)

The generic second-order elliptic interface problem is

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

We keep the KFBIM sign convention from the previous sections: the normal $n$
points from $\Omega_{int}$ to $\Omega_{ext}$ and jumps are interior minus
exterior. In this section write the prescribed interface data as

$$
[u]_\Gamma = a, \qquad [\beta\,\partial_n u]_\Gamma = b.
$$

Since $\beta$ is constant in each phase, division by $\beta$ gives

$$
\left(-\Delta+\lambda_s^2\right)u_s=q_s,\qquad
\lambda_s^2=\frac{\kappa_s^2}{\beta_s},\qquad
q_s=\frac{f_s}{\beta_s},
\qquad s\in\{int,ext\}.
$$

The key distinction is whether the screened ratio $\kappa^2/\beta$ has the
same value on the two sides. The current 2D API for both cases is
`LaplaceTransmission2D`, selected by `LaplaceTransmissionMode2D::CommonRatio`
or `LaplaceTransmissionMode2D::DifferentRatios`.

### 6.1 Common Ratio

First consider the case handled by `LaplaceTransmission2D` in `CommonRatio`
mode:

$$
\frac{\kappa^2_{int}}{\beta_{int}}
=
\frac{\kappa^2_{ext}}{\beta_{ext}}
\equiv \lambda^2.
$$

Then both phases use the same scalar operator:

$$
\left(-\Delta+\lambda^2\right)u=q,\qquad q=\frac{f}{\beta}.
$$

Because the operator is common, the trace jump can be imposed directly with the
double-layer jump primitive:

$$
u = \mathcal{S}_\lambda\psi-\mathcal{D}_\lambda a+\mathcal{V}_\lambda q.
$$

Indeed,

$$
[u]
=
[\mathcal{S}_\lambda\psi]
-[\mathcal{D}_\lambda a]
+[\mathcal{V}_\lambda q]
=0-(-a)+0
=a.
$$

The unknown single-layer density $\psi$ is determined by the flux jump. Using

$$
\partial_n u_{int}
=
\left(K_\lambda^*+\tfrac12 I\right)\psi
-\partial_n\mathcal{D}_\lambda a
+\partial_n\mathcal{V}_\lambda q,
$$

$$
\partial_n u_{ext}
=
\left(K_\lambda^*-\tfrac12 I\right)\psi
-\partial_n\mathcal{D}_\lambda a
+\partial_n\mathcal{V}_\lambda q,
$$

the condition
$\beta_{int}\partial_n u_{int}-\beta_{ext}\partial_n u_{ext}=b$
becomes

$$
\frac{\beta_{int}+\beta_{ext}}{2}\psi
+(\beta_{int}-\beta_{ext})K_\lambda^*\psi
=
b
+(\beta_{int}-\beta_{ext})
\left(
\partial_n\mathcal{D}_\lambda a
-\partial_n\mathcal{V}_\lambda q
\right).
$$

Equivalently,

$$
\boxed{
\left(
I+
2\frac{\beta_{int}-\beta_{ext}}{\beta_{int}+\beta_{ext}}
K_\lambda^*
\right)\psi
=
\frac{2b}{\beta_{int}+\beta_{ext}}
+
2\frac{\beta_{int}-\beta_{ext}}{\beta_{int}+\beta_{ext}}
\left(
\partial_n\mathcal{D}_\lambda a
-\partial_n\mathcal{V}_\lambda q
\right).
}
$$

This scalar second-kind BIE is the natural KFBIM reduction: after division by
$\beta$, the same screened bulk operator applies everywhere. If
$\beta_{int}=\beta_{ext}$, it reduces to $\psi=b/\beta_{int}$.

### 6.2 Different Ratios

When

$$
\lambda_{int}^2
\ne
\lambda_{ext}^2,
$$

there is no single Green's function or single KFBIM bulk solve that represents
both phases. One instead uses phase-dependent layer potentials. Let
$G_s$ solve

$$
\left(-\Delta_x+\lambda_s^2\right)G_s(x,y)=\delta(x-y),
\qquad s\in\{int,ext\},
$$

and define the potentials $\mathcal{S}_s,\mathcal{D}_s$ and boundary operators $S_s,K_s,K_s^*,H_s$ from $G_s$ using the same normal $n$ from
interior to exterior.

Nonzero forcing is handled by splitting off phase-specific volume potentials.
Let

$$
v_s=\mathcal{V}_s q_s,\qquad
\left(-\Delta+\lambda_s^2\right)v_s=q_s
\quad\text{in }\Omega_s,\qquad s\in\{int,ext\}.
$$

The particular solutions $v_s$ need not satisfy the interface jump conditions.
Only their traces enter the BIE. Write

$$
\gamma_s v_s=v_s|_\Gamma,\qquad
\gamma_{n,s}v_s=(\partial_n v_s)|_\Gamma,
$$

where the same normal $n$ from interior to exterior is used in both normal
traces. The full representation is

$$
u_{int}=v_{int}+\mathcal{S}_{int}\psi-\alpha_{int}\mathcal{D}_{int}\phi,
\qquad
u_{ext}=v_{ext}+\mathcal{S}_{ext}\psi-\alpha_{ext}\mathcal{D}_{ext}\phi.
$$

The coefficients are chosen to cancel the hypersingular principal part in the
flux equation:

$$
\boxed{
\alpha_{int}
=
\frac{2\beta_{ext}}{\beta_{int}+\beta_{ext}},
\qquad
\alpha_{ext}
=
\frac{2\beta_{int}}{\beta_{int}+\beta_{ext}}.
}
$$

With this normalization, the forced second-kind BIE is

$$
\boxed{
\begin{aligned}
\left[
I
-
\frac{2\beta_{ext}}{\beta_{int}+\beta_{ext}}K_{int}
+
\frac{2\beta_{int}}{\beta_{int}+\beta_{ext}}K_{ext}
\right]\phi
+
\left(S_{int}-S_{ext}\right)\psi
&=
a-\left(\gamma_{int}v_{int}-\gamma_{ext}v_{ext}\right),
\\[6pt]
\frac{2\beta_{int}\beta_{ext}}{\beta_{int}+\beta_{ext}}
\left(H_{ext}-H_{int}\right)\phi
+
\left[
\frac{\beta_{int}+\beta_{ext}}{2}I
+
\beta_{int}K_{int}^*
-
\beta_{ext}K_{ext}^*
\right]\psi
&=
b-\left(
\beta_{int}\gamma_{n,int}v_{int}
-\beta_{ext}\gamma_{n,ext}v_{ext}
\right).
\end{aligned}
}
$$

Equivalently, define the forcing-corrected jumps

$$
a_h
=
a-\left(\gamma_{int}v_{int}-\gamma_{ext}v_{ext}\right),
\qquad
b_h
=
b-\left(
\beta_{int}\gamma_{n,int}v_{int}
-\beta_{ext}\gamma_{n,ext}v_{ext}
\right).
$$

Then the same operator acts on the homogeneous unknown densities
$(\phi,\psi)$:

$$
\boxed{
\begin{aligned}
\left[
I
-
\frac{2\beta_{ext}}{\beta_{int}+\beta_{ext}}K_{int}
+
\frac{2\beta_{int}}{\beta_{int}+\beta_{ext}}K_{ext}
\right]\phi
+
\left(S_{int}-S_{ext}\right)\psi
&=a_h,\\[6pt]
\frac{2\beta_{int}\beta_{ext}}{\beta_{int}+\beta_{ext}}
\left(H_{ext}-H_{int}\right)\phi
+
\left[
\frac{\beta_{int}+\beta_{ext}}{2}I
+
\beta_{int}K_{int}^*
-
\beta_{ext}K_{ext}^*
\right]\psi
&=b_h.
\end{aligned}
}
$$

Equivalently, scaling the second row to have identity leading coefficient gives

$$
\boxed{
\begin{aligned}
\left[
I
-
\frac{2\beta_{ext}}{\beta_{int}+\beta_{ext}}K_{int}
+
\frac{2\beta_{int}}{\beta_{int}+\beta_{ext}}K_{ext}
\right]\phi
+
\left(S_{int}-S_{ext}\right)\psi
&=a_h,\\[6pt]
\frac{4\beta_{int}\beta_{ext}}{(\beta_{int}+\beta_{ext})^2}
\left(H_{ext}-H_{int}\right)\phi
+
\left[
I
+
\frac{2\beta_{int}}{\beta_{int}+\beta_{ext}}K_{int}^*
-
\frac{2\beta_{ext}}{\beta_{int}+\beta_{ext}}K_{ext}^*
\right]\psi
&=
\frac{2b_h}{\beta_{int}+\beta_{ext}}.
\end{aligned}
}
$$

The choice of $\alpha_{int}$ and $\alpha_{ext}$ enforces
$\beta_{int}\alpha_{int}=\beta_{ext}\alpha_{ext}$, so the leading
hypersingular pieces of $H_{int}$ and $H_{ext}$ cancel. The remaining
$H_{ext}-H_{int}$ term is lower order for a smooth interface, giving a
Fredholm second-kind system for the two unknown densities $(\phi,\psi)$. If
$q_{int}=q_{ext}=0$, then $v_{int}=v_{ext}=0$ and the system reduces to the
homogeneous BIE derived in `generic_if_bie.md`. For nonzero forcing, the
matrix-free BIE operator is unchanged; the phase-specific volume potentials
only shift the right-hand side.

In the C++ implementation, `LaplacePotentialEval2D` uses the jump primitive
`[u]=\phi`, which is the negative of the classical double-layer sign used in
the formulas above. Therefore `LaplaceTransmission2D::DifferentRatios` applies
the equivalent matrix-free rows

$$
\left(I+\alpha_{int}K_{int}-\alpha_{ext}K_{ext}\right)\phi
+(S_{int}-S_{ext})\psi=a_h,
$$

and

$$
\frac{4\beta_{int}\beta_{ext}}{(\beta_{int}+\beta_{ext})^2}
\left(H_{int}-H_{ext}\right)\phi
+
\left[
I+\frac{2\beta_{int}}{\beta_{int}+\beta_{ext}}K_{int}^*
-\frac{2\beta_{ext}}{\beta_{int}+\beta_{ext}}K_{ext}^*
\right]\psi
=\frac{2b_h}{\beta_{int}+\beta_{ext}}.
$$
