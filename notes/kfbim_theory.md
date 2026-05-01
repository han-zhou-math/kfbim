# Mathematical Foundation of the Kernel-Free Boundary Integral Method (KFBIM)

This note documents the mathematical framework for the elliptic PDE solvers in this repository.

## 1. Geometry Representation

The domain $\Omega \subset \mathbb{R}^d$ is partitioned by an interface $\Gamma$ into an interior region $\Omega_{int}$ and an exterior region $\Omega_{ext}$.

*   **Interface $\Gamma$**: Represented by a collection of panels. Each panel in 2D consists of 3 Gauss-Legendre quadrature points.
*   **Normals $\mathbf{n}$**: Unit outward normal pointing from $\Omega_{int}$ to $\Omega_{ext}$.
*   **Domain Labels $\ell$**: 
    *   $\ell(\mathbf{x}) = 1$ if $\mathbf{x} \in \Omega_{int}$.
    *   $\ell(\mathbf{x}) = 0$ if $\mathbf{x} \in \Omega_{ext}$.

## 2. The Constant Interface Problem (Poisson)

Consider the Poisson equation on a rectangular box $B = [0, L_x] \times [0, L_y]$ containing the interface $\Gamma$:

$$
-\Delta u = f \quad \text{in } B \setminus \Gamma
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
    $$-\Delta_{\mathbf{x}} G(\mathbf{x}, \mathbf{y}) = \delta(\mathbf{x} - \mathbf{y}) \quad \text{in } B, \qquad G(\mathbf{x}, \mathbf{y}) = 0 \quad \text{on } \partial B$$
    For the unit box $B = [0,1]^2$ with homogeneous Dirichlet BC, the eigenfunction expansion is:
    $$G(x,y; \xi,\eta) = 4\sum_{m=1}^{\infty}\sum_{n=1}^{\infty} \frac{\sin(m\pi x)\sin(n\pi y)\sin(m\pi \xi)\sin(n\pi \eta)}{\pi^2(m^2 + n^2)}$$

### Interface Problems Satisfied by Each Potential

Each potential is itself the solution to a specific interface problem on the box $B$:

| Potential | PDE in $B\setminus\Gamma$ | BC on $\partial B$ | $[u]$ on $\Gamma$ | $[\partial_n u]$ on $\Gamma$ |
|:---------|:--------------------------|:-------------------|:------------------|:----------------------------|
| $\mathcal{S}\sigma$ | $-\Delta u = 0$ | $u = 0$ | $0$ | $\sigma$ |
| $\mathcal{D}\mu$    | $-\Delta u = 0$ | $u = 0$ | $-\mu$ | $0$ |
| $\mathcal{V}f$      | $-\Delta u = f$ | $u = 0$ | $0$ | $0$ |

The double-layer sign $[u] = -\mu$ follows from the jump relation of the normal derivative of $G$:
$$\lim_{\mathbf{x}\to\Gamma^{\pm}} \int_\Gamma \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,\mu(\mathbf{y})\,ds_y = \mp\frac{1}{2}\mu + \int_\Gamma \frac{\partial G}{\partial n_y}(\mathbf{x},\mathbf{y})\,\mu(\mathbf{y})\,ds_y$$
so that $[\mathcal{D}\mu] = \mathcal{D}\mu|_{\Gamma^-} - \mathcal{D}\mu|_{\Gamma^+} = -\mu$ (with normal $\mathbf{n}$ pointing from $\Omega^+$ to $\Omega^-$).

By linearity, the full representation $u = \mathcal{S}\sigma - \mathcal{D}\mu + \mathcal{V}f$ satisfies:
$$-\Delta u = f \quad \text{in } B\setminus\Gamma, \qquad u = 0 \quad \text{on } \partial B, \qquad [u] = \mu, \qquad [\partial_n u] = \sigma$$
which recovers the constant interface problem of Section 2.

### KFBIM Approach
Instead of evaluating the layer-potential integrals (or the eigenfunction expansion) directly, KFBIM solves the interface problem on a uniform Cartesian grid using the **Immersed Interface Method (IIM)**. The box Green's function $G$ is never explicitly formed; applying it is equivalent to solving $-\Delta u = f$ with homogeneous BCs on $B$, which is done efficiently via FFT. 
1.  **Spread**: Convert jumps $(\mu, \sigma)$ into a defect correction for the bulk RHS $f$.
2.  **Bulk Solve**: Solve the corrected Poisson system $-\Delta_h u = F$ using a fast solver (FFT/ZFFT).
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

## 4. Generic Interface Problems (Variable Coefficients)

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

## 5. Boundary Value Problems (BVPs) to BIEs

BVPs are solved by formulating them as Boundary Integral Equations (BIEs) on $\Gamma$.

### Interior Dirichlet BVP

Find $u$ such that $-\Delta u = f$ in $\Omega_{int}$ with $u|_\Gamma = g$.

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

**KFBIM evaluation.** In each GMRES iteration, the left-hand operator $(\tfrac{1}{2}I - K)$ applied to a density $\phi$ is evaluated by calling the interface solver with jumps $(\mu = \phi,\; \sigma = 0)$ and $f_{bulk} = 0$. The output $u_{int}$ equals $(\tfrac{1}{2}I - K)\phi$ directly:

$$
(\tfrac{1}{2}I - K)\phi = -(\mathcal{D}\phi)|_{\Gamma^+} = u_{int} \quad \text{(from interface solver with } \mu=\phi,\; \sigma=0,\; f=0\text{)}
$$

The right-hand side $g - \mathcal{V}f|_\Gamma$ is assembled by computing $\mathcal{V}f|_\Gamma$ once via the interface solver with jumps $(0, 0)$ and $f_{bulk} = f$.

### Interior Neumann BVP

Find $u$ such that $-\Delta u = f$ in $\Omega_{int}$ with $\partial_n u|_\Gamma = g$.

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

**KFBIM evaluation.** $(\tfrac{1}{2}I + K')\psi$ is the interior normal derivative of $\mathcal{S}\psi$. Call the interface solver with jumps $(\mu = 0,\; \sigma = \psi)$ and $f_{bulk} = 0$; the output $un_{int}$ gives $(\tfrac{1}{2}I + K')\psi$ directly. The volume contribution $\partial_n(\mathcal{V}f)|_\Gamma$ is obtained via the interface solver with jumps $(0, 0)$ and $f_{bulk} = f$.

### Exterior Dirichlet BVP

Find $u$ such that $-\Delta u = f$ in $\Omega_{ext}$ with $u|_\Gamma = g$.

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

**KFBIM evaluation.** Call the interface solver with jumps $(\mu = \phi,\; \sigma = 0)$ and $f_{bulk} = 0$. The output $u_{ext}$ (exterior trace of $-\mathcal{D}\phi$) equals $-(K + \tfrac{1}{2}I)\phi$, so $(K + \tfrac{1}{2}I)\phi = -u_{ext}$.

### Exterior Neumann BVP

Find $u$ such that $-\Delta u = f$ in $\Omega_{ext}$ with $\partial_n u|_\Gamma = g$.

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

**KFBIM evaluation.** Call the interface solver with jumps $(\mu = 0,\; \sigma = \psi)$ and $f_{bulk} = 0$; the output $un_{ext}$ gives $(K' - \tfrac{1}{2}I)\psi$ directly.

### BIE Summary

| BVP | Representation | BIE |
|:----|:---------------|:----|
| Interior Dirichlet | $u = -\mathcal{D}\phi + \mathcal{V}f$ | $(\tfrac{1}{2}I - K)\phi = g - \mathcal{V}f\|_\Gamma$ |
| Interior Neumann   | $u = \mathcal{S}\psi + \mathcal{V}f$  | $(\tfrac{1}{2}I + K')\psi = g - \partial_n(\mathcal{V}f)\|_\Gamma$ |
| Exterior Dirichlet | $u = -\mathcal{D}\phi + \mathcal{V}f$ | $(K + \tfrac{1}{2}I)\phi = \mathcal{V}f\|_\Gamma - g$ |
| Exterior Neumann   | $u = \mathcal{S}\psi + \mathcal{V}f$  | $(K' - \tfrac{1}{2}I)\psi = g - \partial_n(\mathcal{V}f)\|_\Gamma$ |

## 6. Interface Solver Outputs (KFBIM Operator Evaluation)

The interface solver takes jump data $(\mu, \sigma)$ and bulk forcing $f$ and returns the volume solution $u_{bulk}$ together with the interior traces $u_{int}|_\Gamma$ and $\partial_n u_{int}|_\Gamma$. Each BIE operator is evaluated by calling the solver with the appropriate inputs:

| Operator | $(\mu, \sigma)$ | $f$ | Solver Output | Meaning |
|:---------|:---------------|:----|:--------------|:--------|
| $\tfrac{1}{2}I - K$ | $(\phi, 0)$ | $0$ | $u_{int}$ | interior trace of $-\mathcal{D}\phi$ |
| $\tfrac{1}{2}I + K'$ | $(0, \psi)$ | $0$ | $un_{int}$ | interior normal derivative of $\mathcal{S}\psi$ |
| $K + \tfrac{1}{2}I$ | $(\phi, 0)$ | $0$ | $-u_{ext}$ | negative exterior trace of $-\mathcal{D}\phi$ |
| $K' - \tfrac{1}{2}I$ | $(0, \psi)$ | $0$ | $un_{ext}$ | exterior normal derivative of $\mathcal{S}\psi$ |
| $\mathcal{V}$ | $(0, 0)$ | $f$ | $u_{int}$ | volume potential trace ($= u_{ext}$) |
| $\partial_n \mathcal{V}$ | $(0, 0)$ | $f$ | $un_{int}$ | volume potential normal derivative ($= un_{ext}$) |

The exterior traces $u_{ext}$ and $un_{ext}$ are not directly returned by the current solver interface but can be obtained from the interior traces and the jump relations:
$u_{ext} = u_{int} - \mu$, $\;un_{ext} = un_{int} - \sigma$.
