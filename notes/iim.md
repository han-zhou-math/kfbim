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

## Defect Correction for a Generic Discrete Operator

Let the domain labels be $\ell_n \in \{0,1\}$ (0 = exterior, 1 = interior).
Equivalently, let
$$
H_n = \ell_n,\qquad H^-_n = 1-H_n,
$$
where $H_n$ is the discrete Heaviside/indicator of the interior side.

Let $A_h$ be any linear finite-difference operator. At row $n$ write
$$
(A_h U)_n = \sum_{m\in S_n} a_{nm} U_m,
$$
where $S_n$ is the stencil of node $n$. Assume $A_h$ approximates the
piecewise PDE operator $A$ so that, on either smooth branch,
$$
(A_h u_{int})_n = f_{int,n} + O(h^p),\qquad
(A_h u_{ext})_n = f_{ext,n} + O(h^p).
$$
The branch right-hand side at node $n$ is
$$
f_n = H_n f_{int,n} + (1-H_n)f_{ext,n}.
$$

The physical grid function is
$$
U_m = H_m u_{int}(\mathbf{x}_m) + (1-H_m)u_{ext}(\mathbf{x}_m).
$$
For the row at node $n$, however, the finite-difference stencil should be
applied entirely to the smooth branch containing the center node. Define
$$
u^{(\ell_n)} =
\begin{cases}
u_{int}, & \ell_n=1,\\
u_{ext}, & \ell_n=0.
\end{cases}
$$
Using $C=u_{int}-u_{ext}$, every stencil value can be written as
$$
U_m = u^{(\ell_n)}(\mathbf{x}_m) + (H_m-H_n)C_m.
$$
Here $C_m$ is the correction function evaluated at the grid node
$\mathbf{x}_m$, not only on the interface.
Therefore
$$
\begin{aligned}
(A_h U)_n
&= \sum_{m\in S_n} a_{nm}u^{(\ell_n)}(\mathbf{x}_m)
  + \sum_{m\in S_n} a_{nm}(H_m-H_n)C_m \\
&= f_n + \sum_{m\in S_n} a_{nm}(H_m-H_n)C_m + O(h^p).
\end{aligned}
$$

Thus the corrected right-hand side for
$$
A_h U = F
$$
is
$$
\boxed{
F_n = f_n + \sum_{m\in S_n} a_{nm}(H_m-H_n)C_m
}
$$
or, in matrix notation with $H_h=\operatorname{diag}(H_n)$ and $C_h=(C_n)$,
$$
\boxed{
F = f + \bigl(A_h H_h - H_h A_h\bigr)C_h.
}
$$
The correction is the commutator of the discrete operator with the discrete
Heaviside function, applied to the correction function. It vanishes at regular
nodes because $H_m=H_n$ for every stencil point $m\in S_n$.

### Where the Nodal Correction Function Is Needed

The formula above also gives the minimal set of grid nodes where $C_h$ must be
available. For a row $n$, the correction term is
$$
\mathcal{C}_n
= \sum_{m\in S_n} a_{nm}(H_m-H_n)C_m.
$$
Thus $C_m$ is used only if all of the following hold:

1. $m$ belongs to the stencil of some active row $n$,
2. the stencil coefficient $a_{nm}$ is nonzero,
3. $m$ lies on the opposite side of the interface from the row center:
   $H_m\neq H_n$.

Define the irregular row set for the operator $A_h$ by
$$
I_h(A)
= \{n:\exists m\in S_n \text{ with } a_{nm}\neq 0
       \text{ and } H_m\neq H_n\}.
$$
Then the minimal nodal support needed for the RHS correction is
$$
\boxed{
Q_h(A)
= \{m:\exists n\in I_h(A)
       \text{ with } m\in S_n,\ a_{nm}\neq 0,\ H_m\neq H_n\}.
}
$$
Equivalently, $Q_h(A)$ is the set of opposite-side stencil nodes seen by
irregular rows. If $m\notin Q_h(A)$, setting $C_m=0$ does not change the
corrected right-hand side.

For the standard five-point operator $A_h=-\Delta_h$, the diagonal term at
$m=n$ never contributes because $H_n-H_n=0$. Therefore $C_m$ is needed only at
face-neighbor nodes across the interface from an irregular center node. This is
a subset of the one-grid-cell narrow band around $\Gamma$.

There are two useful implementation choices:

1. **Minimal row-sum implementation.** Loop over active rows $n$ and stencil
   nodes $m\in S_n$. Add $a_{nm}(H_m-H_n)C_m$ only when $H_m\neq H_n$. In this
   implementation, $C_m$ only needs valid values on $Q_h(A)$ and can be zero
   elsewhere.

2. **Masked-operator implementation.** Compute the Stokes-paper form
   $-H_n(A_h(CH^-))_n + (1-H_n)(A_h(CH^+))_n$ at irregular rows. This still
   uses only opposite-side stencil values after masking, so it has the same
   mathematical support $Q_h(A)$. In code, however, it is convenient to allocate
   $C_h$ on all grid nodes and fill it only on $Q_h(A)$; the products $CH^-$
   and $CH^+$ automatically erase same-side values.

For higher-order, compact, or staggered-grid operators, the same definition
applies with the appropriate row grid and stencil. The required correction
support is the opposite-side stencil footprint of the irregular rows, not the
entire Cartesian grid. A safe practical rule is to evaluate $C_h$ on every grid
node within one stencil radius of $\Gamma$ and set it to zero outside that
band; the minimal rule above trims this further to only the opposite-side
stencil nodes actually used by crossing rows.

In implementation, this support can be marked before evaluating the local
correction function:

```text
needs_C[:] = false
for each active operator row n:
    Hn = label(row_point[n])
    for each stencil entry (m, a_nm) in S_n:
        if a_nm != 0 and label(node_point[m]) != Hn:
            needs_C[m] = true
```

Then evaluate the local Cauchy/correction polynomial only at nodes with
`needs_C[m] = true`, and set all other entries of `C_h` to zero. The RHS update
can be assembled with the same row loop:

```text
F[n] = f[n]
for each stencil entry (m, a_nm) in S_n:
    F[n] += a_nm * (H[m] - H[n]) * C[m]
```

Rows for which the stencil does not cross $\Gamma$ need no correction. Boundary
treatments should use the actual stencil after boundary conditions have been
eliminated or imposed, because those changes may remove or alter coefficients.

For restriction/interpolation, the same logic applies. To reconstruct the
branch with label $s\in\{0,1\}$ from bulk samples in an interpolation stencil
$R_q$, use
$$
u^{(s)}(\mathbf{x}_m) = U_m + (s-H_m)C_m.
$$
Thus $C_m$ is needed for restriction only when $m\in R_q$ and $H_m\neq s$.
For interior trace interpolation, this means exterior stencil nodes; for
exterior trace interpolation, this means interior stencil nodes. If one
interpolates the average trace instead, use
$$
u_{avg}(\mathbf{x}_m)=U_m+\left(\frac12-H_m\right)C_m,
$$
so every interpolation stencil node that participates in an average-trace
correction needs $C_m$ unless the coefficient multiplying it is known to cancel
in the chosen interpolant.

## Screened-Poisson Surface Decomposition

For a common screened Poisson operator on the two smooth branches,
$$
-\Delta u^\pm + \kappa^2 u^\pm = f^\pm,
$$
the correction function
$$
C = u_{int}-u_{ext}
$$
satisfies, on the interface and by smooth extension from either side,
$$
\boxed{
-\Delta C + \kappa^2 C = [f].
}
$$
Here $[f]=f_{int}-f_{ext}$ uses the same interior-minus-exterior jump
convention as $C=[u]$.

Let $n$ be the outward normal, let $r$ be signed distance in the normal
direction, and let $\Delta_\Gamma$ be the surface Laplace-Beltrami operator.
With the mean-curvature convention
$$
H_\Gamma = \nabla\cdot n
$$
at the interface, the Euclidean Laplacian decomposes at $\Gamma$ as
$$
\boxed{
\Delta C
= \partial_{nn}C + H_\Gamma\,\partial_n C + \Delta_\Gamma C.
}
$$
Therefore the screened correction equation is
$$
\boxed{
-\partial_{nn}C
- H_\Gamma\,\partial_n C
- \Delta_\Gamma C
+ \kappa^2 C
= [f].
}
$$
Equivalently, the second normal derivative needed by a normal Taylor
correction can be eliminated by
$$
\boxed{
\partial_{nn}C
= \kappa^2 C - [f] - H_\Gamma\,\partial_n C - \Delta_\Gamma C.
}
$$
In 2D, $\Gamma$ is a curve, $H_\Gamma$ is the signed curvature
$\nabla\cdot n$, and $\Delta_\Gamma C=\partial_{\tau\tau}C$. In 3D,
$H_\Gamma$ is the sum of the two principal curvatures and
$\Delta_\Gamma$ is the surface Laplacian on the triangular/P2 surface.

## 3D P2 Projection-Point Geometry

For shared quadratic triangular interfaces,
`GridPair3D::project_near_interface_nodes(radius)` supplies broad narrow-band
geometry projection data. The transfer implementation uses
`GridPair3D::project_grid_nodes_to_interface(nodes)` for projection-point
correction so it projects only the grid nodes where `C(x)` is needed.

For each narrow-band grid node, the projection record includes:

- the grid-node index,
- the parent P2 triangle,
- the barycentric coordinate in the reference triangle,
- the projected physical point,
- the oriented surface normal,
- the signed normal distance from projection to grid node,
- a tangential residual, iteration count, and convergence flag.

For the broad radius-band geometry query, the broad-phase seeds are the nearest
of the 16 P2 expansion-center locations per parent triangle. For the explicit
transfer-support query, the nearest expansion center is used only to choose the
parent triangle; the initial parameter is then the closest point on the flat
vertex triangle. A damped Newton iteration solves the local closest-point
normality equations on the curved patch. If Newton fails, the explicit-support
path keeps the flat-triangle parameter with `converged=false`; downstream code
can still use the stored panel and barycentric coordinate for interpolation,
while treating the convergence flag as a diagnostic.

This supports the 3D projection-point IIM correction evaluation. Let
$p$ be the projection of a grid node $x$, let $n(p)$ be the outward normal, and
let
$$
s = (x-p)\cdot n(p).
$$
After interpolating surface data to the returned panel-local coordinate, the
second-order normal Taylor correction is
$$
C(x) \approx C(p) + s\,\partial_n C(p)
        + \frac12 s^2\,\partial_{nn}C(p).
$$
In the implemented projection-point transfer path, `C`, `partial_n C`, and the
surface Laplacian term are interpolated with P2 data at the projected
panel-local coordinate, and `partial_{nn} C` is computed from the surface
decomposition of the screened Laplacian. To avoid the earlier runtime
bottleneck, spread precomputes the union of grid nodes needed by crossing
finite-difference stencil edges and by the restrict interpolation stencils, then
stores one projection cache for both spread and restrict.

The production default remains the existing nearest expansion-center correction
polynomial path. The projection-point path is available through
`LaplaceCorrectionMethod3D::ProjectionPoint` for comparison and further
accuracy work; on the current unit-sphere direct-interface benchmark the
nearest expansion-center path is more accurate on the finest tested levels.

### Equivalent Stokes-Paper Indicator Form

This is the scalar analogue of the correction-function derivation in the
Stokes/Brinkman paper. With $H^+=H$ and $H^-=1-H$, the exact piecewise solution
can be written locally as
$$
U =
\begin{cases}
u_{int} - C H^-, & \mathbf{x}_n\in\Omega^+,\\
u_{ext} + C H^+, & \mathbf{x}_n\in\Omega^-.
\end{cases}
$$
Plugging the exact solution into the discrete operator at an irregular row gives
$$
(A_h U)_n =
\begin{cases}
(A_h u_{int})_n - (A_h(CH^-))_n, & H_n=1,\\
(A_h u_{ext})_n + (A_h(CH^+))_n, & H_n=0.
\end{cases}
$$
Hence the corrected scheme may also be written as
$$
\boxed{
F_n
= f_n
- H_n\,(A_h(CH^-))_n
+ (1-H_n)\,(A_h(CH^+))_n.
}
$$
Since $H^++H^-=1$, this is identical to
$$
F_n = f_n + (A_h(HC))_n - H_n(A_h C)_n
    = f_n + \sum_{m\in S_n}a_{nm}(H_m-H_n)C_m.
$$

## Poisson Specialization

The corrected right-hand side $F_n$ for $-\Delta_h u = F_n$ is:

$$
\boxed{
  F_n = f_n + \sum_{\substack{n_b \sim n \\ \ell_{n_b} \neq \ell_n}}
        \frac{\ell_n - \ell_{n_b}}{h^2}\, C(\mathbf{x}_{n_b})
}
$$

### Detailed Poisson Derivation

We want to solve the Poisson equation $-\Delta u = f$ using a standard central difference scheme on a Cartesian grid with spacing $h$. The numerical solution we are solving for, let's call it $U$, is defined to match the piecewise smooth solution on the exact side of the interface where each node lies:
$$
U_n = \begin{cases}
u_{int}(\mathbf{x}_n) & \text{if node } n \text{ is interior } (\ell_n = 1) \\
u_{ext}(\mathbf{x}_n) & \text{if node } n \text{ is exterior } (\ell_n = 0)
\end{cases}
$$

The standard discrete Laplacian $-\Delta_h$ at node $n$ is:
$$
-\Delta_h U_n = \frac{1}{h^2}\left(2d \cdot U_n - \sum_{n_b \sim n} U_{n_b}\right)
$$
where $d$ is the number of spatial dimensions (e.g., $d=2$ for a 5-point stencil) and the sum is over all face-adjacent neighbors $n_b$ of node $n$.

#### Case 1: Interior Node ($\ell_n = 1$)

Suppose the central node $n$ is inside the interface ($\ell_n = 1$). The physical PDE we want the stencil to approximate at this point is $-\Delta u_{int}(\mathbf{x}_n) = f(\mathbf{x}_n)$.

For the standard difference scheme to yield $O(h^2)$ accuracy for $u_{int}$, it **must** evaluate $u_{int}$ at all stencil nodes. Let's see what the discrete Laplacian actually computes using the global grid values $U$:

$$
\begin{aligned}
-\Delta_h U_n &= \frac{1}{h^2} \left( 2d \cdot u_{int}(\mathbf{x}_n) - \sum_{n_b \sim n} U_{n_b} \right) \\
&= \frac{1}{h^2} \Bigg( 2d \cdot u_{int}(\mathbf{x}_n) - \sum_{\ell_{n_b} = 1} u_{int}(\mathbf{x}_{n_b}) - \sum_{\ell_{n_b} = 0} u_{ext}(\mathbf{x}_{n_b}) \Bigg)
\end{aligned}
$$

For the exterior neighbors ($\ell_{n_b} = 0$), the grid provides $u_{ext}$, but the finite difference stencil *needs* $u_{int}$. We use the jump definition $C = u_{int} - u_{ext}$ to rewrite the grid value in terms of the branch we want:
$$
u_{ext}(\mathbf{x}_{n_b}) = u_{int}(\mathbf{x}_{n_b}) - C(\mathbf{x}_{n_b})
$$

Substitute this back into the discrete Laplacian:
$$
\begin{aligned}
-\Delta_h U_n &= \frac{1}{h^2} \Bigg( 2d \cdot u_{int}(\mathbf{x}_n) - \sum_{\ell_{n_b} = 1} u_{int}(\mathbf{x}_{n_b}) - \sum_{\ell_{n_b} = 0} \Big[ u_{int}(\mathbf{x}_{n_b}) - C(\mathbf{x}_{n_b}) \Big] \Bigg) \\
&= \underbrace{\frac{1}{h^2} \Bigg( 2d \cdot u_{int}(\mathbf{x}_n) - \sum_{n_b \sim n} u_{int}(\mathbf{x}_{n_b}) \Bigg)}_{\text{Pure interior Laplacian } \approx -\Delta u_{int}(\mathbf{x}_n)} + \sum_{\ell_{n_b} = 0} \frac{C(\mathbf{x}_{n_b})}{h^2}
\end{aligned}
$$

Since $-\Delta u_{int}(\mathbf{x}_n) = f_n$, the equation becomes:
$$
-\Delta_h U_n = f_n + \sum_{\ell_{n_b} = 0} \frac{C(\mathbf{x}_{n_b})}{h^2}
$$
To enforce this, we must set our numerical right-hand side $F_n$ to the expression on the right. Note that since $\ell_n = 1$ and $\ell_{n_b} = 0$, the coefficient is $+1$, which perfectly matches $(\ell_n - \ell_{n_b}) = 1 - 0 = 1$.

#### Case 2: Exterior Node ($\ell_n = 0$)

Now suppose the central node $n$ is outside the interface ($\ell_n = 0$). Here, the stencil intends to approximate $-\Delta u_{ext}(\mathbf{x}_n) = f(\mathbf{x}_n)$, so it *needs* $u_{ext}$ at all nodes.

$$
\begin{aligned}
-\Delta_h U_n &= \frac{1}{h^2} \Bigg( 2d \cdot u_{ext}(\mathbf{x}_n) - \sum_{\ell_{n_b} = 0} u_{ext}(\mathbf{x}_{n_b}) - \sum_{\ell_{n_b} = 1} u_{int}(\mathbf{x}_{n_b}) \Bigg)
\end{aligned}
$$

This time, the "wrong" neighbors are the interior ones ($\ell_{n_b} = 1$). We rewrite their grid values $u_{int}$ in terms of the needed branch $u_{ext}$:
$$
u_{int}(\mathbf{x}_{n_b}) = u_{ext}(\mathbf{x}_{n_b}) + C(\mathbf{x}_{n_b})
$$

Substitute this back:
$$
\begin{aligned}
-\Delta_h U_n &= \frac{1}{h^2} \Bigg( 2d \cdot u_{ext}(\mathbf{x}_n) - \sum_{\ell_{n_b} = 0} u_{ext}(\mathbf{x}_{n_b}) - \sum_{\ell_{n_b} = 1} \Big[ u_{ext}(\mathbf{x}_{n_b}) + C(\mathbf{x}_{n_b}) \Big] \Bigg) \\
&= \underbrace{\frac{1}{h^2} \Bigg( 2d \cdot u_{ext}(\mathbf{x}_n) - \sum_{n_b \sim n} u_{ext}(\mathbf{x}_{n_b}) \Bigg)}_{\text{Pure exterior Laplacian } \approx -\Delta u_{ext}(\mathbf{x}_n)} - \sum_{\ell_{n_b} = 1} \frac{C(\mathbf{x}_{n_b})}{h^2}
\end{aligned}
$$

Since $-\Delta u_{ext}(\mathbf{x}_n) = f_n$, we get:
$$
-\Delta_h U_n = f_n - \sum_{\ell_{n_b} = 1} \frac{C(\mathbf{x}_{n_b})}{h^2}
$$
In this scenario, $\ell_n = 0$ and $\ell_{n_b} = 1$, so the coefficient is $-1$, which perfectly matches $(\ell_n - \ell_{n_b}) = 0 - 1 = -1$.

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
