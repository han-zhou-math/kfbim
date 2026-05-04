# Mathematical Foundation of the Kernel-Free Boundary Integral Method (KFBIM)

This note documents the mathematical framework used by the current 2D elliptic
solvers in this repository. It follows the code conventions, which define
potential primitives by imposed jumps and use the active Chebyshev-Lobatto
transfer path.

## 1. Geometry, Traces, and Jumps

The computational box is partitioned by an interface $\Gamma$ into an interior
region $\Omega^+$ (domain label 1) and an exterior region $\Omega^-$ (domain
label 0).

* **Normal $\mathbf n$**: unit normal pointing from $\Omega^+$ to $\Omega^-$.
* **Jump convention**: interior minus exterior,
  $$[u] = u^+ - u^-,$$
  $$[\partial_n u] = \partial_n u^+ - \partial_n u^-.$$

The active 2D panel discretization uses Chebyshev-Lobatto nodes
$s=\{-1,0,1\}$ on each panel. Adjacent panels share endpoint DOFs. Therefore,
for a closed curve with $N_p$ panels, the interface vector length is

$$N_\Gamma = 2N_p,$$

namely $N_p$ shared endpoints plus $N_p$ panel midpoints.

## 2. Restriction Returns the Average Branch

Let $C$ be the local correction polynomial produced by the panel Cauchy solve,
with

$$C=[u],\qquad \partial_n C=[\partial_n u]\quad\text{on }\Gamma.$$

The active restrictor no longer interpolates an interior-side trace and then
post-processes it. Instead, it first maps each side-specific grid sample to the
average branch:

$$u_{\mathrm{avg}} =
\begin{cases}
u_{\mathrm{grid}}-\frac12 C, & \text{interior-side sample},\\
u_{\mathrm{grid}}+\frac12 C, & \text{exterior-side sample}.
\end{cases}$$

The same rule is applied to the local derivative data. Interpolation of these
shifted samples returns the averaged trace and averaged normal derivative
directly:

$$u_{avg}=\frac{u^+ + u^-}{2},\qquad
(\partial_n u)_{avg}=\frac{\partial_n u^+ + \partial_n u^-}{2}.$$

Selected-side values are reconstructed only after restriction:

$$u^\pm = u_{avg}\pm\frac12[u],\qquad
\partial_n u^\pm = (\partial_n u)_{avg}\pm\frac12[\partial_n u].$$

The plus sign is for the interior side and the minus sign is for the exterior
side.

## 3. Constant-Coefficient Poisson/Screened Interface Problem

The active 2D KFBI pipeline solves interface problems for

$$-\Delta u+\eta u=f\quad\text{in}\quad \Omega^+\cup\Omega^-,$$

with jumps

$$[u]=\mu,\qquad [\partial_n u]=\sigma.$$

Here $\eta$ is the screened coefficient. In the BVP tests this is written as
$\kappa^2$ and currently set to $\eta=\kappa^2=1.1$.

`LaplacePotentialEval2D::evaluate()` is the general pipeline:

1. spread arbitrary jumps and RHS derivative data into a corrected bulk RHS;
2. solve the homogeneous-box bulk problem;
3. restrict the bulk solution to averaged trace/flux values.

The specialized potential helpers are thin wrappers:

| Primitive | Imposed jumps | Returned trace/flux operators |
|-----------|---------------|-------------------------------|
| $D[\phi]$ | $[u]=\phi$, $[\partial_n u]=0$, $f=0$ | $K[\phi]=u_{avg}$, $H[\phi]=(\partial_n u)_{avg}$ |
| $S[\psi]$ | $[u]=0$, $[\partial_n u]=\psi$, $f=0$ | $S[\psi]=u_{avg}$, $K'[\psi]=(\partial_n u)_{avg}$ |
| $N[q]$ | $[u]=0$, $[\partial_n u]=0$, $[f]=q$ | $N[q]=u_{avg}$, $\partial_n N[q]=(\partial_n u)_{avg}$ |

Because $D[\phi]$ is defined by the code jump $[u]=\phi$, its $K$ sign is the
code sign. It is the opposite of the classical double-layer sign if the
classical density is defined so that the double-layer potential has jump
$-\phi$.

## 4. Dirichlet BVPs

The Dirichlet wrappers solve

$$-\Delta u+\eta u=f,\qquad u=g\quad\text{on }\Gamma,$$

in the selected physical domain.

They use the double-layer jump unknown $[u]=\phi$, $[\partial_n u]=0$. With
$s=+1$ for the interior side and $s=-1$ for the exterior side, the selected-side
Dirichlet operator is

$$A_D^{(s)}\phi = K[\phi]+\frac{s}{2}\phi.$$

Thus

$$A_D^{(+)}\phi = K[\phi]+\frac12\phi
\quad\text{(interior Dirichlet)},$$

and

$$A_D^{(-)}\phi = K[\phi]-\frac12\phi
\quad\text{(exterior Dirichlet)}.$$

For nonzero volume forcing, the code evaluates the selected-side contribution
of the volume problem, denoted $V_f^{(s)}$, by applying the full affine operator
with zero density. GMRES solves

$$A_D^{(s)}\phi = g - V_f^{(s)}.$$

If the physical domain is exterior, the supplied RHS derivative data represent
the exterior branch and are inserted with a negative jump sign because
$[f]=f^+ - f^-$.

## 5. Neumann BVPs

The Neumann wrappers solve

$$-\Delta u+\eta u=f,\qquad \partial_n u=g\quad\text{on }\Gamma,$$

using the single-layer jump unknown $[\partial_n u]=\psi$, $[u]=0$.

With $s=+1$ for interior and $s=-1$ for exterior, the selected-side Neumann
operator is

$$A_N^{(s)}\psi = K'[\psi]+\frac{s}{2}\psi.$$

That is,

$$A_N^{(+)}\psi = K'[\psi]+\frac12\psi
\quad\text{(interior Neumann)},$$

and

$$A_N^{(-)}\psi = K'[\psi]-\frac12\psi
\quad\text{(exterior Neumann)}.$$

For screened problems ($\eta>0$) there is no constant nullspace and no
projection is applied. For the pure Laplace interior Neumann case ($\eta=0$),
the implementation projects the RHS, the operator input/output, and the final
density to mean zero using the plain vector mean. This is not quadrature
weighted.

## 6. Constant-Ratio Discontinuous-Coefficient Transmission

The implemented first transmission case solves

$$-\nabla\cdot(\beta\nabla u)+\kappa^2u=f,$$

with piecewise constant positive $\beta$ and

$$\frac{\kappa_+^2}{\beta_+}
=\frac{\kappa_-^2}{\beta_-}
=\lambda^2.$$

After division by $\beta$, both sides use the same screened operator

$$-\Delta u+\lambda^2u=q,\qquad q=f/\beta.$$

The interface data are

$$[u]=\mu,\qquad
[\beta\partial_n u]=\sigma_\beta
=\beta_+\partial_n u^+-\beta_-\partial_n u^-.$$

The solver iterates on the ordinary normal-derivative jump

$$\psi=[\partial_n u].$$

Let

$$\gamma=\frac{2(\beta_+-\beta_-)}{\beta_++\beta_-}.$$

Using the screened potential primitives for the reduced operator, the GMRES
equation implemented in `LaplaceTransmissionConstantRatio2D` is

$$\left(I+\gamma K'\right)\psi
=\frac{2}{\beta_++\beta_-}\sigma_\beta
-\gamma\left(H[\mu]+\partial_n N[[q]]\right).$$

Here $[q]=q^+-q^-$ is supplied through `rhs_derivs`, while the grid RHS stores
the piecewise reduced RHS $q$. After $\psi$ is solved, the full jump problem
$[u]=\mu$, $[\partial_n u]=\psi$ is run once to recover the bulk solution.

## 7. Nonzero Outer Cartesian Dirichlet Data

The FFT/DST bulk solver uses homogeneous Dirichlet data on the Cartesian box.
When a problem has nonzero box values $b$, the BVP/transmission wrappers
eliminate them into the finite-difference RHS:

* for interior grid nodes adjacent to the box boundary, add neighboring
  boundary values divided by the corresponding $h^2$;
* set boundary RHS entries to zero for the homogeneous solve;
* after the solve, restore the boundary entries of `u_bulk` to $b$.

This keeps the bulk solver interface homogeneous while allowing manufactured
tests with nonzero outer boundary conditions.

## 8. Current Operator Modes

| Mode/API | Unknown | Applied operator |
|----------|---------|------------------|
| `LaplaceKFBIMode::InteriorDirichlet` | $[u]=\phi$ | $K[\phi]+\frac12\phi$ |
| `LaplaceKFBIMode::ExteriorDirichlet` | $[u]=\phi$ | $K[\phi]-\frac12\phi$ |
| `LaplaceKFBIMode::InteriorNeumann` | $[\partial_n u]=\psi$ | $K'[\psi]+\frac12\psi$ |
| `LaplaceKFBIMode::ExteriorNeumann` | $[\partial_n u]=\psi$ | $K'[\psi]-\frac12\psi$ |
| `LaplaceTransmissionConstantRatio2D` | $\psi=[\partial_n u]$ | $(I+\gamma K')\psi$ |

The active convergence test for the four BVP modes is `tests/test_bvp.cpp`.
It uses the unit circle centered at the origin, box `(-1.7,1.7)^2`,
target interface spacing/h about `1.5`, and a nontrivial sine/cosine exact
solution for the screened equation.
