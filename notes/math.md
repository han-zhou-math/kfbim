# Mathematical Foundation of the Kernel-Free Boundary Integral Method (KFBIM)

This note documents the mathematical framework used by the current 2D elliptic
solvers in this repository. It follows the code conventions, which define
potential primitives by imposed jumps and use the active P2 quadratic-panel
transfer path.

## 1. Geometry, Traces, and Jumps

The computational box is partitioned by an interface $\Gamma$ into an interior
region $\Omega^+$ (domain label 1) and an exterior region $\Omega^-$ (domain
label 0).

* **Normal $\mathbf n$**: unit normal pointing from $\Omega^+$ to $\Omega^-$.
* **Jump convention**: interior minus exterior,
  $$[u] = u^+ - u^-,$$
  $$[\partial_n u] = \partial_n u^+ - \partial_n u^-.$$

The active 2D panel discretization uses P2 quadratic Lagrange nodes
$s=\{-1,0,1\}$ on each panel. Adjacent panels share endpoint DOFs. Therefore,
for a closed curve with $N_p$ panels, the interface vector length is

$$N_\Gamma = 2N_p,$$

namely $N_p$ shared endpoints plus $N_p$ panel midpoints.

The current active convergence programs set the target adjacent P2-node
spacing over the Cartesian grid spacing to $1.2$, equivalently

$$\frac{\text{panel length}}{h}=2.4.$$

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

The active direct interface convergence program, `tests/test_interface.cpp`,
uses this general prescribed-jump path without a GMRES outer solve. It
manufactures distinct interior and exterior sine-mode solutions that vanish on
the outer Cartesian box, prescribes $[u]$, $[\partial_n u]$, and
$[f]=f^+-f^-$ on $\Gamma$, and reports `GMRES=0`.

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

## 6. Discontinuous-Coefficient Transmission

Both transmission modes solve

$$-\nabla\cdot(\beta\nabla u)+\kappa^2u=f,$$

with piecewise constant positive $\beta$. In common-ratio mode,

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
equation implemented by `LaplaceTransmission2D::CommonRatio` is

$$\left(I+\gamma K'\right)\psi
=\frac{2}{\beta_++\beta_-}\sigma_\beta
-\gamma\left(H[\mu]+\partial_n N[[q]]\right).$$

Here $[q]=q^+-q^-$ is supplied through `rhs_derivs`, while the grid RHS stores
the piecewise reduced RHS $q$. After $\psi$ is solved, the full jump problem
$[u]=\mu$, $[\partial_n u]=\psi$ is run once to recover the bulk solution.

### Different Screened Ratios

`LaplaceTransmission2D::DifferentRatios` handles the piecewise-constant case
where

$$\lambda_+^2=\frac{\kappa_+^2}{\beta_+},\qquad
\lambda_-^2=\frac{\kappa_-^2}{\beta_-}$$

may differ. It uses separate reduced screened potential evaluators for the
interior and exterior phases and solves for two densities $(\phi,\psi)$.

Let

$$\alpha_+ = \frac{2\beta_-}{\beta_+ + \beta_-},\qquad
\alpha_- = \frac{2\beta_+}{\beta_+ + \beta_-},\qquad
c_\beta = \frac{2\beta_+\beta_-}{\beta_+ + \beta_-}.$$

With phase-specific operators $K_\pm,H_\pm,S_\pm,K'_\pm$, the homogeneous
two-density operator is

$$A_u(\phi,\psi)
= \phi+\alpha_+K_+[\phi]-\alpha_-K_-[\phi]
  +S_+[\psi]-S_-[\psi],$$

and

$$A_\beta(\phi,\psi)
= \psi+\frac{2}{\beta_+ + \beta_-}
\left(c_\beta(H_+[\phi]-H_-[\phi])
      +\beta_+K'_+[\psi]-\beta_-K'_-[\psi]\right).$$

Volume terms are evaluated separately on each side. If $V_\pm$ denotes the
phase-specific reduced volume solution, GMRES solves

$$A_u(\phi,\psi)=\mu-(V_+-V_-),$$

$$A_\beta(\phi,\psi)
=\frac{2}{\beta_+ + \beta_-}
\left(\sigma_\beta
      -(\beta_+\partial_n V_+ - \beta_-\partial_n V_-)\right).$$

After the densities are found, the code performs separate interior and exterior
potential evaluations and selects the grid value from the matching domain.

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

## 8. Implementation Source Map

The current C++ implementation is split by algorithmic layer:

| Directory | Main responsibility |
|-----------|---------------------|
| `src/grid/` | Cartesian and MAC grid data structures |
| `src/interface/` | 2D/3D interface containers and panel node-layout metadata |
| `src/geometry/` | curve resampling and grid/interface pairing |
| `src/local_cauchy/` | jump data, local polynomials, and Laplace panel Cauchy solver |
| `src/transfer/` | Laplace Lobatto spread and restrict transfer operators |
| `src/bulk_solvers/` | bulk-solver interface, FFT/zFFT engines, Laplace bulk solvers, and IIM helpers |
| `src/potentials/` | `LaplacePotentialEval2D` and reusable potential-evaluation pipeline |
| `src/operators/` | `IKFBIOperator`, `LaplaceBvp2D`, `LaplaceTransmission2D`, and Stokes scaffold |
| `src/gmres/` | matrix-free outer Krylov solver |

The former `src/solver/`, `src/operator/`, and `src/problems/` locations have
been replaced by `src/bulk_solvers/`, `src/potentials/`, and `src/operators/`.

## 9. Current Operator Modes

| Mode/API | Unknown | Applied operator |
|----------|---------|------------------|
| `LaplaceBvpType2D::InteriorDirichlet` | $[u]=\phi$ | $K[\phi]+\frac12\phi$ |
| `LaplaceBvpType2D::ExteriorDirichlet` | $[u]=\phi$ | $K[\phi]-\frac12\phi$ |
| `LaplaceBvpType2D::InteriorNeumann` | $[\partial_n u]=\psi$ | $K'[\psi]+\frac12\psi$ |
| `LaplaceBvpType2D::ExteriorNeumann` | $[\partial_n u]=\psi$ | $K'[\psi]-\frac12\psi$ |
| `LaplaceTransmission2D::CommonRatio` | $\psi=[\partial_n u]$ | $(I+\gamma K')\psi$ |
| `LaplaceTransmission2D::DifferentRatios` | $(\phi,\psi)$ | $(A_u(\phi,\psi),A_\beta(\phi,\psi))$ |

The active 2D convergence programs use P2 quadratic panels with target
adjacent P2-node spacing/h `1.2` (`panel_length/h = 2.4`). The direct
interface and transmission tests use an off-center 3-fold star, while
`tests/test_bvp.cpp` uses an off-center ellipse and reports both max error and
RMS error.
