# Mathematical Foundation of the Kernel-Free Boundary Integral Method (KFBIM)

This note documents the mathematical framework used by the current 2D elliptic
solvers in this repository. It follows the code conventions, which define
potential primitives by their imposed jumps.

## 1. Geometry Representation

The computational box is partitioned by an interface $\Gamma$ into an interior
region $\Omega^+$ (domain label 1) and an exterior region $\Omega^-$ (domain
label 0).

* **Normal $\mathbf n$**: unit normal pointing from $\Omega^+$ to $\Omega^-$.
* **Jump convention**: interior minus exterior,
  $$[u] = u^+ - u^-,$$
  $$[\partial_n u] = \partial_n u^+ - \partial_n u^-.$$

`LaplaceInterfaceSolver2D` restricts the bulk solution to the interior side and
then reports averages:

$$u_{avg}=u^+-\frac12[u], \qquad
(\partial_n u)_{avg}=\partial_n u^+-\frac12[\partial_n u].$$

## 2. Constant-Coefficient Poisson/Screened Interface Problem

The active 2D KFBI pipeline solves interface problems for

$$-\Delta u+\eta u=f\quad\text{in}\quad \Omega^+\cup\Omega^-,$$

with jumps

$$[u]=\mu,\qquad [\partial_n u]=\sigma.$$

The current potential evaluator exposes the following jump primitives:

| Primitive | Imposed jumps | Returned trace/flux operators |
|-----------|---------------|-------------------------------|
| $D[\phi]$ | $[u]=\phi$, $[\partial_n u]=0$, $f=0$ | $K[\phi]=u_{avg}$, $H[\phi]=(\partial_n u)_{avg}$ |
| $S[\psi]$ | $[u]=0$, $[\partial_n u]=\psi$, $f=0$ | $S[\psi]=u_{avg}$, $K'[\psi]=(\partial_n u)_{avg}$ |
| $N[q]$ | $[u]=0$, $[\partial_n u]=0$, $[f]=q$ | $N[q]=u_{avg}$, $\partial_n N[q]=(\partial_n u)_{avg}$ |

Because $D[\phi]$ is defined by the code jump $[u]=\phi$, its $K$ sign is the
code sign. It is the opposite of the classical double-layer sign if the
classical density is defined so that the double-layer potential has jump
$-\phi$.

## 3. Interior Dirichlet BVP

`LaplaceInteriorDirichlet2D` solves

$$-\Delta u+\eta u=f\quad\text{in}\quad \Omega^+,\qquad
u=g\quad\text{on}\quad \Gamma.$$

It uses a double-layer jump unknown $[u]=\phi$, $[\partial_n u]=0$. The matrix
free operator applied by GMRES is the interior trace of this jump primitive,

$$A_D\phi = K[\phi]+\frac12\phi.$$

For nonzero volume forcing, the code first evaluates the interface trace
contribution $V_f$ from `apply_full(0)`, then solves

$$A_D\phi = g - V_f.$$

For $\eta=0$ this is the harmonic interior Dirichlet solver. For $\eta>0$ it is
the screened Poisson version used by `test_laplace_interior_screened_2d.cpp`.

## 4. Interior Neumann BVP Operator

The current operator mode for an interior Neumann formulation uses the
single-layer jump unknown $[\partial_n u]=\psi$, $[u]=0$ and applies

$$A_N\psi = K'[\psi]+\frac12\psi,$$

the interior normal derivative of the single-layer jump primitive. A stable
public Neumann wrapper with compatibility/nullspace handling is still future
work.

## 5. Constant-Ratio Discontinuous-Coefficient Transmission

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

## 6. Nonzero Outer Cartesian Dirichlet Data

The FFT/DST bulk solver uses homogeneous Dirichlet data on the Cartesian box.
When a problem has nonzero box values $b$, the transmission solver eliminates
them into the finite-difference RHS:

* for interior grid nodes adjacent to the box boundary, add neighboring
  boundary values divided by the corresponding $h^2$;
* set boundary RHS entries to zero for the homogeneous solve;
* after the solve, restore the boundary entries of `u_bulk` to $b$.

This keeps the bulk solver interface homogeneous while allowing manufactured
tests with nonzero outer boundary conditions.

## 7. Summary of Current Operator Modes

| Mode/API | Unknown | Applied operator |
|----------|---------|------------------|
| `LaplaceKFBIMode::Dirichlet` | $[u]=\phi$ | $K[\phi]+\frac12\phi$ |
| `LaplaceKFBIMode::Neumann` | $[\partial_n u]=\psi$ | $K'[\psi]+\frac12\psi$ |
| `LaplaceTransmissionConstantRatio2D` | $\psi=[\partial_n u]$ | $(I+\gamma K')\psi$ |
