# KFBIM — Development Plan

## Current state

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | `CartesianGrid2D/3D`, `MACGrid2D/3D` | ✓ done |
| 0 | `Interface2D/3D` (panels, points, normals, weights); `Interface2D` records Chebyshev-Lobatto, legacy Gauss, or raw node layout | ✓ done |
| 0 | `GridPair2D/3D` (domain labeling, closest-pt query); `GridPair2D` labels use oversampled curved-panel polygons | ✓ done |
| 1.5 | `laplace_panel_solver_2d.hpp` — per-panel 6×6 collocation Cauchy solver; Chebyshev-Lobatto center variant stores C derivatives at four expansion centers per panel | ✓ done |
| 2 | `LaplaceFftBulkSolverZfft2D/3D` — DST Poisson/screened solver | ✓ done |
| 1 | `LaplaceLobattoCenterSpread2D` / `LaplaceLobattoCenterRestrict2D` — preferred Chebyshev-Lobatto expansion-center path | ✓ done |
| 1 | `LaplacePanelSpread2D` / `LaplaceQuadraticRestrict2D` — legacy Gauss-point path | ✓ done |
| 3 | `LaplaceKFBIOperator2D/3D` — linear and affine modes | ✓ done |
| 4 | `GMRES` (restarted, Givens) | ✓ done |
| 3 | `LaplacePotentialEval2D` — modular D, S, N potential operators (K, H, K', ∂ₙN); covered by `test_laplace_potential_2d` | ✓ done |
| 5 | `LaplaceInteriorDirichlet2D` — interior Dirichlet BVP for `-Delta u + eta*u = f` | ✓ done |
| 5 | `LaplaceTransmissionConstantRatio2D` — constant-ratio discontinuous-coefficient interface utility | ✓ done |
| — | IIM defect correction (2D, exact C and Taylor path) | ✓ done |
| — | Quasi-uniform curve resampling for stable discretization (`CurveResampler2D`); default is Chebyshev-Lobatto, legacy Gauss is explicit | ✓ done |
| — | `tests/archive/test_laplace_interior_circle_2d_legacy_gauss.cpp`: archived circle-domain regression for grid-alignment and phantom-exterior issues | ✓ done |
| — | Convention: u⁺ = interior, u⁻ = exterior, [u] = u_int − u_ext; labels 0 = Ω⁻, 1 = Ω⁺ | ✓ settled |
| — | Interface Solver Output: Returns averaged trace and normal derivative | ✓ settled |

### Verified Formulations (Laplace 2D)

- **Interface Problem**: Legacy Gauss tests archived (`tests/archive/`) to focus the active suite on Chebyshev-Lobatto components and BVP/interface convergence.
- **Dirichlet BVP ($2^{nd}$-kind BIE)**: Double-layer unknown $[u]=\phi$. Operator mode aligned.
- **Neumann BVP ($2^{nd}$-kind BIE)**: Single-layer unknown $[\partial_n u]=\phi$. Operator mode aligned at the operator level; public wrapper still future work.
- **Interior Dirichlet BVP — preferred Chebyshev-Lobatto panels**: `test_laplace_interior_2d.cpp`. Manufactured solution $u = e^x \cos y$ (harmonic), domain $[-1.8,1.8]^2$, 5-fold star curve centered at $(0.07,-0.04)$ with $r(t)=0.75(1+0.25\cos 5t)$. Panel DOFs are Chebyshev-Lobatto nodes $s=\{-1,0,1\}$; correction expansion centers are $s=\{-0.75,-0.25,0.25,0.75\}$. Latest high-resolution refinement recovers about third-order convergence.
- **Screened interior Dirichlet BVP**: `test_laplace_interior_screened_2d.cpp`. Solves $-\Delta u+u=f$, $u=g$, using manufactured $u=\exp(\sin x)+\cos y$ on a 3-fold star. The outer box is sampled from the interface bounds and the test writes CSV/PNG output under `output/laplace_interior_screened_star3`.
- **Constant-ratio discontinuous-coefficient transmission**: `test_laplace_transmission_constant_ratio_2d.cpp`. Solves $-\nabla\cdot(\beta\nabla u)+\kappa^2u=f$ with $\kappa^2/\beta=\lambda^2$ equal on both sides, so the reduced equation is $-\Delta u+\lambda^2u=q=f/\beta$. The test uses a 5-fold star, nonzero outer Cartesian Dirichlet data, manufactured interior/exterior branches, and Python visualization under `output/laplace_transmission_constant_ratio_2d`.
- **Legacy Gauss regressions**: older Gauss-point transfer tests are in `tests/archive/` and are not part of the active Chebyshev-Lobatto convergence set.

### Preferred 2D panel method

New 2D Laplace work should use Chebyshev-Lobatto panels.

- `CurveResampler2D::discretize()` returns Chebyshev-Lobatto panels.
- `LaplaceInteriorDirichlet2D` defaults to `LaplaceInteriorPanelMethod2D::ChebyshevLobattoCenter`.
- Use `CurveResampler2D::discretize_legacy_gauss()` and `LaplaceInteriorPanelMethod2D::LegacyGaussPanel` only for legacy Gauss comparisons.
- For each Chebyshev-Lobatto panel, the three geometry/DOF nodes are the endpoints and midpoint; the four generated expansion centers are $s=\{-0.75,-0.25,0.25,0.75\}$.
- Active convergence tests use `panel_length/h ~= 4`, so adjacent Chebyshev-node spacing is about $2h$, and GMRES tolerance is $10^{-8}$.

### Public API boundary

- Keep `core/problems/` for internal pipeline utilities that compose lower-layer pieces, such as `LaplaceInterfaceSolver2D` and the current concrete `LaplaceInteriorDirichlet2D` implementation.
- Treat top-level `problems/` as the long-term Layer 5 public API surface for stable user-facing BVP wrappers.
- Do not start Python/MATLAB bindings until the top-level C++ problem API is stable.

### Latest convergence runs (2026-05-02)

`test_laplace_interior_2d`, 5-fold star, Chebyshev-Lobatto path:

| N | max err | rate | GMRES iters |
|---:|--------:|-----:|------------:|
| 32 | 2.7945e-02 | — | 19 |
| 64 | 6.2512e-03 | 2.160 | 28 |
| 128 | 2.3670e-03 | 1.401 | 22 |
| 256 | 2.7520e-04 | 3.105 | 26 |
| 512 | 5.0119e-05 | 2.457 | 20 |
| 1024 | 6.9491e-06 | 2.850 | 20 |

`test_laplace_interior_screened_2d`, 3-fold star, $-\Delta u+u=f$:

| N | max err | rate | GMRES iters |
|---:|--------:|-----:|------------:|
| 32 | 2.2317e-03 | — | 13 |
| 64 | 6.3270e-04 | 1.819 | 14 |
| 128 | 1.7405e-04 | 1.862 | 18 |
| 256 | 2.6148e-05 | 2.735 | 15 |
| 512 | 5.6018e-06 | 2.223 | 17 |
| 1024 | 7.4545e-07 | 2.910 | 10 |

`test_laplace_transmission_constant_ratio_2d`, 5-fold star,
$\beta_{int}=2$, $\beta_{ext}=1$, $\lambda^2=1.1$:

| N | max err | rate | GMRES iters |
|---:|--------:|-----:|------------:|
| 32 | 5.8784e-03 | — | 9 |
| 64 | 7.2765e-04 | 3.014 | 9 |
| 128 | 1.3191e-04 | 2.464 | 9 |
| 256 | 2.4540e-05 | 2.426 | 9 |
| 512 | 3.4423e-06 | 2.834 | 8 |
| 1024 | 6.2691e-07 | 2.457 | 8 |

### Test-setup pitfall: grid/interface alignment
If a grid node lands exactly on the interface (e.g. origin-centered unit circle on $[-2,2]^2$ with $h=4/N$ — axis crossings at $(\pm1,0)$, $(0,\pm1)$ hit node positions for all $N$ divisible by 4), the IIM correction stencil is degenerate (zero distance to interface), producing erratic convergence. Fix: offset the interface center or choose a domain size incommensurate with the interface geometry.

Additionally, an oversized domain amplifies the phantom exterior solution (which the DST bulk solver forces to zero at the box boundary), degrading rates at fine grid levels. Use the tightest domain that still gives $\geq 0.5h$ clearance around the interface.

### Modular Potential Operators (`core/operator/laplace_potential.hpp`)

`LaplacePotentialEval2D` provides reusable evaluation of boundary integral
operators via the KFBI pipeline (Spread → BulkSolve → Restrict). The double-layer
primitive follows the codebase jump convention `[u]=u^+ - u^- = phi`, so its
principal-value output is the average trace of that jump problem.

| Potential | Input | Jumps | Output operators |
|-----------|-------|-------|------------------|
| $D[\phi]$ | $\phi$ | $[u]=\phi, [\partial_n u]=0, f=0$ | $K[\phi]$ (averaged trace), $H[\phi]$ (continuous normal deriv) |
| $S[\psi]$ | $\psi$ | $[u]=0, [\partial_n u]=\psi, f=0$ | $S[\psi]$ (trace), $K'[\psi]$ (adjoint normal deriv) |
| $N[q]$ | $q$ | $[u]=0, [\partial_n u]=0, [f]=q$ | $N[q]$ (trace), $\partial_n N[q]$ (normal deriv) |

These are the building blocks for future GMRES-based BIE solvers (BIEs can be
assembled via linear combinations of K, H, S, K', N, ∂ₙN + identity terms).
General nonzero bulk volume-potential APIs are still future work; current BVP
code handles bulk forcing through `f_bulk` and `rhs_derivs`.

---

## Near-Term Roadmap

The 2D Laplace Chebyshev-Lobatto path is the stable foundation. Expand from
there in small API steps, with direct tests before higher-level BVP wiring.

1. **Verified modular potentials** — keep `test_laplace_potential_2d` green for
   `D`, `S`, and `N` jump-relation consistency.
2. **Screened/interior Dirichlet and constant-ratio transmission tests** — keep
   the active convergence tests green and preserve Python visualization output
   under `output/`.
3. **Interior/exterior Dirichlet wrappers** — expose user-facing Layer 5 APIs
   under top-level `problems/`, using manufactured harmonic solutions for tests.
4. **Interior/exterior Neumann wrappers** — add nullspace/compatibility handling
   explicitly and test projected GMRES behavior.
5. **Forcing and volume-potential APIs** — generalize beyond the current
   `f_bulk` plus interface `rhs_derivs` path.

### Next Agent Continuation Checklist

- Continue the public 2D Laplace BVP API plan; do not pivot to 3D, Stokes,
  general variable coefficients, or bindings unless explicitly requested.
- Start the next implementation under top-level `problems/`, keeping
  `core/problems/` as internal pipeline utilities.
- Use `LaplacePotentialEval2D` for the next Dirichlet/Neumann operator wrapper
  and keep `test_laplace_potential_2d` green while doing so.
- Add one focused BVP test at a time with manufactured harmonic solutions and
  an interface/box setup that avoids exact grid-node alignment.
- Preserve the Chebyshev-Lobatto default for all new 2D Laplace code; select
  `LegacyGaussPanel` only in legacy comparison/regression tests.
- Before handing off, run the known set:
  `test_laplace_potential_2d`,
  `test_laplace_interior_2d`,
  `test_laplace_interior_screened_2d`, and
  `test_laplace_transmission_constant_ratio_2d`.

### Deferred Work

- **3D Laplace BVP verification**: wait until concrete 3D local Cauchy, spread,
  and restrict implementations exist. Current 3D support is strongest at the
  grid/interface/bulk-solver level, not the full KFBI pipeline level.
- **Stokes**: defer until the Laplace BVP API is clean. Stokes has interfaces
  and scaffolding, but no concrete local Cauchy, spread, restrict, or operator
  implementation yet.
- **General discontinuous coefficients**: the constant-ratio case is now
  implemented; defer the case where $\kappa^2/\beta$ differs across the
  interface.

### Test Plan

- Run the known convergence set after structural changes:
  `test_laplace_interior_2d`, `test_laplace_interior_screened_2d`, and
  `test_laplace_transmission_constant_ratio_2d`.
- Run `test_laplace_potential_2d` after changes to potential, restrict, spread,
  interface-solver, or sign-convention code.
- Add BVP tests one API at a time, using manufactured harmonic solutions and
  avoiding exact grid/interface alignment.
