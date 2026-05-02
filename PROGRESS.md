# KFBIM — Development Plan

## Current state

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | `CartesianGrid2D/3D`, `MACGrid2D/3D` | ✓ done |
| 0 | `Interface2D/3D` (panels, points, normals, weights); `Interface2D` records Chebyshev-Lobatto, legacy Gauss, or raw node layout | ✓ done |
| 0 | `GridPair2D/3D` (domain labeling, closest-pt query); `GridPair2D` labels use oversampled curved-panel polygons | ✓ done |
| 1.5 | `laplace_panel_solver_2d.hpp` — per-panel 6×6 collocation Cauchy solver; Chebyshev-Lobatto center variant stores C derivatives at four expansion centers per panel | ✓ done |
| 2 | `LaplaceFftBulkSolverZfft2D/3D` — DST Poisson solver | ✓ done |
| 1 | `LaplaceLobattoCenterSpread2D` / `LaplaceLobattoCenterRestrict2D` — preferred Chebyshev-Lobatto expansion-center path | ✓ done |
| 1 | `LaplacePanelSpread2D` / `LaplaceQuadraticRestrict2D` — legacy Gauss-point path | ✓ done |
| 3 | `LaplaceKFBIOperator2D/3D` — linear and affine modes | ✓ done |
| 4 | `GMRES` (restarted, Givens) | ✓ done |
| 3 | `LaplacePotentialEval2D` — modular D, S, N potential operators (K, H, K', ∂ₙN); covered by `test_laplace_potential_2d` | ✓ done |
| 5 | `LaplaceInteriorDirichlet2D` — High-level BVP API | ✓ done |
| — | IIM defect correction (2D, exact C and Taylor path) | ✓ done |
| — | Quasi-uniform curve resampling for stable discretization (`CurveResampler2D`); default is Chebyshev-Lobatto, legacy Gauss is explicit | ✓ done |
| — | `test_laplace_interior_circle_2d`: circle-domain convergence test (fixed grid-alignment and phantom-exterior issues) | ✓ done |
| — | Convention: u⁺ = interior, u⁻ = exterior, [u] = u_int − u_ext; labels 0 = Ω⁻, 1 = Ω⁺ | ✓ settled |
| — | Interface Solver Output: Returns averaged trace and normal derivative | ✓ settled |

### Verified Formulations (Laplace 2D)

- **Interface Problem**: Tests archived (`tests/archive/`) to focus on component testing.
- **Dirichlet BVP ($2^{nd}$-kind BIE)**: Double-layer unknown $[u]=\phi$. Operator mode aligned. Tests archived.
- **Neumann BVP ($2^{nd}$-kind BIE)**: Single-layer unknown $[\partial_n u]=\phi$. Operator mode aligned. Tests archived.
- **Interior Dirichlet BVP — preferred Chebyshev-Lobatto panels**: `test_laplace_interior_2d.cpp`. Manufactured solution $u = e^x \cos y$ (harmonic), domain $[-1.8,1.8]^2$, 5-fold star curve centered at $(0.07,-0.04)$ with $r(t)=0.75(1+0.25\cos 5t)$. Panel DOFs are Chebyshev-Lobatto nodes $s=\{-1,0,1\}$; correction expansion centers are $s=\{-0.75,-0.25,0.25,0.75\}$. Latest high-resolution refinement recovers about third-order convergence.
- **Interior Dirichlet BVP — legacy Gauss panels**: `test_laplace_interior_2d.cpp` also keeps an explicit legacy comparison using `LegacyGaussPanel`. Rates are good through $N=512$ on the 5-fold star, then flatten at the finest level in the latest run.
- **Interior Dirichlet BVP — offset circle legacy regression**: `test_laplace_interior_circle_2d.cpp`. Manufactured solution $u = e^x \sin y$ (harmonic), domain $[-2,2]^2$, circle center $(0, 0.1)$, radius $1$. Latest rates for $N=32\to1024$ are 1.797, 2.838, 3.108, 1.731, 2.656.

### Preferred 2D panel method

New 2D Laplace work should use Chebyshev-Lobatto panels.

- `CurveResampler2D::discretize()` returns Chebyshev-Lobatto panels.
- `LaplaceInteriorDirichlet2D` defaults to `LaplaceInteriorPanelMethod2D::ChebyshevLobattoCenter`.
- Use `CurveResampler2D::discretize_legacy_gauss()` and `LaplaceInteriorPanelMethod2D::LegacyGaussPanel` only for legacy Gauss comparisons.
- For each Chebyshev-Lobatto panel, the three geometry/DOF nodes are the endpoints and midpoint; the four generated expansion centers are the uniform interval centers in local panel parameter.

### Public API boundary

- Keep `core/problems/` for internal pipeline utilities that compose lower-layer pieces, such as `LaplaceInterfaceSolver2D` and the current concrete `LaplaceInteriorDirichlet2D` implementation.
- Treat top-level `problems/` as the long-term Layer 5 public API surface for stable user-facing BVP wrappers.
- Do not start Python/MATLAB bindings until the top-level C++ problem API is stable.

### Latest convergence runs (2026-05-02)

`test_laplace_interior_2d`, 5-fold star, Chebyshev-Lobatto path:

| N | max err | rate | GMRES iters |
|---:|--------:|-----:|------------:|
| 32 | 1.0721e-01 | — | 19 |
| 64 | 7.0126e-03 | 3.934 | 17 |
| 128 | 9.5195e-04 | 2.881 | 17 |
| 256 | 4.5468e-04 | 1.066 | 16 |
| 512 | 5.6535e-05 | 3.008 | 15 |
| 1024 | 6.1526e-06 | 3.200 | 13 |

`test_laplace_interior_2d`, 5-fold star, legacy Gauss path:

| N | max err | rate | GMRES iters |
|---:|--------:|-----:|------------:|
| 32 | 5.1058e-02 | — | 18 |
| 64 | 7.7224e-03 | 2.725 | 20 |
| 128 | 1.6672e-03 | 2.212 | 18 |
| 256 | 3.3502e-04 | 2.315 | 21 |
| 512 | 6.1310e-05 | 2.450 | 21 |
| 1024 | 4.5125e-05 | 0.442 | 31 |

`test_laplace_interface_solver_2d`:

| N | u_avg err | rate | un_avg err | rate |
|---:|----------:|-----:|-----------:|-----:|
| 32 | 2.8927e-03 | — | 6.4067e-02 | — |
| 64 | 3.9585e-04 | 2.869 | 2.1744e-02 | 1.559 |
| 128 | 1.1855e-04 | 1.739 | 6.0578e-03 | 1.844 |
| 256 | 3.1473e-05 | 1.913 | 1.7755e-03 | 1.771 |

`test_grid_alignment_pitfall_2d`, Poisson interface solve:

| N | bulk err | rate | u_avg err | rate | un_avg err | rate |
|---:|---------:|-----:|----------:|-----:|-----------:|-----:|
| 32 | 1.2622e-02 | — | 1.0352e-02 | — | 6.0732e-02 | — |
| 64 | 2.9028e-03 | 2.120 | 3.5772e-03 | 1.533 | 2.3096e-02 | 1.395 |
| 128 | 6.8632e-04 | 2.080 | 6.5883e-04 | 2.441 | 6.2735e-03 | 1.880 |
| 256 | 8.3146e-05 | 3.045 | 6.5968e-05 | 3.320 | 1.3225e-03 | 2.246 |
| 512 | 1.4276e-05 | 2.542 | 1.2931e-05 | 2.351 | 4.0951e-04 | 1.691 |
| 1024 | 1.9528e-06 | 2.870 | 1.7414e-06 | 2.893 | 1.1584e-04 | 1.822 |

`test_laplace_interior_circle_2d`:

| N | max err | rate | GMRES iters |
|---:|--------:|-----:|------------:|
| 32 | 1.1238e-02 | — | 17 |
| 64 | 3.2333e-03 | 1.797 | 20 |
| 128 | 4.5225e-04 | 2.838 | 12 |
| 256 | 5.2441e-05 | 3.108 | 13 |
| 512 | 1.5793e-05 | 1.731 | 17 |
| 1024 | 2.5051e-06 | 2.656 | 26 |

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
2. **Interior/exterior Dirichlet wrappers** — expose user-facing Layer 5 APIs
   under top-level `problems/`, using manufactured harmonic solutions for tests.
3. **Interior/exterior Neumann wrappers** — add nullspace/compatibility handling
   explicitly and test projected GMRES behavior.
4. **Forcing and volume-potential APIs** — generalize beyond the current
   `f_bulk` plus interface `rhs_derivs` path.

### Next Agent Continuation Checklist

- Continue the public 2D Laplace BVP API plan; do not pivot to 3D, Stokes,
  variable coefficients, or bindings unless explicitly requested.
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
  `test_laplace_interface_solver_2d`,
  `test_grid_alignment_pitfall_2d '[grid_alignment]'`, and
  `test_laplace_interior_circle_2d`.

### Deferred Work

- **3D Laplace BVP verification**: wait until concrete 3D local Cauchy, spread,
  and restrict implementations exist. Current 3D support is strongest at the
  grid/interface/bulk-solver level, not the full KFBI pipeline level.
- **Stokes**: defer until the Laplace BVP API is clean. Stokes has interfaces
  and scaffolding, but no concrete local Cauchy, spread, restrict, or operator
  implementation yet.
- **Discontinuous coefficients**: keep the design notes, but schedule this after
  the constant-coefficient BVP APIs are stable.

### Test Plan

- Run the known convergence set after structural changes:
  `test_laplace_interior_2d`, `test_laplace_interface_solver_2d`,
  `test_grid_alignment_pitfall_2d`, and `test_laplace_interior_circle_2d`.
- Run `test_laplace_potential_2d` after changes to potential, restrict, spread,
  interface-solver, or sign-convention code.
- Add BVP tests one API at a time, using manufactured harmonic solutions and
  avoiding exact grid/interface alignment.
