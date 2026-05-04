# KFBIM - Development Plan

## Current State

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | `CartesianGrid2D/3D`, `MACGrid2D/3D` | done |
| 0 | `Interface2D/3D`; `Interface2D` records Chebyshev-Lobatto, legacy Gauss, or raw node layout | done |
| 0 | `Interface2D` explicit panel connectivity; Chebyshev-Lobatto endpoints are shared between adjacent panels | done |
| 0 | `GridPair2D/3D`; `GridPair2D` labels use oversampled curved-panel polygons | done |
| 1.5 | `laplace_panel_solver_2d.hpp`; per-panel Cauchy solver with Chebyshev-Lobatto expansion centers | done |
| 1 | `LaplaceLobattoCenterSpread2D` / `LaplaceLobattoCenterRestrict2D`; active 2D transfer path | done |
| 2 | `LaplaceFftBulkSolverZfft2D/3D`; DST Poisson/screened solver | done |
| 3 | `LaplacePotentialEval2D`; general arbitrary-jump/RHS KFBI pipeline plus D, S, N helpers | done |
| 3 | `LaplaceKFBIOperator2D/3D`; selected-side Dirichlet/Neumann BIE modes | done |
| 4 | `GMRES`; restarted Givens implementation with residual history | done |
| 5 | `LaplaceInteriorDirichlet2D` | done |
| 5 | `LaplaceExteriorDirichlet2D`, `LaplaceInteriorNeumann2D`, `LaplaceExteriorNeumann2D` | done |
| 5 | `LaplaceTransmissionConstantRatio2D`; constant-ratio discontinuous-coefficient interface utility | done |
| - | IIM defect correction (2D, exact C and Taylor path) | done |
| - | Quasi-uniform curve resampling (`CurveResampler2D`); active wrappers use Chebyshev-Lobatto | done |
| - | Legacy Gauss comparison/reference tests under `tests/archive/` | archived |
| - | Sign convention: u+ = interior, u- = exterior, `[u]=u+ - u-`; labels 0 = exterior, 1 = interior | settled |
| - | Restriction convention: active 2D restrict returns averaged trace and normal derivative directly | settled |

### Verified 2D Laplace Formulations

- **Modular potential evaluator**: `LaplacePotentialEval2D` runs Spread -> BulkSolve -> Restrict for arbitrary jump data and bulk RHS. The restrictor maps side-specific samples to the average branch by subtracting `C/2` on interior samples and adding `C/2` on exterior samples before interpolation.
- **Potential primitives**: `test_potential.cpp` verifies D, S, and N jump relations for averaged trace/flux outputs.
- **Interior Dirichlet**: `LaplaceInteriorDirichlet2D` solves `-Delta u + eta*u = f` using the double-layer unknown `[u]=phi`.
- **Exterior Dirichlet**: `LaplaceExteriorDirichlet2D` uses the same double-layer primitive with exterior-side reconstruction.
- **Interior/exterior Neumann**: `LaplaceInteriorNeumann2D` and `LaplaceExteriorNeumann2D` use the single-layer unknown `[partial_n u]=psi`. Interior Neumann applies the unweighted mean-zero vector projection only for the pure Laplace nullspace case `eta=0`; screened cases do not project.
- **Screened BVP convergence**: `test_bvp.cpp` solves all four Dirichlet/Neumann BVP wrappers with `eta=kappa^2=1.1` on the unit circle centered at the origin in `(-1.7,1.7)^2`, using a nontrivial sine/cosine manufactured solution and N = 32..512.
- **Screened interior Dirichlet star test**: `test_screened.cpp` solves `-Delta u+u=f`, `u=g`, with manufactured `u=exp(sin(x))+cos(y)` on a 3-fold star.
- **Constant-ratio discontinuous-coefficient transmission**: `test_transmission.cpp` solves `-div(beta grad u)+kappa^2 u=f` with `kappa^2/beta=lambda^2` equal on both sides, reducing to a common screened operator.
- **Legacy Gauss regressions**: older Gauss-point transfer and interface-solver tests remain in `tests/archive/` and are not part of the active Chebyshev-Lobatto convergence set.

### Preferred 2D Panel Method

New 2D Laplace work should use Chebyshev-Lobatto panels.

- `CurveResampler2D::discretize()` returns Chebyshev-Lobatto panels.
- Panel-local geometry/DOF nodes are `s={-1,0,1}`.
- Adjacent panels share endpoint DOFs. For a closed curve with `Np` panels, `Interface2D::num_points()` is `2*Np`: `Np` shared endpoints plus `Np` panel midpoints.
- Correction expansion centers are generated at `s={-0.75,-0.25,0.25,0.75}` on each panel.
- Existing star-domain convergence tests use `panel_length/h ~= 4`, so adjacent Chebyshev-node spacing is about `2h`.
- The all-BVP unit-circle test currently uses target interface spacing/h about `1.5` (`panel_length/h ~= 3`).
- `LaplaceInteriorDirichlet2D` defaults to `LaplaceInteriorPanelMethod2D::ChebyshevLobattoCenter`.
- Legacy Gauss comparison code belongs in `tests/archive/`; active problem wrappers do not select Gauss restrictors.

### Public API Boundary

- Keep current problem-level utilities and concrete wrappers in `src/problems/`.
- The active 2D BVP implementation is `LaplaceBvpPipeline2D` plus the public wrappers:
  `LaplaceInteriorDirichlet2D`, `LaplaceExteriorDirichlet2D`,
  `LaplaceInteriorNeumann2D`, and `LaplaceExteriorNeumann2D`.
- `LaplacePotentialEval2D` is the shared Layer 3 pipeline and should remain the reusable base for BIE/operator combinations.
- Do not restart a separate `LaplaceInterfaceSolver2D`; it was merged into `LaplacePotentialEval2D`.
- Do not start Python/MATLAB bindings until the C++ problem API is stable.

### Repository Layout

- `build/` is generated output; recreate it with `cmake -B build` when needed.
- C++ library sources are under `src/`; the library target remains `kfbim_core`.
- Current problem wrappers live in `src/problems/`.
- Visualization and diagnostic scripts live in `python/`.
- Runtime CSV/PNG output is written under `output/` and should not be committed.

## Latest Convergence Runs

The active PDE/convergence suite passed locally after the BVP and potential-evaluation update:
`test_fft`, `test_iim`, `test_potential`, `test_bvp`, `test_dirichlet`,
`test_screened`, and `test_transmission`.

`test_bvp`, unit circle centered at origin, box `(-1.7,1.7)^2`,
`eta=kappa^2=1.1`, target interface spacing/h about `1.5`:

Interior Dirichlet:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 1.7037e-03 | - | 10 |
| 64 | 1.3275e-04 | 3.682 | 9 |
| 128 | 1.9113e-05 | 2.796 | 9 |
| 256 | 2.6885e-06 | 2.830 | 8 |
| 512 | 4.4836e-07 | 2.584 | 6 |

Exterior Dirichlet:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 1.8697e-04 | - | 12 |
| 64 | 2.3980e-05 | 2.963 | 10 |
| 128 | 3.5066e-06 | 2.774 | 10 |
| 256 | 7.0953e-07 | 2.305 | 9 |
| 512 | 1.8515e-07 | 1.938 | 8 |

Interior Neumann:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 8.2754e-03 | - | 10 |
| 64 | 2.0068e-03 | 2.044 | 9 |
| 128 | 6.3105e-04 | 1.669 | 9 |
| 256 | 1.9425e-04 | 1.700 | 8 |
| 512 | 4.9580e-05 | 1.970 | 7 |

Exterior Neumann:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 1.2825e-03 | - | 8 |
| 64 | 3.5106e-04 | 1.869 | 8 |
| 128 | 9.1097e-05 | 1.946 | 7 |
| 256 | 2.2758e-05 | 2.001 | 7 |
| 512 | 5.8315e-06 | 1.964 | 6 |

The test also prints the GMRES residual history for every step and checks that
`num_interface_points == density.size() == 2*num_panels`.

`test_dirichlet`, 5-fold star, Chebyshev-Lobatto path:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 8.7805e-02 | - | 16 |
| 64 | 1.1313e-02 | 2.956 | 15 |
| 128 | 3.9885e-03 | 1.504 | 14 |
| 256 | 4.4891e-04 | 3.151 | 12 |
| 512 | 4.7746e-05 | 3.233 | 12 |
| 1024 | 5.4540e-06 | 3.130 | 11 |

`test_screened`, 3-fold star, `-Delta u+u=f`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 2.7659e-03 | - | 9 |
| 64 | 4.9287e-04 | 2.488 | 10 |
| 128 | 1.0949e-04 | 2.170 | 8 |
| 256 | 1.7159e-05 | 2.674 | 8 |
| 512 | 2.2353e-06 | 2.940 | 8 |
| 1024 | 2.2198e-07 | 3.332 | 8 |

`test_transmission`, 5-fold star,
`beta_int=2`, `beta_ext=1`, `lambda^2=1.1`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 8.3314e-03 | - | 9 |
| 64 | 9.4386e-04 | 3.142 | 8 |
| 128 | 1.9480e-04 | 2.277 | 8 |
| 256 | 1.8549e-05 | 3.393 | 8 |
| 512 | 1.4424e-06 | 3.685 | 8 |
| 1024 | 8.2628e-07 | 0.804 | 8 |

### Test-Setup Pitfall: Grid/Interface Alignment

If a Cartesian grid node lands exactly on the interface, the IIM correction
stencil has zero distance to the interface and the local polynomial fit becomes
degenerate. This produces erratic rates. Avoid exact alignment by offsetting the
interface center or choosing an incommensurate box/interface setup. The current
unit-circle BVP test uses `(-1.7,1.7)^2`, so the axis-crossing points of the
unit circle do not coincide with the power-of-two grid levels used there.

## Modular Potential Operators

`LaplacePotentialEval2D` provides reusable evaluation of boundary integral
operators through the KFBI pipeline. The double-layer primitive follows the
codebase jump convention `[u]=u+ - u- = phi`, so its `K` sign is the code sign.

| Potential | Input | Jumps | Output operators |
|-----------|-------|-------|------------------|
| `D[phi]` | `phi` | `[u]=phi`, `[partial_n u]=0`, `f=0` | `K[phi]` averaged trace, `H[phi]` averaged normal derivative |
| `S[psi]` | `psi` | `[u]=0`, `[partial_n u]=psi`, `f=0` | `S[psi]` trace, `K'[psi]` averaged normal derivative |
| `N[q]` | `q` | `[u]=0`, `[partial_n u]=0`, `[f]=q` | `N[q]` trace, `partial_n N[q]` normal derivative |

These are the building blocks for GMRES-based BIE solvers. Current BVP code
handles forcing through `f_bulk` and per-interface `rhs_derivs`.

## Near-Term Roadmap

1. Keep `test_potential` green while evolving `LaplacePotentialEval2D`.
2. Harden the BVP wrapper APIs: options, diagnostics, residual reporting, and
   pure Laplace Neumann compatibility behavior.
3. Generalize volume-potential/forcing APIs beyond the current `f_bulk` plus
   `rhs_derivs` path.
4. Continue discontinuous-coefficient work from the verified constant-ratio
   case; defer the general case where `kappa^2/beta` differs across the
   interface.

### Next Agent Continuation Checklist

- Continue the public 2D Laplace BVP API plan; do not pivot to 3D, Stokes,
  general variable coefficients, or bindings unless explicitly requested.
- Start the next implementation in `src/problems/`; split out a public API
  directory only when the promoted wrapper is implemented.
- Use `LaplacePotentialEval2D` for new Dirichlet/Neumann operator work.
- Add one focused BVP test at a time with manufactured exact solutions and an
  interface/box setup that avoids exact grid-node alignment.
- Preserve the Chebyshev-Lobatto path for all new active 2D Laplace code.
- Before handing off, run:
  `test_potential`,
  `test_bvp`,
  `test_dirichlet`,
  `test_screened`, and
  `test_transmission`.

### Deferred Work

- **3D Laplace BVP verification**: wait until concrete 3D local Cauchy, spread,
  and restrict implementations exist.
- **Stokes**: defer until the Laplace BVP API is clean. Stokes has interfaces
  and scaffolding, but no concrete local Cauchy, spread, restrict, or operator
  implementation yet.
- **General discontinuous coefficients**: the constant-ratio case is
  implemented; defer the case where `kappa^2/beta` differs across the interface.
