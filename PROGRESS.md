# KFBIM - Development Plan

## 2026-05-06 Update

Recent work extended the scalar Laplace coverage and cleaned up project notes:

- Added a 3D common-ratio transmission convergence test on a shared P2 torus.
- Added torus visualization export and
  `python/visualize_transmission_torus_3d.py`.
- Added a 2D common-ratio transmission convergence test with a bi-periodic
  bulk solver on a cell-centered periodic unit square.
- Added `LaplacePotentialEval2D/3D::eval_layer_combination()` and rewrote
  compatible transmission operator paths to evaluate combined D+S layer
  potentials with fewer bulk solves.
- Extended `LaplaceTransmission2D` with an optional zFFT bulk boundary
  condition argument, defaulting to Dirichlet. The current periodic
  transmission test keeps the interface away from the periodic boundary;
  spread/restrict are not yet general periodic-wrap transfer operators.
- Shortened note filenames under `notes/` and added
  `notes/parallel_plan.md` for the future MFEM/parallel KFBI roadmap.

Current note filenames:

- `notes/bie.md`
- `notes/iim.md`
- `notes/math.md`
- `notes/theory.md`
- `notes/parallel_plan.md`
- `notes/stokes.pdf`

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
| 3 | `LaplacePotentialEval2D/3D`; general arbitrary-jump/RHS KFBI pipeline plus D, S, N, and combined D+S helpers | done |
| 3 | `IKFBIOperator`; implemented directly by active Laplace problem wrappers | done |
| 4 | `GMRES`; restarted Givens implementation with residual history | done |
| 5 | `LaplaceBvp2D`; interior/exterior Dirichlet and Neumann modes | done |
| 5 | `LaplaceTransmission2D/3D`; common-ratio and different-ratio interface modes | done |
| - | IIM defect correction (2D, exact C and Taylor path) | done |
| - | Quasi-uniform curve resampling (`CurveResampler2D`); active wrappers use Chebyshev-Lobatto | done |
| - | Active PDE convergence tests: `test_interface`, `test_bvp`, `test_transmission` | done |
| - | Legacy Gauss comparison/reference tests under `tests/archive/` | archived |
| - | Sign convention: u+ = interior, u- = exterior, `[u]=u+ - u-`; labels 0 = exterior, 1 = interior | settled |
| - | Restriction convention: active 2D restrict returns averaged trace and normal derivative directly | settled |

### Verified 2D Laplace Formulations

- **Modular potential evaluator**: `LaplacePotentialEval2D` runs Spread -> BulkSolve -> Restrict for arbitrary jump data and bulk RHS. The restrictor maps side-specific samples to the average branch by subtracting `C/2` on interior samples and adding `C/2` on exterior samples before interpolation.
- **Potential primitives**: archived component coverage verifies D, S, and N jump relations for averaged trace/flux outputs.
- **BVP modes**: `LaplaceBvp2D` solves interior/exterior Dirichlet and Neumann modes and implements `IKFBIOperator` directly for the homogeneous BIE apply. Interior Neumann applies the unweighted mean-zero vector projection only for the pure Laplace nullspace case `eta=0`; screened cases do not project.
- **Prescribed-jump interface convergence**: `test_interface.cpp` solves a direct constant-coefficient screened interface problem through `LaplacePotentialEval2D`, with manufactured interior/exterior sine modes that vanish on the outer Cartesian box.
- **Screened BVP convergence**: `test_bvp.cpp` solves all four Dirichlet/Neumann BVP wrappers with `eta=kappa^2=1.1`, a nontrivial sine/cosine manufactured solution, exact exterior outer-box Dirichlet elimination/restoration, and N = 32..512.
- **Discontinuous-coefficient transmission**: `LaplaceTransmission2D` implements `IKFBIOperator` directly and `test_transmission.cpp` covers both common-ratio and different-ratio screened operators.
- **Legacy Gauss regressions**: older Gauss-point transfer and interface-solver tests remain in `tests/archive/` and are not part of the active Chebyshev-Lobatto convergence set.
- **Active geometry convention**: all three active convergence programs use the same off-center 3-fold star, sampled-bounds outer box plus margin, target adjacent Chebyshev-node spacing/h `1.5`, and `panel_length/h = 3.0`.

### Preferred 2D Panel Method

New 2D Laplace work should use Chebyshev-Lobatto panels.

- `CurveResampler2D::discretize()` returns Chebyshev-Lobatto panels.
- Panel-local geometry/DOF nodes are `s={-1,0,1}`.
- Adjacent panels share endpoint DOFs. For a closed curve with `Np` panels, `Interface2D::num_points()` is `2*Np`: `Np` shared endpoints plus `Np` panel midpoints.
- Correction expansion centers are generated at `s={-0.75,-0.25,0.25,0.75}` on each panel.
- Active convergence tests use target adjacent Chebyshev-node spacing/h `1.5`, so `panel_length/h = 3.0`.
- `LaplaceBvp2D` defaults to `LaplaceBvpPanelMethod2D::ChebyshevLobattoCenter`.
- Legacy Gauss comparison code belongs in `tests/archive/`; active problem wrappers do not select Gauss restrictors.

### Public API Boundary

- Keep current problem-level utilities and concrete wrappers in `src/operators/`.
- The active 2D BVP implementation is the unified `LaplaceBvp2D` class.
- `LaplacePotentialEval2D` lives in `src/potentials/` and should remain the reusable base for BIE/operator combinations.
- Do not restart a separate `LaplaceInterfaceSolver2D`; it was merged into `LaplacePotentialEval2D`.
- Do not start Python/MATLAB bindings until the C++ problem API is stable.

### Repository Layout

- `build/` is generated output; recreate it with `cmake -B build` when needed.
- C++ library sources are under `src/`; the library target remains `kfbim_core`.
- Current problem wrappers live in `src/operators/`; reusable potential evaluators live in `src/potentials/`.
- Current source modules are:
  `src/grid/`, `src/interface/`, `src/geometry/`, `src/transfer/`,
  `src/local_cauchy/`, `src/bulk_solvers/`, `src/potentials/`,
  `src/operators/`, and `src/gmres/`.
- `src/bulk_solvers/` owns the `IBulkSolver` API, FFT/zFFT engines,
  Laplace FFT/zFFT bulk solvers, boundary-condition enum, and IIM helper.
- `src/potentials/` currently contains `LaplacePotentialEval2D/3D`, the
  reusable Spread -> BulkSolve -> Restrict pipelines.
- `src/operators/` currently contains `IKFBIOperator`, `LaplaceBvp2D/3D`,
  `LaplaceTransmission2D/3D`, and the remaining Stokes scaffold.
- The old `src/solver/`, `src/operator/`, and `src/problems/` paths have been
  replaced by `src/bulk_solvers/`, `src/potentials/`, and `src/operators/`.
- Visualization and diagnostic scripts live in `python/`.
- Runtime CSV/PNG output is written under `output/` and should not be committed.
- Active CMake/CTest registration includes 2D and 3D PDE-level convergence
  programs: `test_interface`, `test_interface_3d`, `test_bvp`, `test_bvp_3d`,
  `test_transmission`, `test_transmission_periodic_2d`,
  `test_transmission_3d`, `test_transmission_ellipsoid_3d`, and
  `test_transmission_torus_3d`. Component and older top-level tests live in
  `tests/archive/`.

## Latest Convergence Runs

The active PDE/convergence suite passed locally after the three-program test
reorganization and grid-ratio update:
`test_interface`, `test_bvp`, and `test_transmission`.

All active tests use the same off-center 3-fold star and target adjacent
Chebyshev-node spacing/h `1.5` (`panel_length/h = 3.0`).

`test_interface`, direct prescribed-jump screened interface problem,
`eta=1.1`, homogeneous outer Cartesian Dirichlet data:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 2.8477e-03 | - | 0 |
| 64 | 6.0155e-04 | 2.243 | 0 |
| 128 | 1.5419e-04 | 1.964 | 0 |
| 256 | 3.6192e-05 | 2.091 | 0 |
| 512 | 9.5274e-06 | 1.926 | 0 |

`test_bvp`, all four screened `LaplaceBvp2D` modes, `eta=1.1`:

Interior Dirichlet:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 1.4467e-03 | - | 13 |
| 64 | 3.5812e-04 | 2.014 | 12 |
| 128 | 4.1393e-05 | 3.113 | 10 |
| 256 | 7.5059e-06 | 2.463 | 10 |
| 512 | 9.6317e-07 | 2.962 | 8 |

Exterior Dirichlet:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 1.3045e-03 | - | 15 |
| 64 | 2.2355e-04 | 2.545 | 14 |
| 128 | 2.8239e-05 | 2.985 | 14 |
| 256 | 3.3656e-06 | 3.069 | 13 |
| 512 | 5.4422e-07 | 2.629 | 12 |

Interior Neumann:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 6.0970e-02 | - | 14 |
| 64 | 7.8588e-03 | 2.956 | 13 |
| 128 | 1.1521e-03 | 2.770 | 12 |
| 256 | 2.0021e-04 | 2.525 | 11 |
| 512 | 6.4093e-05 | 1.643 | 11 |

Exterior Neumann:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 1.9263e-03 | - | 11 |
| 64 | 6.1171e-04 | 1.655 | 10 |
| 128 | 2.1360e-04 | 1.518 | 10 |
| 256 | 2.3042e-05 | 3.213 | 9 |
| 512 | 7.4770e-06 | 1.624 | 9 |

`test_transmission`, `LaplaceTransmission2D::CommonRatio`,
`beta_int=2`, `beta_ext=1`, `lambda^2=1.1`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 8.6829e-04 | - | 8 |
| 64 | 1.4075e-04 | 2.625 | 8 |
| 128 | 3.8581e-05 | 1.867 | 7 |
| 256 | 3.6765e-06 | 3.391 | 7 |
| 512 | 1.2467e-06 | 1.560 | 7 |
| 1024 | 2.7389e-07 | 2.186 | 7 |

`test_transmission`, `LaplaceTransmission2D::DifferentRatios`,
`beta_int=10`, `beta_ext=1`, `kappa_int^2=11`, `kappa_ext^2=0.7`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 2.3241e-02 | - | 15 |
| 64 | 2.9657e-03 | 2.970 | 14 |
| 128 | 5.5521e-04 | 2.417 | 14 |
| 256 | 1.1830e-04 | 2.231 | 13 |
| 512 | 3.3862e-05 | 1.805 | 12 |

`test_transmission_periodic_2d`, `LaplaceTransmission2D::CommonRatio`,
cell-centered bi-periodic unit square, `beta_int=2`, `beta_ext=1`,
`lambda^2=1.3`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 3.4118e-02 | - | 8 |
| 64 | 1.0923e-02 | 1.643 | 7 |
| 128 | 1.8616e-03 | 2.553 | 7 |
| 256 | 3.4085e-04 | 2.449 | 7 |
| 512 | 8.3662e-05 | 2.026 | 7 |

`test_transmission_torus_3d`, `LaplaceTransmission3D::CommonRatio`, shared P2
torus, target P2 `node_spacing/h ~= 1.5`, nonzero outer Cartesian Dirichlet
data, `beta_int=2`, `beta_ext=1`, `lambda^2=1.1`:

| N | panels | iface pts | max err | order | GMRES |
|---:|------:|----------:|--------:|------:|------:|
| 8 | 96 | 192 | 8.4870e-02 | - | 7 |
| 16 | 120 | 240 | 3.7657e-04 | 7.816 | 8 |
| 32 | 240 | 480 | 2.0510e-04 | 0.877 | 9 |
| 64 | 960 | 1920 | 3.6327e-05 | 2.497 | 8 |
| 128 | 3840 | 7680 | 7.5496e-06 | 2.267 | 8 |
| 256 | 15456 | 30912 | 1.1500e-06 | 2.715 | 8 |

### Test-Setup Pitfall: Grid/Interface Alignment

If a Cartesian grid node lands exactly on the interface, the IIM correction
stencil has zero distance to the interface and the local polynomial fit becomes
degenerate. This produces erratic rates. Avoid exact alignment by offsetting the
interface center or choosing an incommensurate box/interface setup. The current
active tests use an off-center star and derive the outer box from sampled
interface bounds plus a margin.

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

1. Keep `test_interface` green while evolving `LaplacePotentialEval2D`.
2. Harden the BVP wrapper APIs: options, diagnostics, residual reporting, and
   pure Laplace Neumann compatibility behavior.
3. Generalize volume-potential/forcing APIs beyond the current `f_bulk` plus
   `rhs_derivs` path.
4. Continue discontinuous-coefficient work from the verified common-ratio and
   different-ratio transmission cases.

### Next Agent Continuation Checklist

- Continue the public 2D Laplace BVP API plan; do not pivot to 3D, Stokes,
  general variable coefficients, or bindings unless explicitly requested.
- Start the next implementation in `src/operators/`; split out a public API
  directory only when the promoted wrapper is implemented.
- Use `LaplacePotentialEval2D` for new Dirichlet/Neumann operator work.
- Add one focused BVP test at a time with manufactured exact solutions and an
  interface/box setup that avoids exact grid-node alignment.
- Preserve the Chebyshev-Lobatto path for all new active 2D Laplace code.
- Before handing off, run:
  `test_interface`,
  `test_bvp`,
  `test_transmission`.

### Deferred Work

- **3D Laplace BVP verification**: wait until concrete 3D local Cauchy, spread,
  and restrict implementations exist.
- **Stokes**: defer until the Laplace BVP API is clean. Stokes has interfaces
  and scaffolding, but no concrete local Cauchy, spread, restrict, or operator
  implementation yet.
- **General discontinuous coefficients**: common-ratio and piecewise
  different-ratio modes are implemented for piecewise constants; defer spatially
  variable coefficients.
