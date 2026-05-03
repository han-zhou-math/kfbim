# KFBIM Library - Project Guide

## Project Goal
Reconstruct problem-dependent KFBIM codes into a clean, reusable C++ library with a public API.
Future: Python/MATLAB bindings (pybind11), possibly Jupyter notebooks.

## Current Status
- Branch `main`; run `git log --oneline -n 3` for the exact current commit.
- This guide reflects the 2026-05-02 2D Laplace Chebyshev-Lobatto,
  screened-Poisson, and constant-ratio transmission work, plus the
  2026-05-03 repository cleanup/reorganization.
- **Completed modules** (active tests passing):
  - Layer 0: `CartesianGrid2D`, `Interface2D`, `GridPair2D`
    - `Interface2D` tracks panel node layout: `ChebyshevLobatto`, `LegacyGaussLegendre`, or `Raw`.
    - `GridPair2D::domain_label()` uses oversampled curved-panel polygons for 3-point panels.
  - Layer 1: preferred Chebyshev-Lobatto transfer path:
    - `LaplaceLobattoCenterSpread2D`
    - `LaplaceLobattoCenterRestrict2D`
  - Layer 1 legacy Gauss transfer path:
    - `LaplacePanelSpread2D`
    - `LaplaceQuadraticRestrict2D`
  - Layer 1.5: `LaplacePanelCauchySolver2D`
    - Legacy Gauss Cauchy solve at the three panel points.
    - Chebyshev-Lobatto Cauchy solve at four generated expansion centers per panel, `s={-0.75,-0.25,0.25,0.75}`.
    - Current collocation offset is `0.05`.
  - Layer 2: `LaplaceFftBulkSolverZfft2D` (DST, Dirichlet BC, optional screened shift)
  - Layer 3: `LaplaceKFBIOperator2D` (delegates to `LaplaceInterfaceSolver2D`), `LaplaceInterfaceSolver2D` (Spread → BulkSolve → Restrict pipeline, arc_h_ratio check)
  - Layer 3 modular potentials: `LaplacePotentialEval2D`
    - Evaluates D/S/N jump primitives through the existing KFBI pipeline.
    - Uses the current restrict convention: restrict returns interior trace/flux; averaged outputs are formed by subtracting half the jump where needed.
    - Covered by `tests/test_potential.cpp`.
  - Layer 4: GMRES outer solver
  - Layer 5: `LaplaceInteriorDirichlet2D` API
    - Solves `-Delta u + eta*u = f` in the interior.
    - Default panel method is `ChebyshevLobattoCenter`.
    - `LegacyGaussPanel` remains available explicitly.
  - Constant-ratio discontinuous coefficient interface utility:
    - `LaplaceTransmissionConstantRatio2D`
    - Solves `-div(beta grad u)+kappa^2 u=f` when `kappa^2/beta=lambda_sq`
      is the same on both sides.
    - Divides by `beta` to use a common screened operator
      `-Delta u + lambda_sq u=q`.
    - Eliminates optional nonzero outer Cartesian Dirichlet data into the RHS,
      then restores boundary values in the output.
- **Recent implementation notes:**
  - Chebyshev-Lobatto panel/expansion-center path is implemented and verified.
  - Gauss-point path is retained as legacy and must be selected explicitly in new tests/code.
  - `LaplacePotentialEval2D` has direct D/S/N jump-relation tests.
  - Active convergence tests target adjacent Chebyshev-node spacing of about
    `2h` (`panel_length/h ~= 4`) and GMRES tolerance `1e-8`.
  - `tests/test_dirichlet.cpp` solves the harmonic interior
    Dirichlet problem on a 5-fold star curve centered at `(0.07,-0.04)`.
  - `tests/test_screened.cpp` solves
    `-Delta u + u = f` with manufactured solution `exp(sin(x))+cos(y)` on a
    3-fold star.
  - `tests/test_transmission.cpp` solves the first
    discontinuous-coefficient interface problem on a 5-fold star.
  - Legacy Gauss tests were moved to `tests/archive/`.
  - Legacy reference code was moved from `old-codes/` to `third_party/old-codes/`.
  - Runtime CSV/PNG output is written under `output/` and should not be committed.
  - `build/` is disposable generated output. Recreate it with CMake when tests
    or local binaries are needed; do not commit it.
  - The C++ source tree is `src/` and the compiled library target remains
    `kfbim_core`.
  - Current concrete problem-level utilities and wrappers live in
    `src/problems/`; the placeholder top-level `problems/` directory was
    removed.
  - Visualization and diagnostic Python scripts live in `python/`.
- **Current convergence test status:** the active convergence binaries passed on 2026-05-03 after the `src/`, `python/`, and shortened-test-name reorganization:
  - `build/tests/test_fft -s`
  - `build/tests/test_iim -s`
  - `build/tests/test_potential -s`
  - `build/tests/test_dirichlet -s`
  - `build/tests/test_screened -s`
  - `build/tests/test_transmission -s`

## Next Agent Handoff — Continue This Plan

When resuming this project, continue the 2D Laplace public-API plan rather than branching into 3D, Stokes, variable coefficients, or bindings.

1. **Keep the foundation stable first.**
   - Preserve `LaplaceInteriorDirichlet2D` behavior and the default `ChebyshevLobattoCenter` panel method.
   - Keep legacy Gauss APIs/tests only for explicit regression or comparison paths.
   - Keep current concrete problem-level utilities and wrappers in `src/problems/`.
   - Do not keep placeholder top-level problem API declarations; create a separate public API directory only when wrappers are implemented.

2. **Next implementation target.**
   - Continue stable 2D Laplace BVP wrappers from concrete implementations.
   - Preferred order: interior/exterior Dirichlet wrappers first, then
     interior/exterior Neumann wrappers with nullspace/compatibility handling,
     then forcing/nonzero-volume-potential cases.
   - Continue discontinuous-coefficient work from the verified constant-ratio
     case; defer the general case where `kappa^2/beta` differs across the
     interface.
   - Use `LaplacePotentialEval2D` as the verified modular building block for D/S/N operator combinations.

3. **Testing requirements for the next change.**
   - Add one focused BVP test at a time with manufactured harmonic solutions.
   - Avoid exact grid/interface alignment by offsetting the interface center or choosing an incommensurate box size.
   - Run at least:
     - `build/tests/test_fft -s`
     - `build/tests/test_iim -s`
     - `build/tests/test_potential -s`
     - `build/tests/test_dirichlet -s`
     - `build/tests/test_screened -s`
     - `build/tests/test_transmission -s`

4. **Deferred work.**
   - Do not start Python/MATLAB bindings until the C++ problem API is stable.
   - Defer 3D BVP verification until concrete 3D local Cauchy, spread, and restrict implementations exist.
   - Defer Stokes until the Laplace BVP API is clean; Stokes currently has scaffolding but not concrete local Cauchy, spread, restrict, or operator implementations.

### Preferred 2D panel method

Use Chebyshev-Lobatto panels for new 2D Laplace work.

- Geometry/DOF nodes per panel: `s={-1,0,1}`.
- Correction expansion centers per panel: `s={-0.75,-0.25,0.25,0.75}`.
- `CurveResampler2D::discretize()` now returns Chebyshev-Lobatto panels.
- Current convergence tests set `panel_length/h ~= 4`, so adjacent
  Chebyshev-node spacing is about `2h`.
- `LaplaceInteriorDirichlet2D` defaults to `LaplaceInteriorPanelMethod2D::ChebyshevLobattoCenter`.
- Use `CurveResampler2D::discretize_legacy_gauss()` and `LaplaceInteriorPanelMethod2D::LegacyGaussPanel` only for legacy Gauss comparisons/regressions.

### Latest convergence snapshot

`test_dirichlet`, 5-fold star, Chebyshev-Lobatto path:

| N | max err | rate | GMRES iters |
|---:|--------:|-----:|------------:|
| 32 | 2.7945e-02 | - | 19 |
| 64 | 6.2512e-03 | 2.160 | 28 |
| 128 | 2.3670e-03 | 1.401 | 22 |
| 256 | 2.7520e-04 | 3.105 | 26 |
| 512 | 5.0119e-05 | 2.457 | 20 |
| 1024 | 6.9491e-06 | 2.850 | 20 |

`test_screened`, 3-fold star,
`u=exp(sin(x))+cos(y)`, `-Delta u + u=f`:

| N | max err | rate | GMRES iters |
|---:|--------:|-----:|------------:|
| 32 | 2.2317e-03 | - | 13 |
| 64 | 6.3270e-04 | 1.819 | 14 |
| 128 | 1.7405e-04 | 1.862 | 18 |
| 256 | 2.6148e-05 | 2.735 | 15 |
| 512 | 5.6018e-06 | 2.223 | 17 |
| 1024 | 7.4545e-07 | 2.910 | 10 |

`test_transmission`, 5-fold star,
`beta_int=2`, `beta_ext=1`, `kappa^2/beta=lambda_sq=1.1`:

| N | max err | rate | GMRES iters |
|---:|--------:|-----:|------------:|
| 32 | 5.8784e-03 | - | 9 |
| 64 | 7.2765e-04 | 3.014 | 9 |
| 128 | 1.3191e-04 | 2.464 | 9 |
| 256 | 2.4540e-05 | 2.426 | 9 |
| 512 | 3.4423e-06 | 2.834 | 8 |
| 1024 | 6.2691e-07 | 2.457 | 8 |

### Known numerical pitfall — grid/interface alignment
When a Cartesian grid node lands exactly on the interface, the IIM correction stencil has zero distance to the interface, making the local polynomial fit degenerate. This produces erratic convergence rates. **Rule:** for convergence tests, ensure no grid node is exactly on the interface by either offsetting the interface center or using a domain size incommensurate with the interface geometry. See `tests/archive/test_laplace_interior_circle_2d_legacy_gauss.cpp` for the archived regression that exposed this pitfall.

## Core Algorithm
Kernel-free boundary integral method for elliptic PDEs on complex interfaces/boundaries in 2D and 3D.

One GMRES iteration = one application of the interface-to-interface operator:
1. Spread interface data → correct bulk RHS
2. Solve PDE on Cartesian grid (FDM + fast solver)
3. Restrict bulk solution → interface data
GMRES iterates on this operator until the interface unknowns converge.

## Target PDE Types
- Laplace / Poisson / screened Poisson
- Constant-ratio discontinuous coefficient transmission
- Stokes
- Elasticity
- Variable-coefficient elliptic
- (Future) Time-dependent models

## Layered Architecture

```
Layer 5    Problem API          LaplaceProblem, StokesProblem, ElasticityProblem
Layer 4    Outer Solver         GMRES (matrix-free), pre/post-processing hooks
Layer 3    Interface Operator   KFBIOperator: Spread → BulkSolve → Restrict
Layer 1.5  Local Cauchy Solver  LaplaceLocalSolver, StokesLocalSolver, ...  (PDE-specific, used by Spread)
Layer 2    Bulk Solver          Abstract BulkSolver; impls: FFT, multigrid, StokesBulkSolver
Layer 1    Transfer Operators   Spread (calls LocalCauchySolver), Restrict (interpolation + jump correction)
Layer 0    Grid & Geometry      CartesianGrid, MACGrid, Interface, GridPair
```

Higher layers depend only on the abstractions of lower layers, not their implementations.

### Layer 0 — Grid & Geometry
- `CartesianGrid`: single scalar grid with one DOF layout (cell-center, face-x, face-y, face-z)
- `MACGrid`: composes multiple `CartesianGrid`s (one per velocity component + pressure)
- `Interface`: quadrature points, normals, weights, panel/element connectivity
- `GridPair`: bulk grid + interface; owns all geometric queries:
  - `ClosestPointQuery`: interface point ↔ nearest bulk grid node
  - `DomainLabeler`: classify bulk nodes by domain (supports multiply-connected domains)
  - `NearInterfaceStencil`: neighborhood info for bulk nodes near the interface

### Layer 1 — Transfer Operators
- `Spread`: interface → bulk correction. Injects a `LocalCauchySolver` to evaluate the
  local PDE solution at nearby bulk nodes from interface jump data (Dirichlet/Neumann jumps)
- `Restrict`: bulk → interface. Interpolates bulk solution to interface quadrature points
  and applies jump correction

### Layer 1.5 — Local Cauchy Solver (the "kernel-free" core)
- Input: jump data at/near an interface quadrature point
- Output: local PDE solution evaluated at specified nearby bulk grid points
- PDE-specific: `LaplaceLocalSolver`, `StokesLocalSolver`, `ElasticityLocalSolver`
- This replaces explicit Green's function kernels with a local PDE solve

### Layer 2 — Bulk Solver
- Abstract `BulkSolver`: corrected RHS → solution on `CartesianGrid` or `MACGrid`
- Concrete impls: FFT-based (constant coeff), multigrid (variable coeff), `StokesBulkSolver`
- Fully interface-agnostic; scalar solvers reused as components inside vector solvers

### Layer 3 — Interface Operator
- `KFBIOperator`: composes Spread → correct RHS → BulkSolve → Restrict
- Matrix-free operator; owns a `LocalCauchySolver` and a `BulkSolver`

### Layer 4 — Outer Solver
- GMRES (or pluggable: BiCGSTAB, etc.) iterating on `KFBIOperator`
- Pre/post-processing hooks (projection, normalization, etc.)

### Layer 5 — Problem API
- `LaplaceProblem`, `StokesProblem`, `ElasticityProblem`
- Each wires up the right `LocalCauchySolver` + `BulkSolver` + BCs
- Target surface for Python/MATLAB bindings (pybind11)

## Open Design Decisions
<!-- Remove entries as decisions are finalized -->

- [x] Interface representation: parametric panels (triangular mesh in 3D, panels in 2D). NURBS on upgrade path but isolated to geometry description layer, not quadrature storage.
- [x] Jump condition entry: pure RHS correction only. Bulk solver is completely interface-agnostic — clean separation between Layers 0-2 and Layer 1.
- [x] Vector PDEs (Stokes/elasticity): both scalar decomposition and coupled solves are used. Primary grid layout is MAC (staggered): velocity components at face centers, pressure at cell centers.
- [x] Spread/Restrict method: Spread solves a local Cauchy problem (PDE-dependent) and evaluates at nearby bulk nodes. Restrict interpolates bulk solution to interface point and applies jump correction. See architecture notes below.
- [x] External dependencies: CGAL for geometric queries (closest point on triangulated surface, domain labeling/inside-outside). Self-written code for the rest. Core otherwise self-contained.

## Settled Decisions
<!-- Move items here once decided -->

- 2D and 3D are separate but consistent modules (not unified by templates)
- `ICartesianGrid2D/3D`: abstract interface with flat-index-only API (`coord(idx)`, `neighbors(idx)`, `num_dofs()`). `BulkSolver` and all layers above depend only on this — enables future adaptive (quadtree/octree) grids without touching upper layers.
- `CartesianGrid2D/3D` implements `ICartesianGrid*` and additionally exposes `(i,j,k)` structured access for stencil builders and `LocalCauchySolver`, which need uniform grid layout knowledge.
- Interface stores quadrature points, normals, weights, and panel/element connectivity explicitly; geometry description (parametric → NURBS upgrade) is isolated above this
- Bulk solver is fully interface-agnostic; interface enters only via corrected RHS (no stencil modification)
- `CartesianGrid` is a single scalar staggered grid (one DOF layout/location); `MACGrid` composes multiple `CartesianGrid`s (one per velocity component + pressure)
- `StokesBulkSolver` operates on `MACGrid`; scalar solvers operate on `CartesianGrid` — scalar Poisson solvers are reused as components inside vector solvers
- Core is self-contained; external libs only for geometry and possibly fast solvers
- C++ core; Python bindings via pybind11 are a future goal, not a current constraint
- CMake build system

## Sign Conventions

- **u⁺** = interior solution (Ω⁺ = inside the interface Γ)
- **u⁻** = exterior solution (Ω⁻ = outside the interface Γ)
- **[u] = u⁺ − u⁻** (interior minus exterior), similarly [∂u/∂n] = ∂u⁺/∂n − ∂u⁻/∂n
- **Normal n** points outward (from interior Ω⁺ to exterior Ω⁻), matching `Interface2D::normals()`
- **Domain label**: `GridPair2D::domain_label(n)` returns 0 = exterior (Ω⁻), 1 = interior (Ω⁺)
- **Correction polynomial** C from panel Cauchy solver: C = [u] on Γ, ∂C/∂n = [∂u/∂n]
- **Restrict**: returns the interior trace/flux used by
  `LaplaceInterfaceSolver2D`
- **Interface average**:
  `u_avg = (u⁺+u⁻)/2 = u_int - [u]/2`, and similarly
  `un_avg = un_int - [∂u/∂n]/2`

```
src/
  grid/           # CartesianGrid2D/3D, MACGrid2D/3D, DofLayout
  interface/      # Interface2D, Interface3D
  geometry/       # GridPair2D/3D  (CGAL-dependent; pimpl hides CGAL headers)
  transfer/       # Spread, Restrict  (Layer 1)
  local_cauchy/   # LaplacePanelCauchySolver2D, jump_data, local_poly  (Layer 1.5)
  solver/         # BulkSolver and impls  (Layer 2)
  operator/       # KFBIOperator  (Layer 3)
  problems/       # Current problem-level utilities and BVP wrappers
  gmres/          # outer solver  (Layer 4)
python/           # Python visualization/diagnostic scripts
bindings/         # pybind11 wrappers (future)
tests/
tests/archive/    # legacy Gauss/reference tests
output/           # generated runtime output, gitignored
examples/
```

## Directory Layout (planned)

## Coding Conventions
<!-- Fill in as decided -->

- Language standard: C++17 (minimum; upgrade if a strong reason arises)
- Naming: `CamelCase` for classes/types, `snake_case` for methods and variables (matches Eigen, CGAL, deal.II)
- Error handling: exceptions for all recoverable errors; `assert` only for truly unreachable internal invariants (asserts disabled in release builds — never rely on them for library correctness)
- Data containers: Eigen internally; public API uses `Eigen::Ref<>` so callers can pass raw arrays, std::vector, or Eigen without copying. Enables pybind11 ↔ numpy and MATLAB interop. Internal custom vectors migrated gradually.

## Build
<!-- Fill in once CMake is set up -->

```bash
cmake -B build && cmake --build build
```
