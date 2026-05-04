# KFBIM Library - Project Guide

## Project Goal
Reconstruct problem-dependent KFBIM codes into a clean, reusable C++ library with a public API.
Future: Python/MATLAB bindings (pybind11), possibly Jupyter notebooks.

## Current Status
- Branch `main`; run `git log --oneline -n 3` for the exact current commit.
- This guide reflects the 2026-05-04 2D Laplace Chebyshev-Lobatto,
  screened-Poisson, constant-ratio transmission, merged potential-evaluation,
  and screened BVP wrapper work.
- **Completed modules** (active tests passing):
  - Layer 0: `CartesianGrid2D`, `Interface2D`, `GridPair2D`
    - `Interface2D` tracks panel node layout: `ChebyshevLobatto`, `LegacyGaussLegendre`, or `Raw`.
    - Chebyshev-Lobatto panels use explicit connectivity with shared adjacent
      endpoints, so a closed curve has `num_points() = 2*num_panels()`.
    - `GridPair2D::domain_label()` uses oversampled curved-panel polygons for 3-point panels.
  - Layer 1: preferred Chebyshev-Lobatto transfer path:
    - `LaplaceLobattoCenterSpread2D`
    - `LaplaceLobattoCenterRestrict2D`
  - Layer 1.5: `LaplacePanelCauchySolver2D`
    - Legacy Gauss Cauchy solve at the three panel points remains for archived/reference paths.
    - Chebyshev-Lobatto Cauchy solve at four generated expansion centers per panel, `s={-0.75,-0.25,0.25,0.75}`.
    - Current collocation offset is `0.05`.
  - Layer 2: `LaplaceFftBulkSolverZfft2D` (DST, Dirichlet BC, optional screened shift)
  - Layer 3: `LaplaceKFBIOperator2D`, backed by `LaplacePotentialEval2D`
    (Spread → BulkSolve → Restrict pipeline, arc_h_ratio check)
  - Layer 3 modular potentials: `LaplacePotentialEval2D`
    - Evaluates arbitrary jumps/RHS and returns bulk values plus averaged trace/flux.
    - Specialized D/S/N helpers are thin calls into the same general pipeline.
    - Uses the current restrict convention: restrict returns averaged trace/flux directly.
    - Covered by `tests/test_potential.cpp`.
  - Layer 4: GMRES outer solver
  - Layer 5: 2D Laplace BVP APIs
    - `LaplaceInteriorDirichlet2D`
    - `LaplaceExteriorDirichlet2D`
    - `LaplaceInteriorNeumann2D`
    - `LaplaceExteriorNeumann2D`
    - Solves `-Delta u + eta*u = f` in the selected interior/exterior domain.
    - Default panel method is `ChebyshevLobattoCenter`.
    - Interior Neumann uses an unweighted mean-zero vector projection only for
      the pure Laplace nullspace case `eta=0`; screened cases do not project.
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
  - Gauss-point restrictor/problem-wrapper paths were removed from active code.
  - `LaplacePotentialEval2D` has direct D/S/N jump-relation tests.
  - Existing star-domain convergence tests target adjacent Chebyshev-node
    spacing of about `2h` (`panel_length/h ~= 4`) and GMRES tolerance `1e-8`.
  - `tests/test_bvp.cpp` covers all four screened Dirichlet/Neumann BVP
    wrappers on the unit circle centered at the origin in the box
    `(-1.7,1.7)^2`, with `eta=kappa^2=1.1`, a sinusoidal/cosine manufactured
    solution, target interface spacing/h about `1.5`, and N = 32..512.
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
- **Current convergence test status:** the active convergence binaries passed on 2026-05-04 after the BVP/potential-evaluation update:
  - `build/tests/test_fft -s`
  - `build/tests/test_iim -s`
  - `build/tests/test_potential -s`
  - `build/tests/test_bvp -s`
  - `build/tests/test_dirichlet -s`
  - `build/tests/test_screened -s`
  - `build/tests/test_transmission -s`

## Next Agent Handoff — Continue This Plan

When resuming this project, continue the 2D Laplace public-API plan rather than branching into 3D, Stokes, variable coefficients, or bindings.

1. **Keep the foundation stable first.**
   - Preserve `LaplaceInteriorDirichlet2D` behavior and the default `ChebyshevLobattoCenter` panel method.
   - Keep the active 2D Laplace path on Chebyshev-Lobatto panels.
   - Keep current concrete problem-level utilities and wrappers in `src/problems/`.
   - Do not keep placeholder top-level problem API declarations; create a separate public API directory only when wrappers are implemented.

2. **Next implementation target.**
   - Harden the stable 2D Laplace BVP wrappers from concrete implementations.
   - Next priority is richer forcing/nonzero-volume-potential APIs, plus
     compatibility handling and diagnostics for pure Laplace Neumann cases.
   - Continue discontinuous-coefficient work from the verified constant-ratio
     case; defer the general case where `kappa^2/beta` differs across the
     interface.
   - Use `LaplacePotentialEval2D` as the verified modular building block for D/S/N operator combinations.

3. **Testing requirements for the next change.**
   - Add one focused BVP test at a time with manufactured exact solutions.
   - Avoid exact grid/interface alignment by offsetting the interface center or choosing an incommensurate box size.
   - Run at least:
     - `build/tests/test_fft -s`
     - `build/tests/test_iim -s`
     - `build/tests/test_potential -s`
     - `build/tests/test_bvp -s`
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
- Adjacent panels share endpoint DOFs. For a closed curve with `Np` panels,
  the unknown/vector length is `2*Np`: `Np` shared endpoints plus `Np` panel
  midpoints.
- Correction expansion centers per panel: `s={-0.75,-0.25,0.25,0.75}`.
- `CurveResampler2D::discretize()` now returns Chebyshev-Lobatto panels.
- Current star-domain convergence tests set `panel_length/h ~= 4`, so adjacent
  Chebyshev-node spacing is about `2h`. The all-BVP circle test currently uses
  target interface spacing/h about `1.5` (`panel_length/h ~= 3`).
- `LaplaceInteriorDirichlet2D` defaults to `LaplaceInteriorPanelMethod2D::ChebyshevLobattoCenter`.
- Legacy Gauss comparison code belongs in `tests/archive/`, not active problem wrappers.

### Latest convergence snapshot

`test_dirichlet`, 5-fold star, Chebyshev-Lobatto path:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 8.7805e-02 | - | 16 |
| 64 | 1.1313e-02 | 2.956 | 15 |
| 128 | 3.9885e-03 | 1.504 | 14 |
| 256 | 4.4891e-04 | 3.151 | 12 |
| 512 | 4.7746e-05 | 3.233 | 12 |
| 1024 | 5.4540e-06 | 3.130 | 11 |

`test_screened`, 3-fold star,
`u=exp(sin(x))+cos(y)`, `-Delta u + u=f`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 2.7659e-03 | - | 9 |
| 64 | 4.9287e-04 | 2.488 | 10 |
| 128 | 1.0949e-04 | 2.170 | 8 |
| 256 | 1.7159e-05 | 2.674 | 8 |
| 512 | 2.2353e-06 | 2.940 | 8 |
| 1024 | 2.2198e-07 | 3.332 | 8 |

`test_transmission`, 5-fold star,
`beta_int=2`, `beta_ext=1`, `kappa^2/beta=lambda_sq=1.1`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 8.3314e-03 | - | 9 |
| 64 | 9.4386e-04 | 3.142 | 8 |
| 128 | 1.9480e-04 | 2.277 | 8 |
| 256 | 1.8549e-05 | 3.393 | 8 |
| 512 | 1.4424e-06 | 3.685 | 8 |
| 1024 | 8.2628e-07 | 0.804 | 8 |

`test_bvp`, unit circle centered at origin, box `(-1.7,1.7)^2`,
`eta=kappa^2=1.1`, target interface spacing/h about `1.5`:

| BVP | N=32 | N=64 | N=128 | N=256 | N=512 |
|-----|-----:|-----:|------:|------:|------:|
| Interior Dirichlet error | 1.7037e-03 | 1.3275e-04 | 1.9113e-05 | 2.6885e-06 | 4.4836e-07 |
| Interior Dirichlet GMRES | 10 | 9 | 9 | 8 | 6 |
| Exterior Dirichlet error | 1.8697e-04 | 2.3980e-05 | 3.5066e-06 | 7.0953e-07 | 1.8515e-07 |
| Exterior Dirichlet GMRES | 12 | 10 | 10 | 9 | 8 |
| Interior Neumann error | 8.2754e-03 | 2.0068e-03 | 6.3105e-04 | 1.9425e-04 | 4.9580e-05 |
| Interior Neumann GMRES | 10 | 9 | 9 | 8 | 7 |
| Exterior Neumann error | 1.2825e-03 | 3.5106e-04 | 9.1097e-05 | 2.2758e-05 | 5.8315e-06 |
| Exterior Neumann GMRES | 8 | 8 | 7 | 7 | 6 |

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
- **Restrict**: maps side-specific grid samples to the interface average branch
  before interpolation: interior samples subtract `C/2`, exterior samples add `C/2`.
- **Interface average**:
  `u_avg = (u⁺+u⁻)/2`, and similarly
  `un_avg = (∂u⁺/∂n + ∂u⁻/∂n)/2`
- **Selected-side reconstruction**:
  `u⁺ = u_avg + [u]/2`, `u⁻ = u_avg - [u]/2`,
  `∂n u⁺ = un_avg + [∂n u]/2`, and
  `∂n u⁻ = un_avg - [∂n u]/2`.

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
