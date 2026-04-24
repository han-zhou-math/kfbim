# KFBIM Library — Project Guide

## Project Goal
Reconstruct problem-dependent KFBIM codes into a clean, reusable C++ library with a public API.
Future: Python/MATLAB bindings (pybind11), possibly Jupyter notebooks.

## Core Algorithm
Kernel-free boundary integral method for elliptic PDEs on complex interfaces/boundaries in 2D and 3D.

One GMRES iteration = one application of the interface-to-interface operator:
1. Spread interface data → correct bulk RHS
2. Solve PDE on Cartesian grid (FDM + fast solver)
3. Restrict bulk solution → interface data
GMRES iterates on this operator until the interface unknowns converge.

## Target PDE Types
- Laplace / Poisson
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

## Directory Layout (planned)
<!-- Fill in as structure is decided -->

```
core/
  grid/           # CartesianGrid2D/3D, MACGrid2D/3D, DofLayout
  interface/      # Interface2D, Interface3D
  geometry/       # GridPair2D/3D  (CGAL-dependent; pimpl hides CGAL headers)
  transfer/       # Spread, Restrict  (Layer 1 — TBD)
  local_cauchy/   # LaplaceLocalSolver, StokesLocalSolver, ...  (Layer 1.5 — TBD)
  solver/         # BulkSolver and impls  (Layer 2 — TBD)
  operator/       # KFBIOperator  (Layer 3 — TBD)
  gmres/          # outer solver  (Layer 4 — TBD)
problems/         # LaplaceProblem, StokesProblem, ...  (Layer 5 — TBD)
bindings/         # pybind11 wrappers (future)
tests/
examples/
```

## Coding Conventions
<!-- Fill in as decided -->

- Language standard: C++17 (minimum; upgrade if a strong reason arises)
- Naming: `CamelCase` for classes/types, `snake_case` for methods and variables (matches Eigen, CGAL, deal.II)
- Error handling: exceptions for all recoverable errors; `assert` only for truly unreachable internal invariants (asserts disabled in release builds — never rely on them for library correctness)
- Data containers: Eigen internally; public API uses `Eigen::Ref<>` so callers can pass raw arrays, std::vector, or Eigen without copying. Enables pybind11 ↔ numpy and MATLAB interop. Internal custom vectors migrated gradually.

## Build
<!-- Fill in once CMake is set up -->

```bash
# placeholder
cmake -B build && cmake --build build
```
