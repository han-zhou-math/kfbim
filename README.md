# KFBIM Recon

A C++17 reconstruction of a Kernel-Free Boundary Integral Method (KFBIM) solver
for elliptic PDEs on Cartesian grids with complex embedded interfaces.

KFBIM replaces explicit Green's-function kernels with local PDE solves. One
interface iteration applies a matrix-free operator:

```text
Spread interface jumps -> solve bulk PDE -> restrict solution to interface
```

The current stable problem wrappers are 2D and 3D Laplace BVP and transmission
solvers. The 2D path uses quadratic Lagrange P2 panels, with the former
Chebyshev-Lobatto API names kept as source-compatible aliases. The 3D path
uses shared P2 triangular patches and is covered on spherical, ellipsoidal,
and torus interfaces. The repository is structured as a reusable library
rather than a one-off solver.

## Status

Implemented:

- Cartesian and MAC grid types in 2D and 3D
- 2D and 3D interface containers with panel connectivity, quadrature points, normals, and weights
- 2D and 3D grid/interface pairing with closest bulk-node queries, active P2
  expansion-center lookup caches, domain labels, and P2 narrow-band projection
  queries
- Quasi-uniform arc-length curve resampling for stable interface discretization
- 2D P2 Laplace panel Cauchy solver with four expansion centers per panel
- 2D P2 Laplace spread and restrict transfer operators with fixed square
  six-point quadratic restrict interpolation
- 2D P2 curve geometry helpers and `GridPair2D` projection cache returning
  panel/local coordinates, projected points, normals, signed distances, and
  convergence flags for narrow-band grid nodes
- 3D P2 triangular-patch Laplace local Cauchy solver with 16 expansion centers
  per parent triangle
- 3D P2 curved-surface geometry helpers and `GridPair3D` projection cache
  returning panel/barycentric coordinates, projected points, normals, and
  signed distances for narrow-band grid nodes
- 3D P2 Laplace spread/restrict transfer operators with fixed square
  ten-point quadratic restrict interpolation and
  `LaplacePotentialEval3D`
- 2D and 3D zFFT-backed Laplace bulk solvers
- Matrix-free `IKFBIOperator` interface implemented by Laplace problem wrappers
- Restarted GMRES outer solver
- Modular `LaplacePotentialEval2D/3D` operators for D/S/N jump primitives and
  combined D+S layer evaluations
- Public forwarding headers under `include/kfbim/` and CMake install/export
  rules for `kfbim_core`
- End-to-end 2D Laplace BVP and transmission tests with current P2 quadratic
  panel convergence coverage, including star boundaries and a bi-periodic
  transmission case
- End-to-end 3D screened Laplace BVP and transmission tests on shared P2
  triangular interfaces, including sphere, ellipsoid, and torus transmission
  coverage
- Current concrete problem APIs: `LaplaceBvp2D`, `LaplaceBvp3D`,
  `LaplaceTransmission2D`, and `LaplaceTransmission3D`
- App-level 2D inverse shape optimization demo for
  `LaplaceTransmission2D::CommonRatio`, with adjoint and finite-difference
  gradients plus CSV/PNG/GIF visualization output

In progress / planned:

- Richer forcing/nonzero-volume-potential APIs and diagnostics for Laplace
  problem wrappers
- 3D runtime profiling and robustness work, especially the expensive
  different-ratio transmission operator
- Parallel KFBI prototype planning around MFEM surface PDEs/meshes, distributed
  bulk-surface search, and adaptive bulk backends
- Variable-coefficient, Stokes, and elasticity extensions
- Python and MATLAB bindings

See [AGENTS.md](AGENTS.md) for current development notes and handoff plans.

## Repository Layout

```text
include/kfbim/    Public forwarding headers for the current library API
src/
  grid/           Cartesian and MAC grids
  interface/      Interface quadrature and panel data
  geometry/       GridPair queries and domain labels
  local_cauchy/   Local panel/patch Cauchy reconstruction
  transfer/       Laplace spread/restrict operators
  bulk_solvers/   FFT and zFFT bulk solvers
  potentials/     Reusable 2D/3D potential evaluators
  operators/      Matrix-free KFBI interface and problem wrappers
  gmres/          Restarted GMRES outer solver
apps/             Standalone demos and diagnostic applications
tests/            Catch2 test suite
python/           Visualization and diagnostics
third_party/zfft/ Vendored zFFT backend
third_party/old-codes/
                  Archived reference implementation snippets
notes/            Derivations and implementation notes
```

## Dependencies

Required:

- CMake 3.20 or newer
- C++17 compiler
- Eigen 3.4 or newer

Optional:

- Catch2 v3 for tests; fetched by CMake if not installed
- FFTW3 for the FFTW-backed solver
- CGAL for raw/legacy GridPair geometry queries and projection fallback
  searches; active 2D/3D P2 transfer paths use internal expansion-center
  spatial hashes for nearest-center lookup

On macOS with Homebrew:

```bash
brew install cmake eigen catch2 fftw cgal
```

If Eigen or Catch2 are not available locally, CMake can fetch them during
configuration.

## Build

```bash
cmake -B build
cmake --build build
```

Disable tests if you only want the library targets:

```bash
cmake -B build -DKFBIM_BUILD_TESTS=OFF
cmake --build build
```

Standalone applications are enabled by default. They can be disabled with:

```bash
cmake -B build -DKFBIM_BUILD_APPS=OFF
cmake --build build
```

## Test

```bash
ctest --test-dir build
```

Run a specific executable directly when you want Catch2 output:

```bash
./build/tests/test_interface -s
```

Primary PDE/convergence executables are `test_interface`, `test_interface_3d`,
`test_bvp`, `test_bvp_3d`, `test_transmission`, `test_transmission_3d`,
`test_transmission_ellipsoid_3d`, `test_transmission_torus_3d`, and
`test_transmission_periodic_2d`. Component coverage includes
`test_projection_2d`, `test_projection_3d`, and `test_refactor_utilities`.
The PDE convergence programs print a `wall_s` column for each grid level and
write the same column to their convergence CSV files. The 3D PDE tests use
power-of-two grid sizes; set `KFBIM_HIGH_RES_3D=1` to include the
highest-resolution 3D levels.

The current periodic 2D transmission test uses a cell-centered periodic bulk
solver and an interface well away from the box edge. The 2D transfer operators
are not yet general periodic-wrap transfer operators for interfaces crossing a
periodic boundary.

## Shape Optimization App

The `apps/` directory contains a small inverse 2D shape-recovery application:

```bash
cmake --build build --target shape_opt_transmission_2d
build/apps/shape_opt_transmission_2d --N 64 --iters 12
python3 apps/visualize_shape_opt_2d.py output/shape_opt_transmission_2d
```

The app generates synthetic exterior measurements from a hidden smooth
inclusion, then recovers a fixed-center radial Fourier boundary using the
public `LaplaceTransmission2D` API. It writes `summary.csv`,
`observations.csv`, `target_boundary.csv`, per-iteration boundary CSV files,
objective plots, boundary-evolution PNGs, and `shape_evolution.gif`. The
default gradient is the continuous adjoint method; use `--gradient fd` for the
finite-difference comparison path or `--gradient-check` for an initial-shape
directional check.

## Visualization Scripts

The `python/` directory contains Python visualization helpers for transfer,
IIM, local Cauchy, grid-pair, and interface-labeling behavior. These scripts are
diagnostic tools and are not required for building the C++ library.

Example:

```bash
python3 python/visualize_laplace_iface_2d.py
```

The torus transmission test can export geometry and solution CSVs with
`KFBIM_TORUS_VIS_N=<N>`, then render them with:

```bash
python3 python/visualize_transmission_torus_3d.py \
  output/laplace_transmission_common_ratio_torus_3d 256
```

Generated PNGs under `python/` are ignored by git.

## Notes

Design and derivation notes live under `notes/`:

- `bie.md`: generic interface BIE derivation
- `iim.md`: IIM correction formula notes
- `math.md`: KFBI mathematical notes
- `theory.md`: broader KFBI theory notes
- `parallel_plan.md`: future parallel MFEM/AMR KFBI project plan
- `shape_opt.tex` / `shape_opt.pdf`: inverse transmission shape optimization
  demo and adjoint derivation
- `stokes.pdf`: Stokes reference paper

## Development Notes

The core architecture is layered so higher-level solvers depend on abstractions
from lower layers:

```text
Problem API
Outer solver
Interface operator
Bulk solver
Transfer operators and local Cauchy solver
Grid, interface, and geometry data
```

Most current tests exercise individual layers plus the full 2D Laplace
interface pipeline and the 3D P2 Laplace BVP/transmission paths. When adding a
new component, prefer a focused component test and one integration test that
verifies convergence or conservation behavior.
For new 2D Laplace work, use `CurveResampler2D::discretize()` and
`LaplaceBvpPanelMethod2D::QuadraticPanelCenter`. The old
`ChebyshevLobattoCenter` and Lobatto transfer names remain compatibility
aliases; keep the legacy Gauss path only for explicit regression or comparison
tests. Active 2D restrict interpolation uses a fixed six-node square quadratic
stencil around the nearest grid node; no least-squares fallback is used.
Active 2D P2 `GridPair2D` geometry lookup is center-only for spread/restrict:
it builds one spatial hash over the four generated expansion centers per panel,
rasterizes a default narrow-band nearest-center cache, labels closed P2 curves
with center-seeded BFS, and keeps nearest interface-DOF lookup as a lazy
compatibility query.
For new 3D Laplace work, use shared six-node P2 triangular patches with
`PanelNodeLayout3D::QuadraticLagrange`, 16 expansion centers per parent
triangle, and target adjacent P2 node spacing over grid spacing of about `1.2`.
Active 3D restrict interpolation uses the analogous fixed ten-node square
quadratic stencil. The default 3D transfer correction is nearest
expansion-center expansion. Active 3D P2 `GridPair3D` uses the same 16
expansion centers as a shared lookup set for labels, spread, restrict, and
projection seeding; nearest interface-DOF lookup is lazy compatibility-only.
Projection-point IIM correction is available as an opt-in comparison path; use
`GridPair3D::project_grid_nodes_to_interface()` for the explicit set of grid
nodes where `C(x)` is needed, or `GridPair3D::project_near_interface_nodes()`
for broader projection-geometry diagnostics.
