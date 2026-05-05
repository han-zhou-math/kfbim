# KFBIM Recon

A C++17 reconstruction of a Kernel-Free Boundary Integral Method (KFBIM) solver
for elliptic PDEs on Cartesian grids with complex embedded interfaces.

KFBIM replaces explicit Green's-function kernels with local PDE solves. One
interface iteration applies a matrix-free operator:

```text
Spread interface jumps -> solve bulk PDE -> restrict solution to interface
```

The current stable problem wrappers are 2D and 3D Laplace BVP and transmission
solvers. The 2D path uses Chebyshev-Lobatto panels, while the 3D path uses
shared P2 triangular patches and is covered on spherical and ellipsoidal
interfaces. The repository is structured as a reusable library rather than a
one-off solver.

## Status

Implemented:

- Cartesian and MAC grid types in 2D and 3D
- 2D and 3D interface containers with panel connectivity, quadrature points, normals, and weights
- 2D and 3D grid/interface pairing with closest-point and domain-label queries
- Quasi-uniform arc-length curve resampling for stable interface discretization
- 2D Laplace panel Cauchy solver for local jump reconstruction
- 2D Laplace spread and restrict transfer operators
- 3D P2 triangular-patch Laplace local Cauchy solver with 16 expansion centers
  per parent triangle
- 3D P2 Laplace spread/restrict transfer operators and
  `LaplacePotentialEval3D`
- 2D and 3D zFFT-backed Laplace bulk solvers
- Matrix-free `IKFBIOperator` interface implemented by Laplace problem wrappers
- Restarted GMRES outer solver
- Modular `LaplacePotentialEval2D` operators for D/S/N jump primitives
- End-to-end 2D Laplace BVP and transmission tests with current
  Chebyshev-Lobatto convergence coverage, including star boundaries
- End-to-end 3D screened Laplace BVP and transmission tests on shared P2
  triangular interfaces, including sphere and ellipsoid transmission coverage
- Current concrete problem APIs: `LaplaceBvp2D`, `LaplaceBvp3D`,
  `LaplaceTransmission2D`, and `LaplaceTransmission3D`

In progress / planned:

- Richer forcing/nonzero-volume-potential APIs and diagnostics for Laplace
  problem wrappers
- 3D runtime profiling and robustness work, especially the expensive
  different-ratio transmission operator
- Variable-coefficient, Stokes, and elasticity extensions
- Python and MATLAB bindings

See [AGENTS.md](AGENTS.md) for current development notes and handoff plans.

## Repository Layout

```text
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
- CGAL for GridPair geometry queries and 3D nearest-neighbor acceleration;
  required for the active geometry-dependent tests

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

## Test

```bash
ctest --test-dir build
```

Run a specific executable directly when you want Catch2 output:

```bash
./build/tests/test_interface -s
```

Primary PDE/convergence executables are `test_interface`, `test_interface_3d`,
`test_bvp`, `test_bvp_3d`, `test_transmission`, `test_transmission_3d`, and
`test_transmission_ellipsoid_3d`. The 3D tests use power-of-two grid sizes;
set `KFBIM_HIGH_RES_3D=1` to include the highest-resolution 3D levels.

## Visualization Scripts

The `python/` directory contains Python visualization helpers for transfer,
IIM, local Cauchy, grid-pair, and interface-labeling behavior. These scripts are
diagnostic tools and are not required for building the C++ library.

Example:

```bash
python3 python/visualize_laplace_iface_2d.py
```

Generated PNGs under `python/` are ignored by git.

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
`LaplaceBvpPanelMethod2D::ChebyshevLobattoCenter`; keep the legacy
Gauss path only for explicit regression or comparison tests.
For new 3D Laplace work, use shared six-node P2 triangular patches with
`PanelNodeLayout3D::QuadraticLagrange`, 16 expansion centers per parent
triangle, and target adjacent P2 node spacing over grid spacing of about `1.5`.
