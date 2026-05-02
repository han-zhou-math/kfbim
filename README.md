# KFBIM Recon

A C++17 reconstruction of a Kernel-Free Boundary Integral Method (KFBIM) solver
for elliptic PDEs on Cartesian grids with complex embedded interfaces.

KFBIM replaces explicit Green's-function kernels with local PDE solves. One
interface iteration applies a matrix-free operator:

```text
Spread interface jumps -> solve bulk PDE -> restrict solution to interface
```

The current code focuses on Laplace interface problems in 2D, with supporting
grid, geometry, transfer, local Cauchy, and FFT bulk-solver components. The
repository is structured as a reusable library rather than a one-off solver.

## Status

Implemented:

- Cartesian and MAC grid types in 2D and 3D
- 2D and 3D interface containers with panel connectivity, quadrature points, normals, and weights
- 2D and 3D grid/interface pairing with closest-point and domain-label queries
- Quasi-uniform arc-length curve resampling for stable interface discretization
- 2D Laplace panel Cauchy solver for local jump reconstruction
- 2D Laplace spread and restrict transfer operators
- 2D and 3D zFFT-backed Laplace bulk solvers
- End-to-end 2D Laplace interface tests with second-order convergence behavior
- Boundary-value problem APIs (e.g., Laplace Interior Dirichlet BVP)

In progress / planned:

- Matrix-free `LaplaceKFBIOperator`
- Restarted GMRES outer solver
- Variable-coefficient, Stokes, and elasticity extensions
- Python and MATLAB bindings

See [PROGRESS.md](PROGRESS.md) for current development notes.

## Repository Layout

```text
core/
  grid/           Cartesian and MAC grids
  interface/      Interface quadrature and panel data
  geometry/       GridPair queries and domain labels
  local_cauchy/   Local panel Cauchy reconstruction
  transfer/       Laplace spread/restrict operators
  solver/         FFT and zFFT bulk solvers
  operator/       Matrix-free KFBI operator interfaces
  gmres/          Outer solver interfaces
problems/         Higher-level problem APIs
tests/            Catch2 test suite
scripts/          Visualization and diagnostics
third_party/zfft/ Vendored zFFT backend
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
- CGAL for future geometry backends; current 2D grid-pair code is self-contained

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
./build/tests/test_laplace_iface_2d -s
```

## Visualization Scripts

The `scripts/` directory contains Python visualization helpers for transfer,
IIM, local Cauchy, grid-pair, and interface-labeling behavior. These scripts are
diagnostic tools and are not required for building the C++ library.

Example:

```bash
python3 scripts/visualize_laplace_iface_2d.py
```

Generated PNGs under `scripts/` are ignored by git.

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
interface pipeline. When adding a new component, prefer a focused component test
and one integration test that verifies convergence or conservation behavior.
