# KFBIM Library - Project Guide

## Project Goal
Reconstruct problem-dependent KFBIM codes into a clean, reusable C++ library with a public API.
Future: Python/MATLAB bindings (pybind11), possibly Jupyter notebooks.

## Current Status
- Branch `main`; run `git log --oneline -n 3` for the exact current commit.
- This guide reflects the 2026-05-04 2D Laplace P2 quadratic panel,
  screened-Poisson, merged potential-evaluation, screened BVP wrapper,
  transmission, three-program PDE convergence-test reorganization work, the
  first verified 3D P2 Laplace potential pipeline, and the 2026-05-05 3D
  screened BVP/transmission wrapper and ellipsoid transmission-test update.
  It also reflects the 2026-05-06 3D torus transmission test, 2D bi-periodic
  transmission test, combined D+S potential-evaluation optimization, shortened
  note filenames, parallel KFBI planning note, 2026-05-07 3D P2
  narrow-band projection geometry, and the 2026-05-07 projection-point
  correction prototype. It also reflects the 2026-05-08 staged organization
  refactor, public forwarding headers, shared grid/operator utilities, 2D/3D
  P2 projection split, 2D P2 geometry/center-sampling unification, and fixed
  square 2D/3D restrict stencils. The default 3D correction path remains
  nearest expansion-center expansion.
- **Completed modules** (active tests passing):
  - Layer 0: `CartesianGrid2D`, `Interface2D`, `GridPair2D`
    - `Interface2D` tracks panel node layout: `QuadraticLagrange`,
      `LegacyGaussLegendre`, or `Raw`. `ChebyshevLobatto` remains a
      source-compatible alias for the P2 quadratic layout.
    - Quadratic Lagrange panels use explicit connectivity with shared adjacent
      endpoints, so a closed curve has `num_points() = 2*num_panels()`.
    - `GridPair2D::domain_label()` uses nearest-sample local normals for P2
      panels, with interface DOFs plus the shared four expansion-center samples
      per panel. Raw/legacy polygon labeling remains available for fallback
      paths.
    - `GridPair2D::project_near_interface_nodes(radius)` returns cached P2
      curve projections for narrow-band grid nodes. Each `CurveProjection2D`
      stores the grid node, parent panel, component, local coordinate,
      projected point, oriented normal, signed distance, distance, tangential
      residual, Newton iteration count, and convergence flag. The Newton
      initial guesses are the same four expansion-center locations per panel
      used by the active 2D local Cauchy path.
    - `GridPair2D::project_grid_nodes_to_interface(nodes)` projects an explicit
      grid-node support set and caches lookup results for future 2D
      projection-point transfer work.
  - Layer 0 3D additions:
    - `Interface3D` tracks panel node layout: `Raw` or `QuadraticLagrange`.
    - P2 triangular patches store explicit six-node connectivity
      `[v0,v1,v2,e01,e12,e20]`, allowing shared vertices and shared edge
      midpoint DOFs.
    - `GridPair3D` uses CGAL kd-tree nearest-neighbor acceleration for closest
      interface/sample queries. For `QuadraticLagrange` P2 surfaces, narrow-band
      distance samples include the interface DOFs and the barycenters of the 16
      twice-subdivided child triangles per parent triangle.
    - `GridPair3D::project_near_interface_nodes(radius)` returns cached P2
      curved-surface projections for narrow-band grid nodes. Each
      `SurfaceProjection3D` stores the grid node, parent panel, component,
      barycentric reference coordinate, projected point, oriented normal,
      signed distance, distance, tangential residual, Newton iteration count,
      and convergence flag. The Newton initial guesses are the same 16
      expansion-center locations per parent triangle used by the 3D P2 local
      Cauchy path. Boundary/edge fallbacks can return `converged=false` while
      still carrying a valid panel-local point.
    - `GridPair3D::project_grid_nodes_to_interface(nodes)` projects an explicit
      grid-node support set. It chooses the parent triangle from the nearest
      expansion center, initializes the parameter by closest point on the flat
      vertex triangle, then applies curved-patch Newton. If Newton fails, it
      returns the flat-triangle parameter with `converged=false`.
    - Current P2 domain labeling is local-normal based at the nearest curved
      surface sample, not a full CGAL inside/outside query on the curved
      tessellation.
  - Layer 1: preferred 2D P2 quadratic transfer path:
    - `LaplaceQuadraticPanelCenterSpread2D`
    - `LaplaceQuadraticPanelCenterRestrict2D`
    - 2D restrict interpolation uses the fixed square six-node quadratic
      stencil around the nearest grid node:
      `(i,j)`, `(i+d1,j)`, `(i,j+d2)`, `(i+d1,j+d2)`,
      `(i-d1,j)`, `(i,j-d2)`, with `d* = (x* > x*_i) ? 1 : -1`.
    - `LaplaceLobattoCenterSpread2D` and `LaplaceLobattoCenterRestrict2D`
      remain source-compatible aliases.
  - Layer 1.5: `LaplacePanelCauchySolver2D`
    - Legacy Gauss Cauchy solve at the three panel points remains for archived/reference paths.
    - Quadratic Lagrange/P2 Cauchy solve at four generated expansion centers
      per panel, `s={-0.75,-0.25,0.25,0.75}`.
    - Current collocation offset is `0.05`.
  - Layer 2: `LaplaceFftBulkSolverZfft2D` (DST, Dirichlet BC, optional screened shift)
  - Layer 3: `IKFBIOperator`, implemented directly by active Laplace problem
    wrappers; `LaplacePotentialEval2D` backs reusable potential evaluation
    (Spread → BulkSolve → Restrict pipeline, arc_h_ratio check)
  - Layer 3 modular potentials: `LaplacePotentialEval2D`
    - Evaluates arbitrary jumps/RHS and returns bulk values plus averaged trace/flux.
    - Specialized D/S/N helpers are thin calls into the same general pipeline.
    - `eval_layer_combination(phi, psi)` evaluates D[phi] + S[psi] with one
      potential solve when both jumps use the same bulk differential operator.
    - Uses the current restrict convention: restrict returns averaged trace/flux directly.
    - Covered by archived component tests; active top-level tests now exercise it
      through PDE-level manufactured problems.
  - Layer 1-3 3D Laplace potential path:
    - `laplace_p2_patch_center_cauchy_3d()` solves one local 10x10 quadratic
      Cauchy system at each of 16 expansion centers per P2 triangle.
    - For each child triangle, Dirichlet collocation uses the three child
      vertices plus three child edge centers; Neumann collocation uses the three
      child edge centers; PDE collocation is imposed at the child barycenter.
      No offset delta is used.
    - `LaplaceQuadraticPatchCenterSpread3D` and
      `LaplaceQuadraticPatchCenterRestrict3D` implement the current transfer
      path for P2 triangular patches. The default correction method is
      `LaplaceCorrectionMethod3D::NearestExpansionCenter`.
      3D restrict interpolation uses the fixed square ten-node quadratic
      stencil matching `third_party/old-codes/Stencil.h::get3DStencil_10`;
      the `restrict_stencil_radius` option is now a compatibility/cache-radius
      parameter, not an interpolation stencil-size control.
      `LaplaceCorrectionMethod3D::ProjectionPoint` is implemented as an
      opt-in method that evaluates `C(x)` from projected P2 surface data and
      precomputes only grid nodes needed by crossing stencil edges and the
      restrict interpolation stencils.
    - `LaplacePotentialEval3D` evaluates arbitrary jumps/RHS and returns bulk
      solution plus averaged trace/flux, matching the 2D potential-evaluation
      convention. It also exposes `eval_layer_combination(phi, psi)`.
  - Layer 4: GMRES outer solver
  - Public API boundary:
    - `include/kfbim/kfbim.hpp` includes focused forwarding headers for grid,
      interface, geometry, potentials, and Laplace operators.
    - `kfbim_core` has CMake install/export rules. Existing `src/...` includes
      still work during the transition.
  - Layer 5: 2D Laplace BVP API
    - `LaplaceBvp2D`
    - Solves `-Delta u + eta*u = f` in the selected interior/exterior domain.
    - Default panel method is `QuadraticPanelCenter`.
      `ChebyshevLobattoCenter` remains a source-compatible alias.
    - Interior Neumann uses an unweighted mean-zero vector projection only for
      the pure Laplace nullspace case `eta=0`; screened cases do not project.
  - Layer 5: 3D Laplace BVP API
    - `LaplaceBvp3D`
    - Solves screened interior/exterior Dirichlet and Neumann BVPs on shared
      P2 triangular interfaces using `LaplacePotentialEval3D`.
    - Interior Neumann uses the same unweighted mean-zero projection as 2D
      only for the pure Laplace nullspace case `eta=0`.
  - Discontinuous coefficient interface utility:
    - `LaplaceTransmission2D`
    - Solves `-div(beta grad u)+kappa^2 u=f` with common-ratio and
      different-ratio modes for `kappa^2/beta`.
    - Divides by `beta` to use a common screened operator
      `-Delta u + lambda_sq u=q`.
    - Eliminates optional nonzero outer Cartesian Dirichlet data into the RHS,
      then restores boundary values in the output.
    - `LaplaceTransmission2D` now accepts an optional zFFT bulk boundary
      condition argument, defaulting to Dirichlet. The active periodic
      transmission test uses a cell-centered periodic grid with the interface
      away from the periodic boundary; 2D spread/restrict are not yet general
      periodic-wrap transfer operators for interfaces crossing the boundary.
    - `LaplaceTransmission3D` mirrors the 2D common-ratio and different-ratio
      density layouts on the 3D P2 potential pipeline.
- **Recent implementation notes:**
  - 2D P2 quadratic panel/expansion-center path is implemented and verified.
    Old Chebyshev-Lobatto names are compatibility aliases.
  - `GridPair2D` uses the same four P2 center samples per panel as the active
    2D expansion-center correction path, plus interface DOF nodes, for its
    nearest-sample kd-tree.
  - The unused 2D P2 projection service is implemented for future
    projection-point correction work and mirrors the 3D projection-cache shape.
  - Active 2D/3D restrict interpolation now uses square 6x6/10x10 quadratic
    interpolation systems from fixed grid stencils; no local least-squares
    restrict fallback remains.
  - Gauss-point restrictor/problem-wrapper paths were removed from active code.
  - Archived component tests include direct D/S/N jump-relation coverage for
    `LaplacePotentialEval2D`.
  - Active convergence tests use the same off-center 3-fold star convention:
    center `(0.07,-0.04)`, P2 quadratic panels, outer box from sampled
    interface bounds plus margin, target adjacent P2-node spacing `1.5h`, and
    `panel_length/h = 3.0`.
  - `tests/test_interface.cpp` solves a direct prescribed-jump constant-
    coefficient screened interface problem using `LaplacePotentialEval2D`;
    it reports `GMRES=0` because there is no outer Krylov solve.
  - `tests/test_bvp.cpp` covers all four screened `LaplaceBvp2D` modes:
    interior/exterior Dirichlet and interior/exterior Neumann, with `eta=1.1`
    and exact outer Cartesian Dirichlet elimination/restoration for exterior
    manufactured solutions.
  - `tests/test_transmission.cpp` covers both `LaplaceTransmission2D` modes:
    `CommonRatio` and `DifferentRatios`.
  - `tests/test_transmission_periodic_2d.cpp` covers
    `LaplaceTransmission2D::CommonRatio` on a cell-centered bi-periodic unit
    square with periodic trigonometric manufactured solutions.
  - `tests/test_projection_2d.cpp` covers 2D P2 naming aliases, the shared
    four-center sample set, and `GridPair2D` P2 projection lookup behavior.
  - `KFBIM_PROFILE_INTERFACE_2D=1 build/tests/test_interface -s` reports
    coarse per-component timings plus `GridPair2D` sample counts.
  - `tests/test_interface_3d.cpp` covers the direct prescribed-jump screened
    3D interface problem on a unit P2 sphere centered at the origin in
    `(-1.5,1.5)^3`, with `eta=1.1`, 16 expansion centers per parent triangle,
    and target adjacent P2 node spacing over grid spacing of about `1.5`.
    The default run uses nearest expansion-center correction. Set
    `KFBIM_INTERFACE_3D_CORRECTION=projection` to run the projection-point
    correction comparison.
  - `tests/test_bvp_3d.cpp` covers all four screened `LaplaceBvp3D` modes on
    an off-center P2 sphere.
  - `tests/test_transmission_3d.cpp` covers both `LaplaceTransmission3D`
    modes on an off-center P2 sphere; the common-ratio default sweep runs
    through `N=64`, while the more expensive two-density different-ratio
    default sweep runs through `N=32`. Set `KFBIM_HIGH_RES_3D=1` for the
    higher-resolution 3D transmission levels.
  - `tests/test_transmission_ellipsoid_3d.cpp` covers
    `LaplaceTransmission3D::CommonRatio` on a shared P2 triaxial ellipsoid
    with axes `(0.61,0.49,0.41)`, center `(0.07,-0.04,0.03)`, nonzero outer
    Cartesian Dirichlet data, and target P2 `node_spacing/h ~= 1.5`.
  - `tests/test_transmission_torus_3d.cpp` covers
    `LaplaceTransmission3D::CommonRatio` on a shared P2 torus. Set
    `KFBIM_TORUS_VIS_N=<N>` to export torus geometry and solution CSVs for
    `python/visualize_transmission_torus_3d.py`.
  - `tests/test_projection_3d.cpp` covers the new
    `GridPair3D::project_near_interface_nodes()` and explicit-support
    `GridPair3D::project_grid_nodes_to_interface()` P2 projection paths on an
    off-center sphere. It verifies cached narrow-band lookup, panel/barycentric
    coordinates, oriented normals, signed distances, Newton residuals, and that
    existing `GridPair3D` label/nearest-point queries are unchanged.
  - Different-ratio transmission GMRES applies now combine compatible D and S
    layer potentials per phase, reducing same-operator layer evaluations from
    four potential solves to two. Common-ratio RHS setup also combines
    compatible double-layer and volume terms.
  - zFFT 3D testing should use power-of-two grid sizes; non-power-of-two 3D
    sizes were observed to hang in earlier exploratory runs.
  - Component/basic/top-level legacy tests were moved to `tests/archive/` and
    are preserved as source files but are not registered in active CMake/CTest.
  - Legacy reference code was moved from `old-codes/` to `third_party/old-codes/`.
  - Runtime CSV/PNG output is written under `output/` and should not be committed.
  - `build/` is disposable generated output. Recreate it with CMake when tests
    or local binaries are needed; do not commit it.
  - The C++ source tree is `src/` and the compiled library target remains
    `kfbim_core`.
  - Current concrete problem-level utilities and wrappers live in
    `src/operators/`.
  - Visualization and diagnostic Python scripts live in `python/`.
  - Notes use short filenames: `notes/bie.md`, `notes/iim.md`,
    `notes/math.md`, `notes/theory.md`, `notes/parallel_plan.md`, and
    `notes/stokes.pdf`.
- **Current convergence test status:**
  - 2026-05-08 fixed square restrict-stencil update:
    - `cmake --build build`
    - `build/tests/test_refactor_utilities`
    - `build/tests/test_projection_2d -s`
    - `build/tests/test_projection_3d`
    - PDE-only timed sweep:
      - `/usr/bin/time -p build/tests/test_interface -s`
      - `/usr/bin/time -p build/tests/test_bvp -s`
      - `/usr/bin/time -p build/tests/test_transmission -s`
      - `/usr/bin/time -p build/tests/test_transmission_periodic_2d -s`
      - `/usr/bin/time -p build/tests/test_interface_3d -s`
      - `/usr/bin/time -p build/tests/test_bvp_3d -s`
      - `/usr/bin/time -p build/tests/test_transmission_3d -s`
      - `/usr/bin/time -p build/tests/test_transmission_ellipsoid_3d -s`
      - `/usr/bin/time -p build/tests/test_transmission_torus_3d -s`
    - `git diff --check`
  - 2026-05-08 staged organization refactor and 2D P2 unification:
    - `cmake -B build`
    - `cmake --build build`
    - `build/tests/test_projection_2d -s`
    - `build/tests/test_projection_3d`
    - `build/tests/test_refactor_utilities`
    - `build/tests/test_interface -s`
    - `build/tests/test_bvp`
    - `build/tests/test_transmission`
    - `build/tests/test_transmission_periodic_2d`
    - `KFBIM_PROFILE_INTERFACE_3D=1 KFBIM_INTERFACE_3D_MAX_N=64 build/tests/test_interface_3d -s`
    - `git diff --check`
  - 2026-05-07 3D P2 projection geometry:
    - `cmake --build build`
    - `build/tests/test_projection_3d`
    - `ctest --test-dir build -R projection --output-on-failure`
    - `git diff --check`
  - 2026-05-07 3D direct-interface correction comparison on the unit sphere:
    - `build/tests/test_interface_3d -s`
    - `KFBIM_INTERFACE_3D_CORRECTION=projection build/tests/test_interface_3d -s`
    - Default remains nearest expansion-center because it was more accurate on
      the finest tested levels for this benchmark.
  - 2026-05-06 transmission suite after the torus/periodic tests and combined
    D+S optimization:
    - `build/tests/test_transmission_periodic_2d -s`
    - `KFBIM_HIGH_RES_3D=1 build/tests/test_transmission_torus_3d -s`
    - `KFBIM_HIGH_RES_3D=1 KFBIM_TORUS_VIS_N=256 build/tests/test_transmission_torus_3d -s`
    - `ctest --test-dir build -R transmission --output-on-failure`
    - `git diff --check`
  - 3D additions passed on 2026-05-05 after the target P2
    `node_spacing/h ~= 1.5` and ellipsoid common-ratio transmission update:
    - `cmake --build build`
    - `build/tests/test_transmission_3d -s`
    - `build/tests/test_transmission_ellipsoid_3d -s`
    - `ctest --test-dir build -R ellipsoid --output-on-failure`
    - `git diff --check`
  - The active 2D/direct-interface baseline previously passed on 2026-05-04:
    - `build/tests/test_interface -s`
    - `build/tests/test_interface_3d -s`
    - `build/tests/test_bvp -s`
    - `build/tests/test_transmission -s`
  - Earlier 2026-05-05 smoke runs also verified `build/tests/test_bvp_3d -s`
    before the final 3D ratio update.

## Next Agent Handoff — Post 2D/3D P2 Refactor

The 3D scalar Laplace wrappers now exist. When resuming this project, preserve
the 2D public API and the verified 3D P2 potential conventions while improving
coverage, runtime, and numerical robustness.

1. **Keep the foundation stable first.**
   - Preserve `LaplaceBvp2D` behavior and the default
     `QuadraticPanelCenter` panel method.
   - Keep the active 2D Laplace path on P2 quadratic panels.
   - Keep old Chebyshev-Lobatto method and transfer names working as
     source-compatible aliases.
   - Preserve the 3D P2 convention: `PanelNodeLayout3D::QuadraticLagrange`,
     six-node shared triangular patches, 16 expansion centers per parent
     triangle, and target P2 `node_spacing/h ~= 1.5`.
   - Keep reusable 3D potential-building code in `src/potentials/`; keep
     problem wrappers in `src/operators/`.

2. **Current 3D operators.**
   - `LaplaceBvp3D` mirrors `LaplaceBvp2D` for screened interior/exterior
     Dirichlet and Neumann BVPs.
   - `LaplaceTransmission3D` mirrors `LaplaceTransmission2D` with common-ratio
     one-density and different-ratio two-density modes.
   - Exterior 3D modes eliminate optional nonzero outer Cartesian Dirichlet data
     into the RHS and restore boundary values in the returned bulk solution.

3. **Next implementation targets.**
   - Keep nearest expansion-center correction as the default 3D transfer path.
     The projection-point IIM correction path is implemented and useful for
     comparison, but currently remains opt-in while its accuracy is assessed.
   - Rerun and refresh the 3D BVP convergence snapshots after the final target
     P2 `node_spacing/h ~= 1.5` change.
   - Profile the 3D different-ratio transmission operator; it is still
     substantially more expensive than common-ratio, though compatible D/S
     layer evaluations are now combined per phase.
   - Keep the different-ratio 3D active default sweep modest, and use
     `KFBIM_HIGH_RES_3D=1` for high-resolution `N=64`/`N=128` runs.

4. **Deferred work.**
   - Do not start Python/MATLAB bindings until the C++ problem API is stable.
   - Defer 3D Stokes/elasticity until scalar 3D Laplace BVP and transmission
     wrappers remain stable under the ratio-1.5 test suite.
   - Defer Stokes until the Laplace BVP API is clean; Stokes currently has
     scaffolding but not concrete local Cauchy, spread, restrict, or operator
     implementations.

### Preferred 2D panel method

Use P2 quadratic Lagrange panels for new 2D Laplace work.

- Geometry/DOF nodes per panel: `s={-1,0,1}`.
- Adjacent panels share endpoint DOFs. For a closed curve with `Np` panels,
  the unknown/vector length is `2*Np`: `Np` shared endpoints plus `Np` panel
  midpoints.
- Correction expansion centers per panel: `s={-0.75,-0.25,0.25,0.75}`.
- `CurveResampler2D::discretize()` now returns P2 quadratic panels.
- Current active convergence tests set target adjacent P2-node spacing over
  grid spacing to `1.5`, so `panel_length/h = 3.0`.
- `LaplaceBvp2D` defaults to `LaplaceBvpPanelMethod2D::QuadraticPanelCenter`.
- `ChebyshevLobatto`, `ChebyshevLobattoCenter`, and old Lobatto transfer class
  names are compatibility aliases.
- Legacy Gauss comparison code belongs in `tests/archive/`, not active problem wrappers.

### Latest convergence snapshot

The active 2D tests use the same off-center 3-fold star and target adjacent
P2-node spacing/h of `1.5` (`panel_length/h = 3.0`). Active 3D sphere tests
target P2 node spacing/h of about `1.5`.

2026-05-08 PDE-only wall times after the fixed square restrict-stencil update:

| Group | Tests | total wall time |
|-------|------:|----------------:|
| 2D PDE tests | 4 | 690.08s |
| 3D PDE tests | 5 | 398.81s |

Per-executable wall times:

| executable | wall time |
|------------|----------:|
| `test_interface -s` | 8.99s |
| `test_bvp -s` | 365.72s |
| `test_transmission -s` | 278.29s |
| `test_transmission_periodic_2d -s` | 37.08s |
| `test_interface_3d -s` | 110.34s |
| `test_bvp_3d -s` | 177.36s |
| `test_transmission_3d -s` | 54.16s |
| `test_transmission_ellipsoid_3d -s` | 40.15s |
| `test_transmission_torus_3d -s` | 16.80s |

`test_interface`, direct prescribed-jump screened interface problem,
`eta=1.1`, homogeneous outer Cartesian Dirichlet data:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 2.8477e-03 | - | 0 |
| 64 | 5.9548e-04 | 2.258 | 0 |
| 128 | 1.5400e-04 | 1.951 | 0 |
| 256 | 3.6164e-05 | 2.090 | 0 |
| 512 | 9.5274e-06 | 1.924 | 0 |

`test_interface_3d`, direct prescribed-jump screened 3D interface problem,
unit P2 sphere centered at the origin in `(-1.5,1.5)^3`, `eta=1.1`, target P2
`node_spacing/h ~= 1.5`. Default correction method:
`NearestExpansionCenter`.

| N | panels | iface pts | max err | order | GMRES |
|---:|------:|----------:|--------:|------:|------:|
| 4 | 32 | 66 | 1.3930e-01 | - | 0 |
| 8 | 32 | 66 | 3.5236e-02 | 1.983 | 0 |
| 16 | 72 | 146 | 2.3289e-02 | 0.597 | 0 |
| 32 | 200 | 402 | 4.0566e-03 | 2.521 | 0 |
| 64 | 800 | 1602 | 5.9520e-04 | 2.769 | 0 |
| 128 | 3200 | 6402 | 1.0797e-04 | 2.463 | 0 |

`test_bvp`, all four screened `LaplaceBvp2D` modes, `eta=1.1`:

| BVP | N=32 | N=64 | N=128 | N=256 | N=512 |
|-----|-----:|-----:|------:|------:|------:|
| Interior Dirichlet error | 4.2920e-04 | 1.2809e-04 | 1.9254e-05 | 2.4861e-06 | 2.3654e-07 |
| Interior Dirichlet GMRES | 12 | 11 | 13 | 11 | 10 |
| Exterior Dirichlet error | 1.8580e-04 | 3.6424e-05 | 6.5979e-06 | 4.5645e-07 | 1.3168e-07 |
| Exterior Dirichlet GMRES | 15 | 15 | 14 | 13 | 12 |
| Interior Neumann error | 2.9581e-02 | 5.9596e-03 | 2.6799e-04 | 1.5751e-04 | 1.9894e-05 |
| Interior Neumann GMRES | 11 | 11 | 11 | 10 | 10 |
| Exterior Neumann error | 6.4472e-04 | 4.5593e-04 | 6.2434e-05 | 9.5711e-06 | 4.4865e-06 |
| Exterior Neumann GMRES | 9 | 9 | 9 | 9 | 9 |

`test_bvp_3d`, all four screened `LaplaceBvp3D` modes on the off-center P2
sphere, `eta=1.1`. Default sweep omits `N=128`; set `KFBIM_HIGH_RES_3D=1`
for the high-resolution level.

| BVP | N=8 | N=16 | N=32 | N=64 |
|-----|----:|-----:|-----:|-----:|
| Interior Dirichlet error | 1.7543e-01 | 3.5501e-03 | 4.2439e-04 | 4.7665e-05 |
| Interior Dirichlet GMRES | 24 | 11 | 12 | 10 |
| Exterior Dirichlet error | 3.0181e-04 | 2.0931e-04 | 2.0338e-05 | 2.7920e-06 |
| Exterior Dirichlet GMRES | 33 | 16 | 15 | 14 |
| Interior Neumann error | 3.0786e-01 | 1.5692e-01 | 2.9862e-02 | 5.7126e-03 |
| Interior Neumann GMRES | 13 | 12 | 11 | 11 |
| Exterior Neumann error | 1.1832e-03 | 1.0794e-03 | 1.0129e-04 | 1.6576e-05 |
| Exterior Neumann GMRES | 11 | 9 | 8 | 8 |

`test_transmission`, `LaplaceTransmission2D::CommonRatio`,
`beta_int=2`, `beta_ext=1`, `lambda^2=1.1`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 8.0911e-04 | - | 7 |
| 64 | 1.2733e-04 | 2.668 | 7 |
| 128 | 3.3576e-05 | 1.923 | 7 |
| 256 | 4.6563e-06 | 2.850 | 7 |
| 512 | 6.1465e-07 | 2.921 | 7 |

`test_transmission`, `LaplaceTransmission2D::DifferentRatios`,
`beta_int=10`, `beta_ext=1`, `kappa_int^2=11`, `kappa_ext^2=0.7`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 1.2074e-02 | - | 14 |
| 64 | 1.9864e-03 | 2.604 | 14 |
| 128 | 3.4135e-04 | 2.541 | 13 |
| 256 | 7.8671e-05 | 2.117 | 13 |
| 512 | 7.7845e-06 | 3.337 | 13 |

`test_transmission_periodic_2d`, cell-centered bi-periodic unit square,
`LaplaceTransmission2D::CommonRatio`, `beta_int=2`, `beta_ext=1`,
`lambda^2=1.3`:

| N | max err | order | GMRES |
|---:|--------:|------:|------:|
| 32 | 8.7480e-03 | - | 7 |
| 64 | 6.1925e-03 | 0.498 | 7 |
| 128 | 8.1011e-04 | 2.934 | 7 |
| 256 | 1.6413e-04 | 2.303 | 7 |
| 512 | 1.2845e-05 | 3.676 | 7 |

`test_transmission_3d`, off-center P2 sphere, target P2
`node_spacing/h ~= 1.5`, nonzero outer Cartesian Dirichlet data,
`LaplaceTransmission3D::CommonRatio`, `beta_int=2`, `beta_ext=1`,
`lambda^2=1.1`:

| N | panels | iface pts | max err | order | GMRES |
|---:|------:|----------:|--------:|------:|------:|
| 8 | 32 | 66 | 1.0010e-03 | - | 8 |
| 16 | 32 | 66 | 7.0723e-04 | 0.501 | 7 |
| 32 | 128 | 258 | 1.1544e-04 | 2.615 | 6 |
| 64 | 512 | 1026 | 1.2939e-05 | 3.157 | 6 |

`test_transmission_3d`, off-center P2 sphere, target P2
`node_spacing/h ~= 1.5`, nonzero outer Cartesian Dirichlet data,
`LaplaceTransmission3D::DifferentRatios`, `beta_int=10`, `beta_ext=1`,
`kappa_int^2=11`, `kappa_ext^2=0.7`:

| N | panels | iface pts | max err | order | GMRES |
|---:|------:|----------:|--------:|------:|------:|
| 8 | 32 | 66 | 1.0234e-01 | - | 21 |
| 16 | 32 | 66 | 5.5583e-02 | 0.881 | 14 |
| 32 | 128 | 258 | 9.3583e-03 | 2.570 | 13 |

`test_transmission_ellipsoid_3d`, off-center P2 triaxial ellipsoid with axes
`(0.61,0.49,0.41)`, target P2 `node_spacing/h ~= 1.5`, nonzero outer
Cartesian Dirichlet data, `LaplaceTransmission3D::CommonRatio`, `beta_int=2`,
`beta_ext=1`, `lambda^2=1.1`:

| N | panels | iface pts | max err | order | GMRES |
|---:|------:|----------:|--------:|------:|------:|
| 8 | 32 | 66 | 1.3215e-02 | - | 8 |
| 16 | 32 | 66 | 1.0124e-03 | 3.706 | 8 |
| 32 | 200 | 402 | 7.1151e-05 | 3.831 | 7 |
| 64 | 648 | 1298 | 1.3557e-05 | 2.392 | 7 |

`test_transmission_torus_3d`, off-center shared P2 torus, target P2
`node_spacing/h ~= 1.5`, nonzero outer Cartesian Dirichlet data,
`LaplaceTransmission3D::CommonRatio`, `beta_int=2`, `beta_ext=1`,
`lambda^2=1.1`:

| N | panels | iface pts | max err | order | GMRES |
|---:|------:|----------:|--------:|------:|------:|
| 8 | 96 | 192 | 5.9263e-03 | - | 9 |
| 16 | 120 | 240 | 4.0886e-04 | 3.857 | 9 |
| 32 | 240 | 480 | 2.0554e-04 | 0.992 | 9 |

The `test_transmission_torus_3d` table above is the default PDE-only run.
Set `KFBIM_HIGH_RES_3D=1` to include `N=64`, `N=128`, and `N=256`.

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
- **Normal n** points outward (from interior Ω⁺ to exterior Ω⁻), matching
  `Interface2D::normals()` and `Interface3D::normals()`.
- **Domain label**: `GridPair2D::domain_label(n)` and
  `GridPair3D::domain_label(n)` return 0 = exterior (Ω⁻), 1 = interior (Ω⁺)
  for single-component tests.
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

## Current Source Layout

The current compiled C++ source tree is:

```
src/
  CMakeLists.txt
  grid/           # CartesianGrid2D/3D, MACGrid2D/3D, DofLayout, grid interfaces
  interface/      # Interface2D, Interface3D and panel node-layout metadata
  geometry/       # Curve2D, CurveResampler2D, GridPair2D/3D
  transfer/       # ISpread/IRestrict plus 2D and 3D P2 Laplace spread/restrict
  local_cauchy/   # Local Cauchy interfaces, jump data, 2D panel and 3D P2 patch solvers, local polynomials
  bulk_solvers/   # BulkSolver API, FFT/zFFT engines, Laplace FFT/zFFT solvers, IIM helpers
  potentials/     # LaplacePotentialEval2D/3D reusable potential pipelines
  operators/      # IKFBIOperator, LaplaceBvp2D/3D, LaplaceTransmission2D/3D, Stokes scaffold
  gmres/          # GMRES and outer-solver interface
```

The old `src/solver/`, `src/operator/`, and `src/problems/` paths have been
replaced by `src/bulk_solvers/`, `src/potentials/`, and `src/operators/`.
New concrete Laplace problem code should live in `src/operators/`; reusable
potential-building code should live in `src/potentials/`.

Other repository paths:

```
include/kfbim/    # public forwarding headers for current library API
python/           # visualization/diagnostic scripts
bindings/         # pybind11 wrappers (future)
tests/            # active PDE convergence programs only
tests/archive/    # archived component/basic/legacy top-level tests
output/           # generated runtime output, gitignored
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
cmake -B build && cmake --build build
```
