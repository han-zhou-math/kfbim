# Parallel KFBI Development Plan

This note sketches a future larger project for kernel-free boundary integral
methods on adaptive Cartesian bulk grids, curved surface meshes, surface PDEs,
and parallel computers. The current repository should remain the serial and
small-scale numerical prototype while the new project is built around parallel
mesh, solver, and search libraries.

## 1. Target

Build a parallel C++ framework that solves coupled bulk/interface/surface
problems of the form:

- bulk elliptic PDEs on adaptive Cartesian grids,
- interface jump and transmission conditions on curved surfaces,
- optional surface PDEs on the same surface mesh,
- time-dependent or nonlinear coupling through outer Krylov/Newton/time
  integrators.

The initial production target should remain scalar elliptic problems:

- Poisson and screened Poisson,
- discontinuous-coefficient transmission,
- surface diffusion or reaction-diffusion on Gamma,
- bulk-surface coupled scalar models.

Stokes, elasticity, and variable-coefficient multiphysics should come after the
scalar bulk-surface coupling is stable.

## 2. Recommended Package Stack

### Surface PDE and Surface Mesh

Use MFEM as the primary surface finite-element layer.

MFEM should own:

- parallel surface mesh representation when feasible,
- high-order surface geometry and surface FE spaces,
- surface mass/stiffness assembly,
- surface PDE solvers,
- surface fields used by KFBI jumps and coefficients.

Relevant MFEM features:

- `ParMesh`,
- `ParFiniteElementSpace`,
- high-order meshes and elements,
- surface meshes embedded in 3D,
- `HypreParMatrix`,
- HYPRE-backed parallel solvers,
- optional PETSc/SUNDIALS integration if needed later.

If surface remeshing, CAD association, or advanced curved parallel mesh
adaptation becomes central, evaluate PUMI/SCOREC as a replacement or companion
for the surface mesh manager. For the first parallel prototype, MFEM is the
better starting point because it also provides the surface PDE discretization.

### Bulk Adaptive Cartesian Grid

Keep two possible backend paths.

Path A, MFEM-first:

- use MFEM quad/hex meshes with nonconforming refinement,
- solve bulk elliptic PDEs with MFEM/HYPRE,
- easiest data-model match with MFEM surface PDEs,
- useful for the first coupled prototype.

Path B, block-structured AMR:

- use AMReX for adaptive Cartesian bulk grids and geometric multigrid,
- better match for finite-difference/finite-volume adaptive Cartesian KFBI,
- stronger long-term option for large-scale Cartesian AMR.

The recommended sequence is:

1. Start with MFEM-first bulk for a controlled parallel prototype.
2. Keep the bulk layer abstract enough to add AMReX later.
3. Add AMReX only after the KFBI coupling contracts are stable.

### Bulk-Surface Interaction Search

Use ArborX for distributed spatial search.

ArborX should build and query:

- surface element bounding boxes,
- bulk cell or patch bounding boxes,
- near-interface bulk-cell lists,
- candidate surface elements for each bulk cell,
- candidate bulk cells for each surface quadrature point.

This avoids all ranks owning the full surface mesh or full bulk grid.

### Local Geometry

Use a custom curved-surface geometry layer, optionally supported by CGAL.

The custom layer should handle the exact geometry needed by KFBI:

- P2 or higher-order curved triangle evaluation,
- normals and tangents,
- closest-point projection by Newton iteration,
- signed distance near the interface,
- surface quadrature and interpolation,
- panel-local coordinates.

CGAL can be used for robust fallback or broad-phase geometric predicates on a
linearized proxy mesh:

- closed-surface inside/outside checks,
- AABB tree debugging,
- robust triangle intersection tests.

Do not rely on CGAL alone for exact P2 curved-patch projection.

### Linear and Nonlinear Solvers

Use the native solvers of each backend first:

- MFEM with HYPRE/BoomerAMG for FE elliptic systems,
- AMReX MLMG for block-structured AMR bulk solves,
- PETSc KSP/SNES/TS later for fully coupled nonlinear systems or monolithic
  saddle-point systems.

The KFBI outer interface solver can initially remain matrix-free GMRES, but the
parallel version should eventually use a solver abstraction that can call:

- MFEM iterative solvers,
- PETSc KSP,
- custom matrix-free GMRES where needed.

## 3. High-Level Architecture

The project should be split into independent modules with narrow interfaces.

### `core`

PDE-independent KFBI contracts:

- jump data types,
- interface averages and side reconstruction,
- matrix-free operator interface,
- convergence and residual reporting,
- common error and diagnostics utilities.

### `surface`

MFEM-backed surface representation:

- `SurfaceMesh`,
- `SurfaceField`,
- `SurfaceQuadrature`,
- `SurfacePdeOperator`,
- `SurfaceGeometryEvaluator`.

This layer exposes geometric and FE data without exposing all MFEM internals to
KFBI operators.

### `bulk`

Backend-independent bulk grid and solver interface:

- `BulkGrid`,
- `BulkField`,
- `BulkRhs`,
- `BulkSolver`,
- `BulkBoundaryCondition`,
- `BulkHierarchy`.

Implementations:

- `MfemBulkSolver`,
- later `AmrexBulkSolver`.

### `search`

Distributed bulk-surface relation builder:

- owns ArborX search structures,
- maps AMR cells or bulk nodes to surface element candidates,
- maps surface quadrature points to nearby bulk cells,
- builds rank-to-rank exchange plans.

### `geometry`

Local curved-patch geometry:

- curved triangle evaluation,
- closest-point projection,
- signed distance,
- normals,
- local coordinates,
- panel-level bounding boxes.

### `coupling`

Cached data needed by spread and restrict:

- near-interface bulk DOFs,
- closest surface elements,
- projection coordinates,
- side labels,
- interpolation/reconstruction stencils,
- local Cauchy stencil metadata,
- cache versioning and invalidation after regrid or surface motion.

### `operators`

KFBI PDE operators:

- Laplace potential evaluation,
- transmission operators,
- BVP wrappers,
- later Stokes and elasticity operators.

### `apps`

Small executable applications:

- manufactured convergence tests,
- bulk-surface reaction-diffusion example,
- moving interface example,
- scaling benchmarks.

## 4. Parallel Data Ownership

Use separate ownership for bulk and surface data.

Bulk ownership:

- owned by MFEM or AMReX bulk backend,
- partitioned by bulk elements, cells, or AMR boxes,
- ghost data managed by the bulk backend.

Surface ownership:

- owned by MFEM `ParMesh`,
- partitioned by surface elements,
- ghost/shared surface DOFs managed by MFEM.

Coupling ownership:

- built from distributed ArborX queries,
- each rank stores only local bulk-surface interaction records,
- cross-rank records are exchanged using explicit communication plans.

Avoid any design that requires every rank to store all surface panels.

## 5. Core Coupling Algorithm

For a fixed bulk grid and fixed surface mesh:

1. Build or update surface geometry data.
2. Build distributed search structures with ArborX.
3. Identify near-interface bulk DOFs.
4. For each near-interface bulk DOF, find candidate surface patches.
5. Run local curved projection and signed-distance classification.
6. Cache side labels, closest points, normals, and interpolation stencils.
7. Given jump data, spread corrections to the bulk RHS.
8. Solve the bulk PDE with the selected bulk solver.
9. Restrict the bulk solution to surface quadrature/DOFs.
10. Apply the matrix-free KFBI interface operator.

For moving surfaces or adaptive regridding, repeat steps 1 through 6 whenever
the mesh hierarchy changes.

## 6. Surface PDE Coupling Pattern

Surface PDEs should be treated as first-class operators on the MFEM surface
mesh.

Example coupled model:

```text
bulk:    -div(beta grad u) + kappa^2 u = f
surface: dc/dt - D_Gamma Delta_Gamma c = R(c, trace(u))
interface jumps: [u] = g(c), [beta du/dn] = h(c, trace(u))
```

At each time step or nonlinear iteration:

1. Solve or update the surface PDE state `c`.
2. Evaluate jump data from `c` and current bulk/interface traces.
3. Apply KFBI bulk/interface solve.
4. Restrict updated bulk traces and fluxes to the surface.
5. Feed the traces/fluxes back into the surface PDE residual.

Start with partitioned fixed-point or block Gauss-Seidel coupling. Move to
Newton/Krylov coupling only after the partitioned method is validated.

## 7. Development Phases

### Phase 0: Freeze Current Prototype

Deliverables:

- current serial 2D/3D Laplace tests passing,
- convergence tables recorded,
- combined D/S potential optimization committed,
- torus and periodic transmission tests committed,
- generated output excluded from commits.

### Phase 1: New Parallel Skeleton

Deliverables:

- standalone CMake project or separate branch,
- MPI-enabled build,
- dependency discovery for MFEM, HYPRE, ArborX, and optional CGAL,
- minimal `core`, `surface`, `bulk`, `search`, `geometry`, `coupling` modules,
- CI or local build scripts for serial and MPI runs.

Do not port all existing KFBI code yet. First prove the package stack builds and
can exchange data.

### Phase 2: MFEM Surface PDE Prototype

Deliverables:

- parallel closed surface mesh,
- surface scalar field,
- Laplace-Beltrami or screened surface PDE solve,
- convergence test on sphere or torus,
- surface field output for visualization,
- MPI test with at least 2 and 4 ranks.

### Phase 3: Bulk Elliptic Prototype

Deliverables:

- adaptive Cartesian-like bulk mesh in MFEM,
- scalar Poisson/screened-Poisson solve,
- Dirichlet, Neumann, and periodic cases as needed,
- convergence tests on smooth manufactured solutions,
- HYPRE preconditioner configuration documented.

Decision point:

- If MFEM bulk works well enough, continue with it.
- If finite-difference/finite-volume AMR and geometric multigrid are required,
  add AMReX as a second bulk backend.

### Phase 4: Distributed Bulk-Surface Search

Deliverables:

- ArborX search over surface element bounding boxes,
- near-interface bulk cell discovery,
- candidate panel lists,
- local P2 projection test,
- signed-distance/domain-label test,
- parallel consistency checks across rank boundaries.

Important tests:

- every near-interface bulk DOF has a valid closest panel,
- signed distance is consistent under MPI repartitioning,
- ghost surface elements are sufficient for all local bulk interactions.

### Phase 5: Parallel KFBI for Scalar Laplace

Deliverables:

- parallel spread on near-interface bulk DOFs,
- parallel bulk solve,
- parallel restrict to surface quadrature/DOFs,
- matrix-free GMRES interface solve,
- manufactured interface problem on sphere,
- manufactured transmission problem on sphere and torus,
- comparison against current uniform-grid prototype.

Use fixed meshes first. Add adaptive regridding after the fixed-grid coupling is
correct.

### Phase 6: Adaptive Regridding

Deliverables:

- error indicators near the interface,
- bulk regrid and solution transfer,
- surface-to-bulk interaction cache invalidation,
- rebuilt ArborX search after regrid,
- convergence test with AMR refinement levels,
- scaling test with moving or static complex geometry.

### Phase 7: Surface PDE Coupling

Deliverables:

- coupled bulk-surface scalar model,
- partitioned time stepper,
- convergence against manufactured solution,
- conservation or balance diagnostics where applicable,
- nonlinear residual monitoring.

### Phase 8: Stokes and More Complex PDEs

Deliverables:

- staggered or mixed bulk Stokes backend decision,
- local Cauchy solver for Stokes,
- spread/restrict for vector fields,
- pressure/velocity trace conventions,
- manufactured Stokes interface tests,
- later elasticity and variable-coefficient elliptic systems.

## 8. Verification Plan

Every major capability needs a serial test and an MPI test.

Geometry tests:

- curved patch evaluation,
- normal orientation,
- closest-point projection,
- signed distance,
- inside/outside labels,
- rank-independent nearest-panel selection.

Surface PDE tests:

- Laplace-Beltrami convergence,
- mass conservation for surface diffusion,
- reaction-diffusion manufactured solution.

Bulk solver tests:

- Poisson and screened Poisson convergence,
- adaptive refinement convergence,
- boundary condition tests.

KFBI tests:

- direct prescribed-jump interface problem,
- BVP problem,
- common-ratio transmission,
- different-ratio transmission,
- torus or multi-component geometry.

Parallel tests:

- result invariance under different MPI rank counts,
- strong scaling,
- weak scaling,
- communication-volume diagnostics,
- load balance near complex surfaces.

## 9. Performance Plan

The main performance risks are bulk-surface search, local geometry projection,
spread/restrict, and repeated local Cauchy setup.

Required optimizations:

- cache local Cauchy factorizations or linear maps,
- cache interaction stencils until mesh or surface changes,
- batch projection queries by surface element,
- use ArborX to avoid global all-to-all search,
- avoid rebuilding search structures inside GMRES iterations,
- reuse bulk solver hierarchy when coefficients and grids are unchanged,
- combine layer-potential evaluations when the same bulk operator is used.

Performance counters to record:

- number of near-interface bulk DOFs,
- number of candidate surface elements per bulk DOF,
- projection failure count,
- spread time,
- bulk solve time,
- restrict time,
- GMRES iterations,
- communication time,
- memory per rank.

## 10. Design Rules

- Keep the current uniform-grid code as the numerical reference.
- Keep package-specific code behind small backend interfaces.
- Do not let MFEM, AMReX, or ArborX types leak through all KFBI layers.
- Treat geometry caches as versioned data tied to a bulk mesh version and a
  surface mesh version.
- Start with fixed interfaces before moving interfaces.
- Start with scalar elliptic PDEs before Stokes.
- Prefer partitioned coupling before monolithic nonlinear coupling.
- Add monolithic PETSc/MFEM block solvers only when a real model requires them.

## 11. Open Decisions

1. Whether the first adaptive bulk backend should be MFEM-only or AMReX.
2. Whether the surface mesh will remain MFEM-owned or eventually move to PUMI.
3. Whether curved surface geometry should be P2-only or generic high-order.
4. Whether the KFBI interface unknowns live at MFEM surface DOFs, quadrature
   points, or a separate KFBI collocation set.
5. Whether outer Krylov solvers should be MFEM-native, PETSc-native, or custom.
6. How to represent coarse-fine adaptive stencils for KFBI correction near AMR
   transitions.

## 12. First Concrete Milestone

The first milestone should be deliberately small:

```text
Parallel scalar bulk-surface prototype:

- MFEM parallel surface mesh: sphere
- MFEM surface PDE: screened Laplace-Beltrami
- MFEM bulk mesh: box with local refinement near sphere
- ArborX search: bulk cells near surface
- custom geometry: closest point to curved P2 surface
- scalar screened Poisson bulk solve
- manufactured jump interface problem
- run with 1, 2, and 4 MPI ranks
```

Success criteria:

- second-order or better convergence in the scalar manufactured test,
- identical error table up to roundoff-level differences across MPI rank counts,
- no global surface replication,
- clean timing split for search, spread, solve, and restrict.
