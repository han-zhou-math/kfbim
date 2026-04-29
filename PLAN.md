# KFBIM — Development Plan

## Current state

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | `CartesianGrid2D/3D`, `MACGrid2D/3D` | ✓ done |
| 0 | `Interface2D/3D` (panels, points, normals, weights) | ✓ done |
| 0 | `GridPair2D/3D` (domain labeling, closest-pt query) | ✓ done |
| 1.5 | `laplace_panel_solver_2d.hpp` — per-panel 6×6 collocation Cauchy solver | ✓ done |
| 2 | `LaplaceFftBulkSolverZfft2D/3D` — DST Poisson solver | ✓ done |
| 1 | `LaplacePanelSpread2D` — panel Cauchy + IIM defect | ✓ done |
| 1 | `LaplaceQuadraticRestrict2D` — quadratic fit + jump correction | ✓ done |
| 3 | `LaplaceKFBIOperator2D/3D` — linear and affine modes | ✓ done |
| 4 | `GMRES` (restarted, Givens) | ✓ done |

### Verified Formulations (Laplace 2D)

- **Interface Problem**: $-\Delta u = f, [u]=a, [\partial_n u]=b$. $O(h^2)$ convergence. ✓
- **Dirichlet BVP ($1^{st}$-kind BIE)**: Single-layer unknown $[\partial_n u]=\sigma$. (Slow convergence). ✓
- **Dirichlet BVP ($2^{nd}$-kind BIE)**: Double-layer unknown $[u]=\phi$. $O(h^2)$ convergence, 15-20 iters for star domain. ✓

---

## Task 1 — Interface problem with discontinuous coefficients

PDE: $-\nabla\cdot(\beta\nabla u) = f$,  where $\beta = \beta^+$ in $\Omega^+$ and $\beta = \beta^-$ in $\Omega^-$.

Jump conditions:
    $[u] = a,    [\beta \partial_n u] = b$

### Changes needed

**IIM correction:** defect at cross-interface neighbor `nb` becomes:
    (label_nb − label_n) × β_side(n) × C(x_nb) / h²
where `β_side(n)` is the coefficient on the side of node `n`.

**Panel Cauchy solver:** Neumann rows enforce $[\beta \partial_n u] = b$, so the
normal-derivative rows are weighted by $\beta$. Specifically, row $l$ ($l=0,2$) becomes:
    $\beta^+ \cdot \nabla\phi_k\cdot n$ (outside)  $-$  $\beta^- \cdot \nabla\phi_k\cdot n$ (inside)  $=  b[l]$
No change to Dirichlet or PDE rows.

**Bulk solver:** For piecewise constant $\beta$, the IIM correction absorbs the jump. Solve $\Delta u = \tilde{f}$ with mean coefficient or treat as a source term for constant-coefficient FFT solver.

### Test

Manufactured solution with $\beta^+ \neq \beta^-$, star interface; verify $O(h^2)$ convergence.

---

## Task 2 — Stokes BVP in BIE form

Extend the BIE machinery to the Stokes equations:
    $-\Delta \mathbf{u} + \nabla p = \mathbf{f}, \quad \nabla \cdot \mathbf{u} = 0$

### Components needed

- `StokesKFBIOperator2D/3D` (implementation of `apply()` and `apply_full()`).
- Stokes-specific Spread and Restrict operators (using MAC grid).
- Verification of $2^{nd}$-kind BIE for Dirichlet velocity BCs.

---

## Task 3 — 3D Laplace BVP Verification

Verify the 3D implementation of `LaplaceKFBIOperator3D` on a smooth surface (e.g., sphere).

---

## Recommended order

1. **Task 1** (discontinuous $\beta$) — localized changes to Spread and Cauchy.
2. **Task 3** (3D Laplace) — verify existing 3D code paths.
3. **Task 2** (Stokes) — significant new implementation.

## File targets

| Task | Files |
|------|-----------|
| 1 | extend `laplace_panel_solver_2d.hpp`, `laplace_spread_2d.cpp`, new test |
| 2 | `core/operator/stokes_kfbi_operator.cpp`, `tests/test_kfbi_stokes_2d.cpp` |
| 3 | `tests/test_kfbi_laplace_3d.cpp` |
