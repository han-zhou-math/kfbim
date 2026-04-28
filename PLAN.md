# KFBIM — Development Plan

## Current state (all 101 tests passing)

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | `CartesianGrid2D/3D`, `MACGrid2D/3D` | ✓ done |
| 0 | `Interface2D/3D` (panels, points, normals, weights) | ✓ done |
| 0 | `GridPair2D/3D` (domain labeling, closest-pt query) | ✓ done |
| 1.5 | `laplace_panel_solver_2d.hpp` — per-panel 6×6 collocation Cauchy solver | ✓ done |
| 2 | `LaplaceFftBulkSolverZfft2D/3D` — DST Poisson solver | ✓ done |
| 1 | `LaplacePanelSpread2D` — panel Cauchy + IIM defect | ✓ done |
| 1 | `LaplaceQuadraticRestrict2D` — quadratic fit + jump correction | ✓ done |
| 3 | `LaplaceKFBIOperator2D/3D` | header only — **not implemented** |
| 4 | `GMRES` (restarted, Givens) | header only — **not implemented** |

End-to-end test: interface problem −Δu = f, [u]=a, [∂ₙu]=b, u=0 on ∂Ω.
~O(h²) convergence over N = {32, 64, 128, 256}. ✓

---

## Task 1 — Implement `LaplaceKFBIOperator2D` + `GMRES` (Layers 3–4)

This is the immediate gap. Headers in `core/operator/` and `core/gmres/` already
define the interface. No design decisions remain open.

### `LaplaceKFBIOperator2D::apply(x, y)`

```
1. Unpack x → LaplaceJumpData at each interface point
       Dirichlet mode: un_jump = x[i],  u_jump = 0  (Dirichlet BC known)
       Neumann   mode: u_jump  = x[i], un_jump = 0  (Neumann  BC known)
2. rhs ← base_rhs_  (bulk forcing, fixed)
   Spread.apply(jumps, rhs)      → correction polys, rhs modified in-place
3. BulkSolver.solve(-rhs, u_bulk)
4. polys ← Restrict.apply(u_bulk, correction_polys)
5. Pack y:
       Dirichlet mode: y[i] = polys[i].coeffs[0]              (trace u|_Γ)
       Neumann   mode: y[i] = polys[i].coeffs[1]*n_x
                             + polys[i].coeffs[2]*n_y          (flux ∂u/∂n|_Γ)
```

### GMRES convergence criterion

`‖r_k‖ / ‖r₀‖ < tol`  (already in header).
Use restart = 50 by default; expose as constructor argument.

### Test

New `tests/test_kfbi_laplace_2d.cpp`:
- Dirichlet BVP on star domain: u = sin(πx)sin(πy), g = u|_Γ known
- GMRES converges in < 30 iterations, solution error < 1e-3 at N=64

---

## Task 2 — Interface problem with discontinuous coefficients

PDE: −∇·(β∇u) = f,  where β = β⁺ in Ω⁺ and β = β⁻ in Ω⁻.

Jump conditions:
    [u] = a,    [β ∂_n u] = b

### Changes needed

**IIM correction:** defect at cross-interface neighbor `nb` becomes
    (label_nb − label_n) × β_side(n) × C(x_nb) / h²
where `β_side(n)` is the coefficient on the side of node `n`.

**Panel Cauchy solver:** Neumann rows enforce `[β ∂_n u] = b`, so the
normal-derivative rows are weighted by β. Specifically, row l (l=0,2) becomes:
    β⁺ · ∇φₖ·n  (outside)  −  β⁻ · ∇φₖ·n  (inside)  =  b[l]
No change to Dirichlet or PDE rows.

**Bulk solver:** for variable β, the constant-coefficient FFT solver is no
longer applicable. Use the KFBIM trick: decompose β = β_mean + δβ and
solve with the mean-coefficient FFT solver, treating δβ terms as an extra
source. Or use a multigrid bulk solver. For constant-per-side β this
simplification suffices: IIM correction absorbs the jump.

### Test

Manufactured solution with β⁺ ≠ β⁻, star interface; verify O(h²) convergence.

---

## Task 3 — Boundary value problem in BIE form with GMRES

The full KFBIM loop for a Dirichlet BVP −Δu = f, u = g on ∂Ω:

### Formulation

Represent the unknown as a single-layer / double-layer density σ on ∂Ω.
KFBIM avoids explicit kernels: instead, treat ∂Ω as an "interface" with

    [u]      = g   (Dirichlet data, known)
    [∂_n u]  = σ   (surface flux, unknown — the GMRES iterate)

The operator maps σ → trace of the computed solution u|_{∂Ω}.
GMRES drives this trace to g.

### Components needed

- `LaplaceKFBIOperator2D` (Task 1) — already handles Dirichlet mode
- Interface representation of ∂Ω as a panel curve (the box boundary, or
  a smooth domain boundary Γ ⊂ (0,1)²)
- Post-processing: once σ is found, one bulk solve gives u everywhere

### Distinction from Task 1

Task 1 tests the operator on a known interface problem (prescribed [u] and
[∂ₙu]). Task 3 uses GMRES to *find* the unknown flux σ that enforces the
boundary condition — the true BIE solve.

### Test

Circle domain, Dirichlet BC u = sin(πx)cos(πy):
- GMRES converges in O(1) iterations (well-conditioned for smooth domains)
- Solution error O(h²) vs exact

---

## Recommended order

1. **Task 1** (operator + GMRES) — closes the loop on the existing layers
2. **Task 2** (discontinuous β) — moderate change, localised to Spread and Cauchy
3. **Task 3** (BIE / GMRES BVP) — depends on Task 1 being reliable

## File targets

| Task | New files |
|------|-----------|
| 1 | `core/operator/laplace_kfbi_operator.cpp`, `core/gmres/gmres.cpp`, `tests/test_kfbi_laplace_2d.cpp` |
| 2 | extend `laplace_panel_solver_2d.hpp`, `laplace_spread_2d`, new test |
| 3 | `problems/laplace_dirichlet_2d.hpp/.cpp`, `tests/test_laplace_bvp_2d.cpp` |
