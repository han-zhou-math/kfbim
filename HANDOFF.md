# KFBIM Project Handoff

## What this project is

A C++ library for the **Kernel-Free Boundary Integral Method (KFBIM)** — an
elliptic PDE solver on complex interfaces that replaces Green's function kernels
with local PDE solves.  See `CLAUDE.md` for the full architecture.

One GMRES iteration = Spread → BulkSolve → Restrict on an interface-to-interface
operator.

---

## Current state (as of last session)

**Layer 1.5 (LocalCauchySolver) is complete and tested.**

### What was built

| File | Status | Description |
|------|--------|-------------|
| `core/local_cauchy/laplace_panel_solver_2d.hpp` | Done | Panel-based collocation Cauchy solver |
| `tests/test_local_cauchy.cpp` | Done | 5 tests, all passing |
| `scripts/visualize_local_cauchy.py` | Done | Visualization of errors and convergence |

### The solver: `laplace_panel_cauchy_2d`

Solves the local Cauchy problem at every interface Gauss point:

    −ΔC = f⁺ − f⁻   in a neighborhood of the interface
    C    = a          (Dirichlet jump [u])
    ∂_n C = b         (Neumann jump [∂_n u])

**Interface representation:** panels of exactly 3 Gauss-Legendre points each
(`iface.points_per_panel() == 3`).

**Per-panel 6×6 collocation system** (one solve per Gauss point as center):

| Rows | Count | Points | Data |
|------|-------|--------|------|
| Dirichlet | 3 | all 3 panel GL pts | `a[g0], a[g1], a[g2]` |
| Neumann | 2 | panel pts 0 and 2 | `b[g0], b[g2]` |
| PDE | 1 | panel midpoint (GL node 1) | `Lu[g1] = f⁺−f⁻` at midpoint |

**Monomial basis** (Taylor-scaled, `dx=(x−cx)/h`, `h`=panel arc-length):

    φ₀=1,  φ₁=dx,  φ₂=dy,  φ₃=½dx²,  φ₄=½dy²,  φ₅=dx·dy

**Output** (after rescaling `c[1:2]/=h`, `c[3:5]/=h²`):

    c[0]=C,  c[1]=Cx,  c[2]=Cy,  c[3]=Cxx,  c[4]=Cyy,  c[5]=Cxy

**Convergence:** Cx/Cy → O(h²),  Cxx/Cyy/Cxy → O(h).

### Public API

```cpp
struct PanelCauchyResult2D {
    Eigen::VectorXd C, Cx, Cy, Cxx, Cyy, Cxy;  // length = iface.num_points()
};

PanelCauchyResult2D laplace_panel_cauchy_2d(
    const Interface2D&     iface,      // must have points_per_panel() == 3
    const Eigen::VectorXd& a,          // [u]     at each Gauss pt
    const Eigen::VectorXd& b,          // [∂_n u] at each Gauss pt
    const Eigen::VectorXd& Lu_iface,   // f⁺−f⁻  at each Gauss pt
    double                 kappa = 0.0
);
```

### Manufactured solution used in tests

```
u⁺ = sin(πx)sin(πy),    f⁺ = 2π²sin(πx)sin(πy)
u⁻ = sin(2πx)sin(2πy),  f⁻ = 8π²sin(2πx)sin(2πy)
C  = u⁺ − u⁻
```

Star interface: center (0.5, 0.5), R=0.28, amplitude A=0.40, K=5 tips.

### IIM plug-in result

Using `laplace_panel_cauchy_2d` output fed into `iim_correct_rhs_taylor` at N=64:
- Exact-Taylor baseline error: 7.55e-04
- Panel-Cauchy error: 9.75e-04  (1.29× ratio, well within 2× bound)

---

## What comes next

**Layer 1 — Transfer Operators (Spread and Restrict)**

These are the next layer up from LocalCauchySolver.

### Spread (interface → bulk correction)

For each irregular bulk node `x` near the interface:
1. Find the nearest interface quadrature point `q`
2. Call `laplace_panel_cauchy_2d` to get `(C_q, Cx_q, Cy_q, Cxx_q, ...)`
3. Taylor-expand to evaluate `C(x)`:
   `C(x) ≈ C_q + Cx_q·dx + Cy_q·dy + ½Cxx_q·dx² + Cxy_q·dx·dy + ½Cyy_q·dy²`
4. Inject into RHS: for each cross-interface neighbor pair `(n, nb)`:
   `F[n] += (lnb − ln) × C(xnb) / h²`

`iim_correct_rhs_taylor` in `core/solver/iim_laplace_2d.hpp` already implements
steps 3–4.  Layer 1 wraps this with the LocalCauchySolver call (step 2).

### Restrict (bulk → interface)

Interpolates the bulk solution to interface quadrature points and applies jump
correction.  Not yet implemented.

### Key existing infrastructure to build on

| File | What it provides |
|------|-----------------|
| `core/solver/iim_laplace_2d.hpp` | `iim_correct_rhs_taylor()`, `iim_irregular_nodes()` |
| `core/geometry/grid_pair_2d.hpp` | `GridPair2D`: nearest-point queries, domain labels |
| `core/local_cauchy/laplace_panel_solver_2d.hpp` | Panel Cauchy solver (Layer 1.5) |
| `core/interface/interface_2d.hpp` | `Interface2D`: panels, points, normals, weights |
| `core/grid/cartesian_grid_2d.hpp` | `CartesianGrid2D` |

---

## Build & test

```bash
cmake -B build && cmake --build build
./build/tests/test_local_cauchy -s   # Layer 1.5 tests
ctest --test-dir build               # all tests
```

All tests currently pass.
