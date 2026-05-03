"""
Visualize the Layer 1 Laplace transfer operators in 2D.

The figure mirrors tests/test_transfer_2d.cpp:
  - Spread: panel Cauchy coefficients are evaluated at cross-interface
    neighbors and accumulated as a 5-point IIM RHS correction.
  - Restrict: a local quadratic least-squares fit is corrected by subtracting
    the spread polynomial, recovering the physical interface expansion.

Output:
  python/transfer_2d_viz.png
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kfbim-matplotlib")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


kPi = np.pi


# ---------------------------------------------------------------------------
# Manufactured solution used by local Cauchy and spread.
# ---------------------------------------------------------------------------


def u_plus(x, y):
    return np.sin(kPi * x) * np.sin(kPi * y)


def u_minus(x, y):
    return np.sin(2.0 * kPi * x) * np.sin(2.0 * kPi * y)


def f_plus(x, y):
    return 2.0 * kPi**2 * np.sin(kPi * x) * np.sin(kPi * y)


def f_minus(x, y):
    return 8.0 * kPi**2 * np.sin(2.0 * kPi * x) * np.sin(2.0 * kPi * y)


def C_fn(x, y):
    return u_plus(x, y) - u_minus(x, y)


def Cx_fn(x, y):
    return (
        kPi * np.cos(kPi * x) * np.sin(kPi * y)
        - 2.0 * kPi * np.cos(2.0 * kPi * x) * np.sin(2.0 * kPi * y)
    )


def Cy_fn(x, y):
    return (
        kPi * np.sin(kPi * x) * np.cos(kPi * y)
        - 2.0 * kPi * np.sin(2.0 * kPi * x) * np.cos(2.0 * kPi * y)
    )


def Cxx_fn(x, y):
    return (
        -kPi**2 * np.sin(kPi * x) * np.sin(kPi * y)
        + 4.0 * kPi**2 * np.sin(2.0 * kPi * x) * np.sin(2.0 * kPi * y)
    )


def Cxy_fn(x, y):
    return (
        kPi**2 * np.cos(kPi * x) * np.cos(kPi * y)
        - 4.0 * kPi**2 * np.cos(2.0 * kPi * x) * np.cos(2.0 * kPi * y)
    )


def Cyy_fn(x, y):
    return (
        -kPi**2 * np.sin(kPi * x) * np.sin(kPi * y)
        + 4.0 * kPi**2 * np.sin(2.0 * kPi * x) * np.sin(2.0 * kPi * y)
    )


# ---------------------------------------------------------------------------
# Star geometry with 3 Gauss-Legendre nodes per panel.
# ---------------------------------------------------------------------------


CX, CY, R_STAR, A_STAR, K_STAR = 0.5, 0.5, 0.28, 0.40, 5
GL_S = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
GL_W = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


def star_r(th):
    return R_STAR * (1.0 + A_STAR * np.cos(K_STAR * th))


def star_contains(x, y):
    rho = np.hypot(x - CX, y - CY)
    th = np.arctan2(y - CY, x - CX)
    return rho < star_r(th)


def star_curve(n=1000):
    th = np.linspace(0.0, 2.0 * kPi, n)
    r = star_r(th)
    return CX + r * np.cos(th), CY + r * np.sin(th)


def make_star_panels(n_panels):
    n_q = 3 * n_panels
    pts = np.zeros((n_q, 2))
    normals = np.zeros((n_q, 2))
    weights = np.zeros(n_q)
    dth = 2.0 * kPi / n_panels

    q = 0
    for p in range(n_panels):
        th_mid = (p + 0.5) * dth
        half_dth = 0.5 * dth
        for i in range(3):
            th = th_mid + half_dth * GL_S[i]
            r = star_r(th)
            drdt = -R_STAR * A_STAR * K_STAR * np.sin(K_STAR * th)
            pts[q, 0] = CX + r * np.cos(th)
            pts[q, 1] = CY + r * np.sin(th)

            tx = drdt * np.cos(th) - r * np.sin(th)
            ty = drdt * np.sin(th) + r * np.cos(th)
            tlen = np.hypot(tx, ty)
            normals[q, 0] = ty / tlen
            normals[q, 1] = -tx / tlen
            weights[q] = GL_W[i] * half_dth * tlen
            q += 1
    return pts, normals, weights


# ---------------------------------------------------------------------------
# Panel Cauchy solver, matching src/local_cauchy/laplace_panel_solver_2d.hpp.
# ---------------------------------------------------------------------------


def solve_local_6x6(bdry1, a, bdry2, bdry2_normals, b, bulk, Lu, center, h):
    h2 = h * h
    A = np.zeros((6, 6))
    rhs = np.zeros(6)

    for row in range(3):
        dx = (bdry1[row, 0] - center[0]) / h
        dy = (bdry1[row, 1] - center[1]) / h
        A[row] = [1.0, dx, dy, 0.5 * dx * dx, 0.5 * dy * dy, dx * dy]
        rhs[row] = a[row]

    for local in range(2):
        row = 3 + local
        dx = (bdry2[local, 0] - center[0]) / h
        dy = (bdry2[local, 1] - center[1]) / h
        nx = bdry2_normals[local, 0]
        ny = bdry2_normals[local, 1]
        A[row] = [0.0, nx, ny, dx * nx, dy * ny, nx * dy + dx * ny]
        rhs[row] = b[local] * h

    A[5] = [0.0, 0.0, 0.0, -1.0, -1.0, 0.0]
    rhs[5] = Lu * h2

    col_scale = np.linalg.norm(A, axis=0)
    col_scale[col_scale < 1e-14] = 1.0
    c_raw = np.linalg.lstsq(A / col_scale, rhs, rcond=None)[0] / col_scale
    return np.array(
        [
            c_raw[0],
            c_raw[1] / h,
            c_raw[2] / h,
            c_raw[3] / h2,
            c_raw[5] / h2,
            c_raw[4] / h2,
        ]
    )


def laplace_panel_cauchy(pts, normals, weights, n_panels):
    n_q = 3 * n_panels
    x = pts[:, 0]
    y = pts[:, 1]
    nx = normals[:, 0]
    ny = normals[:, 1]
    a = C_fn(x, y)
    b = Cx_fn(x, y) * nx + Cy_fn(x, y) * ny
    Lu = f_plus(x, y) - f_minus(x, y)

    coeffs = np.zeros((n_q, 6))
    for p in range(n_panels):
        g = np.array([3 * p, 3 * p + 1, 3 * p + 2])
        h_panel = weights[g].sum()
        bdry1 = pts[g]
        bdry2 = pts[[g[0], g[2]]]
        bdry2_normals = normals[[g[0], g[2]]]
        bulk = pts[g[1]]

        for gi in g:
            coeffs[gi] = solve_local_6x6(
                bdry1,
                a[g],
                bdry2,
                bdry2_normals,
                b[[g[0], g[2]]],
                bulk,
                Lu[g[1]],
                pts[gi],
                h_panel,
            )
    return coeffs


def eval_local_poly(coeffs, centers, q_idx, points):
    q_idx = np.asarray(q_idx)
    dx = points[:, 0] - centers[q_idx, 0]
    dy = points[:, 1] - centers[q_idx, 1]
    c = coeffs[q_idx]
    return (
        c[..., 0]
        + c[..., 1] * dx
        + c[..., 2] * dy
        + 0.5 * c[..., 3] * dx * dx
        + c[..., 4] * dx * dy
        + 0.5 * c[..., 5] * dy * dy
    )


# ---------------------------------------------------------------------------
# GridPair-like helpers used by spread/restrict visualization.
# ---------------------------------------------------------------------------


def make_node_grid(N):
    xs = np.linspace(0.0, 1.0, N + 1)
    ys = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, np.column_stack([X.ravel(), Y.ravel()])


def nearest_interface_indices(grid_pts, iface_pts):
    diff = grid_pts[:, None, :] - iface_pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.argmin(dist2, axis=1), np.sqrt(np.min(dist2, axis=1))


def label_grid(grid_pts, iface_pts, h):
    from collections import deque

    _, dist = nearest_interface_indices(grid_pts, iface_pts)
    labels = np.full(len(grid_pts), -1, dtype=int)
    band = np.where(dist < 4.0 * h)[0]
    for n in band:
        labels[n] = 1 if star_contains(grid_pts[n, 0], grid_pts[n, 1]) else 0

    nx = int(round(1.0 / h)) + 1
    queue = deque(band)
    while queue:
        n = queue.popleft()
        label = labels[n]
        i = n % nx
        j = n // nx
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ii = i + di
            jj = j + dj
            if 0 <= ii < nx and 0 <= jj < nx:
                nb = jj * nx + ii
                if labels[nb] == -1:
                    labels[nb] = label
                    queue.append(nb)
    labels[labels == -1] = 0
    return labels


def is_boundary(idx, nx):
    i = idx % nx
    j = idx // nx
    return i == 0 or i == nx - 1 or j == 0 or j == nx - 1


def spread_rhs_correction(grid_pts, labels, nearest_q, coeffs, iface_pts, N):
    nx = N + 1
    h = 1.0 / N
    rhs = np.zeros(nx * nx)
    active = np.zeros(nx * nx, dtype=bool)
    eval_nodes = np.zeros(nx * nx, dtype=bool)
    side = np.where(labels == 0, 0, 1)

    for n in range(nx * nx):
        if is_boundary(n, nx):
            continue
        i = n % nx
        j = n // nx
        for nb in [j * nx + i - 1, j * nx + i + 1, (j - 1) * nx + i, (j + 1) * nx + i]:
            if is_boundary(nb, nx) or side[nb] == side[n]:
                continue
            q = nearest_q[nb]
            correction = eval_local_poly(coeffs, iface_pts, q, grid_pts[nb : nb + 1])[0]
            rhs[n] += (side[nb] - side[n]) * correction / (h * h)
            active[n] = True
            eval_nodes[nb] = True
    return rhs, active, eval_nodes


def exact_taylor_coeffs_at_interface(iface_pts):
    x = iface_pts[:, 0]
    y = iface_pts[:, 1]
    return np.column_stack(
        [C_fn(x, y), Cx_fn(x, y), Cy_fn(x, y), Cxx_fn(x, y), Cxy_fn(x, y), Cyy_fn(x, y)]
    )


# ---------------------------------------------------------------------------
# Restrict: local quadratic fit and correction subtraction.
# ---------------------------------------------------------------------------


def quad_value(params, x, y):
    c0, cx, cy, cxx, cxy, cyy = params
    return c0 + cx * x + cy * y + 0.5 * cxx * x * x + cxy * x * y + 0.5 * cyy * y * y


def quad_poly_at(params, center):
    c0, cx, cy, cxx, cxy, cyy = params
    x, y = center
    return np.array(
        [
            quad_value(params, x, y),
            cx + cxx * x + cxy * y,
            cy + cxy * x + cyy * y,
            cxx,
            cxy,
            cyy,
        ]
    )


def restrict_fit_bulk_poly(grid_pts, bulk, iface_pts, N, radius=2):
    nx = N + 1
    h = 1.0 / N
    result = np.zeros((len(iface_pts), 6))
    for q, center in enumerate(iface_pts):
        ic = int(round(center[0] / h))
        jc = int(round(center[1] / h))
        rows = []
        rhs = []
        for j in range(max(0, jc - radius), min(nx - 1, jc + radius) + 1):
            for i in range(max(0, ic - radius), min(nx - 1, ic + radius) + 1):
                idx = j * nx + i
                dx = grid_pts[idx, 0] - center[0]
                dy = grid_pts[idx, 1] - center[1]
                rows.append([1.0, dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy])
                rhs.append(bulk[idx])
        result[q] = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    return result


# ---------------------------------------------------------------------------
# Compute visualization data.
# ---------------------------------------------------------------------------


N = 64
xs, ys, grid_pts = make_node_grid(N)
iface_pts, normals, weights = make_star_panels(N)
coeffs_panel = laplace_panel_cauchy(iface_pts, normals, weights, N)
coeffs_exact = exact_taylor_coeffs_at_interface(iface_pts)
nearest_q, _ = nearest_interface_indices(grid_pts, iface_pts)
labels = label_grid(grid_pts, iface_pts, 1.0 / N)

rhs_panel, active_nodes, eval_nodes = spread_rhs_correction(
    grid_pts, labels, nearest_q, coeffs_panel, iface_pts, N
)
rhs_exact, _, _ = spread_rhs_correction(grid_pts, labels, nearest_q, coeffs_exact, iface_pts, N)
rhs_diff = rhs_panel - rhs_exact

panel_C_grid = eval_local_poly(coeffs_panel, iface_pts, nearest_q, grid_pts)
exact_C_grid = C_fn(grid_pts[:, 0], grid_pts[:, 1])
C_err = np.abs(panel_C_grid - exact_C_grid)
C_err_used = np.where(eval_nodes, C_err, 0.0)

N_restrict = 48
xs_r, ys_r, grid_pts_r = make_node_grid(N_restrict)
iface_r, _, _ = make_star_panels(24)
physical = np.array([1.2, -0.7, 2.3, 0.5, -1.1, 0.8])
correction = np.array([-0.4, 0.8, 0.2, 0.25, -0.3, 0.1])
bulk = quad_value(physical, grid_pts_r[:, 0], grid_pts_r[:, 1]) + quad_value(
    correction, grid_pts_r[:, 0], grid_pts_r[:, 1]
)
fitted = restrict_fit_bulk_poly(grid_pts_r, bulk, iface_r, N_restrict)
correction_polys = np.array([quad_poly_at(correction, pt) for pt in iface_r])
restricted = fitted - correction_polys
expected = np.array([quad_poly_at(physical, pt) for pt in iface_r])
restrict_err = np.abs(restricted - expected)
restrict_err_norm = np.linalg.norm(restrict_err, axis=1)

print("Transfer visualization metrics:")
print(f"  spread active nodes        = {active_nodes.sum()}")
print(f"  spread eval nodes          = {eval_nodes.sum()}")
print(f"  max |RHS correction|       = {np.max(np.abs(rhs_panel)):.4e}")
print(f"  max |panel spread diff|    = {np.max(np.abs(rhs_diff)):.4e}")
print(f"  max |panel C - exact C|    = {np.max(C_err_used):.4e}  (used spread nodes)")
print(f"  max restrict coeff error   = {np.max(restrict_err):.4e}")


# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------


fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle(
    "Layer 1 transfer operators: LaplacePanelSpread2D and LaplaceQuadraticRestrict2D",
    fontsize=13,
    fontweight="bold",
)

cx, cy = star_curve()


def field2d(v, n):
    return v.reshape(n + 1, n + 1)


def setup_domain_axis(ax):
    ax.plot(cx, cy, "k-", lw=1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


ax = axes[0, 0]
ax.scatter(grid_pts[labels == 0, 0], grid_pts[labels == 0, 1], s=2, c="#cad3df", label="outside")
ax.scatter(grid_pts[labels == 1, 0], grid_pts[labels == 1, 1], s=2, c="#ffe3a3", label="inside")
ax.scatter(grid_pts[active_nodes, 0], grid_pts[active_nodes, 1], s=16, c="#c1121f", label="spread nodes")
ax.scatter(iface_pts[:, 0], iface_pts[:, 1], s=4, c="#1f77b4", label="GL nodes")
setup_domain_axis(ax)
ax.set_title(f"Cross-interface spread stencil nodes ({active_nodes.sum()} active)")
ax.legend(fontsize=8, loc="upper right")

ax = axes[0, 1]
rhs_abs = np.abs(rhs_panel)
vmax = max(rhs_abs.max(), 1e-12)
norm = mcolors.SymLogNorm(linthresh=vmax * 1e-4, vmin=-vmax, vmax=vmax)
im = ax.pcolormesh(xs, ys, field2d(rhs_panel, N), shading="auto", cmap="RdBu_r", norm=norm)
setup_domain_axis(ax)
ax.set_title("Spread RHS correction accumulated into bulk grid")
cb = plt.colorbar(im, ax=ax)
cb.set_label("rhs correction")

ax = axes[0, 2]
diff_abs = np.abs(rhs_diff)
masked = np.ma.masked_where(diff_abs == 0.0, diff_abs)
if diff_abs.max() > 0:
    diff_norm = mcolors.LogNorm(vmin=max(diff_abs[diff_abs > 0].min(), 1e-12), vmax=diff_abs.max())
else:
    diff_norm = None
im = ax.pcolormesh(xs, ys, field2d(masked.filled(0.0), N), shading="auto", cmap="magma", norm=diff_norm)
setup_domain_axis(ax)
ax.set_title(f"Panel spread error vs exact Taylor, max={diff_abs.max():.2e}")
plt.colorbar(im, ax=ax)

ax = axes[1, 0]
positive_C_err = C_err_used[C_err_used > 0.0]
im = ax.pcolormesh(
    xs,
    ys,
    np.ma.masked_where(field2d(C_err_used, N) <= 0.0, field2d(C_err_used, N)),
    shading="auto",
    cmap="hot_r",
    norm=mcolors.LogNorm(vmin=max(positive_C_err.min(), 1e-12), vmax=max(positive_C_err.max(), 1e-11)),
)
setup_domain_axis(ax)
ax.set_title(
    f"Panel Cauchy value error at spread eval nodes, max={positive_C_err.max():.2e}"
)
plt.colorbar(im, ax=ax)

ax = axes[1, 1]
positive_restrict = np.maximum(restrict_err_norm, 1e-16)
sc = ax.scatter(
    iface_r[:, 0],
    iface_r[:, 1],
    c=positive_restrict,
    s=24,
    cmap="viridis",
    norm=mcolors.LogNorm(vmin=positive_restrict.min(), vmax=positive_restrict.max()),
)
setup_domain_axis(ax)
ax.set_title(f"Restrict corrected coeff error on interface, max={restrict_err.max():.2e}")
cb = plt.colorbar(sc, ax=ax)
cb.set_label("2-norm coeff error")

ax = axes[1, 2]
labels_bar = ["u", "ux", "uy", "uxx", "uxy", "uyy"]
ax.bar(labels_bar, restrict_err.max(axis=0), color="#2c7be5")
ax.set_yscale("log")
ax.set_ylabel("max abs error")
ax.set_title("Restrict exact quadratic recovery after correction")
ax.grid(True, axis="y", which="both", ls=":", alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = "python/transfer_2d_viz.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved -> {out}")
