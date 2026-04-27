"""
Visualize domain labeling on non-trivial surfaces (no simple analytical expression).

Left column — 2D:
  1a. Star polygon (5 tips, non-convex)  — full grid, colored by label
  1b. Star polygon zoomed — grid + boundary curve

Right column — 3D cross-sections (z=0 plane):
  2.  Ellipsoid (a=0.35, b=0.25, c=0.20)
  3.  Torus (R_maj=0.30, R_min=0.14) — the hole at origin should be exterior
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

kPi = np.pi

# ─── colour palette ──────────────────────────────────────────────────────────
C_EXT  = "#b0b8c8"   # label 0 — exterior
C_INT  = "#2c7be5"   # label 1 — interior
C_FACE = "#c0392b"   # surface / boundary curve

# ─── 2D star geometry ────────────────────────────────────────────────────────

def star_r(th, R, A, k):
    return R * (1.0 + A * np.cos(k * th))

def star_contains(x, y, cx, cy, R, A, k):
    rho = np.sqrt((x - cx)**2 + (y - cy)**2)
    th  = np.arctan2(y - cy, x - cx)
    return rho < star_r(th, R, A, k)

def node_grid_2d(lo, hi, n):
    h  = (hi - lo) / n
    xs = np.linspace(lo, hi, n + 1)
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    return np.column_stack([XX.ravel(), YY.ravel()]), h

def node_grid_3d(lo, hi, n):
    h  = (hi - lo) / n
    xs = np.linspace(lo, hi, n + 1)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing='ij')
    return np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]), h

def ellipsoid_contains(x, y, z, a, b, c):
    return (x/a)**2 + (y/b)**2 + (z/c)**2 < 1.0

def torus_contains(x, y, z, R_maj, R_min):
    rho = np.sqrt(x**2 + y**2)
    return (rho - R_maj)**2 + z**2 < R_min**2

# ─── figure ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
fig.suptitle(
    "Domain labeling — non-trivial surfaces\n"
    "(blue = interior, grey = exterior)",
    fontsize=13, fontweight="bold"
)

# ============================================================================
# Panel 1: star polygon — full view
# ============================================================================

ax = axes[0]
ax.set_title("2D · 5-tip star polygon\nr = 0.28(1 + 0.40 cos 5θ)",
             fontsize=10)

cx, cy, R_star, A_star, k_star = 0.5, 0.5, 0.28, 0.40, 5
grid2, h2 = node_grid_2d(0, 1, 64)

labels2 = np.array([
    1 if star_contains(p[0], p[1], cx, cy, R_star, A_star, k_star) else 0
    for p in grid2
])

for lbl, col in [(0, C_EXT), (1, C_INT)]:
    mask = labels2 == lbl
    ax.scatter(grid2[mask, 0], grid2[mask, 1], s=3, c=col, zorder=2)

# Boundary curve
th = np.linspace(0, 2*kPi, 1000)
r_bnd = star_r(th, R_star, A_star, k_star)
ax.plot(cx + r_bnd * np.cos(th), cy + r_bnd * np.sin(th),
        color=C_FACE, lw=1.5, zorder=3)

ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("x"); ax.set_ylabel("y")

# ============================================================================
# Panel 2: star polygon — zoomed, show non-convex indentations clearly
# ============================================================================

ax = axes[1]
ax.set_title("2D · star polygon (zoom)\n— non-convex valleys visible",
             fontsize=10)

lo, hi = 0.05, 0.95
grid2z, h2z = node_grid_2d(lo, hi, 56)
labels2z = np.array([
    1 if star_contains(p[0], p[1], cx, cy, R_star, A_star, k_star) else 0
    for p in grid2z
])

# colour exterior nodes by distance to boundary for depth cue
dist2z = np.array([
    star_r(np.arctan2(p[1]-cy, p[0]-cx), R_star, A_star, k_star) -
    np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)
    for p in grid2z
])

for lbl, col in [(0, C_EXT), (1, C_INT)]:
    mask = labels2z == lbl
    ax.scatter(grid2z[mask, 0], grid2z[mask, 1],
               s=10, c=col, zorder=2, linewidths=0)

ax.plot(cx + r_bnd * np.cos(th), cy + r_bnd * np.sin(th),
        color=C_FACE, lw=2, zorder=3)

# Mark the tip and valley directions
for tip_th in np.linspace(0, 2*kPi, k_star, endpoint=False):
    ax.annotate("", xy=(cx + R_star*(1+A_star)*np.cos(tip_th),
                        cy + R_star*(1+A_star)*np.sin(tip_th)),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-", color="#888888",
                                lw=0.6, linestyle="--"))

ax.set_aspect("equal"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("x"); ax.set_ylabel("y")

# ============================================================================
# Panel 3: ellipsoid (z = 0 cross-section)
# ============================================================================

ax = axes[2]
ax.set_title("3D · ellipsoid  a=0.35, b=0.25, c=0.20\n(z = 0 slice)", fontsize=10)

a_el, b_el, c_el = 0.35, 0.25, 0.20
grid3, h3 = node_grid_3d(-0.5, 0.5, 20)
tol3 = h3 * 0.6
zmask = np.abs(grid3[:, 2]) < tol3

labels3 = np.where(
    ellipsoid_contains(grid3[:, 0], grid3[:, 1], grid3[:, 2], a_el, b_el, c_el),
    1, 0
)

for lbl, col in [(0, C_EXT), (1, C_INT)]:
    mask = (labels3 == lbl) & zmask
    ax.scatter(grid3[mask, 0], grid3[mask, 1], s=12, c=col, zorder=2, linewidths=0)

# Ellipse cross-section at z=0
ph = np.linspace(0, 2*kPi, 400)
ax.plot(a_el * np.cos(ph), b_el * np.sin(ph), color=C_FACE, lw=1.8, zorder=3)
ax.axhline(0, color="#cccccc", lw=0.5, zorder=1)
ax.axvline(0, color="#cccccc", lw=0.5, zorder=1)

ax.set_aspect("equal"); ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x"); ax.set_ylabel("y")

# Mark semi-axes
ax.annotate("", xy=(a_el, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
ax.annotate("", xy=(0, b_el), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
ax.text(a_el/2, -0.05, f"a={a_el}", fontsize=7, ha="center", color="#333")
ax.text(-0.06, b_el/2, f"b={b_el}", fontsize=7, ha="right", color="#333")

# ============================================================================
# Panel 4: torus (z = 0 cross-section)
# ============================================================================

ax = axes[3]
ax.set_title("3D · torus  R=0.30, r=0.14\n(z = 0 slice — hole at origin is exterior)",
             fontsize=10)

R_maj, R_min = 0.30, 0.14
labels_tor = np.where(
    torus_contains(grid3[:, 0], grid3[:, 1], grid3[:, 2], R_maj, R_min),
    1, 0
)

for lbl, col in [(0, C_EXT), (1, C_INT)]:
    mask = (labels_tor == lbl) & zmask
    ax.scatter(grid3[mask, 0], grid3[mask, 1], s=12, c=col, zorder=2, linewidths=0)

# Torus cross-section at z=0: two concentric circles at R_maj ± R_min
for sign in [1, -1]:
    ph = np.linspace(0, 2*kPi, 400)
    r_cross = R_maj + sign * R_min
    ax.plot(r_cross * np.cos(ph), r_cross * np.sin(ph), color=C_FACE, lw=1.8, zorder=3)
    ax.plot(-r_cross * np.cos(ph), -r_cross * np.sin(ph), color=C_FACE, lw=1.8, zorder=3)

# Actually torus at z=0 cross-section is two annuli: draw the ring cross
for sign in [1, -1]:
    r_val = R_maj + sign * R_min
    circle = plt.Circle((0, 0), r_val, fill=False, edgecolor=C_FACE, lw=1.8, zorder=3)
    ax.add_patch(circle)

# Remove duplicate drawn circles from earlier
ax.cla()
ax.set_title("3D · torus  R=0.30, r=0.14\n(z = 0 slice — hole at origin is exterior)",
             fontsize=10)
for lbl, col in [(0, C_EXT), (1, C_INT)]:
    mask = (labels_tor == lbl) & zmask
    ax.scatter(grid3[mask, 0], grid3[mask, 1], s=12, c=col, zorder=2, linewidths=0)
for r_val in [R_maj - R_min, R_maj + R_min]:
    circle = plt.Circle((0, 0), r_val, fill=False, edgecolor=C_FACE, lw=1.8, zorder=3)
    ax.add_patch(circle)

ax.plot(0, 0, 'x', color="#e74c3c", ms=8, mew=2, zorder=5, label="origin (exterior)")
ax.annotate("origin\n(exterior)", xy=(0.01, 0.01), xytext=(0.15, 0.25),
            fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8, color="#555"))

ax.set_aspect("equal"); ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x"); ax.set_ylabel("y")

# ─── shared legend ───────────────────────────────────────────────────────────

legend_elems = [
    mpatches.Patch(color=C_INT,  label="label 1 — interior"),
    mpatches.Patch(color=C_EXT,  label="label 0 — exterior"),
    Line2D([0], [0], color=C_FACE, lw=1.8, label="interface / boundary"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.06, 1, 1])

out = "/Users/zhouhan/programs/kfbim/kfbim-recon/scripts/nontrivial_domain_viz.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
