"""
Visualize domain labeling for GridPair2D / GridPair3D test cases.

2D panels (left three):
  1. Single circle  — label 0 (exterior) and 1 (interior)
  2. Two circles    — labels 0, 1, 2
  3. Three circles  — labels 0, 1, 2, 3

3D panels (right two):
  4. Single sphere  — z=0 cross-section, labels 0 / 1
  5. Two spheres    — z=0 cross-section, labels 0 / 1 / 2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

kPi = np.pi

# ─── colour palette ──────────────────────────────────────────────────────────
LABEL_COLORS = ["#b0b8c8", "#2c7be5", "#e67e22", "#27ae60", "#9b59b6"]
LABEL_NAMES  = ["exterior (0)", "interior 1", "interior 2", "interior 3"]

# ─── geometry helpers ────────────────────────────────────────────────────────

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


def winding_label_2d(grid, circles):
    """
    Ray-casting (rightward +x ray) for a list of circles.
    circles: list of (cx, cy, r) tuples — each is one component.
    Returns array of labels (0 = exterior, 1,2,... = inside component index+1).
    """
    labels = np.zeros(len(grid), dtype=int)
    for comp, (cx, cy, r) in enumerate(circles):
        dist = np.sqrt((grid[:, 0] - cx)**2 + (grid[:, 1] - cy)**2)
        labels[dist < r] = comp + 1
    return labels


def sphere_label_3d(grid, spheres):
    """
    Analytic inside-outside for a list of spheres.
    spheres: list of (cx, cy, cz, r) tuples.
    """
    labels = np.zeros(len(grid), dtype=int)
    for comp, (cx, cy, cz, r) in enumerate(spheres):
        dist = np.sqrt((grid[:, 0]-cx)**2 + (grid[:, 1]-cy)**2 + (grid[:, 2]-cz)**2)
        labels[dist < r] = comp + 1
    return labels


def circle_line(cx, cy, r, N=400):
    th = np.linspace(0, 2*kPi, N)
    return cx + r*np.cos(th), cy + r*np.sin(th)


# ─── figure layout ───────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 8))
fig.suptitle("GridPair domain labeling  (0 = exterior, 1,2,3 = interior components)",
             fontsize=13, fontweight="bold")

# ============================================================================
# Panel 1 — single circle
# ============================================================================

ax = fig.add_subplot(1, 5, 1)
ax.set_title("2D · single circle\n(r = 0.30)", fontsize=10)

grid2, h2 = node_grid_2d(0, 1, 48)
circles1   = [(0.50, 0.50, 0.30)]
lab1       = winding_label_2d(grid2, circles1)

for lbl, col in enumerate(LABEL_COLORS[:2]):
    mask = lab1 == lbl
    ax.scatter(grid2[mask, 0], grid2[mask, 1], s=4, c=col, zorder=2)

cx, cy, r = circles1[0]
ax.plot(*circle_line(cx, cy, r), color="#c0392b", lw=1.5, zorder=3)
ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("x"); ax.set_ylabel("y")

# ============================================================================
# Panel 2 — two circles
# ============================================================================

ax = fig.add_subplot(1, 5, 2)
ax.set_title("2D · two circles\n(r = 0.15 each)", fontsize=10)

circles2 = [(0.25, 0.50, 0.15), (0.75, 0.50, 0.15)]
lab2     = winding_label_2d(grid2, circles2)

for lbl, col in enumerate(LABEL_COLORS[:3]):
    mask = lab2 == lbl
    ax.scatter(grid2[mask, 0], grid2[mask, 1], s=4, c=col, zorder=2)

for cx, cy, r in circles2:
    ax.plot(*circle_line(cx, cy, r), color="#c0392b", lw=1.5, zorder=3)

ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("x")

# ============================================================================
# Panel 3 — three circles
# ============================================================================

ax = fig.add_subplot(1, 5, 3)
ax.set_title("2D · three circles\n(r = 0.12, 0.10, 0.10)", fontsize=10)

circles3 = [(0.20, 0.50, 0.12), (0.55, 0.75, 0.10), (0.70, 0.30, 0.10)]
grid3, h3 = node_grid_2d(0, 1, 64)
lab3      = winding_label_2d(grid3, circles3)

for lbl, col in enumerate(LABEL_COLORS[:4]):
    mask = lab3 == lbl
    ax.scatter(grid3[mask, 0], grid3[mask, 1], s=3, c=col, zorder=2)

for cx, cy, r in circles3:
    ax.plot(*circle_line(cx, cy, r), color="#c0392b", lw=1.5, zorder=3)

ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("x")

# ============================================================================
# Panel 4 — single sphere (z=0 cross-section)
# ============================================================================

ax = fig.add_subplot(1, 5, 4)
ax.set_title("3D · single sphere  z=0\n(R = 0.28, grid 20³)", fontsize=10)

grid3d, h3d = node_grid_3d(-0.5, 0.5, 20)
spheres1     = [(0.0, 0.0, 0.0, 0.28)]
lab3d_1      = sphere_label_3d(grid3d, spheres1)

# Only nodes in the z=0 slab
tol  = h3d * 0.6
zmask = np.abs(grid3d[:, 2]) < tol

for lbl, col in enumerate(LABEL_COLORS[:2]):
    mask = (lab3d_1 == lbl) & zmask
    ax.scatter(grid3d[mask, 0], grid3d[mask, 1], s=12, c=col, zorder=2)

cx, cy, cz, r = spheres1[0]
ax.plot(*circle_line(cx, cy, r), color="#c0392b", lw=1.5, zorder=3)
ax.set_aspect("equal"); ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x"); ax.set_ylabel("y")

# ============================================================================
# Panel 5 — two spheres (z=0 cross-section)
# ============================================================================

ax = fig.add_subplot(1, 5, 5)
ax.set_title("3D · two spheres  z=0\n(R=0.16, grid 22³)", fontsize=10)

grid3d2, h3d2 = node_grid_3d(-0.55, 0.55, 22)
spheres2       = [(-0.28, 0.0, 0.0, 0.16), (0.28, 0.0, 0.0, 0.16)]
lab3d_2        = sphere_label_3d(grid3d2, spheres2)

tol2  = h3d2 * 0.6
zmask2 = np.abs(grid3d2[:, 2]) < tol2

for lbl, col in enumerate(LABEL_COLORS[:3]):
    mask = (lab3d_2 == lbl) & zmask2
    ax.scatter(grid3d2[mask, 0], grid3d2[mask, 1], s=12, c=col, zorder=2)

for cx, cy, cz, r in spheres2:
    ax.plot(*circle_line(cx, cy, r), color="#c0392b", lw=1.5, zorder=3)

ax.set_aspect("equal"); ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55)
ax.set_xlabel("x")

# ─── shared legend ───────────────────────────────────────────────────────────

legend_elems = [
    mpatches.Patch(color=LABEL_COLORS[0], label="label 0 — exterior"),
    mpatches.Patch(color=LABEL_COLORS[1], label="label 1 — interior (comp 0)"),
    mpatches.Patch(color=LABEL_COLORS[2], label="label 2 — interior (comp 1)"),
    mpatches.Patch(color=LABEL_COLORS[3], label="label 3 — interior (comp 2)"),
    plt.Line2D([0], [0], color="#c0392b", lw=1.5, label="interface"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.06, 1, 1])

out = "/Users/zhouhan/programs/kfbim/kfbim-recon/scripts/domain_labels_viz.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
