"""
Visualize the GridPair bounding-box test cases:
  1. closest_bulk_node — interface point → nearest grid node
  2. narrow band (1-layer, 2-layer) — grid nodes near the interface
     and their closest interface point

Mirrors the geometry in tests/test_grid_pair.cpp.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ─── geometry helpers ────────────────────────────────────────────────────────

def make_circle(cx, cy, r, N):
    """N uniformly-spaced points on a circle."""
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([cx + r * np.cos(theta),
                           cy + r * np.sin(theta)])
    return pts


def make_node_grid_2d(lo, hi, n):
    """(n+1) x (n+1) Node grid on [lo,hi]^2; returns (coords Nx2, h)."""
    h = (hi - lo) / n
    xs = np.linspace(lo, hi, n + 1)
    ys = np.linspace(lo, hi, n + 1)
    XX, YY = np.meshgrid(xs, ys)
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    return coords, h


def nearest_node(grid_coords, pt):
    """Return index of the grid node nearest to pt."""
    d2 = np.sum((grid_coords - pt) ** 2, axis=1)
    return int(np.argmin(d2))


def nearest_iface_pt(iface_pts, node_coord):
    """Return index of the interface point nearest to node_coord."""
    d2 = np.sum((iface_pts - node_coord) ** 2, axis=1)
    return int(np.argmin(d2))


def min_iface_dist(iface_pts, node_coord):
    d2 = np.sum((iface_pts - node_coord) ** 2, axis=1)
    return np.sqrt(d2.min())


# ─── build test geometry ─────────────────────────────────────────────────────

HALF   = 0.5
N_GRID = 16           # cells per side  (17×17 = 289 nodes)
R      = 0.25
N_IFACE = 32          # interface quadrature points (fewer = cleaner arrows)
N_IFACE_FINE = 64     # finer circle for narrow-band plot

grid, h = make_node_grid_2d(-HALF, HALF, N_GRID)
iface32 = make_circle(0.0, 0.0, R, N_IFACE)
iface64 = make_circle(0.0, 0.0, R, N_IFACE_FINE)

# ─── figure layout ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.suptitle("GridPair bounding-box test cases", fontsize=13, fontweight="bold")

GRID_COLOR  = "#b0b8c8"
NODE_COLOR  = "#4a6fa5"
IFACE_COLOR = "#c0392b"
BAND1_COLOR = "#f39c12"
BAND2_COLOR = "#27ae60"
ARROW_KW    = dict(arrowstyle="-|>", mutation_scale=7, linewidth=0.8)

circle_theta = np.linspace(0, 2 * np.pi, 400)
circle_x = R * np.cos(circle_theta)
circle_y = R * np.sin(circle_theta)

# ─── Panel 1: closest_bulk_node ──────────────────────────────────────────────

ax = axes[0]
ax.set_title("1 · closest_bulk_node\n(interface pt → nearest grid node)",
             fontsize=10)

# Grid
ax.scatter(grid[:, 0], grid[:, 1], s=8, color=GRID_COLOR, zorder=1)

# Interface
ax.plot(circle_x, circle_y, color=IFACE_COLOR, lw=1.5, zorder=3)
ax.scatter(iface32[:, 0], iface32[:, 1], s=18, color=IFACE_COLOR, zorder=4)

# Arrows: each interface point → its nearest grid node
for q, pt in enumerate(iface32):
    ni = nearest_node(grid, pt)
    nc = grid[ni]
    ax.annotate("", xy=nc, xytext=pt,
                arrowprops=dict(color="#2980b9", **ARROW_KW), zorder=5)

# Highlight the nearest grid nodes
nn_set = {nearest_node(grid, pt) for pt in iface32}
nn_coords = grid[list(nn_set)]
ax.scatter(nn_coords[:, 0], nn_coords[:, 1], s=35, color=NODE_COLOR,
           zorder=6, label="nearest nodes")

ax.set_xlim(-HALF - 0.05, HALF + 0.05)
ax.set_ylim(-HALF - 0.05, HALF + 0.05)
ax.set_aspect("equal")
ax.set_xlabel("x");  ax.set_ylabel("y")

legend_elems = [
    mpatches.Patch(color=IFACE_COLOR,  label="interface pts"),
    mpatches.Patch(color=NODE_COLOR,   label="nearest grid nodes"),
    mpatches.Patch(color=GRID_COLOR,   label="grid"),
]
ax.legend(handles=legend_elems, fontsize=7, loc="lower right")

# ─── Panel 2: narrow band 1-layer ────────────────────────────────────────────

ax = axes[1]
ax.set_title(f"2 · narrow band (radius = 1.5 h)\nclosest_interface_point",
             fontsize=10)

ax.scatter(grid[:, 0], grid[:, 1], s=8, color=GRID_COLOR, zorder=1)
ax.plot(circle_x, circle_y, color=IFACE_COLOR, lw=1.5, zorder=3)
ax.scatter(iface64[:, 0], iface64[:, 1], s=12, color=IFACE_COLOR, zorder=4)

# Band nodes
band1 = [n for n in range(len(grid))
         if min_iface_dist(iface64, grid[n]) < 1.5 * h]

# Colour all band nodes
ax.scatter(grid[band1, 0], grid[band1, 1], s=40, color=BAND1_COLOR,
           zorder=5, label=f"band nodes ({len(band1)})")

# Arrows: band node → nearest interface point
for n in band1:
    nc = grid[n]
    qi = nearest_iface_pt(iface64, nc)
    ip = iface64[qi]
    ax.annotate("", xy=ip, xytext=nc,
                arrowprops=dict(color=BAND1_COLOR, alpha=0.6, **ARROW_KW),
                zorder=6)

ax.set_xlim(-HALF - 0.05, HALF + 0.05)
ax.set_ylim(-HALF - 0.05, HALF + 0.05)
ax.set_aspect("equal")
ax.set_xlabel("x")

legend_elems = [
    mpatches.Patch(color=IFACE_COLOR,  label="interface pts"),
    mpatches.Patch(color=BAND1_COLOR,  label=f"1-layer band ({len(band1)} nodes)"),
    mpatches.Patch(color=GRID_COLOR,   label="grid"),
]
ax.legend(handles=legend_elems, fontsize=7, loc="lower right")

# ─── Panel 3: narrow band 2-layer ────────────────────────────────────────────

ax = axes[2]
ax.set_title(f"3 · narrow band (radius = 2.5 h)\nclosest_interface_point",
             fontsize=10)

ax.scatter(grid[:, 0], grid[:, 1], s=8, color=GRID_COLOR, zorder=1)
ax.plot(circle_x, circle_y, color=IFACE_COLOR, lw=1.5, zorder=3)
ax.scatter(iface64[:, 0], iface64[:, 1], s=12, color=IFACE_COLOR, zorder=4)

band2 = [n for n in range(len(grid))
         if min_iface_dist(iface64, grid[n]) < 2.5 * h]

# Draw outer ring (in band2 but not band1) first, then inner ring on top
outer = [n for n in band2 if n not in set(band1)]
ax.scatter(grid[outer,  0], grid[outer,  1], s=40, color=BAND2_COLOR,
           zorder=5, alpha=0.8)
ax.scatter(grid[band1,  0], grid[band1,  1], s=40, color=BAND1_COLOR,
           zorder=6)

for n in band2:
    nc = grid[n]
    qi = nearest_iface_pt(iface64, nc)
    ip = iface64[qi]
    color = BAND1_COLOR if n in set(band1) else BAND2_COLOR
    ax.annotate("", xy=ip, xytext=nc,
                arrowprops=dict(color=color, alpha=0.55, **ARROW_KW),
                zorder=7)

ax.set_xlim(-HALF - 0.05, HALF + 0.05)
ax.set_ylim(-HALF - 0.05, HALF + 0.05)
ax.set_aspect("equal")
ax.set_xlabel("x")

legend_elems = [
    mpatches.Patch(color=IFACE_COLOR,  label="interface pts"),
    mpatches.Patch(color=BAND1_COLOR,  label=f"1-layer band ({len(band1)})"),
    mpatches.Patch(color=BAND2_COLOR,  label=f"2-layer band ({len(band2)})"),
    mpatches.Patch(color=GRID_COLOR,   label="grid"),
]
ax.legend(handles=legend_elems, fontsize=7, loc="lower right")

# ─── save ────────────────────────────────────────────────────────────────────

out = "/Users/zhouhan/programs/kfbim/kfbim-recon/scripts/grid_pair_test_viz.png"
plt.tight_layout()
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
