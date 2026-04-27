"""
Visualize 3D GridPair bounding-box test cases (mirrors test_grid_pair.cpp):
  1. closest_bulk_node   — interface centroid → nearest grid node
  2. narrow band 1-layer — grid nodes within 1.5 h of the sphere
  3. narrow band 2-layer — grid nodes within 2.5 h of the sphere

Grid: [-0.5, 0.5]^3, n=8 cells (9^3 = 729 nodes), h = 0.125
Interface: UV sphere, r=0.3, M=4 latitude rings, N=8 longitudes → 64 centroids
(Test uses n=16; n=8 chosen here for visual clarity — algorithm is identical.)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ─── geometry helpers ────────────────────────────────────────────────────────

def make_node_grid_3d(lo, hi, n):
    h = (hi - lo) / n
    xs = np.linspace(lo, hi, n + 1)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing='ij')
    return np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]), h


def make_sphere_uv(cx, cy, cz, r, M, N):
    """
    UV sphere: M latitude rings, N longitude segments.
    Returns (vertices Nv×3, triangles Nt×3, centroids Nt×3).
    Same geometry as the C++ make_sphere_uv in test_grid_pair.cpp.
    """
    Nv = 2 + M * N
    verts = np.zeros((Nv, 3))
    verts[0] = [cx, cy, cz + r]
    verts[1] = [cx, cy, cz - r]
    for m in range(M):
        phi  = np.pi * (m + 1) / (M + 1)
        z_r  = cz + r * np.cos(phi)
        rxy  = r * np.sin(phi)
        for nn in range(N):
            th = 2 * np.pi * nn / N
            verts[2 + m * N + nn] = [cx + rxy * np.cos(th),
                                     cy + rxy * np.sin(th), z_r]
    tris = []
    for nn in range(N):
        tris.append([0, 2 + nn, 2 + (nn + 1) % N])
    for m in range(M - 1):
        for nn in range(N):
            a, b = 2+m*N+nn, 2+m*N+(nn+1)%N
            c, d = 2+(m+1)*N+nn, 2+(m+1)*N+(nn+1)%N
            tris.append([a, c, b])
            tris.append([b, c, d])
    for nn in range(N):
        tris.append([1, 2+(M-1)*N+(nn+1)%N, 2+(M-1)*N+nn])
    tris     = np.array(tris)
    centroids = (verts[tris[:,0]] + verts[tris[:,1]] + verts[tris[:,2]]) / 3.0
    return verts, tris, centroids


def min_dist_matrix(grid, iface_pts):
    """(N_nodes,) array of min distances to the interface point set."""
    diff = grid[:, None, :] - iface_pts[None, :, :]   # (N, M, 3)
    return np.min(np.linalg.norm(diff, axis=2), axis=1)


def nearest_node_idx(grid, pt):
    return int(np.argmin(np.linalg.norm(grid - pt, axis=1)))


def nearest_iface_idx(iface_pts, node):
    return int(np.argmin(np.linalg.norm(iface_pts - node, axis=1)))


# ─── smooth sphere surface (for wireframe backdrop) ──────────────────────────

def sphere_surface(r=0.3, Nu=40, Nv=20):
    u = np.linspace(0, 2 * np.pi, Nu)
    v = np.linspace(0, np.pi, Nv)
    X = r * np.outer(np.cos(u), np.sin(v))
    Y = r * np.outer(np.sin(u), np.sin(v))
    Z = r * np.outer(np.ones(Nu), np.cos(v))
    return X, Y, Z


# ─── shared drawing helpers ──────────────────────────────────────────────────

def add_sphere_wire(ax, alpha=0.08, color='#c0392b', lw=0.4):
    X, Y, Z = sphere_surface()
    ax.plot_wireframe(X, Y, Z, alpha=alpha, color=color, linewidth=lw, zorder=1)


def set_axes(ax, title, elev=22, azim=38):
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("x", fontsize=7, labelpad=1)
    ax.set_ylabel("y", fontsize=7, labelpad=1)
    ax.set_zlabel("z", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=elev, azim=azim)


# ─── build data ──────────────────────────────────────────────────────────────

R = 0.3
grid, h = make_node_grid_3d(-0.5, 0.5, 8)       # 9^3 = 729 nodes, h = 0.125
verts, tris, iface = make_sphere_uv(0, 0, 0, R, M=4, N=8)   # 64 centroids

print(f"Grid nodes: {len(grid)}   Interface pts: {len(iface)}   h = {h:.4f}")

# Per-node min distance to any interface centroid
node_dist = min_dist_matrix(grid, iface)

band1 = np.where(node_dist < 1.5 * h)[0]
band2 = np.where(node_dist < 2.5 * h)[0]
outer = np.setdiff1d(band2, band1)

print(f"1-layer band: {len(band1)} nodes   2-layer band: {len(band2)} nodes")

GRID_C   = "#b0b8c8"
NODE_C   = "#2c7be5"
IFACE_C  = "#c0392b"
BAND1_C  = "#f39c12"
BAND2_C  = "#27ae60"

# ─── figure ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 5.5))
fig.suptitle("GridPair 3D bounding-box test cases  (sphere r=0.3, grid 9³, h=0.125)",
             fontsize=12, fontweight="bold")

# ─── Panel 1: closest_bulk_node ──────────────────────────────────────────────

ax1 = fig.add_subplot(1, 3, 1, projection='3d')

add_sphere_wire(ax1)

# All grid nodes (tiny, faded)
ax1.scatter(*grid.T, s=3, c=GRID_C, alpha=0.3, depthshade=False, zorder=2)

# Lines: interface centroid → nearest grid node
nn_indices = [nearest_node_idx(grid, pt) for pt in iface]
segs = [[pt, grid[ni]] for pt, ni in zip(iface, nn_indices)]
lc = Line3DCollection(segs, linewidths=0.8, colors=NODE_C, alpha=0.7, zorder=4)
ax1.add_collection3d(lc)

# Nearest grid nodes (blue, larger)
nn_set = np.unique(nn_indices)
ax1.scatter(*grid[nn_set].T,  s=30, c=NODE_C,  alpha=1.0, depthshade=False, zorder=5)

# Interface centroids (red)
ax1.scatter(*iface.T, s=18, c=IFACE_C, alpha=1.0, depthshade=False, zorder=6)

set_axes(ax1, "1 · closest_bulk_node\n(interface centroid → nearest grid node)")

# ─── Panel 2: narrow band 1-layer ────────────────────────────────────────────

ax2 = fig.add_subplot(1, 3, 2, projection='3d')

add_sphere_wire(ax2)

# Band-1 nodes
ax2.scatter(*grid[band1].T, s=25, c=BAND1_C, alpha=0.9, depthshade=False, zorder=3)

# Arrows: subsample every 2nd band node → nearest interface centroid
stride = 2
for n in band1[::stride]:
    nc = grid[n]
    qi = nearest_iface_idx(iface, nc)
    ip = iface[qi]
    d  = ip - nc
    ax2.quiver(*nc, *d, length=1.0, normalize=False,
               color=BAND1_C, alpha=0.55, arrow_length_ratio=0.3,
               linewidth=0.7, zorder=4)

# Interface centroids
ax2.scatter(*iface.T, s=14, c=IFACE_C, alpha=0.9, depthshade=False, zorder=5)

set_axes(ax2,
         f"2 · narrow band (1-layer, r<1.5h)\nclosest_interface_point  [{len(band1)} nodes]")

# ─── Panel 3: narrow band 2-layer ────────────────────────────────────────────

ax3 = fig.add_subplot(1, 3, 3, projection='3d')

add_sphere_wire(ax3)

# Outer ring (band2 \ band1) in green
ax3.scatter(*grid[outer].T,  s=22, c=BAND2_C, alpha=0.85, depthshade=False, zorder=3)
# Inner ring (band1) in orange, on top
ax3.scatter(*grid[band1].T,  s=25, c=BAND1_C, alpha=0.95, depthshade=False, zorder=4)

# Arrows (every 2nd node, both layers)
all_band = np.concatenate([outer, band1])
for n in all_band[::2]:
    nc = grid[n]
    qi = nearest_iface_idx(iface, nc)
    ip = iface[qi]
    color = BAND1_C if n in set(band1) else BAND2_C
    ax3.quiver(*nc, *(ip - nc), length=1.0, normalize=False,
               color=color, alpha=0.45, arrow_length_ratio=0.3,
               linewidth=0.7, zorder=5)

ax3.scatter(*iface.T, s=14, c=IFACE_C, alpha=0.9, depthshade=False, zorder=6)

set_axes(ax3,
         f"3 · narrow band (2-layer, r<2.5h)\n"
         f"1-layer: {len(band1)}  |  2-layer: {len(band2)}")

# ─── shared legend ───────────────────────────────────────────────────────────

import matplotlib.patches as mpatches
legend_elems = [
    mpatches.Patch(color=IFACE_C,  label="interface centroids (64)"),
    mpatches.Patch(color=NODE_C,   label="nearest grid nodes  (panel 1)"),
    mpatches.Patch(color=BAND1_C,  label=f"1-layer band  ({len(band1)} nodes, r<1.5h)"),
    mpatches.Patch(color=BAND2_C,  label=f"2-layer outer ({len(outer)} nodes, r<2.5h)"),
    mpatches.Patch(color=GRID_C,   label="grid nodes"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=5,
           fontsize=8, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])

out = "/Users/zhouhan/programs/kfbim/kfbim-recon/scripts/grid_pair_test_viz_3d.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
