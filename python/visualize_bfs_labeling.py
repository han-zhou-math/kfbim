"""
Visualize the narrow-band + BFS domain-labeling algorithm.

Each panel shows two node categories:
  Band nodes   — within the narrow band (labeled by exact geometry test)
  BFS nodes    — outside the band (labeled by flood-fill from the band)
Further split by inside / outside, giving four visual layers.

Panels:
  1. 2D circle — no-boundary invariant
  2. 2D small circle on large grid — far-field BFS accuracy
  3. 2D star polygon (non-convex) — BFS in re-entrant geometry
  4. 3D sphere  z=0 slice — no-boundary invariant
  5. 3D sphere  z=0 slice — exhaustive far-field check
  6. 3D small sphere on large grid z=0 — far-field BFS accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

kPi = np.pi

# ─── colours ─────────────────────────────────────────────────────────────────
C_BAND_INT  = "#1a5276"   # band  + interior  — deep navy
C_BAND_EXT  = "#626567"   # band  + exterior  — slate
C_BFS_INT   = "#85c1e9"   # BFS   + interior  — sky blue
C_BFS_EXT   = "#d5d8dc"   # BFS   + exterior  — pale grey
C_IFACE     = "#c0392b"   # interface curve / sphere cross-section

# ─── helpers ─────────────────────────────────────────────────────────────────

def node_grid_2d(lo, hi, n):
    h  = (hi - lo) / n
    xs = np.linspace(lo, hi, n + 1)
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    return np.column_stack([XX.ravel(), YY.ravel()]), h

def node_grid_3d(lo, hi, n):
    h  = (hi - lo) / n
    xs = np.linspace(lo, hi, n + 1)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing='ij')
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    return pts, h

def min_dist_to_pts(grid_pts, iface_pts):
    """Brute-force min distance from each grid node to the nearest interface point."""
    d = np.full(len(grid_pts), np.inf)
    for q in iface_pts:
        diff = grid_pts - q
        d2   = (diff**2).sum(axis=1)
        d    = np.minimum(d, np.sqrt(d2))
    return d

def circle_pts(cx, cy, r, N):
    th = np.linspace(0, 2*kPi, N, endpoint=False)
    return np.column_stack([cx + r*np.cos(th), cy + r*np.sin(th)])

def sphere_uv_pts(cx, cy, cz, r, M, N):
    """Centroid quadrature points of UV-sphere triangles (matches make_sphere_uv)."""
    pts = []
    # Build vertices
    V = [[cx, cy, cz + r], [cx, cy, cz - r]]  # poles
    for m in range(M):
        phi = kPi * (m + 1) / (M + 1)
        for n in range(N):
            th = 2 * kPi * n / N
            V.append([cx + r*np.sin(phi)*np.cos(th),
                       cy + r*np.sin(phi)*np.sin(th),
                       cz + r*np.cos(phi)])
    V = np.array(V)

    def vidx(m, n): return 2 + m * N + (n % N)

    for n in range(N):  # top cap
        tri = np.array([V[0], V[vidx(0, n)], V[vidx(0, n+1)]])
        pts.append(tri.mean(axis=0))
    for m in range(M - 1):
        for n in range(N):
            tri1 = np.array([V[vidx(m,n)], V[vidx(m+1,n)],   V[vidx(m,n+1)]])
            tri2 = np.array([V[vidx(m,n+1)], V[vidx(m+1,n)], V[vidx(m+1,n+1)]])
            pts.append(tri1.mean(axis=0))
            pts.append(tri2.mean(axis=0))
    for n in range(N):  # bottom cap
        tri = np.array([V[1], V[vidx(M-1, n+1)], V[vidx(M-1, n)]])
        pts.append(tri.mean(axis=0))
    return np.array(pts)

def star_r(th, R, A, k):
    return R * (1.0 + A * np.cos(k * th))

def star_contains(x, y, cx, cy, R, A, k):
    rho = np.sqrt((x - cx)**2 + (y - cy)**2)
    th  = np.arctan2(y - cy, x - cx)
    return rho < star_r(th, R, A, k)

def star_pts(cx, cy, R, A, k, N):
    th = np.linspace(0, 2*kPi, N, endpoint=False)
    r  = star_r(th, R, A, k)
    return np.column_stack([cx + r*np.cos(th), cy + r*np.sin(th)])

def label_2d(grid_pts, iface_pts, contains_fn, h, band_mult=4.0):
    """domain_label and band flag for each node, mimicking BFS algorithm."""
    d     = min_dist_to_pts(grid_pts, iface_pts)
    band  = d < band_mult * h
    label = np.zeros(len(grid_pts), dtype=int)
    for n, p in enumerate(grid_pts):
        if contains_fn(p[0], p[1]):
            label[n] = 1
    return label, band

def label_3d_sphere(grid_pts, iface_pts, cx, cy, cz, R, h, band_mult=4.0):
    d     = min_dist_to_pts(grid_pts, iface_pts)
    band  = d < band_mult * h
    dist  = np.sqrt((grid_pts[:,0]-cx)**2 + (grid_pts[:,1]-cy)**2 + (grid_pts[:,2]-cz)**2)
    label = (dist < R).astype(int)
    return label, band

def plot_panel(ax, grid_pts, label, band, title, iface_curve_2d=None, dot_size=6):
    """Scatter plot with four-colour scheme."""
    cats = {
        (True,  1): C_BAND_INT,
        (True,  0): C_BAND_EXT,
        (False, 1): C_BFS_INT,
        (False, 0): C_BFS_EXT,
    }
    order = [(False, 0), (False, 1), (True, 0), (True, 1)]  # band on top
    for (b, l), col in [(k, cats[k]) for k in order]:
        mask = (band == b) & (label == l)
        if mask.any():
            ax.scatter(grid_pts[mask, 0], grid_pts[mask, 1],
                       s=dot_size, c=col, linewidths=0, zorder=2 + b)
    if iface_curve_2d is not None:
        for curve in iface_curve_2d:
            ax.plot(curve[:, 0], curve[:, 1], color=C_IFACE, lw=1.5, zorder=5)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
    ax.tick_params(labelsize=7)

# ─── figure ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    "Narrow-band + BFS domain labeling\n"
    "navy/blue = interior · grey = exterior · dark = band (geometry test) · light = BFS flood-fill",
    fontsize=11, fontweight="bold"
)

# ============================================================================
# Panel 1: 2D circle — no-boundary invariant
# ============================================================================
ax = axes[0, 0]
grid2, h2 = node_grid_2d(0.0, 1.0, 32)
cx, cy, r_c = 0.50, 0.50, 0.30
ipts_c = circle_pts(cx, cy, r_c, 512)
lbl, band = label_2d(grid2, ipts_c, lambda x, y: np.sqrt((x-cx)**2+(y-cy)**2) < r_c, h2)

th = np.linspace(0, 2*kPi, 400)
curve_c = [np.column_stack([cx + r_c*np.cos(th), cy + r_c*np.sin(th)])]
plot_panel(ax, grid2, lbl, band,
           "2D · circle  r=0.30\nno-boundary invariant (h=1/32)",
           iface_curve_2d=curve_c)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ============================================================================
# Panel 2: 2D small circle on large grid — far-field BFS accuracy
# ============================================================================
ax = axes[0, 1]
grid2b, h2b = node_grid_2d(-1.0, 1.0, 40)
r_s = 0.15
ipts_cs = circle_pts(0.0, 0.0, r_s, 256)
lbl_b, band_b = label_2d(grid2b, ipts_cs, lambda x, y: np.sqrt(x**2+y**2) < r_s, h2b)

curve_s = [np.column_stack([r_s*np.cos(th), r_s*np.sin(th)])]
plot_panel(ax, grid2b, lbl_b, band_b,
           "2D · small circle  r=0.15 on [-1,1]²\nfar-field BFS accuracy (h=0.05)",
           iface_curve_2d=curve_s, dot_size=8)
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

# ============================================================================
# Panel 3: 2D star polygon (non-convex) — BFS in re-entrant geometry
# ============================================================================
ax = axes[0, 2]
grid2c, h2c = node_grid_2d(0.0, 1.0, 64)
cx_st, cy_st, R_st, A_st, k_st = 0.5, 0.5, 0.28, 0.40, 5
ipts_st = star_pts(cx_st, cy_st, R_st, A_st, k_st, 512)
lbl_c, band_c = label_2d(grid2c, ipts_st,
                          lambda x, y: star_contains(x, y, cx_st, cy_st, R_st, A_st, k_st),
                          h2c)

th_st  = np.linspace(0, 2*kPi, 1000)
r_bnd  = star_r(th_st, R_st, A_st, k_st)
curve_st = [np.column_stack([cx_st + r_bnd*np.cos(th_st), cy_st + r_bnd*np.sin(th_st)])]
plot_panel(ax, grid2c, lbl_c, band_c,
           "2D · star polygon (5 tips, non-convex)\nBFS in re-entrant geometry (h=1/64)",
           iface_curve_2d=curve_st, dot_size=3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ============================================================================
# Panel 4: 3D sphere z=0 — no-boundary invariant
# ============================================================================
ax = axes[1, 0]
grid3a, h3a = node_grid_3d(-0.5, 0.5, 20)
R3a = 0.25
ipts_3a = sphere_uv_pts(0, 0, 0, R3a, 10, 20)
lbl_3a, band_3a = label_3d_sphere(grid3a, ipts_3a, 0, 0, 0, R3a, h3a)

zmask_a = np.abs(grid3a[:, 2]) < 0.6 * h3a
plot_panel(ax, grid3a[zmask_a, :2], lbl_3a[zmask_a], band_3a[zmask_a],
           "3D · sphere  R=0.25  z=0 slice\nno-boundary invariant (h=1/20)",
           iface_curve_2d=[np.column_stack([R3a*np.cos(th), R3a*np.sin(th)])],
           dot_size=14)
ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)

# ============================================================================
# Panel 5: 3D sphere exhaustive far-field check
# ============================================================================
ax = axes[1, 1]
grid3b, h3b = node_grid_3d(-0.5, 0.5, 24)
R3b = 0.28
ipts_3b = sphere_uv_pts(0, 0, 0, R3b, 12, 24)
lbl_3b, band_3b = label_3d_sphere(grid3b, ipts_3b, 0, 0, 0, R3b, h3b)

zmask_b = np.abs(grid3b[:, 2]) < 0.6 * h3b
plot_panel(ax, grid3b[zmask_b, :2], lbl_3b[zmask_b], band_3b[zmask_b],
           "3D · sphere  R=0.28  z=0 slice\nexhaustive far-field check (h=1/24)",
           iface_curve_2d=[np.column_stack([R3b*np.cos(th), R3b*np.sin(th)])],
           dot_size=12)
ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)

# ============================================================================
# Panel 6: 3D small sphere on large grid — far-field BFS accuracy
# ============================================================================
ax = axes[1, 2]
grid3c, h3c = node_grid_3d(-2.0, 2.0, 20)
R3c = 0.30
ipts_3c = sphere_uv_pts(0, 0, 0, R3c, 8, 16)
lbl_3c, band_3c = label_3d_sphere(grid3c, ipts_3c, 0, 0, 0, R3c, h3c)

zmask_c = np.abs(grid3c[:, 2]) < 0.6 * h3c
plot_panel(ax, grid3c[zmask_c, :2], lbl_3c[zmask_c], band_3c[zmask_c],
           "3D · small sphere  R=0.30 on [-2,2]³  z=0 slice\nfar-field BFS accuracy (h=0.2)",
           iface_curve_2d=[np.column_stack([R3c*np.cos(th), R3c*np.sin(th)])],
           dot_size=16)
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)

# ─── shared legend ───────────────────────────────────────────────────────────
legend_elems = [
    mpatches.Patch(color=C_BAND_INT,  label="band · interior  (geometry test)"),
    mpatches.Patch(color=C_BAND_EXT,  label="band · exterior  (geometry test)"),
    mpatches.Patch(color=C_BFS_INT,   label="BFS  · interior  (flood-fill)"),
    mpatches.Patch(color=C_BFS_EXT,   label="BFS  · exterior  (flood-fill)"),
    Line2D([0], [0], color=C_IFACE, lw=1.8, label="interface"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout(rect=[0, 0.06, 1, 1])

out = "python/bfs_labeling_viz.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved → {out}")
plt.show()
