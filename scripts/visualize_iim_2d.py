"""
Visualize the IIM defect-correction Poisson solve on a star-shaped interface.

Panels:
  (0,0)  Irregular nodes — which nodes get the 1/h² correction
  (0,1)  Exact solution u (piecewise smooth)
  (0,2)  Computed solution u_h (IIM-corrected FFT solve)
  (1,0)  Pointwise error |u_h − u|
  (1,1)  Convergence: ‖error‖_∞ vs h  (log-log, slope ≈ 2)
  (1,2)  CPU time breakdown vs N
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.fft import dstn

kPi = np.pi

# ─── manufactured solution ───────────────────────────────────────────────────

def u_plus(x, y):  return np.sin(kPi*x) * np.sin(kPi*y)
def u_minus(x, y): return np.sin(2*kPi*x) * np.sin(2*kPi*y)
def f_plus(x, y):  return 2.0*kPi**2 * np.sin(kPi*x) * np.sin(kPi*y)
def f_minus(x, y): return 8.0*kPi**2 * np.sin(2*kPi*x) * np.sin(2*kPi*y)
def C_fn(x, y):    return u_plus(x,y) - u_minus(x,y)

# ─── star geometry ───────────────────────────────────────────────────────────

CX, CY, R_STAR, A_STAR, K_STAR = 0.5, 0.5, 0.28, 0.40, 5

def star_r(th):
    return R_STAR * (1.0 + A_STAR * np.cos(K_STAR * th))

def star_contains(x, y):
    rho = np.sqrt((x - CX)**2 + (y - CY)**2)
    th  = np.arctan2(y - CY, x - CX)
    return rho < star_r(th)

def star_iface_pts(N_pts):
    th = np.linspace(0, 2*kPi, N_pts, endpoint=False)
    r  = star_r(th)
    return np.column_stack([CX + r*np.cos(th), CY + r*np.sin(th)])

# ─── domain labeling: narrow band (exact test) + BFS flood-fill ──────────────

def label_grid(pts, iface_pts, h):
    """Returns labels[n] = 1 (inside) or 0 (outside) for each grid point."""
    N_pts = len(pts)
    # min distance to interface quadrature points
    d = np.full(N_pts, np.inf)
    for q in iface_pts:
        dd = np.sqrt(((pts - q)**2).sum(axis=1))
        d = np.minimum(d, dd)

    band_r = 4.0 * h
    labels = np.full(N_pts, -1, dtype=int)

    # (a) label band nodes analytically
    band_idx = np.where(d < band_r)[0]
    for n in band_idx:
        labels[n] = 1 if star_contains(pts[n,0], pts[n,1]) else 0

    # (b) BFS from band
    from collections import deque
    nx = ny = int(round(1.0 / h)) + 1
    queue = deque(band_idx)
    while queue:
        n = queue.popleft()
        lbl = labels[n]
        i, j = n % nx, n // nx
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < nx and 0 <= nj < ny:
                nb = nj*nx + ni
                if labels[nb] == -1:
                    labels[nb] = lbl
                    queue.append(nb)
    labels[labels == -1] = 0
    return labels

# ─── irregular node detection ────────────────────────────────────────────────

def find_irregular(labels, nx, ny):
    """Returns boolean array; True = irregular interior node."""
    N = nx * ny
    irreg = np.zeros(N, dtype=bool)
    for n in range(N):
        i, j = n % nx, n // nx
        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            continue
        ln = labels[n]
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i+di, j+dj
            nb = nj*nx + ni
            if 1 <= ni <= nx-2 and 1 <= nj <= ny-2 and labels[nb] != ln:
                irreg[n] = True
                break
    return irreg

# ─── IIM correction ──────────────────────────────────────────────────────────

def iim_correct(f, C, labels, nx, ny, h):
    """Returns corrected F = f + (lnb - ln) * C[nb] / h² for irregular nodes."""
    F = f.copy()
    h2 = h * h
    for n in range(nx * ny):
        i, j = n % nx, n // nx
        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            continue
        ln = labels[n]
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i+di, j+dj
            if 1 <= ni <= nx-2 and 1 <= nj <= ny-2:
                nb = nj*nx + ni
                lnb = labels[nb]
                if lnb != ln:
                    F[n] += (lnb - ln) * C[nb] / h2
    return F

# ─── DST Poisson solver: Δ_h u = rhs with homogeneous Dirichlet ──────────────

def solve_poisson(rhs_full, N, h):
    """rhs_full: (N+1)*(N+1) flat array, row-major j*(N+1)+i.
    Returns u_full same shape (zero at boundary)."""
    nx = ny = N + 1
    # Extract interior rhs: shape (N-1, N-1), row=j-1, col=i-1
    rhs_int = np.zeros((N-1, N-1))
    for j in range(1, N):
        for i in range(1, N):
            rhs_int[j-1, i-1] = rhs_full[j*nx + i]

    # Eigenvalues of Δ_h (discrete positive Laplacian / h²):
    p = np.arange(1, N)
    lam1d = (2*np.cos(p*kPi/N) - 2.0) / h**2   # negative values
    LAM = lam1d[:, None] + lam1d[None, :]        # shape (N-1, N-1)

    # DST-I (ortho) diagonalizes the 1D stencil
    rhs_hat = dstn(rhs_int, type=1, norm='ortho')
    u_hat   = rhs_hat / LAM
    u_int   = dstn(u_hat, type=1, norm='ortho')

    u_full = np.zeros(nx * ny)
    for j in range(1, N):
        for i in range(1, N):
            u_full[j*nx + i] = u_int[j-1, i-1]
    return u_full

# ─── full IIM pipeline at resolution N ───────────────────────────────────────

def iim_solve(N, timed=False):
    h   = 1.0 / N
    nx  = ny = N + 1
    n_total = nx * ny

    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    # indexing='xy' (default): XI[j,i]=xs[i], YI[j,i]=ys[j]
    # ravel order: j*nx+i  →  pts[j*nx+i] = (xs[i], ys[j])  ✓ matches C++ convention
    XI, YI = np.meshgrid(xs, ys)
    pts = np.column_stack([XI.ravel(), YI.ravel()])  # (n_total, 2)

    t0 = time.perf_counter()

    # 1. Domain labeling
    iface_pts = star_iface_pts(8 * N)
    labels = label_grid(pts, iface_pts, h)

    t_label = time.perf_counter()

    # 2. Build f, C, u_exact
    f_arr      = np.zeros(n_total)
    u_exact    = np.zeros(n_total)
    C_arr      = np.zeros(n_total)
    for n in range(n_total):
        x, y = pts[n]
        i, j = n % nx, n // nx
        bdy  = (i == 0 or i == nx-1 or j == 0 or j == ny-1)
        lbl  = labels[n]
        u_exact[n] = u_minus(x,y) if lbl == 1 else u_plus(x,y)
        C_arr[n]   = C_fn(x, y)
        if not bdy:
            f_arr[n] = f_minus(x,y) if lbl == 1 else f_plus(x,y)

    # 3. IIM correction
    F = iim_correct(f_arr, C_arr, labels, nx, ny, h)

    t_correct = time.perf_counter()

    # 4. Solve (solver convention: pass -F for -Δu = f ↔ Δu = -f)
    u_sol = solve_poisson(-F, N, h)

    t_solve = time.perf_counter()

    # 5. Max-norm error at interior nodes
    err = 0.0
    for j in range(1, N):
        for i in range(1, N):
            n = j*nx + i
            err = max(err, abs(u_sol[n] - u_exact[n]))

    t_end = time.perf_counter()

    timing = {
        'label':   t_label   - t0,
        'correct': t_correct - t_label,
        'solve':   t_solve   - t_correct,
        'total':   t_end     - t0,
    }
    return labels, u_exact, u_sol, pts, h, nx, ny, err, timing

# ============================================================================
# Compute data
# ============================================================================

print("Computing N=64 solution for panels 1–4 …")
labels64, u_exact64, u_sol64, pts64, h64, nx64, ny64, err64, timing64 = iim_solve(64, timed=True)
irreg64 = find_irregular(labels64, nx64, ny64)

print("  N=64  max_err = {:.4e}  t_total = {:.3f}s".format(err64, timing64['total']))

print("Computing convergence table …")
Ns     = [16, 32, 64, 128, 256]
errors = []
times  = {'label': [], 'correct': [], 'solve': [], 'total': []}

for N in Ns:
    *_, err, tim = iim_solve(N)
    errors.append(err)
    for k in times: times[k].append(tim[k])
    print("  N={:4d}  err={:.4e}  t_total={:.3f}s".format(N, err, tim['total']))

# ============================================================================
# Figure
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "IIM defect correction — star interface Poisson problem\n"
    r"$-\Delta u = f,\quad u^+ = \sin(\pi x)\sin(\pi y),\quad u^- = \sin(2\pi x)\sin(2\pi y)$",
    fontsize=12, fontweight='bold'
)

th_curve = np.linspace(0, 2*kPi, 800)
r_curve  = star_r(th_curve)
cx_curve = CX + r_curve * np.cos(th_curve)
cy_curve = CY + r_curve * np.sin(th_curve)

# ─── helper: reshape flat array to 2D (x-fast) for pcolormesh ───────────────
def to_grid(arr, nx, ny):
    # arr[j*nx+i] → shape (ny, nx) for imshow/pcolormesh
    return arr.reshape(ny, nx).T  # now shape (nx, ny)? let's keep it as (ny,nx)

def to_2d(arr, nx, ny):
    return arr.reshape(ny, nx)  # row=j (y), col=i (x)  → matches imshow

# ─── Panel (0,0): irregular nodes ────────────────────────────────────────────
ax = axes[0, 0]

cols_reg  = {0: '#d5d8dc', 1: '#85c1e9'}   # regular exterior / interior
cols_irr  = {0: '#e74c3c', 1: '#1a5276'}   # irreg exterior / interior

# Sort: regular first (background), irregular on top
for lbl in [0, 1]:
    mask_reg = (~irreg64) & (labels64 == lbl)
    ax.scatter(pts64[mask_reg, 0], pts64[mask_reg, 1],
               s=4, c=cols_reg[lbl], linewidths=0, zorder=2)
for lbl in [0, 1]:
    mask_irr = irreg64 & (labels64 == lbl)
    ax.scatter(pts64[mask_irr, 0], pts64[mask_irr, 1],
               s=16, c=cols_irr[lbl], linewidths=0, zorder=3,
               label=f"irregular, {'inside' if lbl else 'outside'}")

ax.plot(cx_curve, cy_curve, 'k-', lw=1.2, zorder=4)
ax.set_title(f"Irregular nodes  (N={nx64-1}, {irreg64.sum()} irregular)", fontsize=10)
ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xlabel('x'); ax.set_ylabel('y')

from matplotlib.patches import Patch
legend_elems = [
    Patch(color=cols_reg[1],  label='regular interior'),
    Patch(color=cols_reg[0],  label='regular exterior'),
    Patch(color=cols_irr[1],  label='irregular interior'),
    Patch(color=cols_irr[0],  label='irregular exterior'),
]
ax.legend(handles=legend_elems, fontsize=7, loc='upper right')

# ─── Panel (0,1): exact solution ─────────────────────────────────────────────
ax = axes[0, 1]
Z_exact = to_2d(u_exact64, nx64, ny64)  # (ny, nx)
xs = np.linspace(0, 1, nx64)
ys = np.linspace(0, 1, ny64)
im = ax.pcolormesh(xs, ys, Z_exact, shading='auto', cmap='RdBu_r')
ax.plot(cx_curve, cy_curve, 'k-', lw=1.2)
ax.set_title("Exact solution $u$", fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax)

# ─── Panel (0,2): computed solution ──────────────────────────────────────────
ax = axes[0, 2]
Z_sol = to_2d(u_sol64, nx64, ny64)
im2 = ax.pcolormesh(xs, ys, Z_sol, shading='auto', cmap='RdBu_r',
                    vmin=Z_exact.min(), vmax=Z_exact.max())
ax.plot(cx_curve, cy_curve, 'k-', lw=1.2)
ax.set_title("Computed solution $u_h$ (IIM-corrected FFT)", fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im2, ax=ax)

# ─── Panel (1,0): pointwise error ────────────────────────────────────────────
ax = axes[1, 0]
err_field = np.abs(u_sol64 - u_exact64)
# zero out boundary
for n in range(nx64 * ny64):
    i, j = n % nx64, n // nx64
    if i == 0 or i == nx64-1 or j == 0 or j == ny64-1:
        err_field[n] = 0.0
Z_err = to_2d(err_field, nx64, ny64)
im3 = ax.pcolormesh(xs, ys, Z_err, shading='auto', cmap='hot_r',
                    norm=mcolors.LogNorm(vmin=max(Z_err.max()*1e-4, 1e-10),
                                         vmax=max(Z_err.max(), 1e-9)))
ax.plot(cx_curve, cy_curve, 'w-', lw=1.2)
ax.set_title(f"Pointwise error $|u_h - u|$  (max={err64:.2e})", fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im3, ax=ax)

# ─── Panel (1,1): convergence ────────────────────────────────────────────────
ax = axes[1, 1]
hs   = [1.0/N for N in Ns]
ax.loglog(hs, errors, 'o-', color='#2c7be5', linewidth=2, markersize=7, label='IIM error')

# Reference O(h²) line through the first point
h_ref = np.array([hs[0], hs[-1]])
ax.loglog(h_ref, errors[0] * (h_ref / hs[0])**2, 'k--', lw=1, label='$O(h^2)$')

# Annotate rates
rates = [np.log2(errors[i-1]/errors[i]) / np.log2(hs[i-1]/hs[i])
         for i in range(1, len(Ns))]
for i, (h_mid, rate) in enumerate(zip([(hs[i]+hs[i+1])*0.5 for i in range(len(Ns)-1)], rates)):
    ax.text(h_mid, (errors[i]*errors[i+1])**0.5 * 1.4,
            f'{rate:.2f}', fontsize=8, ha='center', color='#2c7be5')

ax.set_xlabel('$h$', fontsize=11)
ax.set_ylabel(r'$\|u_h - u\|_\infty$', fontsize=11)
ax.set_title('Convergence rate (log-log)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, which='both', linestyle=':', alpha=0.5)

# Second x-axis showing N
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xscale('log')
ax2.set_xticks(hs)
ax2.set_xticklabels([str(N) for N in Ns], fontsize=8)
ax2.set_xlabel('N', fontsize=9)

# ─── Panel (1,2): CPU time breakdown ─────────────────────────────────────────
ax = axes[1, 2]
Ns_arr = np.array(Ns)

colors = {'label': '#e67e22', 'correct': '#2c7be5', 'solve': '#27ae60'}
labels_legend = {'label': 'domain labeling', 'correct': 'IIM correction', 'solve': 'DST solve'}
bottom = np.zeros(len(Ns))

for key in ['label', 'correct', 'solve']:
    vals = np.array(times[key])
    ax.bar(range(len(Ns)), vals, bottom=bottom,
           color=colors[key], label=labels_legend[key], alpha=0.85)
    bottom += vals

ax.plot(range(len(Ns)), times['total'], 'ko-', markersize=5, linewidth=1.5, label='total')
ax.set_xticks(range(len(Ns)))
ax.set_xticklabels([str(N) for N in Ns])
ax.set_xlabel('N', fontsize=11)
ax.set_ylabel('CPU time (s)', fontsize=11)
ax.set_title('CPU time breakdown (Python)', fontsize=10)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, axis='y', linestyle=':', alpha=0.5)

# Annotate each bar with total time
for i, t in enumerate(times['total']):
    ax.text(i, t + times['total'][-1]*0.01, f'{t:.2f}s', ha='center', fontsize=7)

# ─── layout ──────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.95])

out = "/Users/zhouhan/programs/kfbim/kfbim-recon/scripts/iim_2d_viz.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out}")
plt.show()
