"""
IIM defect-correction with Taylor-expanded correction function.

C(x) = u⁺(x) − u⁻(x) is known only at interface quadrature points
(value + derivatives up to 2nd order).  For each cross-interface bulk
neighbor, we find the nearest interface point q and approximate:

  C(x_nb) ≈ C_q + Cx_q·dx + Cy_q·dy
           + ½Cxx_q·dx² + Cxy_q·dx·dy + ½Cyy_q·dy²

Panels:
  (0,0)  C approximation error at grid nodes (Taylor vs exact)
  (0,1)  Exact solution u (piecewise smooth)
  (0,2)  Computed solution u_h (Taylor-IIM-corrected FFT solve)
  (1,0)  Pointwise error |u_h − u|
  (1,1)  Convergence comparison: exact-C vs Taylor-C
  (1,2)  Distance from grid node to its nearest interface point
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.fft import dstn

kPi = np.pi

# ─── manufactured solution ───────────────────────────────────────────────────

def u_plus(x, y):   return np.sin(kPi*x) * np.sin(kPi*y)
def u_minus(x, y):  return np.sin(2*kPi*x) * np.sin(2*kPi*y)
def f_plus(x, y):   return 2.0*kPi**2 * np.sin(kPi*x) * np.sin(kPi*y)
def f_minus(x, y):  return 8.0*kPi**2 * np.sin(2*kPi*x) * np.sin(2*kPi*y)

def C_fn(x, y):     return u_plus(x, y) - u_minus(x, y)
def Cx_fn(x, y):    return ( kPi*np.cos(kPi*x)*np.sin(kPi*y)
                            - 2*kPi*np.cos(2*kPi*x)*np.sin(2*kPi*y))
def Cy_fn(x, y):    return ( kPi*np.sin(kPi*x)*np.cos(kPi*y)
                            - 2*kPi*np.sin(2*kPi*x)*np.cos(2*kPi*y))
def Cxx_fn(x, y):   return (-kPi**2*np.sin(kPi*x)*np.sin(kPi*y)
                            + 4*kPi**2*np.sin(2*kPi*x)*np.sin(2*kPi*y))
def Cxy_fn(x, y):   return ( kPi**2*np.cos(kPi*x)*np.cos(kPi*y)
                            - 4*kPi**2*np.cos(2*kPi*x)*np.cos(2*kPi*y))
def Cyy_fn(x, y):   return (-kPi**2*np.sin(kPi*x)*np.sin(kPi*y)
                            + 4*kPi**2*np.sin(2*kPi*x)*np.sin(2*kPi*y))

# ─── star geometry ───────────────────────────────────────────────────────────

CX, CY, R_STAR, A_STAR, K_STAR = 0.5, 0.5, 0.28, 0.40, 5

def star_r(th):   return R_STAR * (1.0 + A_STAR * np.cos(K_STAR * th))

def star_contains(x, y):
    rho = np.sqrt((x - CX)**2 + (y - CY)**2)
    return rho < star_r(np.arctan2(y - CY, x - CX))

def star_iface_pts(N_pts):
    th = np.linspace(0, 2*kPi, N_pts, endpoint=False)
    r  = star_r(th)
    return np.column_stack([CX + r*np.cos(th), CY + r*np.sin(th)])

# ─── domain labeling: narrow band + BFS flood-fill ───────────────────────────

def label_grid(pts, iface_pts, h):
    N_pts = len(pts)
    d = np.full(N_pts, np.inf)
    for q in iface_pts:
        dd = np.sqrt(((pts - q)**2).sum(axis=1))
        d = np.minimum(d, dd)

    band_r = 4.0 * h
    labels = np.full(N_pts, -1, dtype=int)

    band_idx = np.where(d < band_r)[0]
    for n in band_idx:
        labels[n] = 1 if star_contains(pts[n,0], pts[n,1]) else 0

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

# ─── nearest interface point for every grid node ─────────────────────────────

def nearest_iface_index(pts, iface_pts):
    """Returns (nearest_idx, min_dist) for each grid point."""
    # iface_pts: (Nq, 2),  pts: (N, 2)
    # Vectorised via broadcasting; fine for Nq, N ~ a few thousand.
    diff = pts[:, None, :] - iface_pts[None, :, :]   # (N, Nq, 2)
    dist = np.sqrt((diff**2).sum(axis=2))             # (N, Nq)
    idx  = dist.argmin(axis=1)                        # (N,)
    return idx, dist[np.arange(len(pts)), idx]

# ─── Taylor-expanded C at bulk node ──────────────────────────────────────────

def taylor_C_at_nodes(pts, iface_pts, C_data):
    """
    C_data: dict with keys 'C','Cx','Cy','Cxx','Cxy','Cyy',
            each a (Nq,) array of values at iface_pts.
    Returns C_approx (N,) and the nearest-point indices + distances.
    """
    idx, dists = nearest_iface_index(pts, iface_pts)

    xq = iface_pts[idx, 0];  yq = iface_pts[idx, 1]
    dx = pts[:, 0] - xq;     dy = pts[:, 1] - yq

    C_approx = (C_data['C'][idx]
                + C_data['Cx'][idx]  * dx
                + C_data['Cy'][idx]  * dy
                + 0.5 * C_data['Cxx'][idx] * dx**2
                + C_data['Cxy'][idx] * dx * dy
                + 0.5 * C_data['Cyy'][idx] * dy**2)
    return C_approx, idx, dists

# ─── IIM corrections ─────────────────────────────────────────────────────────

def iim_correct(f, C, labels, nx, ny, h):
    """Exact-C correction: C evaluated analytically at every node."""
    F = f.copy()
    h2 = h * h
    for n in range(nx * ny):
        i, j = n % nx, n // nx
        if i == 0 or i == nx-1 or j == 0 or j == ny-1: continue
        ln = labels[n]
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni, nj = i+di, j+dj
            if 1 <= ni <= nx-2 and 1 <= nj <= ny-2:
                nb = nj*nx + ni
                lnb = labels[nb]
                if lnb != ln:
                    F[n] += (lnb - ln) * C[nb] / h2
    return F

def iim_correct_taylor(f, C_taylor, labels, nx, ny, h):
    """Taylor-C correction: C_taylor is the approximation at every node."""
    return iim_correct(f, C_taylor, labels, nx, ny, h)

# ─── DST Poisson solver ───────────────────────────────────────────────────────

def solve_poisson(rhs_full, N, h):
    nx = ny = N + 1
    rhs_int = np.zeros((N-1, N-1))
    for j in range(1, N):
        for i in range(1, N):
            rhs_int[j-1, i-1] = rhs_full[j*nx + i]
    p   = np.arange(1, N)
    lam = (2*np.cos(p*kPi/N) - 2.0) / h**2
    LAM = lam[:, None] + lam[None, :]
    rhs_hat = dstn(rhs_int, type=1, norm='ortho')
    u_int   = dstn(rhs_hat / LAM, type=1, norm='ortho')
    u_full  = np.zeros(nx * ny)
    for j in range(1, N):
        for i in range(1, N):
            u_full[j*nx + i] = u_int[j-1, i-1]
    return u_full

# ─── full pipeline ────────────────────────────────────────────────────────────

def iim_solve_both(N):
    h   = 1.0 / N
    nx  = ny = N + 1
    n_total = nx * ny

    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XI, YI = np.meshgrid(xs, ys)
    pts = np.column_stack([XI.ravel(), YI.ravel()])

    iface_pts = star_iface_pts(4 * N)

    labels = label_grid(pts, iface_pts, h)

    # Build f, exact u, exact C at every grid node
    f_arr   = np.zeros(n_total)
    u_exact = np.zeros(n_total)
    C_exact = np.zeros(n_total)
    for n in range(n_total):
        x, y = pts[n]
        i, j = n % nx, n // nx
        bdy  = (i == 0 or i == nx-1 or j == 0 or j == ny-1)
        lbl  = labels[n]
        u_exact[n] = u_minus(x,y) if lbl == 1 else u_plus(x,y)
        C_exact[n] = C_fn(x, y)
        if not bdy:
            f_arr[n] = f_minus(x,y) if lbl == 1 else f_plus(x,y)

    # C and its derivatives at interface quadrature points only
    xq, yq = iface_pts[:,0], iface_pts[:,1]
    C_data = {
        'C'  : C_fn  (xq, yq),
        'Cx' : Cx_fn (xq, yq),
        'Cy' : Cy_fn (xq, yq),
        'Cxx': Cxx_fn(xq, yq),
        'Cxy': Cxy_fn(xq, yq),
        'Cyy': Cyy_fn(xq, yq),
    }

    # Taylor-approximate C at every grid node
    C_taylor, near_idx, near_dist = taylor_C_at_nodes(pts, iface_pts, C_data)

    # Exact-C solve
    F_exact  = iim_correct(f_arr, C_exact,  labels, nx, ny, h)
    u_exact_solve = solve_poisson(-F_exact, N, h)

    # Taylor-C solve
    F_taylor = iim_correct_taylor(f_arr, C_taylor, labels, nx, ny, h)
    u_taylor_solve = solve_poisson(-F_taylor, N, h)

    def max_interior_err(u_sol):
        err = 0.0
        for j in range(1, N):
            for i in range(1, N):
                err = max(err, abs(u_sol[j*nx+i] - u_exact[j*nx+i]))
        return err

    return dict(
        pts=pts, nx=nx, ny=ny, h=h, labels=labels,
        iface_pts=iface_pts, near_idx=near_idx, near_dist=near_dist,
        C_exact=C_exact, C_taylor=C_taylor,
        u_exact=u_exact,
        u_exact_solve=u_exact_solve,
        u_taylor_solve=u_taylor_solve,
        err_exact=max_interior_err(u_exact_solve),
        err_taylor=max_interior_err(u_taylor_solve),
    )

# ============================================================================
# Compute
# ============================================================================

print("Computing N=64 solution …")
r64 = iim_solve_both(64)
print(f"  exact-C  max_err = {r64['err_exact']:.4e}")
print(f"  Taylor-C max_err = {r64['err_taylor']:.4e}")

print("Computing convergence table …")
Ns = [16, 32, 64, 128, 256]
errs_exact  = []
errs_taylor = []
for N in Ns:
    r = iim_solve_both(N)
    errs_exact.append(r['err_exact'])
    errs_taylor.append(r['err_taylor'])
    print(f"  N={N:4d}  exact={r['err_exact']:.4e}  taylor={r['err_taylor']:.4e}")

# ============================================================================
# Figure
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "IIM with Taylor-expanded correction function\n"
    r"$C(x_\mathrm{nb}) \approx C_q + \nabla C_q \cdot \Delta x "
    r"+ \frac{1}{2} \Delta x^T H_q \Delta x$   "
    r"(value + 2nd-order derivatives at interface points only)",
    fontsize=11, fontweight='bold'
)

nx64, ny64 = r64['nx'], r64['ny']
pts64      = r64['pts']
iface_pts64 = r64['iface_pts']
xs = np.linspace(0, 1, nx64)
ys = np.linspace(0, 1, ny64)

th_c = np.linspace(0, 2*kPi, 800)
r_c  = star_r(th_c)
cx_c = CX + r_c*np.cos(th_c)
cy_c = CY + r_c*np.sin(th_c)

def to_2d(arr, nx, ny): return arr.reshape(ny, nx)

# ─── Panel (0,0): C approximation error at every grid node ───────────────────
ax = axes[0, 0]
C_err = np.abs(r64['C_taylor'] - r64['C_exact'])
C_err_2d = to_2d(C_err, nx64, ny64)

im0 = ax.pcolormesh(xs, ys, C_err_2d, shading='auto', cmap='hot_r',
                    norm=mcolors.LogNorm(
                        vmin=max(C_err[C_err>0].min(), 1e-10),
                        vmax=C_err.max()))
ax.plot(cx_c, cy_c, 'w-', lw=1.2)
ax.scatter(iface_pts64[:,0], iface_pts64[:,1], s=2, c='cyan', zorder=3,
           label=f'{len(iface_pts64)} interface pts')
ax.set_title(r"$|C_\mathrm{Taylor} - C_\mathrm{exact}|$ at grid nodes  (N=64)", fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im0, ax=ax)
ax.legend(fontsize=7, loc='upper right')

# ─── Panel (0,1): exact solution ─────────────────────────────────────────────
ax = axes[0, 1]
Z_exact = to_2d(r64['u_exact'], nx64, ny64)
im1 = ax.pcolormesh(xs, ys, Z_exact, shading='auto', cmap='RdBu_r')
ax.plot(cx_c, cy_c, 'k-', lw=1.2)
ax.set_title("Exact solution $u$", fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im1, ax=ax)

# ─── Panel (0,2): Taylor-C computed solution ─────────────────────────────────
ax = axes[0, 2]
Z_taylor = to_2d(r64['u_taylor_solve'], nx64, ny64)
im2 = ax.pcolormesh(xs, ys, Z_taylor, shading='auto', cmap='RdBu_r',
                    vmin=Z_exact.min(), vmax=Z_exact.max())
ax.plot(cx_c, cy_c, 'k-', lw=1.2)
ax.set_title("Computed solution $u_h$  (Taylor-C IIM)", fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im2, ax=ax)

# ─── Panel (1,0): pointwise error |u_h - u|  (Taylor-C) ─────────────────────
ax = axes[1, 0]
err_field = np.abs(r64['u_taylor_solve'] - r64['u_exact'])
for n in range(nx64 * ny64):
    i, j = n % nx64, n // nx64
    if i == 0 or i == nx64-1 or j == 0 or j == ny64-1:
        err_field[n] = 0.0
Z_err = to_2d(err_field, nx64, ny64)
im3 = ax.pcolormesh(xs, ys, Z_err, shading='auto', cmap='hot_r',
                    norm=mcolors.LogNorm(
                        vmin=max(Z_err.max()*1e-4, 1e-10),
                        vmax=max(Z_err.max(), 1e-9)))
ax.plot(cx_c, cy_c, 'w-', lw=1.2)
ax.set_title(f"Pointwise error $|u_h - u|$  (Taylor-C, max={r64['err_taylor']:.2e})", fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im3, ax=ax)

# ─── Panel (1,1): convergence comparison ─────────────────────────────────────
ax = axes[1, 1]
hs = [1.0/N for N in Ns]

ax.loglog(hs, errs_exact,  'o-', color='#2c7be5', lw=2, ms=7, label='exact C (analytical)')
ax.loglog(hs, errs_taylor, 's--', color='#e74c3c', lw=2, ms=7, label='Taylor C (interface pts only)')

h_ref = np.array([hs[0], hs[-1]])
ax.loglog(h_ref, errs_exact[0]*(h_ref/hs[0])**2, 'k:', lw=1, label='$O(h^2)$')

# Annotate Taylor rates
for i in range(1, len(Ns)):
    rate = np.log2(errs_taylor[i-1]/errs_taylor[i])
    h_mid = (hs[i-1]*hs[i])**0.5
    ax.text(h_mid, (errs_taylor[i-1]*errs_taylor[i])**0.5 * 1.5,
            f'{rate:.2f}', fontsize=8, ha='center', color='#e74c3c')

ax.set_xlabel('$h$', fontsize=11)
ax.set_ylabel(r'$\|u_h - u\|_\infty$', fontsize=11)
ax.set_title('Convergence: exact-C vs Taylor-C', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, which='both', linestyle=':', alpha=0.5)

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim()); ax2.set_xscale('log')
ax2.set_xticks(hs); ax2.set_xticklabels([str(N) for N in Ns], fontsize=8)
ax2.set_xlabel('N', fontsize=9)

# ─── Panel (1,2): nearest interface point distance at grid nodes ──────────────
ax = axes[1, 2]
dist_2d = to_2d(r64['near_dist'], nx64, ny64)
im5 = ax.pcolormesh(xs, ys, dist_2d, shading='auto', cmap='viridis')
ax.plot(cx_c, cy_c, 'w-', lw=1.2)
ax.scatter(iface_pts64[:,0], iface_pts64[:,1], s=2, c='red', zorder=3)
ax.set_title("Distance to nearest interface point  (N=64)", fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
cb = plt.colorbar(im5, ax=ax)
cb.set_label('distance', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.93])

out = "python/iim_2d_taylor_viz.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out}")
plt.show()
