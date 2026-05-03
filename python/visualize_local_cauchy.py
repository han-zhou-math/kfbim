"""
Panel-based local Cauchy solver for C = u⁺ − u⁻.

At each 3-point-per-panel interface node, a 6×6 collocation system
(3 Dirichlet + 2 Neumann + 1 PDE row) is solved for a quadratic
polynomial approximation of C and its derivatives.

Layout:
  (0,0)  Interface scatter: |C − C_true|  at N=64 panels
  (0,1)  Interface scatter: |Cx − Cx_true| and |Cy − Cy_true|
  (0,2)  Interface scatter: |Cxx|, |Cyy|, |Cxy| errors
  (1,0)  Gradient convergence: Cx, Cy  (O(h²))
  (1,1)  2nd-derivative convergence: Cxx, Cyy, Cxy  (O(h))
  (1,2)  IIM plug-in: pointwise solution error — panel Cauchy vs exact Taylor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.fft import dstn

kPi = np.pi

# ── manufactured solution ─────────────────────────────────────────────────────

def u_plus (x, y): return np.sin(kPi*x)*np.sin(kPi*y)
def u_minus(x, y): return np.sin(2*kPi*x)*np.sin(2*kPi*y)
def f_plus (x, y): return 2.0*kPi**2*np.sin(kPi*x)*np.sin(kPi*y)
def f_minus(x, y): return 8.0*kPi**2*np.sin(2*kPi*x)*np.sin(2*kPi*y)

def C_fn  (x, y): return u_plus(x,y) - u_minus(x,y)
def Cx_fn (x, y): return (  kPi*np.cos(kPi*x)*np.sin(kPi*y)
                           - 2*kPi*np.cos(2*kPi*x)*np.sin(2*kPi*y))
def Cy_fn (x, y): return (  kPi*np.sin(kPi*x)*np.cos(kPi*y)
                           - 2*kPi*np.sin(2*kPi*x)*np.cos(2*kPi*y))
def Cxx_fn(x, y): return ( -kPi**2*np.sin(kPi*x)*np.sin(kPi*y)
                           + 4*kPi**2*np.sin(2*kPi*x)*np.sin(2*kPi*y))
def Cxy_fn(x, y): return (  kPi**2*np.cos(kPi*x)*np.cos(kPi*y)
                           - 4*kPi**2*np.cos(2*kPi*x)*np.cos(2*kPi*y))
def Cyy_fn(x, y): return ( -kPi**2*np.sin(kPi*x)*np.sin(kPi*y)
                           + 4*kPi**2*np.sin(2*kPi*x)*np.sin(2*kPi*y))

# ── star geometry ──────────────────────────────────────────────────────────────

CX, CY, R_STAR, A_STAR, K_STAR = 0.5, 0.5, 0.28, 0.40, 5

kGL_s = np.array([-0.7745966692414834, 0.0, +0.7745966692414834])
kGL_w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

def make_star_panels(N_panels):
    """Build (pts, nml, wts) for a star interface with 3 GL points per panel."""
    Nq  = 3 * N_panels
    pts = np.zeros((Nq, 2))
    nml = np.zeros((Nq, 2))
    wts = np.zeros(Nq)
    dth = 2.0 * kPi / N_panels
    q   = 0
    for p in range(N_panels):
        th_mid   = (p + 0.5) * dth
        half_dth = 0.5 * dth
        for i in range(3):
            th   = th_mid + half_dth * kGL_s[i]
            r    = R_STAR * (1.0 + A_STAR * np.cos(K_STAR * th))
            drdt = -R_STAR * A_STAR * K_STAR * np.sin(K_STAR * th)
            pts[q, 0] = CX + r * np.cos(th)
            pts[q, 1] = CY + r * np.sin(th)
            tx   = drdt * np.cos(th) - r * np.sin(th)
            ty   = drdt * np.sin(th) + r * np.cos(th)
            tlen = np.hypot(tx, ty)
            nml[q, 0] =  ty / tlen
            nml[q, 1] = -tx / tlen
            wts[q]    = kGL_w[i] * half_dth * tlen
            q += 1
    return pts, nml, wts

def star_curve(N=800):
    th = np.linspace(0, 2*kPi, N)
    r  = R_STAR * (1.0 + A_STAR * np.cos(K_STAR * th))
    return CX + r*np.cos(th), CY + r*np.sin(th)

def star_contains(x, y):
    rho = np.hypot(x - CX, y - CY)
    th  = np.arctan2(y - CY, x - CX)
    return rho < R_STAR * (1.0 + A_STAR * np.cos(K_STAR * th))

# ── panel Cauchy solver ────────────────────────────────────────────────────────

def solve_local_6x6(bdry1, a, bdry2, bdry2_nml, b, bulk, Lu, center, kappa, h):
    """Solve the 6×6 collocation system for one interface node."""
    h2     = h * h
    kap_h2 = kappa * h2
    A   = np.zeros((6, 6))
    rhs = np.zeros(6)

    # Rows 0-2: Dirichlet
    for l in range(3):
        dx = (bdry1[l, 0] - center[0]) / h
        dy = (bdry1[l, 1] - center[1]) / h
        A[l] = [1.0, dx, dy, 0.5*dx*dx, 0.5*dy*dy, dx*dy]
        rhs[l] = a[l]

    # Rows 3-4: Neumann (scaled by h)
    for l in range(2):
        dx = (bdry2[l, 0] - center[0]) / h
        dy = (bdry2[l, 1] - center[1]) / h
        nx = bdry2_nml[l, 0]
        ny = bdry2_nml[l, 1]
        A[3+l] = [0.0, nx, ny, dx*nx, dy*ny, nx*dy + dx*ny]
        rhs[3+l] = b[l] * h

    # Row 5: PDE (-Δ + κ) (scaled by h²)
    dx = (bulk[0] - center[0]) / h
    dy = (bulk[1] - center[1]) / h
    A[5] = [kap_h2, kap_h2*dx, kap_h2*dy,
            kap_h2*0.5*dx*dx - 1.0,
            kap_h2*0.5*dy*dy - 1.0,
            kap_h2*dx*dy]
    rhs[5] = Lu * h2

    # Column-norm preconditioning
    col_scale        = np.linalg.norm(A, axis=0)
    col_scale[col_scale < 1e-14] = 1.0
    c_raw, *_        = np.linalg.lstsq(A / col_scale, rhs, rcond=None)
    c_raw           /= col_scale

    return np.array([c_raw[0],
                     c_raw[1] / h,  c_raw[2] / h,
                     c_raw[3] / h2, c_raw[4] / h2, c_raw[5] / h2])
    # indices: 0=C, 1=Cx, 2=Cy, 3=Cxx, 4=Cyy, 5=Cxy

def laplace_panel_cauchy(pts, nml, wts, a, b, Lu, N_panels, kappa=0.0):
    """Solve at every interface node; returns (C, Cx, Cy, Cxx, Cyy, Cxy)."""
    Nq  = 3 * N_panels
    out = [np.zeros(Nq) for _ in range(6)]  # C, Cx, Cy, Cxx, Cyy, Cxy

    for p in range(N_panels):
        g  = [3*p, 3*p+1, 3*p+2]
        h  = wts[g[0]] + wts[g[1]] + wts[g[2]]

        bdry1     = pts[g]
        av        = a[g]
        bdry2     = pts[[g[0], g[2]]]
        bdry2_nml = nml[[g[0], g[2]]]
        bv        = b[[g[0], g[2]]]

        bulk    = pts[g[1]]   # panel midpoint — PDE collocation point
        Lu_bulk = Lu[g[1]]

        for gi in g:
            center = pts[gi]
            c      = solve_local_6x6(bdry1, av, bdry2, bdry2_nml, bv,
                                     bulk, Lu_bulk, center, kappa, h)
            for k in range(6):
                out[k][gi] = c[k]

    return tuple(out)  # (C, Cx, Cy, Cxx, Cyy, Cxy)

def run_panel_cauchy(N_panels):
    pts, nml, wts = make_star_panels(N_panels)
    x, y  = pts[:, 0], pts[:, 1]
    nx, ny = nml[:, 0], nml[:, 1]
    a  =  C_fn(x, y)
    b  =  Cx_fn(x, y)*nx + Cy_fn(x, y)*ny
    Lu =  f_plus(x, y) - f_minus(x, y)
    C, Cx, Cy, Cxx, Cyy, Cxy = laplace_panel_cauchy(pts, nml, wts, a, b, Lu, N_panels)
    errors = dict(
        eC   = np.max(np.abs(C   - C_fn  (x, y))),
        eCx  = np.max(np.abs(Cx  - Cx_fn (x, y))),
        eCy  = np.max(np.abs(Cy  - Cy_fn (x, y))),
        eCxx = np.max(np.abs(Cxx - Cxx_fn(x, y))),
        eCyy = np.max(np.abs(Cyy - Cyy_fn(x, y))),
        eCxy = np.max(np.abs(Cxy - Cxy_fn(x, y))),
    )
    return pts, nml, wts, (C, Cx, Cy, Cxx, Cyy, Cxy), errors

# ── IIM utilities ─────────────────────────────────────────────────────────────

def label_grid(pts_grid, iface_pts, h):
    """BFS domain labeling: 1 = inside star, 0 = outside."""
    from collections import deque
    N_grid = len(pts_grid)
    dist   = np.full(N_grid, np.inf)
    for q in iface_pts:
        d = np.hypot(pts_grid[:, 0] - q[0], pts_grid[:, 1] - q[1])
        np.minimum(dist, d, out=dist)

    nx = ny = int(round(1.0 / h)) + 1
    labels  = np.full(N_grid, -1, dtype=int)
    band    = np.where(dist < 4.0 * h)[0]
    for n in band:
        labels[n] = 1 if star_contains(pts_grid[n, 0], pts_grid[n, 1]) else 0

    queue = deque(band)
    while queue:
        n   = queue.popleft()
        lbl = labels[n]
        i, j = n % nx, n // nx
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni2, nj2 = i+di, j+dj
            if 0 <= ni2 < nx and 0 <= nj2 < ny:
                nb = nj2*nx + ni2
                if labels[nb] == -1:
                    labels[nb] = lbl
                    queue.append(nb)
    labels[labels == -1] = 0
    return labels

def nearest_iface_index(pts_grid, iface_pts):
    diff = pts_grid[:, None, :] - iface_pts[None, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    idx  = dist.argmin(axis=1)
    return idx, dist[np.arange(len(pts_grid)), idx]

def taylor_C_at_nodes(pts_grid, iface_pts, C_data):
    idx, _ = nearest_iface_index(pts_grid, iface_pts)
    dx = pts_grid[:, 0] - iface_pts[idx, 0]
    dy = pts_grid[:, 1] - iface_pts[idx, 1]
    return (C_data['C'][idx]
            + C_data['Cx'][idx]*dx + C_data['Cy'][idx]*dy
            + 0.5*C_data['Cxx'][idx]*dx**2
            + C_data['Cxy'][idx]*dx*dy
            + 0.5*C_data['Cyy'][idx]*dy**2)

def iim_correct(f, C_nodes, labels, nx, ny, h):
    F  = f.copy()
    h2 = h * h
    for n in range(nx * ny):
        i, j = n % nx, n // nx
        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            continue
        ln = labels[n]
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni2, nj2 = i+di, j+dj
            if 1 <= ni2 <= nx-2 and 1 <= nj2 <= ny-2:
                nb  = nj2*nx + ni2
                lnb = labels[nb]
                if lnb != ln:
                    F[n] += (lnb - ln) * C_nodes[nb] / h2
    return F

def solve_poisson(rhs_full, N, h):
    nx = ny = N + 1
    rhs_int = rhs_full.reshape(ny, nx)[1:N, 1:N].copy()
    p   = np.arange(1, N)
    lam = (2*np.cos(p*kPi/N) - 2.0) / h**2
    LAM = lam[:, None] + lam[None, :]
    u_int = dstn(dstn(rhs_int, type=1, norm='ortho') / LAM, type=1, norm='ortho')
    u_full = np.zeros(nx * ny)
    u_full.reshape(ny, nx)[1:N, 1:N] = u_int
    return u_full

def iim_solve_panel_vs_exact(N):
    h  = 1.0 / N
    nx = ny = N + 1

    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XI, YI   = np.meshgrid(xs, ys)
    pts_grid = np.column_stack([XI.ravel(), YI.ravel()])

    # Panel interface (3N pts)
    iface_pts, iface_nml, iface_wts = make_star_panels(N)
    Nq = len(iface_pts)

    labels  = label_grid(pts_grid, iface_pts, h)

    f_arr   = np.zeros(nx * ny)
    u_exact = np.zeros(nx * ny)
    for n in range(nx * ny):
        x, y = pts_grid[n]
        i, j = n % nx, n // nx
        if i == 0 or i == nx-1 or j == 0 or j == ny-1:
            u_exact[n] = 0.0
            continue
        lbl        = labels[n]
        u_exact[n] = u_minus(x, y) if lbl == 1 else u_plus(x, y)
        f_arr[n]   = f_minus(x, y) if lbl == 1 else f_plus(x, y)

    # Analytical C at interface
    xq, yq = iface_pts[:, 0], iface_pts[:, 1]
    C_data_exact = dict(
        C   = C_fn  (xq, yq),
        Cx  = Cx_fn (xq, yq),
        Cy  = Cy_fn (xq, yq),
        Cxx = Cxx_fn(xq, yq),
        Cxy = Cxy_fn(xq, yq),
        Cyy = Cyy_fn(xq, yq),
    )

    # Panel Cauchy C at interface
    nx_q = iface_nml[:, 0]
    ny_q = iface_nml[:, 1]
    a_q  = C_fn(xq, yq)
    b_q  = Cx_fn(xq, yq)*nx_q + Cy_fn(xq, yq)*ny_q
    Lu_q = f_plus(xq, yq) - f_minus(xq, yq)
    C_pc, Cx_pc, Cy_pc, Cxx_pc, Cyy_pc, Cxy_pc = laplace_panel_cauchy(
        iface_pts, iface_nml, iface_wts, a_q, b_q, Lu_q, N)
    C_data_panel = dict(C=C_pc, Cx=Cx_pc, Cy=Cy_pc, Cxx=Cxx_pc, Cxy=Cxy_pc, Cyy=Cyy_pc)

    def solve_with(C_data_iface):
        C_nodes = taylor_C_at_nodes(pts_grid, iface_pts, C_data_iface)
        F = iim_correct(f_arr, C_nodes, labels, nx, ny, h)
        u = solve_poisson(-F, N, h)
        mask = np.zeros(nx*ny, bool)
        mask.reshape(ny, nx)[1:N, 1:N] = True
        err = np.max(np.abs((u - u_exact)[mask]))
        return u, err

    u_exact_solve, err_exact = solve_with(C_data_exact)
    u_panel_solve, err_panel = solve_with(C_data_panel)

    return dict(
        pts_grid=pts_grid, nx=nx, ny=ny, h=h,
        labels=labels, iface_pts=iface_pts,
        u_exact=u_exact,
        u_exact_solve=u_exact_solve, err_exact=err_exact,
        u_panel_solve=u_panel_solve, err_panel=err_panel,
    )

# ── compute ───────────────────────────────────────────────────────────────────

N_VIZ = 64

print(f"Running panel Cauchy solver at N_panels={N_VIZ} …")
pts64, nml64, wts64, derivs64, errs64 = run_panel_cauchy(N_VIZ)
C64, Cx64, Cy64, Cxx64, Cyy64, Cxy64 = derivs64
x64, y64 = pts64[:, 0], pts64[:, 1]

print("Convergence study …")
Nps  = [16, 32, 64, 128]
keys = ['eC', 'eCx', 'eCy', 'eCxx', 'eCyy', 'eCxy']
table = {k: [] for k in keys}
for Np in Nps:
    _, _, _, _, err = run_panel_cauchy(Np)
    for k in keys:
        table[k].append(err[k])

print(f"  {'N':>6}  {'eCx':>10}  rate  {'eCy':>10}  rate  {'eCxx':>10}  rate")
for i, Np in enumerate(Nps):
    if i == 0:
        print(f"  {Np:>6}  {table['eCx'][0]:>10.4e}     —  {table['eCy'][0]:>10.4e}     —  {table['eCxx'][0]:>10.4e}     —")
    else:
        rx  = np.log2(table['eCx' ][i-1]/table['eCx' ][i])
        ry  = np.log2(table['eCy' ][i-1]/table['eCy' ][i])
        rxx = np.log2(table['eCxx'][i-1]/table['eCxx'][i])
        print(f"  {Np:>6}  {table['eCx'][i]:>10.4e}  {rx:.2f}  {table['eCy'][i]:>10.4e}  {ry:.2f}  {table['eCxx'][i]:>10.4e}  {rxx:.2f}")

print(f"\nRunning IIM plug-in at N={N_VIZ} …")
r_iim = iim_solve_panel_vs_exact(N_VIZ)
print(f"  exact-Taylor err  = {r_iim['err_exact']:.4e}")
print(f"  panel-Cauchy err  = {r_iim['err_panel']:.4e}  (ratio {r_iim['err_panel']/r_iim['err_exact']:.2f}×)")

# ── figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "Panel-based collocation Cauchy solver   "
    r"$-\Delta C = f^+ - f^-$,  $C = a$,  $\partial_n C = b$ on $\Gamma$"
    "\n"
    r"Basis: $\{1,\,dx,\,dy,\,\frac{1}{2}dx^2,\,\frac{1}{2}dy^2,\,dx\,dy\}$"
    r"  (6$\times$6 per-panel collocation, 3 Dirichlet + 2 Neumann + 1 PDE row)",
    fontsize=11, fontweight='bold'
)

cx_c, cy_c = star_curve()

def scatter_iface(ax, pts, vals, label, cmap='hot_r', log=True, **kw):
    vmin = vals[vals > 0].min() if log and (vals > 0).any() else vals.min()
    vmax = vals.max()
    norm = mcolors.LogNorm(vmin=max(vmin, 1e-14), vmax=max(vmax, 1e-13)) if log else None
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, s=18, cmap=cmap, norm=norm, zorder=3, **kw)
    ax.plot(cx_c, cy_c, 'k-', lw=1.0, alpha=0.4)
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(label, fontsize=9)
    return sc

# ─── (0,0): C error at interface ─────────────────────────────────────────────
ax = axes[0, 0]
eC_pts = np.abs(C64 - C_fn(x64, y64))
scatter_iface(ax, pts64, eC_pts, r'$|C - C_\mathrm{true}|$')
ax.set_title(fr'$|C - C_\mathrm{{true}}|$ at interface  (N={N_VIZ} panels,  max={errs64["eC"]:.2e})', fontsize=10)

# ─── (0,1): Cx and Cy errors ──────────────────────────────────────────────────
ax = axes[0, 1]
eCx_pts = np.abs(Cx64 - Cx_fn(x64, y64))
eCy_pts = np.abs(Cy64 - Cy_fn(x64, y64))
grad_err = np.sqrt(eCx_pts**2 + eCy_pts**2)
scatter_iface(ax, pts64, grad_err, r'$\|\nabla C_\mathrm{err}\|_2$', cmap='plasma')
ax.set_title(fr'Gradient error $\|\nabla(C - C_\mathrm{{true}})\|_2$'
             f'\n(N={N_VIZ}, Cx max={errs64["eCx"]:.2e}, Cy max={errs64["eCy"]:.2e})', fontsize=10)

# ─── (0,2): 2nd-derivative errors ────────────────────────────────────────────
ax = axes[0, 2]
eCxx_pts = np.abs(Cxx64 - Cxx_fn(x64, y64))
eCyy_pts = np.abs(Cyy64 - Cyy_fn(x64, y64))
eCxy_pts = np.abs(Cxy64 - Cxy_fn(x64, y64))
d2_err = eCxx_pts + eCyy_pts + eCxy_pts
scatter_iface(ax, pts64, d2_err, r'$|C_{xx}|+|C_{yy}|+|C_{xy}|$ error', cmap='inferno')
ax.set_title(fr'2nd-derivative error sum  (N={N_VIZ}'
             f'\nCxx max={errs64["eCxx"]:.2e}, Cyy max={errs64["eCyy"]:.2e})', fontsize=10)

# ─── (1,0): gradient convergence ─────────────────────────────────────────────
ax = axes[1, 0]
hs = [1.0/Np for Np in Nps]
ax.loglog(hs, table['eCx'], 'o-',  color='#2c7be5', lw=2, ms=7, label=r'$|C_x - C_{x,\mathrm{true}}|_\infty$')
ax.loglog(hs, table['eCy'], 's--', color='#e74c3c', lw=2, ms=7, label=r'$|C_y - C_{y,\mathrm{true}}|_\infty$')
h_ref = np.array([hs[0], hs[-1]])
ref_cx = table['eCx'][0] * (np.array(h_ref) / hs[0])**2
ax.loglog(h_ref, ref_cx, 'k:', lw=1.2, label='$O(h^2)$')
for i in range(1, len(Nps)):
    rate = np.log2(table['eCx'][i-1] / table['eCx'][i])
    hm   = (hs[i-1]*hs[i])**0.5
    em   = (table['eCx'][i-1]*table['eCx'][i])**0.5
    ax.text(hm, em*1.6, f'{rate:.2f}', fontsize=8, ha='center', color='#2c7be5')
for i in range(1, len(Nps)):
    rate = np.log2(table['eCy'][i-1] / table['eCy'][i])
    hm   = (hs[i-1]*hs[i])**0.5
    em   = (table['eCy'][i-1]*table['eCy'][i])**0.5
    ax.text(hm, em*0.6, f'{rate:.2f}', fontsize=8, ha='center', color='#e74c3c')
ax.set_xlabel('panel arc-length $h$', fontsize=11)
ax.set_ylabel(r'$\max$ error', fontsize=11)
ax.set_title('Gradient convergence (formal order 2)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.5)
ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xscale('log')
ax2.set_xticks(hs); ax2.set_xticklabels([str(Np) for Np in Nps], fontsize=8)
ax2.set_xlabel('N panels', fontsize=9)

# ─── (1,1): 2nd-derivative convergence ───────────────────────────────────────
ax = axes[1, 1]
ax.loglog(hs, table['eCxx'], 'o-',  color='#2c7be5', lw=2, ms=6, label=r'$C_{xx}$')
ax.loglog(hs, table['eCyy'], 's--', color='#e74c3c', lw=2, ms=6, label=r'$C_{yy}$')
ax.loglog(hs, table['eCxy'], '^:', color='#27ae60', lw=2, ms=6, label=r'$C_{xy}$')
ref_cxx = table['eCxx'][0] * (np.array(h_ref) / hs[0])**1
ax.loglog(h_ref, ref_cxx, 'k:', lw=1.2, label='$O(h)$')
for key, color in zip(['eCxx', 'eCyy', 'eCxy'], ['#2c7be5', '#e74c3c', '#27ae60']):
    for i in range(1, len(Nps)):
        rate = np.log2(table[key][i-1] / table[key][i])
        hm   = (hs[i-1]*hs[i])**0.5
        em   = (table[key][i-1]*table[key][i])**0.5
        ax.text(hm, em*1.4, f'{rate:.2f}', fontsize=7, ha='center', color=color)
ax.set_xlabel('panel arc-length $h$', fontsize=11)
ax.set_ylabel(r'$\max$ error', fontsize=11)
ax.set_title('2nd-derivative convergence (formal order 1)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.5)
ax3 = ax.twiny(); ax3.set_xlim(ax.get_xlim()); ax3.set_xscale('log')
ax3.set_xticks(hs); ax3.set_xticklabels([str(Np) for Np in Nps], fontsize=8)
ax3.set_xlabel('N panels', fontsize=9)

# ─── (1,2): IIM plug-in solution error ───────────────────────────────────────
ax = axes[1, 2]
ni, nj = r_iim['nx'], r_iim['ny']
xs_iim = np.linspace(0, 1, ni)
ys_iim = np.linspace(0, 1, nj)

err_panel_field = np.abs(r_iim['u_panel_solve'] - r_iim['u_exact'])
err_exact_field = np.abs(r_iim['u_exact_solve'] - r_iim['u_exact'])
# zero out boundary
for arr in [err_panel_field, err_exact_field]:
    arr.reshape(nj, ni)[[0, -1], :] = 0
    arr.reshape(nj, ni)[:, [0, -1]] = 0

vmax = max(err_panel_field.max(), err_exact_field.max(), 1e-10)
vmin = max(err_panel_field[err_panel_field > 0].min(), 1e-10) if (err_panel_field > 0).any() else 1e-10
norm_err = mcolors.LogNorm(vmin=vmin, vmax=vmax)

im = ax.pcolormesh(xs_iim, ys_iim, err_panel_field.reshape(nj, ni),
                   shading='auto', cmap='hot_r', norm=norm_err)
ax.plot(cx_c, cy_c, 'w-', lw=1.2)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im, ax=ax)
ax.set_title(
    f'IIM plug-in: pointwise error $|u_h - u|$  (N={N_VIZ})\n'
    f'panel Cauchy = {r_iim["err_panel"]:.2e}'
    f',  exact Taylor = {r_iim["err_exact"]:.2e}'
    f'  (ratio {r_iim["err_panel"]/r_iim["err_exact"]:.2f}×)',
    fontsize=10
)

plt.tight_layout(rect=[0, 0, 1, 0.93])

out = "python/local_cauchy_viz.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out}")
plt.show()
