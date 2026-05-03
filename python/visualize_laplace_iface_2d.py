"""
Full end-to-end visualization: -Δu = f (piecewise smooth) on [0,1]²,
interface jump conditions [u]=a and [∂_n u]=b on star-shaped Γ, u=0 on ∂Ω.

Pipeline (mirrors test_laplace_iface_2d.cpp):
  1. Jump data a, b, [f] at panel Gauss points.
  2. Panel Cauchy solver → C, Cx, Cy, Cxx, Cyy, Cxy at Gauss points.
  3. IIM defect correction: for each cross-interface node pair, add
         (label_nb − label_n) × C_Taylor(x_nb) / h²  to RHS.
  4. FFT Poisson solve.

Layout (2×3):
  (0,0)  Exact solution u (piecewise smooth, jump at Γ)
  (0,1)  Computed solution u_h  (N=64, panel Cauchy + FFT)
  (0,2)  Pointwise error |u_h − u|
  (1,0)  IIM RHS correction field  (shows where defect is applied)
  (1,1)  Convergence: max-norm error vs N  (log-log, O(h²) reference)
  (1,2)  C = u⁺ − u⁻ at interface Gauss points (panel Cauchy vs exact)
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
def Cx_fn (x, y): return  kPi*np.cos(kPi*x)*np.sin(kPi*y) - 2*kPi*np.cos(2*kPi*x)*np.sin(2*kPi*y)
def Cy_fn (x, y): return  kPi*np.sin(kPi*x)*np.cos(kPi*y) - 2*kPi*np.sin(2*kPi*x)*np.cos(2*kPi*y)
def Cxx_fn(x, y): return -kPi**2*np.sin(kPi*x)*np.sin(kPi*y) + 4*kPi**2*np.sin(2*kPi*x)*np.sin(2*kPi*y)
def Cxy_fn(x, y): return  kPi**2*np.cos(kPi*x)*np.cos(kPi*y) - 4*kPi**2*np.cos(2*kPi*x)*np.cos(2*kPi*y)
def Cyy_fn(x, y): return -kPi**2*np.sin(kPi*x)*np.sin(kPi*y) + 4*kPi**2*np.sin(2*kPi*x)*np.sin(2*kPi*y)

# ── star geometry ─────────────────────────────────────────────────────────────

CX, CY, R_STAR, A_STAR, K_STAR = 0.5, 0.5, 0.28, 0.40, 5

kGL_s = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
kGL_w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

def make_star_panels(N_panels):
    Nq  = 3 * N_panels
    pts = np.zeros((Nq, 2))
    nml = np.zeros((Nq, 2))
    wts = np.zeros(Nq)
    dth = 2*kPi / N_panels
    q   = 0
    for p in range(N_panels):
        th_mid = (p + 0.5) * dth
        hd     = 0.5 * dth
        for i in range(3):
            th   = th_mid + hd * kGL_s[i]
            r    = R_STAR * (1 + A_STAR * np.cos(K_STAR * th))
            drdt = -R_STAR * A_STAR * K_STAR * np.sin(K_STAR * th)
            pts[q, 0] = CX + r * np.cos(th)
            pts[q, 1] = CY + r * np.sin(th)
            tx = drdt * np.cos(th) - r * np.sin(th)
            ty = drdt * np.sin(th) + r * np.cos(th)
            tlen = np.hypot(tx, ty)
            nml[q, 0] =  ty / tlen
            nml[q, 1] = -tx / tlen
            wts[q]    = kGL_w[i] * hd * tlen
            q += 1
    return pts, nml, wts

def star_curve(N=800):
    th = np.linspace(0, 2*kPi, N)
    r  = R_STAR * (1 + A_STAR * np.cos(K_STAR * th))
    return CX + r*np.cos(th), CY + r*np.sin(th)

def star_contains(x, y):
    rho = np.hypot(x - CX, y - CY)
    return rho < R_STAR * (1 + A_STAR * np.cos(K_STAR * np.arctan2(y - CY, x - CX)))

# ── panel Cauchy solver ───────────────────────────────────────────────────────

def solve_local_6x6(bdry1, a, bdry2, bdry2_nml, b, bulk, Lu, center, kappa, h):
    h2     = h * h
    kap_h2 = kappa * h2
    A   = np.zeros((6, 6))
    rhs = np.zeros(6)
    for l in range(3):
        dx = (bdry1[l, 0] - center[0]) / h
        dy = (bdry1[l, 1] - center[1]) / h
        A[l] = [1, dx, dy, 0.5*dx*dx, 0.5*dy*dy, dx*dy]
        rhs[l] = a[l]
    for l in range(2):
        dx = (bdry2[l, 0] - center[0]) / h
        dy = (bdry2[l, 1] - center[1]) / h
        nx, ny = bdry2_nml[l]
        A[3+l] = [0, nx, ny, dx*nx, dy*ny, nx*dy + dx*ny]
        rhs[3+l] = b[l] * h
    dx = (bulk[0] - center[0]) / h
    dy = (bulk[1] - center[1]) / h
    A[5] = [kap_h2, kap_h2*dx, kap_h2*dy,
            kap_h2*0.5*dx*dx - 1, kap_h2*0.5*dy*dy - 1, kap_h2*dx*dy]
    rhs[5] = Lu * h2
    col_scale = np.linalg.norm(A, axis=0)
    col_scale[col_scale < 1e-14] = 1.0
    c_raw, *_ = np.linalg.lstsq(A / col_scale, rhs, rcond=None)
    c_raw /= col_scale
    return np.array([c_raw[0], c_raw[1]/h, c_raw[2]/h,
                     c_raw[3]/h2, c_raw[4]/h2, c_raw[5]/h2])

def laplace_panel_cauchy(pts, nml, wts, a, b, Lu, N_panels):
    Nq  = 3 * N_panels
    out = [np.zeros(Nq) for _ in range(6)]
    for p in range(N_panels):
        g  = [3*p, 3*p+1, 3*p+2]
        h  = wts[g[0]] + wts[g[1]] + wts[g[2]]
        bdry1 = pts[g]; av = a[g]
        bdry2 = pts[[g[0], g[2]]]; bdry2_nml = nml[[g[0], g[2]]]; bv = b[[g[0], g[2]]]
        bulk  = pts[g[1]]; Lu_bulk = Lu[g[1]]
        for gi in g:
            c = solve_local_6x6(bdry1, av, bdry2, bdry2_nml, bv,
                                 bulk, Lu_bulk, pts[gi], 0.0, h)
            for k in range(6):
                out[k][gi] = c[k]
    return tuple(out)   # C, Cx, Cy, Cxx, Cyy, Cxy

# ── domain labeling ───────────────────────────────────────────────────────────

def label_grid(pts_grid, iface_pts, h):
    from collections import deque
    dist = np.full(len(pts_grid), np.inf)
    for q in iface_pts:
        d = np.hypot(pts_grid[:,0]-q[0], pts_grid[:,1]-q[1])
        np.minimum(dist, d, out=dist)
    nx = ny = int(round(1/h)) + 1
    labels = np.full(len(pts_grid), -1, dtype=int)
    band = np.where(dist < 4*h)[0]
    for n in band:
        labels[n] = 1 if star_contains(pts_grid[n,0], pts_grid[n,1]) else 0
    queue = deque(band)
    while queue:
        n = queue.popleft()
        lbl = labels[n]
        i, j = n % nx, n // nx
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni2, nj2 = i+di, j+dj
            if 0 <= ni2 < nx and 0 <= nj2 < ny:
                nb = nj2*nx + ni2
                if labels[nb] == -1:
                    labels[nb] = lbl; queue.append(nb)
    labels[labels == -1] = 0
    return labels

# ── Taylor-expanded C at bulk nodes ──────────────────────────────────────────

def taylor_C(pts_grid, iface_pts, C_data):
    diff = pts_grid[:,None,:] - iface_pts[None,:,:]
    idx  = np.hypot(diff[:,:,0], diff[:,:,1]).argmin(axis=1)
    dx   = pts_grid[:,0] - iface_pts[idx,0]
    dy   = pts_grid[:,1] - iface_pts[idx,1]
    return (C_data['C'][idx] + C_data['Cx'][idx]*dx + C_data['Cy'][idx]*dy
            + 0.5*C_data['Cxx'][idx]*dx**2 + C_data['Cxy'][idx]*dx*dy
            + 0.5*C_data['Cyy'][idx]*dy**2)

# ── IIM RHS correction ────────────────────────────────────────────────────────

def iim_correct(f, C_nodes, labels, nx, ny, h):
    F  = f.copy()
    h2 = h * h
    for n in range(nx * ny):
        i, j = n % nx, n // nx
        if i == 0 or i == nx-1 or j == 0 or j == ny-1: continue
        ln = labels[n]
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            ni2, nj2 = i+di, j+dj
            if 1 <= ni2 <= nx-2 and 1 <= nj2 <= ny-2:
                nb  = nj2*nx + ni2
                lnb = labels[nb]
                if lnb != ln:
                    F[n] += (lnb - ln) * C_nodes[nb] / h2
    return F

# ── DST Poisson solver ────────────────────────────────────────────────────────

def solve_poisson(rhs_full, N, h):
    nx = ny = N + 1
    rhs_int = rhs_full.reshape(ny, nx)[1:N, 1:N].copy()
    p   = np.arange(1, N)
    lam = (2*np.cos(p*kPi/N) - 2) / h**2
    LAM = lam[:,None] + lam[None,:]
    u_int = dstn(dstn(rhs_int, type=1, norm='ortho') / LAM, type=1, norm='ortho')
    u = np.zeros(nx * ny)
    u.reshape(ny, nx)[1:N, 1:N] = u_int
    return u

# ── full pipeline ─────────────────────────────────────────────────────────────

def full_solve(N):
    h  = 1.0 / N
    nx = ny = N + 1
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XI, YI = np.meshgrid(xs, ys)
    pts_grid = np.column_stack([XI.ravel(), YI.ravel()])

    iface_pts, iface_nml, iface_wts = make_star_panels(N)
    Nq = len(iface_pts)

    labels  = label_grid(pts_grid, iface_pts, h)
    x_g, y_g = pts_grid[:,0], pts_grid[:,1]

    f_arr   = np.zeros(nx * ny)
    u_exact = np.zeros(nx * ny)
    for n in range(nx * ny):
        x, y = pts_grid[n]
        i, j = n % nx, n // nx
        lbl  = labels[n]
        u_exact[n] = u_minus(x,y) if lbl == 1 else u_plus(x,y)
        if i==0 or i==nx-1 or j==0 or j==ny-1: continue
        f_arr[n] = f_minus(x,y) if lbl == 1 else f_plus(x,y)

    # Jump data
    xq, yq   = iface_pts[:,0], iface_pts[:,1]
    nx_q     = iface_nml[:,0]; ny_q = iface_nml[:,1]
    a_q      = C_fn(xq, yq)
    b_q      = Cx_fn(xq,yq)*nx_q + Cy_fn(xq,yq)*ny_q
    Lu_q     = f_plus(xq,yq) - f_minus(xq,yq)

    # Panel Cauchy solve
    C_pc, Cx_pc, Cy_pc, Cxx_pc, Cyy_pc, Cxy_pc = \
        laplace_panel_cauchy(iface_pts, iface_nml, iface_wts, a_q, b_q, Lu_q, N)

    C_data = dict(C=C_pc, Cx=Cx_pc, Cy=Cy_pc, Cxx=Cxx_pc, Cxy=Cxy_pc, Cyy=Cyy_pc)

    # Taylor-expand C at every grid node
    C_nodes = taylor_C(pts_grid, iface_pts, C_data)

    # IIM RHS correction
    F_corrected = iim_correct(f_arr, C_nodes, labels, nx, ny, h)
    rhs_correction = F_corrected - f_arr  # just the delta, for visualization

    # Poisson solve
    u_h = solve_poisson(-F_corrected, N, h)

    mask = np.zeros(nx*ny, bool)
    mask.reshape(ny, nx)[1:N, 1:N] = True
    err = np.max(np.abs((u_h - u_exact)[mask]))

    return dict(
        h=h, nx=nx, ny=ny, N=N, xs=xs, ys=ys,
        pts_grid=pts_grid, labels=labels,
        iface_pts=iface_pts, iface_nml=iface_nml,
        u_exact=u_exact, u_h=u_h,
        C_pc=C_pc, C_exact_iface=a_q,
        rhs_correction=rhs_correction,
        err=err,
    )

# ── compute ───────────────────────────────────────────────────────────────────

N_VIZ = 64
print(f"Solving at N={N_VIZ} …")
r = full_solve(N_VIZ)
print(f"  max_err = {r['err']:.4e}")

Ns = [16, 32, 64, 128, 256]
print("Convergence study …")
errs = []
for N in Ns:
    res = full_solve(N)
    errs.append(res['err'])
    rate_str = f"  rate={np.log2(errs[-2]/errs[-1]):.2f}" if len(errs) > 1 else ""
    print(f"  N={N:4d}  err={errs[-1]:.4e}{rate_str}")

# ── figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    r"Full interface problem: $-\Delta u = f$ (piecewise smooth),  "
    r"$[u]=a$,  $[\partial_n u]=b$ on $\Gamma$,  $u=0$ on $\partial\Omega$"
    "\n"
    r"Panel Cauchy (Layer 1.5) $\to$ Spread/IIM correction (Layer 1) $\to$ FFT solve (Layer 2)",
    fontsize=11, fontweight='bold'
)

nx, ny = r['nx'], r['ny']
xs, ys = r['xs'], r['ys']
cx_c, cy_c = star_curve()

def to2d(arr): return arr.reshape(ny, nx)

# ─── (0,0): exact solution ───────────────────────────────────────────────────
ax = axes[0, 0]
Z_exact = to2d(r['u_exact'])
im0 = ax.pcolormesh(xs, ys, Z_exact, shading='auto', cmap='RdBu_r')
ax.plot(cx_c, cy_c, 'k-', lw=1.2)
ax.set_title(fr'Exact solution $u$  (N={N_VIZ})', fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im0, ax=ax)

# ─── (0,1): computed solution ─────────────────────────────────────────────────
ax = axes[0, 1]
Z_h = to2d(r['u_h'])
im1 = ax.pcolormesh(xs, ys, Z_h, shading='auto', cmap='RdBu_r',
                    vmin=Z_exact.min(), vmax=Z_exact.max())
ax.plot(cx_c, cy_c, 'k-', lw=1.2)
ax.set_title(fr'Computed solution $u_h$  (panel Cauchy + FFT, N={N_VIZ})', fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im1, ax=ax)

# ─── (0,2): pointwise error ───────────────────────────────────────────────────
ax = axes[0, 2]
err_field = np.abs(r['u_h'] - r['u_exact'])
for n in range(nx*ny):
    i, j = n % nx, n // nx
    if i==0 or i==nx-1 or j==0 or j==ny-1: err_field[n] = 0.0
Z_err = to2d(err_field)
vmax_e = max(Z_err.max(), 1e-10)
vmin_e = max(Z_err[Z_err > 0].min(), 1e-10) if (Z_err > 0).any() else 1e-10
im2 = ax.pcolormesh(xs, ys, Z_err, shading='auto', cmap='hot_r',
                    norm=mcolors.LogNorm(vmin=vmin_e, vmax=vmax_e))
ax.plot(cx_c, cy_c, 'w-', lw=1.2)
ax.set_title(fr'Pointwise error $|u_h - u|$  (max={r["err"]:.2e})', fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im2, ax=ax)

# ─── (1,0): IIM correction field ─────────────────────────────────────────────
ax = axes[1, 0]
corr = np.abs(r['rhs_correction'])
for n in range(nx*ny):
    i, j = n % nx, n // nx
    if i==0 or i==nx-1 or j==0 or j==ny-1: corr[n] = 0.0
Z_corr = to2d(corr)
vmax_c = max(Z_corr.max(), 1e-10)
vmin_c = max(Z_corr[Z_corr > 0].min(), 1e-10) if (Z_corr > 0).any() else 1e-10
im3 = ax.pcolormesh(xs, ys, Z_corr, shading='auto', cmap='viridis',
                    norm=mcolors.LogNorm(vmin=vmin_c, vmax=vmax_c))
ax.plot(cx_c, cy_c, 'w-', lw=1.2)
ax.set_title(fr'IIM RHS correction $|\Delta F|$  (nonzero only at irregular nodes)', fontsize=10)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
plt.colorbar(im3, ax=ax)

# ─── (1,1): convergence plot ──────────────────────────────────────────────────
ax = axes[1, 1]
hs = [1.0/N for N in Ns]
ax.loglog(hs, errs, 'o-', color='#2c7be5', lw=2, ms=8, label='panel Cauchy + FFT')
h_ref = np.array([hs[0], hs[-1]])
ax.loglog(h_ref, errs[0]*(np.array(h_ref)/hs[0])**2, 'k:', lw=1.5, label=r'$O(h^2)$')
ax.loglog(h_ref, errs[0]*(np.array(h_ref)/hs[0])**1, 'k--', lw=1, alpha=0.4, label=r'$O(h)$')
for i in range(1, len(Ns)):
    rate = np.log2(errs[i-1]/errs[i])
    hm   = (hs[i-1]*hs[i])**0.5
    em   = (errs[i-1]*errs[i])**0.5
    ax.text(hm, em*1.5, f'{rate:.2f}', fontsize=9, ha='center', color='#2c7be5')
ax.set_xlabel(r'$h = 1/N$', fontsize=11)
ax.set_ylabel(r'$\|u_h - u\|_\infty$', fontsize=11)
ax.set_title('Convergence: end-to-end pipeline', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, which='both', ls=':', alpha=0.5)
ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xscale('log')
ax2.set_xticks(hs); ax2.set_xticklabels([str(N) for N in Ns], fontsize=8)
ax2.set_xlabel('N', fontsize=9)

# ─── (1,2): C at interface — panel Cauchy vs exact ────────────────────────────
ax = axes[1, 2]
iface_pts = r['iface_pts']
C_pc_iface = r['C_pc']
C_ex_iface = r['C_exact_iface']

# Arc-length parameter for ordering
th = np.arctan2(iface_pts[:,1] - CY, iface_pts[:,0] - CX)
order = np.argsort(th)

ax.plot(th[order], C_ex_iface[order], 'k-', lw=1.5, label=r'$C_\mathrm{exact} = u^+ - u^-$')
ax.plot(th[order], C_pc_iface[order], 'o', color='#e74c3c', ms=3,
        label=fr'Panel Cauchy $C$ (N={N_VIZ})')
ax.set_xlabel(r'$\theta$ (angle)', fontsize=11)
ax.set_ylabel(r'$C = u^+ - u^-$ on $\Gamma$', fontsize=11)
ax.set_title(r'Correction function $C$ at interface Gauss points', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.5)
ax.set_xlim(-kPi, kPi)

plt.tight_layout(rect=[0, 0, 1, 0.93])

out = "python/laplace_iface_2d_viz.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out}")
plt.show()
