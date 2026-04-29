"""
Visualization of Laplace interior Dirichlet BVP for a 3-fold star interface.
Representation: u = D[phi] (double-layer potential).
BIE: (K - 1/2 I) phi = g  (for interior trace)

Exact solution: u_exact(x,y) = exp(x)sin(y)  (Laplace: Δu = 0).
Boundary data:  g = u_exact|Γ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.fft import dstn
from scipy.sparse.linalg import gmres, LinearOperator

kPi = np.pi

# ── geometry: 3-fold star ──────────────────────────────────────────────────

CX, CY = 0.5, 0.5
R_STAR, A_STAR, K_STAR = 0.28, 0.40, 3  # K=3 for 3-fold star

def get_star_r(th):
    return R_STAR * (1 + A_STAR * np.cos(K_STAR * th))

def get_star_drdt(th):
    return -R_STAR * A_STAR * K_STAR * np.sin(K_STAR * th)

def star_contains(x, y):
    rho = np.hypot(x - CX, y - CY)
    th = np.arctan2(y - CY, x - CX)
    return rho < get_star_r(th)

def star_curve(N=800):
    th = np.linspace(0, 2*kPi, N)
    r = get_star_r(th)
    return CX + r*np.cos(th), CY + r*np.sin(th)

# Gauss points (3 per panel)
kGL_s = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
kGL_w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

def make_star_panels(N_panels):
    Nq = 3 * N_panels
    pts = np.zeros((Nq, 2))
    nml = np.zeros((Nq, 2))
    wts = np.zeros(Nq)
    dth = 2*kPi / N_panels
    for p in range(N_panels):
        th_mid = (p + 0.5) * dth
        hd = 0.5 * dth
        for i in range(3):
            q = 3*p + i
            th = th_mid + hd * kGL_s[i]
            r = get_star_r(th)
            drdt = get_star_drdt(th)
            pts[q, 0] = CX + r * np.cos(th)
            pts[q, 1] = CY + r * np.sin(th)
            tx = drdt * np.cos(th) - r * np.sin(th)
            ty = drdt * np.sin(th) + r * np.cos(th)
            tlen = np.hypot(tx, ty)
            nml[q, 0] =  ty / tlen
            nml[q, 1] = -tx / tlen
            wts[q] = kGL_w[i] * hd * tlen
    return pts, nml, wts

# ── manufactured solution ─────────────────────────────────────────────────────
# u = exp(x)sin(y)

def u_exact_fn(x, y): return np.exp(x) * np.sin(y)

# ── internal solver components ───────────────────────────────────────────────

def solve_local_6x6(bdry1, a, bdry2, bdry2_nml, b, bulk, Lu, center, h):
    h2 = h*h
    A = np.zeros((6, 6))
    rhs = np.zeros(6)
    for l in range(3):
        dx, dy = (bdry1[l] - center) / h
        A[l] = [1, dx, dy, 0.5*dx*dx, 0.5*dy*dy, dx*dy]
        rhs[l] = a[l]
    for l in range(2):
        dx, dy = (bdry2[l] - center) / h
        nx, ny = bdry2_nml[l]
        A[3+l] = [0, nx, ny, dx*nx, dy*ny, nx*dy + dx*ny]
        rhs[3+l] = b[l] * h
    # ΔC = 0 for Laplace double-layer
    A[5] = [0, 0, 0, -1, -1, 0] 
    rhs[5] = 0.0
    c_raw, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return np.array([c_raw[0], c_raw[1]/h, c_raw[2]/h, c_raw[3]/h2, c_raw[4]/h2, c_raw[5]/h2])

def panel_cauchy(pts, nml, wts, u_jump, un_jump, f_jump, N_panels):
    # For DL potential, [u]=phi, [un]=0, [f]=0
    Nq = 3 * N_panels
    C_data = np.zeros((Nq, 6))
    for p in range(N_panels):
        g = [3*p, 3*p+1, 3*p+2]
        h_panel = np.sum(wts[g]) / 2.0 # approx panel length
        for i in range(3):
            q = g[i]
            C_data[q] = solve_local_6x6(pts[g], u_jump[g], 
                                         pts[[g[0], g[2]]], nml[[g[0], g[2]]], un_jump[[g[0], g[2]]],
                                         pts[g[1]], f_jump[g[1]], pts[q], h_panel)
    return C_data

def apply_kfbim(phi, params, mode='interior'):
    N, pts, nml, wts, labels = params['N'], params['pts'], params['nml'], params['wts'], params['labels']
    h = 1.0/N
    nx = ny = N+1
    Nq = len(pts)
    N_panels = Nq // 3

    # 1. Spread: [u]=phi, [un]=0, [f]=0
    C_data = panel_cauchy(pts, nml, wts, phi, np.zeros(Nq), np.zeros(Nq), N_panels)
    
    # Correction at grid nodes
    grid_pts = params['grid_pts']
    dist = np.hypot(grid_pts[:,None,0] - pts[None,:,0], grid_pts[:,None,1] - pts[None,:,1])
    idx = dist.argmin(axis=1)
    
    dx = grid_pts[:,0] - pts[idx,0]
    dy = grid_pts[:,1] - pts[idx,1]
    C_nodes = (C_data[idx,0] + C_data[idx,1]*dx + C_data[idx,2]*dy 
               + 0.5*C_data[idx,3]*dx**2 + 0.5*C_data[idx,4]*dy**2 + C_data[idx,5]*dx*dy)

    # Corrected RHS
    F = np.zeros(nx*ny) # Laplace: base_f = 0
    h2 = h*h
    for n in range(nx*ny):
        i, j = n%nx, n//nx
        if i==0 or i==nx-1 or j==0 or j==ny-1: continue
        ln = labels[n]
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            nb = (j+dj)*nx + (i+di)
            lnb = labels[nb]
            if lnb != ln:
                F[n] += (lnb - ln) * C_nodes[nb] / h2

    # 2. Bulk Solve
    u_h = solve_poisson(-F, N, h)

    # 3. Restrict (Quadratic fit)
    u_trace_bulk = np.zeros(Nq)
    for q in range(Nq):
        qc = idx_grid_closest(pts[q], h, nx)
        ic, jc = qc%nx, qc//nx
        stencil = []
        for dj in [-2,-1,0,1,2]: # use slightly larger stencil for robustness
            for di in [-2,-1,0,1,2]:
                ni, nj = ic+di, jc+dj
                if 0<=ni<nx and 0<=nj<ny: stencil.append(nj*nx+ni)
        
        S_pts = grid_pts[stencil]
        S_val = u_h[stencil]
        dxs = S_pts[:,0] - pts[q,0]
        dys = S_pts[:,1] - pts[q,1]
        V = np.column_stack([np.ones(len(stencil)), dxs, dys, 0.5*dxs**2, 0.5*dys**2, dxs*dys])
        coeffs, *_ = np.linalg.lstsq(V, S_val, rcond=None)
        u_trace_bulk[q] = coeffs[0]

    if mode == 'interior':
        return u_trace_bulk - C_data[:,0] # u- = (K - 1/2 I) phi
    else:
        return u_trace_bulk # u+ = (K + 1/2 I) phi

def idx_grid_closest(pt, h, nx):
    i = int(round(pt[0]/h))
    j = int(round(pt[1]/h))
    return j*nx + i

def solve_poisson(rhs_full, N, h):
    nx = ny = N + 1
    rhs_int = rhs_full.reshape(ny, nx)[1:N, 1:N].copy()
    p = np.arange(1, N)
    lam = (2*np.cos(p*kPi/N) - 2) / h**2
    LAM = lam[:,None] + lam[None,:]
    u_int = dstn(dstn(rhs_int, type=1, norm='ortho') / LAM, type=1, norm='ortho')
    u = np.zeros(nx * ny)
    u.reshape(ny, nx)[1:N, 1:N] = u_int
    return u

# ── main loop ─────────────────────────────────────────────────────────────────

Ns = [32, 64, 128]
gmres_iters = []
results = {}

print(f"Solving interior Laplace Dirichlet BVP on 3-fold star...")

for N in Ns:
    h = 1.0/N
    nx = ny = N+1
    grid_x, grid_y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # Interface: keep arc ~ h
    N_panels = int(round(1.7 * N))
    pts, nml, wts = make_star_panels(N_panels)
    Nq = len(pts)
    
    labels = np.array([1 if star_contains(x,y) else 0 for x,y in grid_pts])
    g = np.array([u_exact_fn(x,y) for x,y in pts])
    
    params = {'N':N, 'pts':pts, 'nml':nml, 'wts':wts, 'labels':labels, 'grid_pts':grid_pts}
    
    def matvec(phi):
        return apply_kfbim(phi, params, mode='interior')

    A_op = LinearOperator((Nq, Nq), matvec=matvec)
    
    iters = []
    def callback(pr): iters.append(pr)
    # Using atol=1e-10. Note: (K - 1/2 I) phi = g
    phi_sol, info = gmres(A_op, g, atol=1e-10, restart=50, callback=callback)
    gmres_iters.append(len(iters))
    
    print(f"  N={N:3d}  panels={N_panels:3d}  GMRES iters={gmres_iters[-1]}")
    
    if N == 64:
        # Save for visualization
        C_data = panel_cauchy(pts, nml, wts, phi_sol, np.zeros(Nq), np.zeros(Nq), N_panels)
        dist = np.hypot(grid_pts[:,None,0] - pts[None,:,0], grid_pts[:,None,1] - pts[None,:,1])
        idx = dist.argmin(axis=1)
        dx, dy = grid_pts[:,0] - pts[idx,0], grid_pts[:,1] - pts[idx,1]
        C_nodes = (C_data[idx,0] + C_data[idx,1]*dx + C_data[idx,2]*dy 
                   + 0.5*C_data[idx,3]*dx**2 + 0.5*C_data[idx,4]*dy**2 + C_data[idx,5]*dx*dy)
        F = np.zeros(nx*ny)
        for n in range(nx*ny):
            i, j = n%nx, n//nx
            if i==0 or i==nx-1 or j==0 or j==ny-1: continue
            ln = labels[n]
            for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb = (j+dj)*nx + (i+di)
                if labels[nb] != ln: F[n] += (labels[nb] - ln) * C_nodes[nb] / (h*h)
        u_bulk = solve_poisson(-F, N, h)
        
        # In KFBIM, u_bulk represents u_smooth on each side.
        # But for visualization of D[phi], we want u- = u_bulk - C and u+ = u_bulk
        u_final = u_bulk.copy()
        u_final[labels == 1] -= C_nodes[labels == 1]
        
        results['N64'] = {'u':u_final, 'labels':labels, 'pts':pts, 'phi':phi_sol, 'nx':nx, 'ny':ny}

# ── visualization ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2)

# (0,0) Bulk solution (D[phi] everywhere)
ax0 = fig.add_subplot(gs[0, 0])
res = results['N64']
nx, ny = res['nx'], res['ny']
u_plot = res['u'].reshape(ny, nx)
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, ny)
im = ax0.pcolormesh(xs, ys, u_plot, shading='auto', cmap='RdBu_r')
plt.colorbar(im, ax=ax0)
cx, cy = star_curve()
ax0.plot(cx, cy, 'k-', lw=1.5)
ax0.set_title("Double-layer Potential $u = D[\phi]$ (N=64)")
ax0.set_aspect('equal')

# (0,1) Density phi
ax1 = fig.add_subplot(gs[0, 1])
th = np.arctan2(res['pts'][:,1]-CY, res['pts'][:,0]-CX)
order = np.argsort(th)
ax1.plot(th[order], res['phi'][order], 'o-', ms=2, color='darkblue')
ax1.set_title("Double-layer density $\phi$ vs polar angle")
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$\phi$")
ax1.grid(True, alpha=0.3)

# (1,0) Interior Error
ax2 = fig.add_subplot(gs[1, 0])
# We need to compute u_exact on the N=64 grid
grid_pts64 = np.column_stack([np.meshgrid(np.linspace(0,1,64+1), np.linspace(0,1,64+1))[0].ravel(),
                               np.meshgrid(np.linspace(0,1,64+1), np.linspace(0,1,64+1))[1].ravel()])
u_exact_grid64 = np.array([u_exact_fn(x,y) for x,y in grid_pts64])
err = np.abs(res['u'] - u_exact_grid64)
err[res['labels'] == 0] = 0 # only interior
im2 = ax2.pcolormesh(xs, ys, err.reshape(ny, nx), shading='auto', cmap='hot_r', norm=mcolors.LogNorm(vmin=1e-8, vmax=1e-2))
plt.colorbar(im2, ax=ax2)
ax2.plot(cx, cy, 'k-', lw=1)
ax2.set_title("Interior Error $|u_h - \exp(x)\sin(y)|$")
ax2.set_aspect('equal')

# (1,1) GMRES statistics
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
table_data = [["N", "Panels", "GMRES Iters"]]
for i, N in enumerate(Ns):
    table_data.append([str(N), str(int(round(1.7*N))), str(gmres_iters[i])])

table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
table.set_fontsize(14)
table.scale(1, 4)
ax3.set_title("GMRES Iteration Counts ($2^{nd}$-kind BIE)", y=0.8)

plt.tight_layout()
out_png = "scripts/laplace_dl_3fold_viz.png"
plt.savefig(out_png, dpi=150)
print(f"Visualization saved to {out_png}")
