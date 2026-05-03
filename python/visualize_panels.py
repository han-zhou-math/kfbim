"""
Interface panel structure visualization.

Shows how the star-shaped interface is discretized into panels, how the
3 Gauss-Legendre points are placed within each panel, and how those points
map to the rows of the 6x6 local Cauchy collocation system.

Layout (1x3 + lower row):
  (0,0)  Full star interface: panel boundaries, GL points, outward normals
  (0,1)  Close-up of one panel: GL points labeled by collocation role
           Diri (blue)   — rows 0-2: all 3 GL points,  enforce [u] = a
           Neum (orange) — rows 3-4: pts 0 and 2,       enforce [beta d_n u] = b
           PDE  (green)  — row  5:   midpoint (pt 1),   enforce -Delta C = Lu
           Bulk (green X)— interior bulk point for the center-pt solve shown
  (0,2)  Arc-length weight distribution along the interface
  (1,0)  Normal vectors zoom-in around one tip of the star
  (1,1)  Panel arc-length h_p vs panel index (shows resolution variation)
  (1,2)  Convergence reminder: Nq = 3*N_panels vs geometric arc-length
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

kPi = np.pi

# ── star geometry ─────────────────────────────────────────────────────────────

CX, CY, R_STAR, A_STAR, K_STAR = 0.5, 0.5, 0.28, 0.40, 5

kGL_s = np.array([-0.7745966692414834, 0.0, +0.7745966692414834])
kGL_w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

def make_star_panels(N_panels):
    Nq  = 3 * N_panels
    pts = np.zeros((Nq, 2))
    nml = np.zeros((Nq, 2))
    wts = np.zeros(Nq)
    panel_h = np.zeros(N_panels)
    panel_mid_idx = np.zeros(N_panels, dtype=int)  # index of GL node 1 in each panel
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
        panel_h[p] = wts[3*p] + wts[3*p+1] + wts[3*p+2]
        panel_mid_idx[p] = 3*p + 1
    return pts, nml, wts, panel_h, panel_mid_idx

def star_curve(N=1200):
    th = np.linspace(0, 2*kPi, N)
    r  = R_STAR * (1 + A_STAR * np.cos(K_STAR * th))
    return CX + r*np.cos(th), CY + r*np.sin(th)

def panel_boundary_pts(N_panels):
    """Angular positions of panel boundaries → points on the star curve."""
    dth = 2*kPi / N_panels
    ths = np.arange(N_panels + 1) * dth
    r   = R_STAR * (1 + A_STAR * np.cos(K_STAR * ths))
    return CX + r*np.cos(ths), CY + r*np.sin(ths)

# ── colours and markers ───────────────────────────────────────────────────────

C_DIRI  = '#2c7be5'   # blue  — Dirichlet
C_NEUM  = '#e67e22'   # orange — Neumann
C_PDE   = '#27ae60'   # green  — PDE midpoint
C_BULK  = '#8e44ad'   # purple — interior bulk point
C_PANEL = '#aaaaaa'   # grey   — panel boundary ticks

# ── compute ───────────────────────────────────────────────────────────────────

N_OVERVIEW = 20   # panels for the overview plot
N_ZOOM     = 20   # panels for close-up (same so we can pick a nice panel)

pts_ov, nml_ov, wts_ov, h_ov, mid_ov = make_star_panels(N_OVERVIEW)
cx_c, cy_c = star_curve()
bx, by     = panel_boundary_pts(N_OVERVIEW)

# Pick the panel closest to angle 0 (rightmost, cleanest for the close-up)
# Panel p spans theta = [p*dth, (p+1)*dth]; pick p=0
PANEL_IDX = 0
g = [3*PANEL_IDX, 3*PANEL_IDX+1, 3*PANEL_IDX+2]
h_panel = h_ov[PANEL_IDX]

# Interior bulk point for the midpoint solve (center = GL node 1)
center_pt = pts_ov[g[1]]
bulk_pt   = center_pt - 0.5 * h_panel * nml_ov[g[1]]

# ── figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 13))
fig.suptitle(
    "Interface panel structure and collocation roles\n"
    r"Star interface: $r(\theta)=R(1+A\cos K\theta)$,  "
    fr"$R={R_STAR}$, $A={A_STAR}$, $K={K_STAR}$,  "
    fr"$N_\mathrm{{panels}}={N_OVERVIEW}$,  $N_q=3N_\mathrm{{panels}}={3*N_OVERVIEW}$ Gauss points",
    fontsize=11, fontweight='bold'
)

# ─── (0,0): full interface overview ──────────────────────────────────────────
ax = axes[0, 0]
ax.plot(cx_c, cy_c, '-', color='#cccccc', lw=1.0, zorder=1)

# Panel boundary ticks (short line perpendicular to tangent)
tick_len = 0.012
for i in range(len(bx)-1):   # skip last (= first)
    ax.plot(bx[i], by[i], '|', color=C_PANEL, ms=8, mew=1.2, zorder=2)

# Outward normals (subsampled for clarity)
nml_scale = 0.030
ax.quiver(pts_ov[:,0], pts_ov[:,1],
          nml_ov[:,0]*nml_scale, nml_ov[:,1]*nml_scale,
          color='#888888', scale=1, scale_units='xy',
          width=0.002, headwidth=4, headlength=5, zorder=3)

# GL points coloured by role
for p in range(N_OVERVIEW):
    gp = [3*p, 3*p+1, 3*p+2]
    # pts 0 and 2: Dirichlet + Neumann
    for gi in [gp[0], gp[2]]:
        ax.plot(pts_ov[gi,0], pts_ov[gi,1], 'o',
                color=C_NEUM, ms=6, zorder=4)
    # Dirichlet outer ring
    for gi in gp:
        ax.plot(pts_ov[gi,0], pts_ov[gi,1], 'o',
                color=C_DIRI, ms=9, mfc='none', mew=1.2, zorder=4)
    # Midpoint: PDE role
    ax.plot(pts_ov[gp[1],0], pts_ov[gp[1],1], 's',
            color=C_PDE, ms=6, zorder=5)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0.1, 0.9); ax.set_ylim(0.1, 0.9)
ax.set_title(fr'Full interface  ($N_\mathrm{{panels}}={N_OVERVIEW}$, $N_q={3*N_OVERVIEW}$)',
             fontsize=10)
ax.set_xlabel('x'); ax.set_ylabel('y')

legend_elems = [
    mpatches.Patch(color=C_DIRI,  label='Dirichlet (all 3 GL pts, rows 0–2)'),
    mpatches.Patch(color=C_NEUM,  label='Neumann (pts 0 & 2, rows 3–4)'),
    mpatches.Patch(color=C_PDE,   label='PDE midpoint (pt 1, row 5)'),
    mpatches.Patch(color=C_PANEL, label='Panel boundary'),
]
ax.legend(handles=legend_elems, fontsize=8, loc='upper right')

# ─── (0,1): close-up of panel PANEL_IDX ──────────────────────────────────────
ax = axes[0, 1]

# Draw star curve near the panel
dth_full = 2*kPi / N_OVERVIEW
th_a = PANEL_IDX * dth_full - dth_full * 0.5
th_b = (PANEL_IDX + 1) * dth_full + dth_full * 0.5
th_local = np.linspace(th_a, th_b, 300)
r_local  = R_STAR * (1 + A_STAR * np.cos(K_STAR * th_local))
xl = CX + r_local * np.cos(th_local)
yl = CY + r_local * np.sin(th_local)
ax.plot(xl, yl, '-', color='#cccccc', lw=1.2, zorder=1)

# Panel boundary markers
bx_p, by_p = panel_boundary_pts(N_OVERVIEW)
for i in [PANEL_IDX, PANEL_IDX+1]:
    ax.plot(bx_p[i], by_p[i], 'D', color=C_PANEL, ms=8, zorder=2,
            label='Panel boundary' if i == PANEL_IDX else '')

# Adjacent panel GL points (dimmed)
for p in [PANEL_IDX-1, PANEL_IDX+1]:
    pp = p % N_OVERVIEW
    for gi in [3*pp, 3*pp+1, 3*pp+2]:
        ax.plot(pts_ov[gi,0], pts_ov[gi,1], 'o',
                color='#dddddd', ms=7, zorder=3)

# Current panel: normals
nml_scale_z = 0.025
for i, gi in enumerate(g):
    ax.annotate('', xy=(pts_ov[gi,0] + nml_ov[gi,0]*nml_scale_z,
                        pts_ov[gi,1] + nml_ov[gi,1]*nml_scale_z),
                xytext=(pts_ov[gi,0], pts_ov[gi,1]),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2))

# GL pts 0 and 2: Dirichlet + Neumann
for i, gi in enumerate([g[0], g[2]]):
    ax.plot(pts_ov[gi,0], pts_ov[gi,1], 'o',
            color=C_NEUM, ms=12, zorder=5, label='Dirichlet + Neumann' if i==0 else '')
    ax.plot(pts_ov[gi,0], pts_ov[gi,1], 'o',
            color=C_DIRI, ms=17, mfc='none', mew=2, zorder=5)
    lbl = f'GL {2*i}\n[u]=a\n[β∂ₙu]=b'
    off = (-0.032, -0.028) if i == 0 else (0.010, -0.028)
    ax.text(pts_ov[gi,0]+off[0], pts_ov[gi,1]+off[1], lbl,
            fontsize=7.5, ha='center', color='#333333')

# GL pt 1 (midpoint): PDE role
ax.plot(pts_ov[g[1],0], pts_ov[g[1],1], 's',
        color=C_PDE, ms=12, zorder=6, label='PDE midpoint')
ax.text(pts_ov[g[1],0]+0.002, pts_ov[g[1],1]+0.022,
        'GL 1\n−ΔC=Lu', fontsize=7.5, ha='center', color=C_PDE)

# Interior bulk point for the GL-1 (center) solve
ax.plot(bulk_pt[0], bulk_pt[1], 'X',
        color=C_BULK, ms=11, zorder=7, label='Bulk PDE pt (center=GL 1)')
ax.annotate('bulk\n(0.5h inward)', xy=bulk_pt,
            xytext=(bulk_pt[0]+0.030, bulk_pt[1]-0.018),
            fontsize=7.5, color=C_BULK,
            arrowprops=dict(arrowstyle='->', color=C_BULK, lw=1))

# Dashed line connecting panel GL points
ax.plot([pts_ov[g[0],0], pts_ov[g[1],0], pts_ov[g[2],0]],
        [pts_ov[g[0],1], pts_ov[g[1],1], pts_ov[g[2],1]],
        '--', color='#bbbbbb', lw=1, zorder=2)

ax.set_aspect('equal', adjustable='box')
# Zoom tightly around the panel
px = pts_ov[g, 0]; py = pts_ov[g, 1]
cx_mid = px.mean(); cy_mid = py.mean()
hw = max(px.max()-px.min(), py.max()-py.min()) * 1.8 + 0.04
ax.set_xlim(cx_mid - hw, cx_mid + hw)
ax.set_ylim(cy_mid - hw, cy_mid + hw)
ax.set_title(f'Close-up: panel {PANEL_IDX}  (6×6 collocation roles)', fontsize=10)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend(fontsize=8, loc='lower right')

# ─── (0,2): arc-length weight distribution ───────────────────────────────────
ax = axes[0, 2]
q_idx = np.arange(3 * N_OVERVIEW)
colors_wt = []
for p in range(N_OVERVIEW):
    colors_wt += [C_DIRI, C_PDE, C_DIRI]  # 0=Diri, 1=PDE, 2=Diri
ax.bar(q_idx, wts_ov, color=colors_wt, width=0.8, alpha=0.8)

# Annotate Neumann pts (0 and 2 of each panel)
for p in range(N_OVERVIEW):
    for local_i in [0, 2]:
        gi = 3*p + local_i
        ax.bar(gi, wts_ov[gi], color=C_NEUM, width=0.8, alpha=0.6)

ax.set_xlabel('Gauss point index $q$', fontsize=10)
ax.set_ylabel('Arc-length weight $w_q$', fontsize=10)
ax.set_title('Arc-length weights (= $w_i \\cdot \\frac{\\Delta\\theta}{2} \\cdot |\\mathbf{t}|$)', fontsize=10)

legend_wt = [
    mpatches.Patch(color=C_DIRI,  alpha=0.8, label='Dirichlet (pts 0, 2)'),
    mpatches.Patch(color=C_NEUM,  alpha=0.6, label='+ Neumann (pts 0, 2)'),
    mpatches.Patch(color=C_PDE,   alpha=0.8, label='PDE midpoint (pt 1)'),
]
ax.legend(handles=legend_wt, fontsize=8)
ax.grid(True, axis='y', ls=':', alpha=0.5)

# Mark panel boundaries with vertical dashed lines
for p in range(1, N_OVERVIEW):
    ax.axvline(3*p - 0.5, color=C_PANEL, lw=0.8, ls='--', alpha=0.6)

# ─── (1,0): normals zoom around one star tip ─────────────────────────────────
ax = axes[1, 0]

# Tip of first arm is near theta ≈ 0
th_tip_range = 2*kPi / K_STAR   # one tip period
th_zoom = np.linspace(-th_tip_range/2, th_tip_range/2, 600)
r_zoom  = R_STAR * (1 + A_STAR * np.cos(K_STAR * th_zoom))
xz = CX + r_zoom * np.cos(th_zoom)
yz = CY + r_zoom * np.sin(th_zoom)
ax.plot(xz, yz, '-', color='#aaaaaa', lw=1.5)

# Show GL points near the tip (panels 0, N-1, and neighbours)
tip_panels = list(range(N_OVERVIEW - 2, N_OVERVIEW)) + list(range(0, 3))
shown = set()
for p in tip_panels:
    for local_i in range(3):
        gi = 3*p + local_i
        if gi in shown: continue
        shown.add(gi)
        c = C_PDE if local_i == 1 else (C_NEUM if local_i in [0,2] else C_DIRI)
        ax.plot(pts_ov[gi,0], pts_ov[gi,1], 'o', color=c, ms=7, zorder=4)
        scale = 0.035
        ax.annotate('', xy=(pts_ov[gi,0]+nml_ov[gi,0]*scale,
                             pts_ov[gi,1]+nml_ov[gi,1]*scale),
                    xytext=(pts_ov[gi,0], pts_ov[gi,1]),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0.65, 0.92); ax.set_ylim(0.39, 0.62)
ax.set_title('Outward normals near a star tip', fontsize=10)
ax.set_xlabel('x'); ax.set_ylabel('y')

# ─── (1,1): panel arc-length h_p vs panel index ──────────────────────────────
ax = axes[1, 1]

# Compute for several refinement levels
for N_ref, color, lbl in [(10, '#e74c3c', 'N=10'),
                           (20, '#2c7be5', 'N=20'),
                           (40, '#27ae60', 'N=40')]:
    _, _, _, h_ref, _ = make_star_panels(N_ref)
    ax.plot(np.arange(N_ref), h_ref, '-o', color=color, ms=4, lw=1.5, label=lbl)

ax.set_xlabel('Panel index $p$', fontsize=10)
ax.set_ylabel(r'Panel arc-length $h_p = \sum_i w_{p,i}$', fontsize=10)
ax.set_title('Panel arc-length variation (non-uniform due to curvature)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, ls=':', alpha=0.5)
# Mark star tips (where curvature is highest, panels are shorter)
ax.axhline(2*kPi*R_STAR*(1+A_STAR)/20, color='#aaaaaa', ls='--', lw=0.8,
           label='uniform (circle, N=20)')

# ─── (1,2): Nq = 3N vs interface arc-length accuracy ────────────────────────
ax = axes[1, 2]

# Total arc-length of star curve (numerical)
th_dense = np.linspace(0, 2*kPi, 100000)
r_dense  = R_STAR * (1 + A_STAR * np.cos(K_STAR * th_dense))
drdt_d   = -R_STAR * A_STAR * K_STAR * np.sin(K_STAR * th_dense)
tlen_d   = np.hypot(drdt_d * np.cos(th_dense) - r_dense * np.sin(th_dense),
                    drdt_d * np.sin(th_dense) + r_dense * np.cos(th_dense))
arc_total = np.trapz(tlen_d, th_dense)

Nps = [4, 8, 12, 16, 20, 32, 48, 64, 96, 128]
arc_errs = []
for Np in Nps:
    _, _, wts_ref, _, _ = make_star_panels(Np)
    arc_errs.append(abs(wts_ref.sum() - arc_total))

ax.loglog(Nps, arc_errs, 'o-', color='#2c7be5', lw=2, ms=7)
h_ref_arr = np.array([Nps[0], Nps[-1]], dtype=float)
ax.loglog(h_ref_arr, arc_errs[0]*(h_ref_arr/Nps[0])**(-4), 'k:', lw=1.2,
          label=r'$O(N^{-4})$')
for i in range(1, len(Nps)):
    if arc_errs[i] < 1e-15 or arc_errs[i-1] < 1e-15:
        continue
    rate = np.log2(arc_errs[i-1]/arc_errs[i]) / np.log2(Nps[i]/Nps[i-1])
    Nm   = (Nps[i-1]*Nps[i])**0.5
    em   = (arc_errs[i-1]*arc_errs[i])**0.5
    if i % 2 == 1:
        ax.text(Nm, em*1.6, f'{rate:.1f}', fontsize=8, ha='center', color='#2c7be5')

ax.set_xlabel('$N_\\mathrm{panels}$', fontsize=11)
ax.set_ylabel('|computed arc-length − exact|', fontsize=10)
ax.set_title('GL quadrature accuracy for arc-length\n(3-point GL → 4th-order convergence)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, which='both', ls=':', alpha=0.5)

plt.subplots_adjust(top=0.91, bottom=0.06, left=0.06, right=0.97,
                    hspace=0.42, wspace=0.32)

out = "python/panels_viz.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved → {out}")
plt.show()
