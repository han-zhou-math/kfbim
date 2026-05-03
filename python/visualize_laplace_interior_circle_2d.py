import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

Ns = [32, 64, 128, 256, 512, 1024]
csv_dir = "build/tests"

# Circle interface for overlay: center (0, 0.1), radius 1
theta  = np.linspace(0, 2 * np.pi, 500)
circ_x = np.cos(theta)
circ_y = 0.1 + np.sin(theta)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Laplace Interior Dirichlet — circle, $[-2,2]^2$\n"
             "Pointwise error $|u_h - u|$ (interior only, log scale)",
             fontsize=13, fontweight='bold')

for ax, N in zip(axes.flat, Ns):
    csv_file = os.path.join(csv_dir, f"laplace_interior_circle_2d_N{N}.csv")
    if not os.path.exists(csv_file):
        ax.set_title(f"N={N} — CSV missing"); ax.axis('off'); continue

    df     = pd.read_csv(csv_file)
    x      = df['x'].values
    y      = df['y'].values
    label  = df['label'].values
    err    = np.abs(df['u_bulk'].values - df['u_exact'].values)

    xs = np.unique(x); ys = np.unique(y)
    nx_g, ny_g = len(xs), len(ys)
    X     = x.reshape(ny_g, nx_g)
    Y     = y.reshape(ny_g, nx_g)
    Label = label.reshape(ny_g, nx_g)
    Err   = err.reshape(ny_g, nx_g)

    interior = (Label == 1)
    emax = Err[interior].max()
    emin = max(emax * 1e-4, 1e-12)
    Err_vis = np.where(interior, Err, np.nan)

    norm = mcolors.LogNorm(vmin=emin, vmax=emax)
    im   = ax.pcolormesh(X, Y, Err_vis, norm=norm, shading='nearest', cmap='hot_r')
    ax.plot(circ_x, circ_y, 'w-', lw=0.7)
    ax.set_title(f"$N={N}$,  max err = {emax:.2e}", fontsize=11)
    ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out = "python/laplace_interior_circle_2d_results.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out}")
