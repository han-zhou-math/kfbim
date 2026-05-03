import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

csv_file = "laplace_interior_2d_N128.csv"

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found. Please run the test first.")
    exit(1)

df = pd.read_csv(csv_file)

# Extract variables
x = df['x'].values
y = df['y'].values
u_bulk = df['u_bulk'].values
u_exact = df['u_exact'].values
label = df['label'].values

# The grid is structured, so we can reshape to 2D
N = int(np.sqrt(len(x))) - 1 # N is intervals, so N+1 nodes
nx = ny = N + 1

X = x.reshape(ny, nx)
Y = y.reshape(ny, nx)
U_bulk = u_bulk.reshape(ny, nx)
U_exact = u_exact.reshape(ny, nx)
Label = label.reshape(ny, nx)

# We will visualize the full domain, as the potentials define a field everywhere.
U_bulk_viz = U_bulk
U_exact_viz = U_exact
Err = np.abs(U_bulk_viz - U_exact_viz)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Laplace Interior Dirichlet BVP - Full Potential Field (N={N})", fontsize=14, fontweight='bold')

# Plot Exact Solution
im0 = axes[0].pcolormesh(X, Y, U_exact_viz, shading='auto', cmap='RdBu_r')
axes[0].set_title('Exact Solution (Zero-extended outside)')
axes[0].set_aspect('equal')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
fig.colorbar(im0, ax=axes[0])

# Plot Computed Solution
im1 = axes[1].pcolormesh(X, Y, U_bulk_viz, shading='auto', cmap='RdBu_r', 
                         vmin=U_exact_viz.min(), vmax=U_exact_viz.max())
axes[1].set_title('Computed Numerical Solution (Full Field)')
axes[1].set_aspect('equal')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
fig.colorbar(im1, ax=axes[1])

# Plot Error
vmax_err = max(Err.max(), 1e-10)
im2 = axes[2].pcolormesh(X, Y, Err, shading='auto', cmap='hot_r', vmax=vmax_err)
axes[2].set_title(f'Pointwise Error (Max = {Err.max():.2e})')
axes[2].set_aspect('equal')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
out_file = "python/laplace_interior_2d_results.png"
plt.savefig(out_file, dpi=150, bbox_inches='tight')
print(f"Visualization saved to {out_file}")
