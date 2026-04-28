# IIM Defect Correction for the 2D Interface Poisson Problem

## Problem Statement

Solve $-\Delta u = f$ on $\Omega = [0,1]^2$ with homogeneous Dirichlet BC, where
$u$ and $f$ are piecewise smooth across an interface $\Gamma$:

$$
[u]_\Gamma = u^+ - u^-\ , \qquad
[\partial_n u]_\Gamma = \partial_n u^+ - \partial_n u^-
$$

The solution is $u = u^-$ inside $\Gamma$ and $u = u^+$ outside.

## Grid and Stencil

Uniform grid with spacing $h$, node $(i,j)$ at $(x_i, y_j) = (ih, jh)$.
Standard 5-point stencil:

$$
L_h[u]_{i,j}
  = \frac{4u_{i,j} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1}}{h^2}
  \approx -\Delta u_{i,j} + O(h^2)
$$

## Regular vs Irregular Nodes

- **Regular node**: all four face-adjacent neighbors are on the same side of $\Gamma$.  
  Standard FDM is $O(h^2)$ accurate.

- **Irregular node**: at least one neighbor is on the opposite side.  
  The stencil picks up a value from the wrong branch, introducing an $O(1/h^2)$ error that destroys convergence.

## Correction Function

Define the **correction function** (smooth everywhere):

$$
C(\mathbf{x}) = u^+(\mathbf{x}) - u^-(\mathbf{x})
$$

At every grid node, regardless of which side contains the node,

$$
C_n = C(\mathbf{x}_n)
    = u^+(\mathbf{x}_n) - u^-(\mathbf{x}_n).
$$

Thus $C_n$ is not the physical jump in the computed piecewise solution at node
$n$; it is the value of the smooth extension difference $u^+ - u^-$ evaluated
at that node.

For the manufactured solution below,

$$
C_{i,j}
  = \sin(\pi x_i)\sin(\pi y_j)
    - \sin(2\pi x_i)\sin(2\pi y_j).
$$

## Defect Correction Derivation

Let the discrete Heaviside/indicator be

$$
H_n =
\begin{cases}
1, & \mathbf{x}_n \in \Omega^-,\\
0, & \mathbf{x}_n \in \Omega^+.
\end{cases}
$$

Equivalently, if a level set $\phi$ is negative inside, then
$H_n = \mathcal{H}(-\phi_n)$ with the nodal Heaviside
$\mathcal{H}(z)=1$ for $z>0$ and $\mathcal{H}(z)=0$ for $z<0$.

The piecewise solution and source may be written as

$$
u_n = H_n u^-_n + (1-H_n)u^+_n
    = u^+_n - H_n C_n,
$$

$$
f_n = H_n f^-_n + (1-H_n)f^+_n.
$$

### Case 1: $n \in \Omega^-$, neighbor $n_b \in \Omega^+$

The actual value $u(\mathbf{x}_{n_b}) = u^+(\mathbf{x}_{n_b}) = u^-(\mathbf{x}_{n_b}) + C(\mathbf{x}_{n_b})$.

Applying $L_h$ at node $n$, the stencil uses the true (piecewise) $u$:

$$
L_h[u]_n
  = L_h[u^-]_n - \frac{C(\mathbf{x}_{n_b})}{h^2}
  = f^-_n + O(h^2) - \frac{C(\mathbf{x}_{n_b})}{h^2}
$$

The $O(1/h^2)$ term is the defect.

### Case 2: $n \in \Omega^+$, neighbor $n_b \in \Omega^-$

The actual value $u(\mathbf{x}_{n_b}) = u^-(\mathbf{x}_{n_b}) = u^+(\mathbf{x}_{n_b}) - C(\mathbf{x}_{n_b})$.

$$
L_h[u]_n
  = L_h[u^+]_n + \frac{C(\mathbf{x}_{n_b})}{h^2}
  = f^+_n + O(h^2) + \frac{C(\mathbf{x}_{n_b})}{h^2}
$$

### Unified Formula

Using domain labels $\ell_n \in \{0,1\}$ (0 = outside, 1 = inside):

$$
\boxed{
  F_n = f_n + \sum_{\substack{n_b \sim n,\; n_b \in \mathrm{int} \\ \ell_{n_b} \neq \ell_n}}
        \frac{\ell_{n_b} - \ell_n}{h^2}\, C(\mathbf{x}_{n_b})
}
$$

where the sum is over all four face-adjacent interior neighbors that cross the interface.

- $n \in \Omega^-,\; n_b \in \Omega^+$: $\ell_{n_b}-\ell_n = -1$, correction $= -C_{n_b}/h^2$
- $n \in \Omega^+,\; n_b \in \Omega^-$: $\ell_{n_b}-\ell_n = +1$, correction $= +C_{n_b}/h^2$

At regular nodes no neighbors cross, so $F_n = f_n$.

### Heaviside Form

With $H_n=\ell_n$, the same correction can be written without an explicit
case split:

$$
\boxed{
F_n
  = f_n
    + \frac{1}{h^2}
      \sum_{\substack{n_b \sim n,\; n_b \in \mathrm{int}}}
      \left(H_{n_b}-H_n\right) C_{n_b}
}
$$

Only edges whose endpoints lie on opposite sides contribute, because
$H_{n_b}-H_n=0$ for same-side neighbors. The two possible signs are

$$
\begin{aligned}
H_n=1,\ H_{n_b}=0
&\Rightarrow
\left(H_{n_b}-H_n\right)C_{n_b}
  = -C_{n_b},\\
H_n=0,\ H_{n_b}=1
&\Rightarrow
\left(H_{n_b}-H_n\right)C_{n_b}
  = +C_{n_b}.
\end{aligned}
$$

This sign comes directly from the neighbor coefficient $-1/h^2$ in the
5-point stencil: if a neighbor value has to be replaced by the other smooth
branch, the resulting defect enters with the negative of the neighbor-value
difference.

## Nodewise Finite Difference Scheme

For an interior grid node $(i,j)$, define

$$
H_{i,j} =
\begin{cases}
1, & (x_i,y_j)\in\Omega^-,\\
0, & (x_i,y_j)\in\Omega^+,
\end{cases}
\qquad
C_{i,j}=u^+(x_i,y_j)-u^-(x_i,y_j).
$$

The corrected right-hand side is

$$
\boxed{
\begin{aligned}
F_{i,j}
  = f_{i,j}
    + \frac{1}{h^2}\Big[
      &(H_{i-1,j}-H_{i,j})C_{i-1,j}
       +(H_{i+1,j}-H_{i,j})C_{i+1,j}\\
      &+(H_{i,j-1}-H_{i,j})C_{i,j-1}
       +(H_{i,j+1}-H_{i,j})C_{i,j+1}
    \Big],
\end{aligned}
}
$$

where terms corresponding to non-interior Dirichlet boundary nodes are omitted
or moved into the usual boundary contribution.

Then the finite difference equation is

$$
\boxed{
\frac{
4u_{i,j}
-u_{i-1,j}
-u_{i+1,j}
-u_{i,j-1}
-u_{i,j+1}
}{h^2}
= F_{i,j}
}
$$

for all interior nodes. In implementation form:

```text
F[i,j] = f[i,j]
for each face neighbor (p,q) of (i,j):
    if (p,q) is an interior node:
        F[i,j] += (H[p,q] - H[i,j]) * C[p,q] / h^2
```

## Easier Implementation When Branch Values Are Available

For a manufactured solution, or for any problem where both smooth extensions
$u^+$ and $u^-$ can be evaluated at grid nodes, do not form $C$ from the
interface jump data $[u]$ and $[\partial_n u]$. Instead, evaluate

$$
C_{i,j}=u^+(x_i,y_j)-u^-(x_i,y_j)
$$

directly at grid nodes and use the finite difference formula above. This is
the simplest and least error-prone implementation.

Equivalently, the correction can be computed edge by edge without storing a
separate $C$ array:

```text
F[i,j] = f[i,j]
for each face neighbor (p,q) of (i,j):
    if (p,q) is an interior node and H[p,q] != H[i,j]:
        Cpq = u_plus(x[p], y[q]) - u_minus(x[p], y[q])
        F[i,j] += (H[p,q] - H[i,j]) * Cpq / h^2
```

or, using the two cases explicitly,

```text
if node (i,j) is inside and neighbor (p,q) is outside:
    F[i,j] -= (u_plus(p,q) - u_minus(p,q)) / h^2

if node (i,j) is outside and neighbor (p,q) is inside:
    F[i,j] += (u_plus(p,q) - u_minus(p,q)) / h^2
```

This avoids closest-point projection, curvature, tangential derivatives, and
Taylor expansion of the jump data.

For a general interface problem where only $[u]$ and $[\partial_n u]$ are
prescribed on $\Gamma$, one cannot avoid approximating the off-interface value
$C_{p,q}$ in some form. The correction needs the difference between the two
smooth branches at the neighboring grid node, not only the jump at the closest
interface point. In that setting the usual IIM step is a local Taylor extension
from the interface data, for example

$$
C(\mathbf{x}_{p,q})
  \approx [u](\mathbf{x}_\Gamma)
        + d\, [\partial_n u](\mathbf{x}_\Gamma)
$$

where $\mathbf{x}_\Gamma$ is the closest interface point and $d$ is the signed
normal distance from $\mathbf{x}_\Gamma$ to $\mathbf{x}_{p,q}$. Higher-order
versions add tangential and second-normal terms. Thus the easy branch-value
method is preferred when $u^\pm$ are known; otherwise the Taylor reconstruction
of $C$ is the information needed to make the correction.

## Modified System

The corrected system $L_h[\mathbf{u}] = \mathbf{F}$ is solved by FFT (DST-I for homogeneous Dirichlet BC). Note the solver convention: `LaplaceFftBulkSolverZfft2D` solves $\Delta_h u = \text{rhs}$, so pass **$-F$**:

```
solver.solve(-F, u)
```

## Local Truncation Error After Correction

After substituting the corrected RHS, the modified equation is satisfied with $O(h^2)$ residual at every interior node (regular and irregular), giving **global 2nd order convergence** in the $\ell^\infty$ norm.

## Manufactured Solution (Test Case)

Box $[0,1]^2$, star interface $r(\theta) = 0.28(1 + 0.40\cos 5\theta)$ centered at $(0.5,0.5)$.

| Region  | Exact solution | $-\Delta u$ |
|---------|---------------|-------------|
| $\Omega^+$ (outside) | $u^+ = \sin(\pi x)\sin(\pi y)$ | $f^+ = 2\pi^2 u^+$ |
| $\Omega^-$ (inside)  | $u^- = \sin(2\pi x)\sin(2\pi y)$ | $f^- = 8\pi^2 u^-$ |

Correction function: $C = u^+ - u^- = \sin(\pi x)\sin(\pi y) - \sin(2\pi x)\sin(2\pi y)$

Both $u^\pm$ vanish on $\partial\Omega$, so homogeneous Dirichlet BC is satisfied exactly.

## Observed Convergence

| $N$ | $\|e\|_\infty$ | Rate |
|-----|---------------|------|
| 32  | 1.64 × 10⁻³  | —    |
| 64  | 4.00 × 10⁻⁴  | 2.03 |
| 128 | 9.95 × 10⁻⁵  | 2.01 |
| 256 | 2.49 × 10⁻⁵  | 2.00 |
