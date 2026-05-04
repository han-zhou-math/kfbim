# Second-kind boundary integral equation for an elliptic interface problem

Consider the interface problem

$$
-\nabla\cdot(\beta\nabla u)+\kappa^2 u=f,
$$

with piecewise constant coefficients

$$
\beta=\beta^\pm,\qquad \kappa=\kappa^\pm,
$$

on the two sides of a smooth closed interface $\Gamma$. Let the unit normal $n$ point from $\Omega^-$ to $\Omega^+$, and define the jumps

$$
[u]=u^+-u^-=a,
$$

$$
[\beta u_n]=\beta^+\partial_n u^+ - \beta^-\partial_n u^-=b.
$$

In each phase, since $\beta^\pm$ is constant,

$$
\left(-\Delta+(\lambda^\pm)^2\right)u^\pm=\frac{f^\pm}{\beta^\pm},
\qquad
\lambda^\pm=\frac{\kappa^\pm}{\sqrt{\beta^\pm}}.
$$

For the homogeneous part, let $G^\pm$ be the Green's function satisfying

$$
\left(-\Delta_x+(\lambda^\pm)^2\right)G^\pm(x,y)=\delta(x-y).
$$

Define the layer potentials

$$
S^\pm\sigma(x)=\int_\Gamma G^\pm(x,y)\sigma(y)\,ds_y,
$$

$$
D^\pm\mu(x)=\int_\Gamma \partial_{n_y}G^\pm(x,y)\mu(y)\,ds_y.
$$

---

## 1. General ansatz

Take

$$
u^+(x)=\alpha^+D^+\mu(x)-S^+\sigma(x),
\qquad x\in\Omega^+,
$$

$$
u^-(x)=\alpha^-D^-\mu(x)-S^-\sigma(x),
\qquad x\in\Omega^-.
$$

The constants $\alpha^+$ and $\alpha^-$ are left undetermined initially.

---

## 2. Trace jump

Using the jump relations

$$
\gamma^+D^+\mu=\left(\frac12 I+K^+\right)\mu,
$$

$$
\gamma^-D^-\mu=\left(-\frac12 I+K^-\right)\mu,
$$

and the continuity of the single-layer trace,

$$
\gamma^\pm S^\pm\sigma=S^\pm\sigma,
$$

we obtain

$$
u^+|_\Gamma
=
\alpha^+\left(\frac12 I+K^+\right)\mu-S^+\sigma,
$$

$$
u^-|_\Gamma
=
\alpha^-\left(-\frac12 I+K^-\right)\mu-S^-\sigma.
$$

Hence

$$
[u]=u^+-u^-
$$

becomes

$$
[u]
=
\left[
\frac{\alpha^++\alpha^-}{2}I
+
\alpha^+K^+
-
\alpha^-K^-
\right]\mu
-
\left(S^+-S^-\right)\sigma.
$$

Thus the first interface condition gives

$$
\boxed{
\left[
\frac{\alpha^++\alpha^-}{2}I
+
\alpha^+K^+
-
\alpha^-K^-
\right]\mu
-
\left(S^+-S^-\right)\sigma
=
a.
}
$$

---

## 3. Flux jump

For the single layer,

$$
\partial_n^+S^+\sigma
=
-\frac12\sigma+(K^+)^*\sigma,
$$

$$
\partial_n^-S^-\sigma
=
\frac12\sigma+(K^-)^*\sigma.
$$

For the double layer,

$$
\partial_n^+D^+\mu=N^+\mu,
$$

$$
\partial_n^-D^-\mu=N^-\mu,
$$

where $N^\pm$ are the hypersingular boundary operators.

Therefore

$$
\partial_n^+u^+
=
\alpha^+N^+\mu
+
\frac12\sigma
-
(K^+)^*\sigma,
$$

and

$$
\partial_n^-u^-
=
\alpha^-N^-\mu
-
\frac12\sigma
-
(K^-)^*\sigma.
$$

The flux jump is

$$
[\beta u_n]
=
\beta^+\partial_n u^+
-
\beta^-\partial_n u^-.
$$

Therefore

$$
[\beta u_n]
=
\left(
\beta^+\alpha^+N^+
-
\beta^-\alpha^-N^-
\right)\mu
+
\frac{\beta^++\beta^-}{2}\sigma
-
\beta^+(K^+)^*\sigma
+
\beta^-(K^-)^*\sigma.
$$

Thus the second interface condition gives

$$
\boxed{
\left(
\beta^+\alpha^+N^+
-
\beta^-\alpha^-N^-
\right)\mu
+
\left[
\frac{\beta^++\beta^-}{2}I
-
\beta^+(K^+)^*
+
\beta^-(K^-)^*
\right]\sigma
=
b.
}
$$

---

## 4. General boundary integral system

The ansatz

$$
u^\pm=\alpha^\pm D^\pm\mu-S^\pm\sigma
$$

leads to

$$
\boxed{
\begin{aligned}
\left[
\frac{\alpha^++\alpha^-}{2}I
+
\alpha^+K^+
-
\alpha^-K^-
\right]\mu
-
\left(S^+-S^-\right)\sigma
&=a,\\[6pt]
\left(
\beta^+\alpha^+N^+
-
\beta^-\alpha^-N^-
\right)\mu
+
\left[
\frac{\beta^++\beta^-}{2}I
-
\beta^+(K^+)^*
+
\beta^-(K^-)^*
\right]\sigma
&=b.
\end{aligned}
}
$$

---

## 5. Condition for canceling the hypersingular principal part

The potentially noncompact term is

$$
\beta^+\alpha^+N^+ - \beta^-\alpha^-N^-.
$$

The hypersingular operators $N^+$ and $N^-$ have the same order-$+1$ principal singularity. Therefore the order-$+1$ part cancels if and only if

$$
\boxed{
\beta^+\alpha^+=\beta^-\alpha^-.
}
$$

Let the common value be $c$:

$$
\beta^+\alpha^+=\beta^-\alpha^-=c.
$$

Then

$$
\alpha^+=\frac{c}{\beta^+},
\qquad
\alpha^-=\frac{c}{\beta^-},
$$

and

$$
\beta^+\alpha^+N^+-\beta^-\alpha^-N^-
=
c(N^+-N^-).
$$

Since $N^+$ and $N^-$ have the same principal part,

$$
N^+-N^-
$$

is lower order, hence compact on a smooth closed interface in the usual Sobolev setting.

Without the condition

$$
\beta^+\alpha^+=\beta^-\alpha^-,
$$

the flux equation contains an uncanceled hypersingular order-$+1$ operator and the formulation is not genuinely second kind.

---

## 6. Convenient normalization

The first equation has identity coefficient

$$
\frac{\alpha^++\alpha^-}{2}I.
$$

A natural normalization is

$$
\frac{\alpha^++\alpha^-}{2}=1.
$$

Together with

$$
\beta^+\alpha^+=\beta^-\alpha^-,
$$

this gives

$$
\boxed{
\alpha^+=\frac{2\beta^-}{\beta^++\beta^-},
\qquad
\alpha^-=\frac{2\beta^+}{\beta^++\beta^-}.
}
$$

Then the second-kind system becomes

$$
\boxed{
\begin{aligned}
\left[
I
+
\frac{2\beta^-}{\beta^++\beta^-}K^+
-
\frac{2\beta^+}{\beta^++\beta^-}K^-
\right]\mu
-
\left(S^+-S^-\right)\sigma
&=a,\\[6pt]
\frac{2\beta^+\beta^-}{\beta^++\beta^-}
\left(N^+-N^-\right)\mu
+
\left[
\frac{\beta^++\beta^-}{2}I
-
\beta^+(K^+)^*
+
\beta^-(K^-)^*
\right]\sigma
&=b.
\end{aligned}
}
$$

This is second kind because the leading diagonal part is

$$
\begin{pmatrix}
I & 0\\
0 & \dfrac{\beta^++\beta^-}{2}I
\end{pmatrix},
$$

and the remaining operators are compact for smooth $\Gamma$.

---

## 7. Identity scaling in both equations

If desired, multiply the second equation by

$$
\frac{2}{\beta^++\beta^-}.
$$

Then the system becomes

$$
\boxed{
\begin{aligned}
\left[
I
+
\frac{2\beta^-}{\beta^++\beta^-}K^+
-
\frac{2\beta^+}{\beta^++\beta^-}K^-
\right]\mu
-
\left(S^+-S^-\right)\sigma
&=a,\\[6pt]
\frac{4\beta^+\beta^-}{(\beta^++\beta^-)^2}
\left(N^+-N^-\right)\mu
+
\left[
I
-
\frac{2\beta^+}{\beta^++\beta^-}(K^+)^*
+
\frac{2\beta^-}{\beta^++\beta^-}(K^-)^*
\right]\sigma
&=
\frac{2b}{\beta^++\beta^-}.
\end{aligned}
}
$$

---

## 8. Summary

Starting with

$$
u^\pm=\alpha^\pm D^\pm\mu-S^\pm\sigma,
$$

the flux equation contains

$$
\beta^+\alpha^+N^+-\beta^-\alpha^-N^-.
$$

To cancel the hypersingular principal part, choose $\alpha^\pm$ so that

$$
\boxed{
\beta^+\alpha^+=\beta^-\alpha^-.
}
$$

A convenient normalization is

$$
\boxed{
\alpha^+=\frac{2\beta^-}{\beta^++\beta^-},
\qquad
\alpha^-=\frac{2\beta^+}{\beta^++\beta^-}.
}
$$

With this choice, the hypersingular term becomes proportional to

$$
N^+-N^-,
$$

whose principal parts cancel. The resulting BIE is Fredholm second kind for a smooth closed interface.
