#pragma once

#include <cassert>
#include <Eigen/Dense>
#include "../interface/interface_2d.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Panel-based collocation Cauchy solver for the 2D Laplace equation.
//
// Solves the local Cauchy problem at each interface Gauss point q:
//   −ΔC = f⁺ − f⁻   (in a small neighbourhood of the interface)
//   C    = a          (Dirichlet jump on the interface)
//   ∂_n C = b         (Neumann jump on the interface)
//
// For each Gauss point (center), a 6×6 collocation system is assembled
// using points sampled from the containing panel:
//   Rows 0–2  Dirichlet : all 3 panel Gauss pts         → value [u]
//   Rows 3–4  Neumann   : panel pts 0 and 2              → normal-deriv [∂_n u]·h
//   Row  5    PDE       : center − 0.5h · outward normal → (−Δ+κ)C · h²
//
// Monomial basis (Taylor-scaled):
//   φ₀=1,  φ₁=dx,  φ₂=dy,  φ₃=½dx²,  φ₄=½dy²,  φ₅=dx·dy
// where dx=(x−cx)/h, dy=(y−cy)/h, h=panel arc-length.
//
// After QR solve and rescaling (c[1:2]/=h, c[3:5]/=h²), the output is:
//   c[0]=C,  c[1]=Cx,  c[2]=Cy,  c[3]=Cxx,  c[4]=Cyy,  c[5]=Cxy
// all evaluated at the center point.
//
// Reference: solve2DCauchyProblem2 in old-codes/CauchyProblemSolvers.h
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Per-node 6×6 solve  (internal helper, mirrors solve2DCauchyProblem2)
//
// bdry1[3][2]     : coords of 3 Dirichlet collocation pts
// a[3]            : [u] at bdry1 pts
// bdry2[2][2]     : coords of 2 Neumann collocation pts
// bdry2_nml[2][2] : outward normals at bdry2 pts
// b[2]            : [∂_n u] at bdry2 pts
// bulk[2]         : coord of 1 interior PDE collocation pt
// Lu              : (f⁺ − f⁻) at bulk pt
// center[2]       : expansion origin (the target interface point)
// kappa           : coefficient of u in PDE (0 for pure Laplace)
// h               : scaling parameter (panel arc-length)
// c[6]            : output Taylor coefficients at center (see header doc)
// ---------------------------------------------------------------------------
inline void solve_local_6x6_2d(
    const double bdry1[3][2], const double a[3],
    const double bdry2[2][2], const double bdry2_nml[2][2], const double b[2],
    const double bulk[2], double Lu,
    const double center[2], double kappa, double h,
    double c[6])
{
    const double h2      = h * h;
    const double kap_h2  = kappa * h2;

    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 1> rhs;
    A.setZero();

    int m = 0;

    // ── Rows 0–2: Dirichlet  [u] = a ─────────────────────────────────────
    for (int l = 0; l < 3; ++l) {
        const double dx = (bdry1[l][0] - center[0]) / h;
        const double dy = (bdry1[l][1] - center[1]) / h;
        A(m,0) = 1.0;
        A(m,1) = dx;          A(m,2) = dy;
        A(m,3) = 0.5*dx*dx;   A(m,4) = 0.5*dy*dy;  A(m,5) = dx*dy;
        rhs[m] = a[l];
        ++m;
    }

    // ── Rows 3–4: Neumann  [∂_n u] = b  (scaled by h) ───────────────────
    for (int l = 0; l < 2; ++l) {
        const double dx = (bdry2[l][0] - center[0]) / h;
        const double dy = (bdry2[l][1] - center[1]) / h;
        const double nx = bdry2_nml[l][0];
        const double ny = bdry2_nml[l][1];
        A(m,0) = 0.0;
        A(m,1) = nx;               A(m,2) = ny;
        A(m,3) = dx*nx;            A(m,4) = dy*ny;
        A(m,5) = nx*dy + dx*ny;
        rhs[m] = b[l] * h;
        ++m;
    }

    // ── Row 5: PDE  (−Δ + κ)C = Lu  (scaled by h²) ──────────────────────
    {
        const double dx = (bulk[0] - center[0]) / h;
        const double dy = (bulk[1] - center[1]) / h;
        A(m,0) = kap_h2;
        A(m,1) = kap_h2*dx;           A(m,2) = kap_h2*dy;
        A(m,3) = kap_h2*0.5*dx*dx - 1.0;
        A(m,4) = kap_h2*0.5*dy*dy - 1.0;
        A(m,5) = kap_h2*dx*dy;
        rhs[m] = Lu * h2;
        ++m;
    }

    // ── Column-norm preconditioning (improves QR conditioning) ───────────
    Eigen::Matrix<double,6,1> col_scale;
    for (int k = 0; k < 6; ++k) {
        col_scale[k] = A.col(k).norm();
        if (col_scale[k] > 1e-14) A.col(k) /= col_scale[k];
        else                       col_scale[k] = 1.0;
    }

    // ── QR solve ─────────────────────────────────────────────────────────
    Eigen::Matrix<double,6,1> c_raw =
        A.colPivHouseholderQr().solve(rhs);

    for (int k = 0; k < 6; ++k)
        c_raw[k] /= col_scale[k];

    // ── Rescale: raw poly coefficients → physical derivatives at center ───
    // c[k] from QR: coefficient of φₖ(dx,dy)
    // After rescaling: c[k] = k-th partial derivative of C at center
    //   c[0]  = C
    //   c[1] /= h   → Cx
    //   c[2] /= h   → Cy
    //   c[3] /= h²  → Cxx   (basis was ½dx², d²/dx²= 1/h² factor)
    //   c[4] /= h²  → Cyy
    //   c[5] /= h²  → Cxy
    c[0] = c_raw[0];
    c[1] = c_raw[1] / h;
    c[2] = c_raw[2] / h;
    c[3] = c_raw[3] / h2;
    c[4] = c_raw[4] / h2;
    c[5] = c_raw[5] / h2;
}

// ---------------------------------------------------------------------------
// Output type: per-quadrature-point derivatives of C
// ---------------------------------------------------------------------------
struct PanelCauchyResult2D {
    Eigen::VectorXd C;    // C     at each Gauss pt
    Eigen::VectorXd Cx;   // ∂C/∂x
    Eigen::VectorXd Cy;   // ∂C/∂y
    Eigen::VectorXd Cxx;  // ∂²C/∂x²
    Eigen::VectorXd Cyy;  // ∂²C/∂y²
    Eigen::VectorXd Cxy;  // ∂²C/∂x∂y
};

// ---------------------------------------------------------------------------
// laplace_panel_cauchy_2d
//
// Solves the local Cauchy problem at every Gauss point of the interface.
// Requires iface.points_per_panel() == 3.
//
// Inputs:
//   iface      — Interface2D with 3 Gauss points per panel
//   a          — [u]     at each Gauss point (length Nq)
//   b          — [∂_n u] at each Gauss point (length Nq)
//   Lu_iface   — f⁺−f⁻  at each Gauss point (length Nq);
//                used as a constant approximation of Lu at the interior bulk
//                point (O(h) error, consistent with order-2 accuracy)
//   kappa      — coefficient of u in the PDE (0 for pure Laplace)
//
// Returns PanelCauchyResult2D with C, Cx, Cy, Cxx, Cyy, Cxy at every node.
// ---------------------------------------------------------------------------
inline PanelCauchyResult2D laplace_panel_cauchy_2d(
    const Interface2D&     iface,
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& Lu_iface,
    double                 kappa = 0.0)
{
    assert(iface.points_per_panel() == 3);
    const int Nq = iface.num_points();
    const int Np = iface.num_panels();

    PanelCauchyResult2D res;
    res.C   = Eigen::VectorXd::Zero(Nq);
    res.Cx  = Eigen::VectorXd::Zero(Nq);
    res.Cy  = Eigen::VectorXd::Zero(Nq);
    res.Cxx = Eigen::VectorXd::Zero(Nq);
    res.Cyy = Eigen::VectorXd::Zero(Nq);
    res.Cxy = Eigen::VectorXd::Zero(Nq);

    for (int p = 0; p < Np; ++p) {

        // Global indices of the 3 Gauss points on this panel
        const int g0 = iface.point_index(p, 0);
        const int g1 = iface.point_index(p, 1);
        const int g2 = iface.point_index(p, 2);

        // Panel arc-length (sum of quadrature weights) = scaling parameter h
        const double h = iface.weights()[g0] + iface.weights()[g1] + iface.weights()[g2];

        // Dirichlet collocation: all 3 panel Gauss points
        double bdry1[3][2] = {
            { iface.points()(g0, 0), iface.points()(g0, 1) },
            { iface.points()(g1, 0), iface.points()(g1, 1) },
            { iface.points()(g2, 0), iface.points()(g2, 1) }
        };
        double av[3] = { a[g0], a[g1], a[g2] };

        // Neumann collocation: panel points 0 and 2 (span the panel endpoints)
        double bdry2[2][2] = {
            { iface.points()(g0, 0), iface.points()(g0, 1) },
            { iface.points()(g2, 0), iface.points()(g2, 1) }
        };
        double bdry2_nml[2][2] = {
            { iface.normals()(g0, 0), iface.normals()(g0, 1) },
            { iface.normals()(g2, 0), iface.normals()(g2, 1) }
        };
        double bv[2] = { b[g0], b[g2] };

        // Solve for each of the 3 Gauss points as the local center
        const int gidx[3] = { g0, g1, g2 };
        for (int i = 0; i < 3; ++i) {
            const int gi = gidx[i];

            double center[2] = { iface.points()(gi, 0), iface.points()(gi, 1) };
            double nx_i = iface.normals()(gi, 0);
            double ny_i = iface.normals()(gi, 1);

            // Interior bulk point: shift half a panel arc-length inward
            double bulk[2] = {
                center[0] - 0.5 * h * nx_i,
                center[1] - 0.5 * h * ny_i
            };

            // Constant approximation of Lu at bulk point (O(h) error)
            double Lu = Lu_iface[gi];

            double c[6];
            solve_local_6x6_2d(bdry1, av,
                                bdry2, bdry2_nml, bv,
                                bulk, Lu,
                                center, kappa, h,
                                c);

            res.C  [gi] = c[0];
            res.Cx [gi] = c[1];
            res.Cy [gi] = c[2];
            res.Cxx[gi] = c[3];
            res.Cyy[gi] = c[4];
            res.Cxy[gi] = c[5];
        }
    }

    return res;
}

} // namespace kfbim
