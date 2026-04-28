// Full end-to-end test: -Δu = f (piecewise smooth), [u] = a, [∂_n u] = b on Γ,
// u = 0 on ∂Ω.
//
// Pipeline:
//   1. Package jump data into LaplaceJumpData2D at each Gauss point.
//   2. LaplacePanelSpread2D::apply  → RHS correction (IIM defect) via panel Cauchy solver.
//   3. LaplaceFftBulkSolverZfft2D::solve  → numerical solution u_h.
//   4. Check convergence over N = {32, 64, 128, 256}, expect ≥ O(h^1.5).

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>

#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/grid/cartesian_grid_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "core/solver/zfft_bc_type.hpp"
#include "core/local_cauchy/jump_data.hpp"
#include "core/transfer/laplace_spread_2d.hpp"

using namespace kfbim;

static constexpr double kPi = 3.14159265358979323846;

// ── Manufactured solution ────────────────────────────────────────────────────
//
//   u⁺ = sin(πx)sin(πy)        outside Γ,   -Δu⁺ = 2π²u⁺
//   u⁻ = sin(2πx)sin(2πy)      inside  Γ,   -Δu⁻ = 8π²u⁻
//   C  = u⁺ − u⁻                                   (correction function)

static double u_plus (double x, double y) { return std::sin(kPi*x)*std::sin(kPi*y); }
static double u_minus(double x, double y) { return std::sin(2*kPi*x)*std::sin(2*kPi*y); }
static double f_plus (double x, double y) { return 2.0*kPi*kPi*std::sin(kPi*x)*std::sin(kPi*y); }
static double f_minus(double x, double y) { return 8.0*kPi*kPi*std::sin(2*kPi*x)*std::sin(2*kPi*y); }

static double C_fn (double x, double y) { return u_plus(x,y) - u_minus(x,y); }
static double Cx_fn(double x, double y) {
    return  kPi*std::cos(kPi*x)*std::sin(kPi*y)
          - 2*kPi*std::cos(2*kPi*x)*std::sin(2*kPi*y);
}
static double Cy_fn(double x, double y) {
    return  kPi*std::sin(kPi*x)*std::cos(kPi*y)
          - 2*kPi*std::sin(2*kPi*x)*std::cos(2*kPi*y);
}

// ── Star interface with 3 Gauss-Legendre points per panel ────────────────────

static const double kGL_s[3] = {-0.7745966692414834, 0.0, +0.7745966692414834};
static const double kGL_w[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};

static Interface2D make_star_panels(double cx, double cy,
                                    double R, double A, int K, int N_panels)
{
    const int Nq = 3 * N_panels;
    Eigen::MatrixX2d pts(Nq, 2), nml(Nq, 2);
    Eigen::VectorXd  wts(Nq);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(N_panels);
    const double dth = 2.0 * kPi / N_panels;
    int q = 0;
    for (int p = 0; p < N_panels; ++p) {
        const double th_mid   = (p + 0.5) * dth;
        const double half_dth = 0.5 * dth;
        for (int i = 0; i < 3; ++i) {
            const double th   = th_mid + half_dth * kGL_s[i];
            const double r    = R * (1.0 + A * std::cos(K * th));
            const double drdt = -R * A * K * std::sin(K * th);
            pts(q, 0) = cx + r * std::cos(th);
            pts(q, 1) = cy + r * std::sin(th);
            const double tx   = drdt * std::cos(th) - r * std::sin(th);
            const double ty   = drdt * std::sin(th) + r * std::cos(th);
            const double tlen = std::sqrt(tx*tx + ty*ty);
            nml(q, 0) =  ty / tlen;
            nml(q, 1) = -tx / tlen;
            wts[q]    = kGL_w[i] * half_dth * tlen;
            ++q;
        }
    }
    return {pts, nml, wts, 3, comp};
}

// ── Full pipeline helper ──────────────────────────────────────────────────────

static double solve_and_measure(int N)
{
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];

    // Interface: N panels of 3 Gauss points each
    auto iface = make_star_panels(0.5, 0.5, 0.28, 0.40, 5, N);
    GridPair2D gp(grid, iface);

    // Domain labels
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n) labels[n] = gp.domain_label(n);

    // RHS f and exact solution at every grid node
    Eigen::VectorXd f_arr(nx * ny), u_exact(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        double x = c[0], y = c[1];
        int lbl = labels[n];
        u_exact[n] = (lbl == 1) ? u_minus(x, y) : u_plus(x, y);
        bool bdy   = (n % nx == 0 || n % nx == nx-1 || n / nx == 0 || n / nx == ny-1);
        f_arr[n]   = bdy ? 0.0 : ((lbl == 1) ? f_minus(x, y) : f_plus(x, y));
    }

    // Jump data at interface Gauss points: [u]=a, [∂_n u]=b, [f] = f⁺−f⁻
    const int Nq = iface.num_points();
    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x  = iface.points()(q, 0), y  = iface.points()(q, 1);
        double nx_q = iface.normals()(q, 0), ny_q = iface.normals()(q, 1);
        jumps[q].u_jump     = C_fn(x, y);
        jumps[q].un_jump    = Cx_fn(x, y)*nx_q + Cy_fn(x, y)*ny_q;
        jumps[q].rhs_derivs = Eigen::VectorXd::Constant(1, f_plus(x,y) - f_minus(x,y));
    }

    // Spread: panel Cauchy solve + IIM RHS correction
    LaplacePanelSpread2D spread(gp);
    Eigen::VectorXd rhs_correction = Eigen::VectorXd::Zero(nx * ny);
    spread.apply(jumps, rhs_correction);

    // FFT solve: Δ_h u = −(f + rhs_correction)
    LaplaceFftBulkSolverZfft2D solver(grid, ZfftBcType::Dirichlet);
    Eigen::VectorXd u_h;
    solver.solve(-(f_arr + rhs_correction), u_h);

    // Max-norm error on interior nodes
    double err = 0.0;
    for (int j = 1; j < ny-1; ++j)
        for (int i = 1; i < nx-1; ++i)
            err = std::max(err, std::abs(u_h[j*nx+i] - u_exact[j*nx+i]));
    return err;
}

// ── Test ─────────────────────────────────────────────────────────────────────

TEST_CASE("Laplace interface problem: end-to-end convergence — panel Cauchy + FFT",
          "[laplace][iface][convergence][2d]")
{
    const int    Ns[]     = {32, 64, 128, 256};
    const int    n_levels = 4;
    double       errors[n_levels];

    std::printf("\n  Full interface problem: -Δu = f,  [u]=a,  [∂_n u]=b,  u=0 on ∂Ω\n");
    std::printf("  Pipeline: panel Cauchy (Layer 1.5) → Spread (Layer 1) → FFT solve\n");
    std::printf("  %6s  %12s  %8s\n", "N", "max_err", "rate");

    for (int l = 0; l < n_levels; ++l) {
        errors[l] = solve_and_measure(Ns[l]);
        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s\n", Ns[l], errors[l], "—");
        } else {
            double rate = std::log2(errors[l-1] / errors[l]);
            std::printf("  %6d  %12.4e  %8.3f\n", Ns[l], errors[l], rate);
        }
    }

    for (int l = 1; l < n_levels; ++l) {
        double rate = std::log2(errors[l-1] / errors[l]);
        REQUIRE(rate > 1.5);
    }
}
