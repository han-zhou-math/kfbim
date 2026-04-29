// Interior Dirichlet BVP:  -Δu = 0  in Ω_int (star domain),  u = g on Γ.
//
// Manufactured solution: u = (x-½)² - (y-½)²  (harmonic, Δu = 0).
// Formulation: single-layer potential.
//   GMRES iterate: σ = [∂u/∂n]  (flux jump, unknown)
//   Fixed jump:    [u] = 0
//   Operator T:    σ → trace(u|_Γ) from interior
//   System:        T σ = g,  g[q] = u_exact(x_q, y_q)
//
// After convergence the bulk solution should match u_exact on interior nodes.

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/grid/cartesian_grid_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "core/solver/zfft_bc_type.hpp"
#include "core/local_cauchy/jump_data.hpp"
#include "core/transfer/laplace_spread_2d.hpp"
#include "core/transfer/laplace_restrict_2d.hpp"
#include "core/operator/laplace_kfbi_operator.hpp"
#include "core/gmres/gmres.hpp"

using namespace kfbim;

static constexpr double kPi = 3.14159265358979323846;

// ── Manufactured solution ─────────────────────────────────────────────────────
// u = (x-½)² - (y-½)²   (harmonic: Δu = 2 - 2 = 0)

static double u_exact(double x, double y)
{
    double dx = x - 0.5, dy = y - 0.5;
    return dx * dx - dy * dy;
}

// ── Star interface (same geometry as other 2D tests) ─────────────────────────

static const double kGL_s[3] = {-0.7745966692414834, 0.0, +0.7745966692414834};
static const double kGL_w[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};

static Interface2D make_star_panels(int N_panels)
{
    const double cx = 0.5, cy = 0.5, R = 0.28, A = 0.40;
    const int K = 5;
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

// ── Full KFBI pipeline ────────────────────────────────────────────────────────

static double run_kfbi_dirichlet(int N)
{
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];

    auto iface = make_star_panels(N);
    GridPair2D gp(grid, iface);
    const int Nq = iface.num_points();

    // Domain labels for each grid node
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n)
        labels[n] = gp.domain_label(n);

    // base_rhs = 0  (harmonic problem, no body force)
    Eigen::VectorXd base_rhs = Eigen::VectorXd::Zero(nx * ny);

    // rhs_derivs = 0 at each interface point (f jump = 0)
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));

    // Build components
    LaplacePanelSpread2D        spread(gp);
    LaplaceFftBulkSolverZfft2D  bulk_solver(grid, ZfftBcType::Dirichlet);
    LaplaceQuadraticRestrict2D  restrict_op(gp);

    // KFBI operator in Dirichlet mode: maps σ=[∂u/∂n] → trace u|_Γ
    LaplaceKFBIOperator2D op(spread, bulk_solver, restrict_op,
                              base_rhs, rhs_derivs,
                              LaplaceKFBIMode::Dirichlet);

    // GMRES RHS: g[q] = u_exact at each interface point
    Eigen::VectorXd gmres_rhs(Nq);
    for (int q = 0; q < Nq; ++q)
        gmres_rhs[q] = u_exact(iface.points()(q, 0), iface.points()(q, 1));

    // Solve
    Eigen::VectorXd sigma = Eigen::VectorXd::Zero(Nq);
    GMRES gmres(200, 1e-10, 50);
    int iters = gmres.solve(op, gmres_rhs, sigma);

    std::printf("    N=%3d  GMRES iters=%3d  converged=%d  rel_res=%.2e\n",
                N, iters, static_cast<int>(gmres.converged()),
                gmres.residuals().empty() ? 0.0 : gmres.residuals().back());

    // Final bulk solve with converged σ
    std::vector<LaplaceJumpData2D> final_jumps(Nq);
    for (int q = 0; q < Nq; ++q) {
        final_jumps[q].u_jump     = 0.0;
        final_jumps[q].un_jump    = sigma[q];
        final_jumps[q].rhs_derivs = Eigen::VectorXd::Zero(1);
    }
    Eigen::VectorXd rhs = base_rhs;
    auto corr_polys = spread.apply(final_jumps, rhs);
    Eigen::VectorXd u_bulk;
    bulk_solver.solve(-rhs, u_bulk);

    // Max-norm error on interior nodes
    double err = 0.0;
    for (int j = 1; j < ny-1; ++j) {
        for (int i = 1; i < nx-1; ++i) {
            int n = j * nx + i;
            if (labels[n] != 1) continue;  // only interior nodes
            auto c = grid.coord(n);
            err = std::max(err, std::abs(u_bulk[n] - u_exact(c[0], c[1])));
        }
    }
    return err;
}

// ── Test ──────────────────────────────────────────────────────────────────────

TEST_CASE("LaplaceKFBIOperator2D: interior Dirichlet BVP, GMRES convergence",
          "[kfbi][laplace][gmres][2d]")
{
    SECTION("N=64: GMRES converges and interior error < 1e-2") {
        std::printf("\n  Interior Dirichlet BVP:  -Δu = 0,  u = (x-½)²-(y-½)²  on star Γ\n");
        std::printf("  Pipeline: KFBI operator (Dirichlet) + GMRES(50)\n");
        double err = run_kfbi_dirichlet(64);
        std::printf("    Interior max error: %.4e\n", err);
        REQUIRE(err < 1e-2);
    }
}

TEST_CASE("LaplaceKFBIOperator2D: interior Dirichlet BVP, grid convergence",
          "[kfbi][laplace][convergence][2d]")
{
    const int Ns[] = {32, 64, 128};
    const int n_levels = 3;
    double errors[n_levels];

    std::printf("\n  Interior Dirichlet BVP convergence:  -Δu = 0,  u = (x-½)²-(y-½)²\n");
    std::printf("  %6s  %12s  %8s\n", "N", "max_err", "rate");

    for (int l = 0; l < n_levels; ++l) {
        errors[l] = run_kfbi_dirichlet(Ns[l]);
        if (l == 0)
            std::printf("  %6d  %12.4e  %8s\n", Ns[l], errors[l], "—");
        else {
            double rate = std::log2(errors[l-1] / errors[l]);
            std::printf("  %6d  %12.4e  %8.3f\n", Ns[l], errors[l], rate);
            REQUIRE(rate > 1.0);
        }
    }
}
