#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/grid/cartesian_grid_2d.hpp"
#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "core/solver/zfft_bc_type.hpp"
#include "core/local_cauchy/jump_data.hpp"
#include "core/transfer/laplace_spread_2d.hpp"

using namespace kfbim;

static constexpr double kPi = 3.14159265358979323846;

// ============================================================================
// Manufactured solution
// ============================================================================

static double sol_u_int(double x, double y)
{
    return std::sin(2*kPi*x) * std::sin(2*kPi*y);
}

static double sol_u_ext(double x, double y)
{
    return std::sin(kPi*x) * std::sin(kPi*y);
}

static double sol_f_int(double x, double y)
{
    return 8.0 * kPi*kPi * std::sin(2*kPi*x) * std::sin(2*kPi*y);
}

static double sol_f_ext(double x, double y)
{
    return 2.0 * kPi*kPi * std::sin(kPi*x) * std::sin(kPi*y);
}

static double sol_dudn_int(double x, double y, double nx, double ny)
{
    double ux = 2 * kPi * std::cos(2*kPi*x) * std::sin(2*kPi*y);
    double uy = 2 * kPi * std::sin(2*kPi*x) * std::cos(2*kPi*y);
    return ux * nx + uy * ny;
}

static double sol_dudn_ext(double x, double y, double nx, double ny)
{
    double ux = kPi * std::cos(kPi*x) * std::sin(kPi*y);
    double uy = kPi * std::sin(kPi*x) * std::cos(kPi*y);
    return ux * nx + uy * ny;
}

// ============================================================================
// Geometry (same as test_iim_2d.cpp)
// ============================================================================

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

// ============================================================================
// Core IIM Spread Test
// ============================================================================

static double run_iim_spread_test(int N)
{
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d = grid.dof_dims();
    int nx = d[0], ny = d[1];

    auto iface = make_star_panels(N);
    GridPair2D gp(grid, iface);
    const int Nq = iface.num_points();

    // Domain labels
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n)
        labels[n] = gp.domain_label(n);

    // Build piecewise RHS
    Eigen::VectorXd base_rhs(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        int lbl = labels[n];
        bool bdy = (n % nx == 0 || n % nx == nx-1 || n / nx == 0 || n / nx == ny-1);
        base_rhs[n] = bdy ? 0.0 : ((lbl == 1) ? sol_f_int(c[0], c[1]) : sol_f_ext(c[0], c[1]));
    }

    // Build jump data and RHS derivatives at interface points
    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0);
        double y = iface.points()(q, 1);
        double nx_q = iface.normals()(q, 0);
        double ny_q = iface.normals()(q, 1);

        jumps[q].u_jump  = sol_u_int(x, y) - sol_u_ext(x, y);
        jumps[q].un_jump = sol_dudn_int(x, y, nx_q, ny_q) - sol_dudn_ext(x, y, nx_q, ny_q);
        
        // [f] = f_int - f_ext
        jumps[q].rhs_derivs.resize(1);
        jumps[q].rhs_derivs[0] = sol_f_int(x, y) - sol_f_ext(x, y);
    }

    // Apply LaplacePanelSpread2D
    LaplacePanelSpread2D spread(gp);
    Eigen::VectorXd rhs = base_rhs;
    spread.apply(jumps, rhs);

    // Solve using bulk solver
    LaplaceFftBulkSolverZfft2D bulk_solver(grid, ZfftBcType::Dirichlet);
    Eigen::VectorXd u_bulk;
    bulk_solver.solve(-rhs, u_bulk);

    // Verify error
    double max_err = 0.0;
    for (int j = 1; j < ny-1; ++j) {
        for (int i = 1; i < nx-1; ++i) {
            int n = j * nx + i;
            auto c = grid.coord(n);
            double u_exact = (labels[n] == 1) ? sol_u_int(c[0], c[1]) : sol_u_ext(c[0], c[1]);
            max_err = std::max(max_err, std::abs(u_bulk[n] - u_exact));
        }
    }

    return max_err;
}

TEST_CASE("LaplacePanelSpread2D + BulkSolver: Poisson interface problem manufactured solution",
          "[iim][spread][laplace][2d]")
{
    const int Ns[] = {32, 64, 128};
    const int n_levels = 3;
    double errors[n_levels];

    std::printf("\n  IIM Spread + BulkSolver convergence (star interface):\n");
    std::printf("  %6s  %12s  %8s\n", "N", "max_err", "rate");

    for (int l = 0; l < n_levels; ++l) {
        errors[l] = run_iim_spread_test(Ns[l]);
        if (l == 0)
            std::printf("  %6d  %12.4e  %8s\n", Ns[l], errors[l], "—");
        else {
            double rate = std::log2(errors[l-1] / errors[l]);
            std::printf("  %6d  %12.4e  %8.3f\n", Ns[l], errors[l], rate);
            REQUIRE(rate > 1.5);
        }
    }
}
