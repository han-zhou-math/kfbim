#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/grid/cartesian_grid_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "core/solver/zfft_bc_type.hpp"
#include "core/local_cauchy/jump_data.hpp"
#include "core/transfer/laplace_spread_2d.hpp"
#include "core/transfer/laplace_restrict_2d.hpp"
#include "core/problems/laplace_interface_solver_2d.hpp"

using namespace kfbim;

static constexpr double kPi = 3.14159265358979323846;

// ── Manufactured solution ────────────────────────────────────────────────────
//
//   u_int = sin(πx) sin(πy)       interior (label 1, inside the circle)
//   u_ext = sin(2πx) sin(2πy)     exterior (label 0, outside the circle)
//
// Both vanish on the box boundary x∈{0,1} or y∈{0,1}.

static double sol_u_int (double x, double y) { return std::sin(kPi*x)*std::sin(kPi*y); }
static double sol_u_ext (double x, double y) { return std::sin(2*kPi*x)*std::sin(2*kPi*y); }
static double sol_f_int (double x, double y) { return 2.0*kPi*kPi*std::sin(kPi*x)*std::sin(kPi*y); }
static double sol_f_ext (double x, double y) { return 8.0*kPi*kPi*std::sin(2*kPi*x)*std::sin(2*kPi*y); }

static double sol_ux_int (double x, double y) { return  kPi*std::cos(kPi*x)*std::sin(kPi*y); }
static double sol_uy_int (double x, double y) { return  kPi*std::sin(kPi*x)*std::cos(kPi*y); }
static double sol_ux_ext (double x, double y) { return 2*kPi*std::cos(2*kPi*x)*std::sin(2*kPi*y); }
static double sol_uy_ext (double x, double y) { return 2*kPi*std::sin(2*kPi*x)*std::cos(2*kPi*y); }

static double C_fn(double x, double y)   { return sol_u_int(x,y) - sol_u_ext(x,y); }

static double Cx_fn(double x, double y) {
    return sol_ux_int(x,y) - sol_ux_ext(x,y);
}
static double Cy_fn(double x, double y) {
    return sol_uy_int(x,y) - sol_uy_ext(x,y);
}

// ── Circle interface with 3 Gauss-Legendre points per panel ──────────────────

static const double kGL_s[3] = {-0.7745966692414834, 0.0, +0.7745966692414834};
static const double kGL_w[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};

static Interface2D make_circle_panels(double cx, double cy, double R, int N_panels)
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
            const double th = th_mid + half_dth * kGL_s[i];
            pts(q, 0) = cx + R * std::cos(th);
            pts(q, 1) = cy + R * std::sin(th);
            // Outward normal (pointing from interior to exterior)
            nml(q, 0) =  std::cos(th);
            nml(q, 1) =  std::sin(th);
            wts[q]    = kGL_w[i] * half_dth * R;
            ++q;
        }
    }
    return {pts, nml, wts, 3, comp};
}

// ── Complete solve-and-measure helper ─────────────────────────────────────────

struct ConvergenceData {
    double bulk_err;     // max |u_h − u_exact| on interior nodes
    double u_avg_err;    // max |u_avg - (u_int+u_ext)/2|
    double un_avg_err;   // max |un_avg - (∂u_int/∂n+∂u_ext/∂n)/2|
};

static ConvergenceData solve_and_measure(int N)
{
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];

    // Circle interface: radius 0.3 centered at (0.5, 0.5), N panels
    auto iface = make_circle_panels(0.5, 0.5, 0.3, N);
    GridPair2D gp(grid, iface);
    const int Nq = iface.num_points();

    // Domain labels
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n)
        labels[n] = gp.domain_label(n);

    // f_bulk at every grid node (RHS of −Δu = f) and exact solution
    Eigen::VectorXd f_bulk(nx * ny), u_exact(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        double x = c[0], y = c[1];
        int lbl = labels[n];
        u_exact[n] = (lbl == 1) ? sol_u_int(x, y) : sol_u_ext(x, y);
        bool bdy = (n % nx == 0 || n % nx == nx-1 || n / nx == 0 || n / nx == ny-1);
        f_bulk[n] = bdy ? 0.0 : ((lbl == 1) ? sol_f_int(x, y) : sol_f_ext(x, y));
    }

    // Jump data at interface quadrature points
    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0), y = iface.points()(q, 1);
        double nx_q = iface.normals()(q, 0), ny_q = iface.normals()(q, 1);
        jumps[q].u_jump  = C_fn(x, y);
        jumps[q].un_jump = Cx_fn(x, y) * nx_q + Cy_fn(x, y) * ny_q;
        jumps[q].rhs_derivs = Eigen::VectorXd::Constant(1, sol_f_int(x,y) - sol_f_ext(x,y));
    }

    // Build solver and run
    LaplacePanelSpread2D       spread(gp);
    LaplaceFftBulkSolverZfft2D bulk_solver(grid, ZfftBcType::Dirichlet);
    LaplaceQuadraticRestrict2D restrict_op(gp);
    LaplaceInterfaceSolver2D   solver(spread, bulk_solver, restrict_op);

    auto res = solver.solve(jumps, f_bulk);

    // --- Bulk error: max |u_h − u_exact| on interior nodes ---
    double bulk_err = 0.0;
    for (int j = 1; j < ny-1; ++j) {
        for (int i = 1; i < nx-1; ++i) {
            int n = j * nx + i;
            bulk_err = std::max(bulk_err, std::abs(res.u_bulk[n] - u_exact[n]));
        }
    }

    // --- Average errors ---
    double u_avg_err  = 0.0;
    double un_avg_err = 0.0;

    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0), y = iface.points()(q, 1);
        double nx_q = iface.normals()(q, 0), ny_q = iface.normals()(q, 1);

        double uint = sol_u_int(x,y), uext = sol_u_ext(x,y);
        double unint = sol_ux_int(x,y)*nx_q + sol_uy_int(x,y)*ny_q;
        double unext = sol_ux_ext(x,y)*nx_q + sol_uy_ext(x,y)*ny_q;

        double ex_u_avg  = 0.5 * (uint + uext);
        double ex_un_avg = 0.5 * (unint + unext);

        u_avg_err  = std::max(u_avg_err,  std::abs(res.u_avg[q]  - ex_u_avg));
        un_avg_err = std::max(un_avg_err, std::abs(res.un_avg[q] - ex_un_avg));
    }

    return {bulk_err, u_avg_err, un_avg_err};
}

// ── Tests ────────────────────────────────────────────────────────────────────

TEST_CASE("LaplaceInterfaceSolver2D: trace averages convergence",
          "[laplace][interface_solver][average][convergence][2d]")
{
    const int Ns[]     = {32, 64, 128, 256};
    const int n_levels = 4;
    ConvergenceData data[n_levels];

    std::printf("\n  LaplaceInterfaceSolver2D: interface average values convergence\n");
    std::printf("  Solver u_avg  vs Exact (u⁺+u⁻)/2\n");
    std::printf("  Solver un_avg vs Exact (∂u⁺/∂n+∂u⁻/∂n)/2\n");
    std::printf("  %6s  %12s  %8s  %12s  %8s\n",
                "N", "u_avg_err", "rate", "un_avg_err", "rate");

    for (int l = 0; l < n_levels; ++l) {
        data[l] = solve_and_measure(Ns[l]);
        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s  %12.4e  %8s\n",
                        Ns[l], data[l].u_avg_err, "—", data[l].un_avg_err, "—");
        } else {
            double rate_u  = std::log2(data[l-1].u_avg_err  / data[l].u_avg_err);
            double rate_un = std::log2(data[l-1].un_avg_err / data[l].un_avg_err);
            std::printf("  %6d  %12.4e  %8.3f  %12.4e  %8.3f\n",
                        Ns[l], data[l].u_avg_err, rate_u,
                        data[l].un_avg_err, rate_un);
            REQUIRE(rate_u  > 1.5);
            REQUIRE(rate_un > 1.0);
        }
    }
}
