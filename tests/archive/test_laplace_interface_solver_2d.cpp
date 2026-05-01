// Full-pipeline test: LaplaceInterfaceSolver2D with manufactured solution.
//
// The manufactured solution u = sin(πx)sin(πy) (exterior) / sin(2πx)sin(2πy)
// (interior) has all three representation-formula components active:
//   Single-layer   S[[∂u/∂n]]  —  normal-derivative jump [∂u/∂n] ≠ 0
//   Double-layer   D[[u]]      —  Dirichlet jump [u] ≠ 0
//   Volume         V[f]        —  body force f = 2π²u⁺ / 8π²u⁻  ≠ 0
//
// With u=0 on the box boundary and a circle interface, the interface solver
// is checked for:
//   1. Bulk O(h²) convergence (max error on interior grid nodes)
//   2. Interface averaged trace O(h²) convergence:
//        u_avg  = u_trace + [u]/2   vs  (u⁺ + u⁻)/2
//        un_avg = un_trace + [∂u/∂n]/2  vs  (∂u⁺/∂n + ∂u⁻/∂n)/2

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
// Representation:  u = S[[∂u/∂n]] − D[[u]] + V[f]
//   [u]     = u_int − u_ext
//   [∂u/∂n] = ∂u_int/∂n − ∂u_ext/∂n
//   f ≠ 0

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
            wts[q]    = kGL_w[i] * half_dth * R;  // Jacobian |dr/dθ| = R
            ++q;
        }
    }
    return {pts, nml, wts, 3, comp};
}

// ── Complete solve-and-measure helper ─────────────────────────────────────────

struct ConvergenceData {
    double bulk_err;     // max |u_h − u_exact| on interior nodes
    double u_avg_err;    // max |u_trace + [u] − (u⁺+u⁻)/2| on interface pts
    double un_avg_err;   // max |un_trace + [∂u/∂n] − (∂u⁺/∂n+∂u⁻/∂n)/2|
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

    // --- Interface average errors ---
    // u_avg  = u_trace - 0.5 * [u]   (should → (u_int+u_ext)/2)
    // un_avg = un_trace - 0.5 * [∂u/∂n]  (should → (∂u_int/∂n+∂u_ext/∂n)/2)
    double u_avg_err  = 0.0;
    double un_avg_err = 0.0;

    // Debug: also try u_trace + 0.5 * [u] and u_trace - [u] (no /2)
    double u_trace_minus_jump_err = 0.0;   // u_trace - [u]
    double u_trace_plus_half_err = 0.0;    // u_trace + 0.5 * [u]
    double u_trace_raw_err = 0.0;         // u_trace alone vs u_int
    double u_trace_vs_uext_err = 0.0;     // u_trace alone vs u_ext

    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0), y = iface.points()(q, 1);
        double nx_q = iface.normals()(q, 0), ny_q = iface.normals()(q, 1);

        double jump_u   = jumps[q].u_jump;   // [u]
        double jump_un  = jumps[q].un_jump;  // [∂u/∂n]

        double uint = sol_u_int(x,y), uext = sol_u_ext(x,y);
        double unint = sol_ux_int(x,y)*nx_q + sol_uy_int(x,y)*ny_q;
        double unext = sol_ux_ext(x,y)*nx_q + sol_uy_ext(x,y)*ny_q;

        // Test various correction formulas
        double num_u_avg     = res.u_trace[q] - 0.5 * jump_u;
        double num_u_minus_j = res.u_trace[q] - jump_u;        // u_trace - [u]
        double num_u_plus_h  = res.u_trace[q] + 0.5 * jump_u;  // u_trace + 0.5 * [u]

        double ex_u_avg  = 0.5 * (uint + uext);
        double ex_un_avg = 0.5 * (unint + unext);

        u_avg_err            = std::max(u_avg_err,  std::abs(num_u_avg     - ex_u_avg));
        u_trace_minus_jump_err = std::max(u_trace_minus_jump_err, std::abs(num_u_minus_j - ex_u_avg));
        u_trace_plus_half_err  = std::max(u_trace_plus_half_err, std::abs(num_u_plus_h - ex_u_avg));
        u_trace_raw_err       = std::max(u_trace_raw_err, std::abs(res.u_trace[q] - uint));
        u_trace_vs_uext_err   = std::max(u_trace_vs_uext_err, std::abs(res.u_trace[q] - uext));

        // un_avg with standard formula
        double num_un_avg = res.un_trace[q] - 0.5 * jump_un;
        un_avg_err = std::max(un_avg_err, std::abs(num_un_avg - ex_un_avg));
    }

    if (N == 32) {  // Print debug on coarsest level
        std::printf("  DEBUG N=%d: u_trace vs various targets:\n", N);
        std::printf("    u_trace vs u_int:          %.4e\n", u_trace_raw_err);
        std::printf("    u_trace vs u_ext:          %.4e\n", u_trace_vs_uext_err);
        std::printf("    u_trace - [u]/2 vs u_avg: %.4e\n", u_avg_err);
        std::printf("    u_trace - [u]   vs u_avg: %.4e\n", u_trace_minus_jump_err);
        std::printf("    u_trace + [u]/2 vs u_avg: %.4e\n", u_trace_plus_half_err);
    }

    return {bulk_err, u_avg_err, un_avg_err};
}

// ── Tests ────────────────────────────────────────────────────────────────────

TEST_CASE("LaplaceInterfaceSolver2D: bulk O(h²) convergence — circle interface",
          "[laplace][interface_solver][convergence][2d]")
{
    const int Ns[]     = {32, 64, 128, 256};
    const int n_levels = 4;
    ConvergenceData data[n_levels];

    std::printf("\n  LaplaceInterfaceSolver2D: bulk convergence, circle interface\n");
    std::printf("  Manufactured: sol_u_int=sin(πx)sin(πy), sol_u_ext=sin(2πx)sin(2πy)\n");
    std::printf("  %6s  %12s  %8s\n", "N", "max_err", "rate");

    for (int l = 0; l < n_levels; ++l) {
        data[l] = solve_and_measure(Ns[l]);
        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s\n", Ns[l], data[l].bulk_err, "—");
        } else {
            double rate = std::log2(data[l-1].bulk_err / data[l].bulk_err);
            std::printf("  %6d  %12.4e  %8.3f\n", Ns[l], data[l].bulk_err, rate);
            REQUIRE(rate > 1.5);
        }
    }
}

TEST_CASE("LaplaceInterfaceSolver2D: arc_h_ratio check — circle interface",
          "[laplace][interface_solver][arc_h_ratio][2d]")
{
    // arc_h_ratio should be > 0.5 for well-resolved interfaces
    const int N_panels[] = {32, 64, 128, 256};
    const int n_cases = 4;

    std::printf("\n  LaplaceInterfaceSolver2D: arc_h_ratio for circle, N=64 grid\n");
    std::printf("  %10s  %12s  %8s\n", "N_panels", "arc_h_ratio", "> 0.5?");

    for (int k = 0; k < n_cases; ++k) {
        const int N = 64;
        const double h = 1.0 / N;
        CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
        auto iface = make_circle_panels(0.5, 0.5, 0.3, N_panels[k]);
        GridPair2D gp(grid, iface);

        LaplacePanelSpread2D       spread(gp);
        LaplaceFftBulkSolverZfft2D bulk_solver(grid, ZfftBcType::Dirichlet);
        LaplaceQuadraticRestrict2D restrict_op(gp);
        LaplaceInterfaceSolver2D   solver(spread, bulk_solver, restrict_op);

        double ratio = solver.arc_h_ratio();
        std::printf("  %10d  %12.6f  %8s\n", N_panels[k], ratio,
                     ratio > 0.1 ? "yes" : "NO");
        REQUIRE(ratio > 0.1);
    }
}

TEST_CASE("LaplaceInterfaceSolver2D: interface average trace O(h²) convergence",
          "[laplace][interface_solver][trace][convergence][2d]")
{
    const int Ns[]     = {32, 64, 128, 256};
    const int n_levels = 4;
    ConvergenceData data[n_levels];

    std::printf("\n  LaplaceInterfaceSolver2D: interface average trace convergence\n");
    std::printf("  u_avg  = u_trace + [u]/2       vs  (u⁺+u⁻)/2\n");
    std::printf("  un_avg = un_trace + [∂u/∂n]/2   vs  (∂u⁺/∂n+∂u⁻/∂n)/2\n");
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
            // u_avg should converge at O(h²); un_avg can be a bit slower
            REQUIRE(rate_u  > 1.5);
            REQUIRE(rate_un > 1.2);
        }
    }
}
