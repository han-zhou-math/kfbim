#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

#include "core/grid/cartesian_grid_2d.hpp"
#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "core/solver/zfft_bc_type.hpp"
#include "core/solver/iim_laplace_2d.hpp"

using namespace kfbim;
using Catch::Matchers::WithinAbs;

// ============================================================================
// Constants and helpers
// ============================================================================

static constexpr double kPi = 3.14159265358979323846;

// Star polygon: r(θ) = R * (1 + A * cos(k*θ)).
// Outward normal = 90° CW rotation of the unit tangent.
static Interface2D make_star(double cx, double cy, double R, double A, int k, int N)
{
    Eigen::MatrixX2d pts(N, 2), nml(N, 2);
    Eigen::VectorXd  wts(N);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(N);
    const double dth = 2.0 * kPi / N;
    for (int i = 0; i < N; ++i) {
        double th   = i * dth;
        double r    = R * (1.0 + A * std::cos(k * th));
        double drdt = -R * A * k * std::sin(k * th);
        pts(i, 0) = cx + r * std::cos(th);
        pts(i, 1) = cy + r * std::sin(th);
        double tx = drdt * std::cos(th) - r * std::sin(th);
        double ty = drdt * std::sin(th) + r * std::cos(th);
        double tlen = std::sqrt(tx*tx + ty*ty);
        nml(i, 0) =  ty / tlen;
        nml(i, 1) = -tx / tlen;
        wts(i)    = tlen * dth;
    }
    return {pts, nml, wts, 1, comp};
}

// ============================================================================
// Manufactured solution
//
// Box: [0,1]²   Star interface: centre (0.5,0.5), R=0.28, A=0.40, k=5.
//
//   u_ext(x,y) = sin(πx) sin(πy)           (outside star; vanishes on box boundary)
//   u_int(x,y) = sin(2πx) sin(2πy)         (inside star)
//
//   f_ext = −Δu_ext = 2π² sin(πx) sin(πy)
//   f_int = −Δu_int = 8π² sin(2πx) sin(2πy)
//
//   C(x,y)  = u_int − u_ext  = sin(2πx)sin(2πy) − sin(πx)sin(πy)
// ============================================================================

static double sol_u_ext(double x, double y)
{
    return std::sin(kPi*x) * std::sin(kPi*y);
}

static double sol_u_int(double x, double y)
{
    return std::sin(2*kPi*x) * std::sin(2*kPi*y);
}

static double sol_f_ext(double x, double y)
{
    return 2.0 * kPi*kPi * std::sin(kPi*x) * std::sin(kPi*y);
}

static double sol_f_int(double x, double y)
{
    return 8.0 * kPi*kPi * std::sin(2*kPi*x) * std::sin(2*kPi*y);
}

static double correction_fn(double x, double y)
{
    return sol_u_int(x, y) - sol_u_ext(x, y);
}

// Partial derivatives of C = u_int − u_ext
static double Cx_fn(double x, double y)
{
    return  2*kPi*std::cos(2*kPi*x)*std::sin(2*kPi*y)
          - kPi*std::cos(kPi*x)*std::sin(kPi*y);
}
static double Cy_fn(double x, double y)
{
    return  2*kPi*std::sin(2*kPi*x)*std::cos(2*kPi*y)
          - kPi*std::sin(kPi*x)*std::cos(kPi*y);
}
static double Cxx_fn(double x, double y)
{
    return -4*kPi*kPi*std::sin(2*kPi*x)*std::sin(2*kPi*y)
           + kPi*kPi*std::sin(kPi*x)*std::sin(kPi*y);
}
static double Cxy_fn(double x, double y)
{
    return  4*kPi*kPi*std::cos(2*kPi*x)*std::cos(2*kPi*y)
          - kPi*kPi*std::cos(kPi*x)*std::cos(kPi*y);
}
static double Cyy_fn(double x, double y)
{
    return -4*kPi*kPi*std::sin(2*kPi*x)*std::sin(2*kPi*y)
           + kPi*kPi*std::sin(kPi*x)*std::sin(kPi*y);
}

// ============================================================================
// Error and convergence helper
//
// Build GridPair2D, classify nodes, apply IIM correction, solve with FFT,
// return the max-norm error at interior nodes.
// ============================================================================

static double iim_error(int N)
{
    // Node grid on [0,1]² — (N+1)×(N+1) DOFs, spacing h=1/N.
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);

    // Star interface with 8*N quadrature points for good resolution.
    auto iface = make_star(0.5, 0.5, 0.28, 0.40, 5, 8 * N);
    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];  // = N+1

    // Collect domain labels.
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n)
        labels[n] = gp.domain_label(n);

    // Build piecewise RHS and exact solution at all nodes.
    Eigen::VectorXd f(nx * ny), u_exact(nx * ny), C(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto   c = grid.coord(n);
        double x = c[0], y = c[1];
        int    lbl = labels[n];
        u_exact[n]  = (lbl == 1) ? sol_u_int(x, y) : sol_u_ext(x, y);
        C[n]        = correction_fn(x, y);
        bool bdy = (n % nx == 0 || n % nx == nx-1 || n / nx == 0 || n / nx == ny-1);
        f[n] = bdy ? 0.0 : ((lbl == 1) ? sol_f_int(x, y) : sol_f_ext(x, y));
    }

    // Apply IIM defect correction.
    Eigen::VectorXd F = iim_correct_rhs(grid, f, C, labels);

    // Solver convention: solves Δ_h u = rhs, so pass rhs = −F (= Δu, not −Δu).
    LaplaceFftBulkSolverZfft2D solver(grid, ZfftBcType::Dirichlet);
    Eigen::VectorXd u_sol;
    solver.solve(-F, u_sol);

    // Max-norm error at interior nodes.
    double err = 0.0;
    for (int j = 1; j < ny-1; ++j)
        for (int i = 1; i < nx-1; ++i)
            err = std::max(err, std::abs(u_sol[j*nx+i] - u_exact[j*nx+i]));
    return err;
}

// ============================================================================
// Tests
// ============================================================================

// ─── 1. Irregular node classification ────────────────────────────────────────

TEST_CASE("IIM 2D: irregular node classification — star interface",
          "[iim][2d][classification]")
{
    const int N = 64;
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto iface = make_star(0.5, 0.5, 0.28, 0.40, 5, 512);
    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n) labels[n] = gp.domain_label(n);

    auto irreg = iim_irregular_nodes(grid, labels);

    // No boundary nodes in the irregular set.
    for (int n : irreg) {
        int i = n % nx, j = n / nx;
        REQUIRE(i > 0); REQUIRE(i < nx-1);
        REQUIRE(j > 0); REQUIRE(j < ny-1);
    }

    // All irregular nodes must have at least one cross-interface neighbor.
    for (int n : irreg) {
        bool found = false;
        int ln = labels[n];
        for (int nb : grid.neighbors(n)) {
            if (nb < 0) continue;
            int inb = nb % nx, jnb = nb / nx;
            if (inb == 0 || inb == nx-1 || jnb == 0 || jnb == ny-1) continue;
            if (labels[nb] != ln) { found = true; break; }
        }
        REQUIRE(found);
    }

    // Irregular nodes ~ perimeter / h.  Star perimeter ≈ 2π * R * (integral
    // correction) ≈ 2.1.  Each interface cell contributes ~2 irregular nodes.
    // Expect 2*2.1/h = 4.2*N ≈ 269 for N=64; use loose bounds [100, 1000].
    REQUIRE(irreg.size() > 100);
    REQUIRE(irreg.size() < 1000);

    // All irregular nodes must be near the interface (within 3h).
    for (int n : irreg)
        REQUIRE(gp.is_near_interface(n, 3.0 * h));
}

// ─── 2. Regular nodes are unmodified ─────────────────────────────────────────

TEST_CASE("IIM 2D: regular nodes unchanged by correction",
          "[iim][2d][correction]")
{
    const int N = 32;
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto iface = make_star(0.5, 0.5, 0.28, 0.40, 5, 256);
    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n) labels[n] = gp.domain_label(n);

    auto irreg_set = iim_irregular_nodes(grid, labels);
    std::vector<bool> is_irreg(nx * ny, false);
    for (int n : irreg_set) is_irreg[n] = true;

    // Build dummy f and C.
    Eigen::VectorXd f(nx * ny), C(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        f[n] = std::sin(kPi * c[0]) * std::sin(kPi * c[1]);
        C[n] = 1.0;  // non-zero so any leakage would be detectable
    }

    Eigen::VectorXd F = iim_correct_rhs(grid, f, C, labels);

    for (int n = 0; n < nx * ny; ++n) {
        if (!is_irreg[n])
            REQUIRE_THAT(F[n], WithinAbs(f[n], 1e-14));
    }
}

// ─── 3. Correction sign and magnitude at a known irregular node ───────────────
//
// Find an interior node n with exactly one cross-interface neighbor nb, and
// verify that F[n] = f[n] + (label[nb] - label[n]) * C[nb] / h².

TEST_CASE("IIM 2D: correction term sign and magnitude",
          "[iim][2d][correction]")
{
    const int N = 64;
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto iface = make_star(0.5, 0.5, 0.28, 0.40, 5, 512);
    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n) labels[n] = gp.domain_label(n);

    // Distinct f and C so accidental cancellation is visible.
    Eigen::VectorXd f(nx * ny), C(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        f[n] = 2.0 * c[0] + 3.0 * c[1];
        C[n] = 5.0 * c[0] * c[0] + 7.0 * c[1];
    }

    Eigen::VectorXd F = iim_correct_rhs(grid, f, C, labels);

    // Find a node with exactly one cross-interface interior neighbor and verify.
    bool found_test_node = false;
    for (int n = 0; n < nx * ny && !found_test_node; ++n) {
        int i = n % nx, j = n / nx;
        if (i == 0 || i == nx-1 || j == 0 || j == ny-1) continue;
        int ln = labels[n];

        int n_cross = 0;
        double expected_corr = 0.0;
        for (int nb : grid.neighbors(n)) {
            if (nb < 0) continue;
            int inb = nb % nx, jnb = nb / nx;
            if (inb == 0 || inb == nx-1 || jnb == 0 || jnb == ny-1) continue;
            int lnb = labels[nb];
            if (lnb != ln) {
                ++n_cross;
                expected_corr += (ln - lnb) * C[nb] / (h * h);
            }
        }

        if (n_cross == 1) {
            REQUIRE_THAT(F[n], WithinAbs(f[n] + expected_corr, 1e-12));
            found_test_node = true;
        }
    }
    REQUIRE(found_test_node);  // sanity: at least one such node exists
}

// ─── 4. 2nd order convergence ─────────────────────────────────────────────────
//
// Solve −Δu = f on [0,1]² with homogeneous Dirichlet BC and a 5-tip star
// interface.  IIM correction gives 2nd order accuracy everywhere.

TEST_CASE("IIM 2D: 2nd order convergence — star interface manufactured solution",
          "[iim][2d][convergence]")
{
    // N = 32 → 64 → 128 → 256.  Expect ratio ≈ 4 each doubling.
    const int    Ns[]     = {32, 64, 128, 256};
    const int    n_levels = 4;
    double errs[n_levels];
    for (int l = 0; l < n_levels; ++l)
        errs[l] = iim_error(Ns[l]);

    // Print for human inspection (visible with -s flag).
    std::printf("\n  IIM 2D convergence (star interface):\n");
    std::printf("  %6s  %12s  %8s\n", "N", "max_err", "rate");
    std::printf("  %6d  %12.4e  %8s\n", Ns[0], errs[0], "—");
    for (int l = 1; l < n_levels; ++l) {
        double rate = std::log2(errs[l-1] / errs[l]);
        std::printf("  %6d  %12.4e  %8.3f\n", Ns[l], errs[l], rate);
    }

    // Assert 2nd order convergence at every refinement step.
    for (int l = 1; l < n_levels; ++l) {
        double rate = std::log2(errs[l-1] / errs[l]);
        REQUIRE(rate > 1.85);
        REQUIRE(rate < 2.20);
    }
}

// ─── Error helper for the Taylor-expansion variant ───────────────────────────

static double iim_error_taylor(int N)
{
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);

    // Use 4N interface points: same geometry quality as the exact-C test.
    // The Taylor truncation error is O(D³/h²) ≈ O(h) per correction, but in
    // 2D this gets smoothed by the discrete Green's function G_h(x,n) ~ h²logN,
    // yielding O(D³ logN) ≈ O(h³ logN) per node.  Summed over O(N) irregular
    // nodes the u-error is O(h² logN) — essentially 2nd-order, but with a
    // larger pre-constant than the exact-C variant (see test 7 below).
    auto iface = make_star(0.5, 0.5, 0.28, 0.40, 5, 4 * N);
    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];

    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n)
        labels[n] = gp.domain_label(n);

    Eigen::VectorXd f(nx * ny), u_exact(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        double x = c[0], y = c[1];
        int lbl  = labels[n];
        u_exact[n] = (lbl == 1) ? sol_u_int(x, y) : sol_u_ext(x, y);
        bool bdy = (n % nx == 0 || n % nx == nx-1 || n / nx == 0 || n / nx == ny-1);
        f[n] = bdy ? 0.0 : ((lbl == 1) ? sol_f_int(x, y) : sol_f_ext(x, y));
    }

    // Build C and its derivatives at each interface quadrature point.
    int Nq = iface.num_points();
    Eigen::VectorXd C_q(Nq), Cx_q(Nq), Cy_q(Nq), Cxx_q(Nq), Cxy_q(Nq), Cyy_q(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0), y = iface.points()(q, 1);
        C_q[q]   = correction_fn(x, y);
        Cx_q[q]  = Cx_fn(x, y);
        Cy_q[q]  = Cy_fn(x, y);
        Cxx_q[q] = Cxx_fn(x, y);
        Cxy_q[q] = Cxy_fn(x, y);
        Cyy_q[q] = Cyy_fn(x, y);
    }

    Eigen::VectorXd F = iim_correct_rhs_taylor(grid, gp, iface, f,
                                                C_q, Cx_q, Cy_q,
                                                Cxx_q, Cxy_q, Cyy_q,
                                                labels);

    LaplaceFftBulkSolverZfft2D solver(grid, ZfftBcType::Dirichlet);
    Eigen::VectorXd u_sol;
    solver.solve(-F, u_sol);

    double err = 0.0;
    for (int j = 1; j < ny-1; ++j)
        for (int i = 1; i < nx-1; ++i)
            err = std::max(err, std::abs(u_sol[j*nx+i] - u_exact[j*nx+i]));
    return err;
}

// ─── 5. Without correction — O(1) error at irregular nodes ───────────────────
//
// Verify that skipping the correction produces large errors (non-convergent),
// confirming the correction is doing real work.

TEST_CASE("IIM 2D: uncorrected solve has O(1) error at irregular nodes",
          "[iim][2d][convergence]")
{
    const int N = 64;
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto iface = make_star(0.5, 0.5, 0.28, 0.40, 5, 8 * N);
    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n) labels[n] = gp.domain_label(n);

    Eigen::VectorXd f(nx * ny), u_exact(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c   = grid.coord(n);
        double x = c[0], y = c[1];
        int   lbl = labels[n];
        u_exact[n] = (lbl == 1) ? sol_u_int(x, y) : sol_u_ext(x, y);
        bool  bdy  = (n % nx == 0 || n % nx == nx-1 || n / nx == 0 || n / nx == ny-1);
        f[n] = bdy ? 0.0 : ((lbl == 1) ? sol_f_int(x, y) : sol_f_ext(x, y));
    }

    // Solve WITHOUT correction (still need to negate: solver solves Δu = rhs).
    LaplaceFftBulkSolverZfft2D solver(grid, ZfftBcType::Dirichlet);
    Eigen::VectorXd u_uncorr;
    solver.solve(-f, u_uncorr);

    double err_uncorr = 0.0;
    for (int j = 1; j < ny-1; ++j)
        for (int i = 1; i < nx-1; ++i)
            err_uncorr = std::max(err_uncorr, std::abs(u_uncorr[j*nx+i] - u_exact[j*nx+i]));

    // Solve WITH correction.
    Eigen::VectorXd C(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        C[n] = correction_fn(c[0], c[1]);
    }
    Eigen::VectorXd F = iim_correct_rhs(grid, f, C, labels);
    Eigen::VectorXd u_corr;
    solver.solve(-F, u_corr);

    double err_corr = 0.0;
    for (int j = 1; j < ny-1; ++j)
        for (int i = 1; i < nx-1; ++i)
            err_corr = std::max(err_corr, std::abs(u_corr[j*nx+i] - u_exact[j*nx+i]));

    std::printf("\n  IIM 2D uncorrected vs corrected (N=%d):\n", N);
    std::printf("    uncorrected max err = %.4e\n", err_uncorr);
    std::printf("    corrected   max err = %.4e\n", err_corr);

    // The corrected solve must be dramatically more accurate.
    REQUIRE(err_corr < err_uncorr * 0.01);
    // Corrected error is O(h²); uncorrected is O(1).
    REQUIRE(err_corr   < 1e-2);
    REQUIRE(err_uncorr > 1e-2);
}

// ─── 6. Taylor-expansion IIM: ~2nd-order convergence (larger constant) ────────
//
// Replace global C(x) with a degree-2 Taylor expansion from the nearest
// interface point.  The per-correction RHS error is O(D³/h²) ≈ O(h), but the
// 2D discrete Poisson Green's function G_h(x,n) ~ h²logN smooths each
// correction's u-contribution to O(h³ logN).  Summing over O(N) irregular
// nodes yields solution error O(h² logN) — still ~2nd order, but with a
// larger pre-constant than the exact-C variant.

TEST_CASE("IIM 2D (Taylor): ~2nd-order convergence with degree-2 Taylor C",
          "[iim][2d][taylor][convergence]")
{
    const int    Ns[]     = {32, 64, 128, 256};
    const int    n_levels = 4;
    double errs[n_levels];
    for (int l = 0; l < n_levels; ++l)
        errs[l] = iim_error_taylor(Ns[l]);

    std::printf("\n  IIM 2D Taylor convergence (degree-2 from nearest interface pt):\n");
    std::printf("  %6s  %12s  %8s\n", "N", "max_err", "rate");
    std::printf("  %6d  %12.4e  %8s\n", Ns[0], errs[0], "—");
    for (int l = 1; l < n_levels; ++l) {
        double rate = std::log2(errs[l-1] / errs[l]);
        std::printf("  %6d  %12.4e  %8.3f\n", Ns[l], errs[l], rate);
    }

    // Expect ~2nd order (same rate as exact-C, larger constant due to Taylor
    // truncation error); accept rates in [1.7, 2.5].
    for (int l = 1; l < n_levels; ++l) {
        double rate = std::log2(errs[l-1] / errs[l]);
        REQUIRE(rate > 1.7);
        REQUIRE(rate < 2.5);
    }
    // All errors should be small (well below the O(1) uncorrected baseline).
    for (int l = 0; l < n_levels; ++l)
        REQUIRE(errs[l] < 0.1);
}

// ─── 7. Taylor vs exact-C error comparison at same N ─────────────────────────
//
// At N=64, Taylor-C gives larger error than exact-C, but both are well below
// the uncorrected O(1) baseline.

TEST_CASE("IIM 2D (Taylor): Taylor error larger than exact-C at same N",
          "[iim][2d][taylor]")
{
    const int N = 64;
    double err_exact  = iim_error(N);
    double err_taylor = iim_error_taylor(N);

    std::printf("\n  IIM 2D N=%d:  exact-C err=%.4e   Taylor-C err=%.4e   ratio=%.2f\n",
                N, err_exact, err_taylor, err_taylor / err_exact);

    // Taylor is less accurate (larger error).
    REQUIRE(err_taylor > err_exact);
    // But both should converge (well below O(1) baseline).
    REQUIRE(err_taylor < 0.1);
    REQUIRE(err_exact  < 0.1);
}
