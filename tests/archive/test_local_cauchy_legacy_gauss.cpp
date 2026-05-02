#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/grid/cartesian_grid_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "core/solver/zfft_bc_type.hpp"
#include "core/solver/iim_laplace_2d.hpp"
#include "core/local_cauchy/laplace_panel_solver_2d.hpp"

using namespace kfbim;
using Catch::Matchers::WithinAbs;

static constexpr double kPi = 3.14159265358979323846;

// ============================================================================
// Manufactured solution (same as test_iim_2d)
// ============================================================================

static double u_plus (double x, double y) { return std::sin(kPi*x)*std::sin(kPi*y); }
static double u_minus(double x, double y) { return std::sin(2*kPi*x)*std::sin(2*kPi*y); }
static double f_plus (double x, double y) { return 2.0*kPi*kPi*std::sin(kPi*x)*std::sin(kPi*y); }
static double f_minus(double x, double y) { return 8.0*kPi*kPi*std::sin(2*kPi*x)*std::sin(2*kPi*y); }

static double C_fn  (double x, double y) { return u_plus(x,y) - u_minus(x,y); }
static double Cx_fn (double x, double y) {
    return  kPi*std::cos(kPi*x)*std::sin(kPi*y)
          - 2*kPi*std::cos(2*kPi*x)*std::sin(2*kPi*y);
}
static double Cy_fn (double x, double y) {
    return  kPi*std::sin(kPi*x)*std::cos(kPi*y)
          - 2*kPi*std::sin(2*kPi*x)*std::cos(2*kPi*y);
}
static double Cxx_fn(double x, double y) {
    return -kPi*kPi*std::sin(kPi*x)*std::sin(kPi*y)
           +4*kPi*kPi*std::sin(2*kPi*x)*std::sin(2*kPi*y);
}
static double Cxy_fn(double x, double y) {
    return  kPi*kPi*std::cos(kPi*x)*std::cos(kPi*y)
           -4*kPi*kPi*std::cos(2*kPi*x)*std::cos(2*kPi*y);
}
static double Cyy_fn(double x, double y) {
    return -kPi*kPi*std::sin(kPi*x)*std::sin(kPi*y)
           +4*kPi*kPi*std::sin(2*kPi*x)*std::sin(2*kPi*y);
}

// ============================================================================
// Star interface builder with 3 Gauss-Legendre points per panel
//
// N_panels equal panels cover θ ∈ [0, 2π].
// Within each panel [θ_a, θ_b], the 3 GL nodes use weights/abscissae:
//   s = {-√(3/5), 0, +√(3/5)},  w = {5/9, 8/9, 5/9}
//
// Star curve: r(θ) = R * (1 + A * cos(K * θ)).
// ============================================================================

static const double kGL_s[3] = {
    -0.7745966692414834,   // -sqrt(3/5)
     0.0,
    +0.7745966692414834    // +sqrt(3/5)
};
static const double kGL_w[3] = { 5.0/9.0, 8.0/9.0, 5.0/9.0 };

static const double kLobatto_s[3] = {-1.0, 0.0, 1.0};
static const double kLobatto_w[3] = {1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0};

static Interface2D make_star_panels(double cx, double cy,
                                    double R, double A, int K,
                                    int N_panels)
{
    const int Nq = 3 * N_panels;
    Eigen::MatrixX2d pts(Nq, 2), nml(Nq, 2);
    Eigen::VectorXd  wts(Nq);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(N_panels);

    const double dth = 2.0 * kPi / N_panels;  // panel angular width

    int q = 0;
    for (int p = 0; p < N_panels; ++p) {
        double th_a = p * dth;
        double th_b = th_a + dth;
        double th_mid  = 0.5 * (th_a + th_b);
        double half_dth = 0.5 * dth;

        for (int i = 0; i < 3; ++i) {
            double th    = th_mid + half_dth * kGL_s[i];
            double r     = R * (1.0 + A * std::cos(K * th));
            double drdt  = -R * A * K * std::sin(K * th);

            pts(q, 0) = cx + r * std::cos(th);
            pts(q, 1) = cy + r * std::sin(th);

            // Tangent vector at θ: d/dθ (r cosθ, r sinθ)
            double tx = drdt * std::cos(th) - r * std::sin(th);
            double ty = drdt * std::sin(th) + r * std::cos(th);
            double tlen = std::sqrt(tx*tx + ty*ty);

            // Outward normal = 90° CW rotation of unit tangent
            nml(q, 0) =  ty / tlen;
            nml(q, 1) = -tx / tlen;

            // Arc-length weight = GL_weight × half_dth × |tangent|
            wts[q] = kGL_w[i] * half_dth * tlen;
            ++q;
        }
    }
    return {pts, nml, wts, 3, comp};
}

static Interface2D make_star_lobatto_panels(double cx, double cy,
                                            double R, double A, int K,
                                            int N_panels)
{
    const int Nq = 3 * N_panels;
    Eigen::MatrixX2d pts(Nq, 2), nml(Nq, 2);
    Eigen::VectorXd  wts(Nq);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(N_panels);

    const double dth = 2.0 * kPi / N_panels;

    int q = 0;
    for (int p = 0; p < N_panels; ++p) {
        const double th_a = p * dth;
        const double th_b = th_a + dth;
        const double th_mid = 0.5 * (th_a + th_b);
        const double half_dth = 0.5 * dth;

        for (int i = 0; i < 3; ++i) {
            const double th = th_mid + half_dth * kLobatto_s[i];
            const double r = R * (1.0 + A * std::cos(K * th));
            const double drdt = -R * A * K * std::sin(K * th);

            pts(q, 0) = cx + r * std::cos(th);
            pts(q, 1) = cy + r * std::sin(th);

            const double tx = drdt * std::cos(th) - r * std::sin(th);
            const double ty = drdt * std::sin(th) + r * std::cos(th);
            const double tlen = std::sqrt(tx * tx + ty * ty);

            nml(q, 0) =  ty / tlen;
            nml(q, 1) = -tx / tlen;

            wts[q] = kLobatto_w[i] * half_dth * tlen;
            ++q;
        }
    }

    return {pts, nml, wts, 3, comp, PanelNodeLayout2D::ChebyshevLobatto};
}

// Overload for a circle (A=0, K=0) — used in regression tests
static Interface2D make_circle_panels(double cx, double cy, double R, int N_panels)
{
    return make_star_panels(cx, cy, R, 0.0, 0, N_panels);
}

// ============================================================================
// Helper: run panel Cauchy solver on star and return max-norm derivative errors
// ============================================================================

struct DerivErrors {
    double eCx, eCy, eCxx, eCyy, eCxy;
};

static DerivErrors panel_cauchy_errors(int N_panels)
{
    auto iface = make_star_panels(0.5, 0.5, 0.28, 0.40, 5, N_panels);
    int Nq = iface.num_points();

    Eigen::VectorXd a(Nq), b(Nq), Lu(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0);
        double y = iface.points()(q, 1);
        double nx = iface.normals()(q, 0);
        double ny = iface.normals()(q, 1);
        a[q]  = C_fn(x, y);
        b[q]  = Cx_fn(x, y)*nx + Cy_fn(x, y)*ny;
        Lu[q] = f_plus(x, y) - f_minus(x, y);
    }

    auto res = laplace_panel_cauchy_2d(iface, a, b, Lu);

    DerivErrors e{};
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0);
        double y = iface.points()(q, 1);
        e.eCx  = std::max(e.eCx,  std::abs(res.Cx [q] - Cx_fn (x,y)));
        e.eCy  = std::max(e.eCy,  std::abs(res.Cy [q] - Cy_fn (x,y)));
        e.eCxx = std::max(e.eCxx, std::abs(res.Cxx[q] - Cxx_fn(x,y)));
        e.eCyy = std::max(e.eCyy, std::abs(res.Cyy[q] - Cyy_fn(x,y)));
        e.eCxy = std::max(e.eCxy, std::abs(res.Cxy[q] - Cxy_fn(x,y)));
    }
    return e;
}

static DerivErrors lobatto_center_cauchy_errors(int N_panels)
{
    auto iface = make_star_lobatto_panels(0.5, 0.5, 0.28, 0.40, 5, N_panels);
    const int Nq = iface.num_points();

    Eigen::VectorXd a(Nq), b(Nq), Lu(Nq);
    for (int q = 0; q < Nq; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const double nx = iface.normals()(q, 0);
        const double ny = iface.normals()(q, 1);
        a[q]  = C_fn(x, y);
        b[q]  = Cx_fn(x, y) * nx + Cy_fn(x, y) * ny;
        Lu[q] = f_plus(x, y) - f_minus(x, y);
    }

    const auto res = laplace_panel_lobatto_center_cauchy_2d(iface, a, b, Lu);

    DerivErrors e{};
    for (int c = 0; c < res.centers.rows(); ++c) {
        const double x = res.centers(c, 0);
        const double y = res.centers(c, 1);
        e.eCx  = std::max(e.eCx,  std::abs(res.Cx [c] - Cx_fn (x, y)));
        e.eCy  = std::max(e.eCy,  std::abs(res.Cy [c] - Cy_fn (x, y)));
        e.eCxx = std::max(e.eCxx, std::abs(res.Cxx[c] - Cxx_fn(x, y)));
        e.eCyy = std::max(e.eCyy, std::abs(res.Cyy[c] - Cyy_fn(x, y)));
        e.eCxy = std::max(e.eCxy, std::abs(res.Cxy[c] - Cxy_fn(x, y)));
    }
    return e;
}

// ============================================================================
// Tests
// ============================================================================

// ─── 1. Dirichlet recovery ────────────────────────────────────────────────────
//
// For each target Gauss point t, the shifted collocation system uses
// X(t-delta), X(t), X(t+delta) for Dirichlet rows.  The center row still
// collapses to c[0] = a(t) because dx=dy=0 there.
// Verify C[q] = a[q] to machine precision at every node.

TEST_CASE("LaplaceLocalSolver2D: Dirichlet recovery — C[q] equals a[q]",
          "[local_cauchy][2d][dirichlet]")
{
    const int N_panels = 32;
    auto iface = make_star_panels(0.5, 0.5, 0.28, 0.40, 5, N_panels);
    int Nq = iface.num_points();

    Eigen::VectorXd a(Nq), b(Nq), Lu(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0), y = iface.points()(q, 1);
        double nx = iface.normals()(q, 0), ny = iface.normals()(q, 1);
        a[q]  = C_fn(x, y);
        b[q]  = Cx_fn(x, y)*nx + Cy_fn(x, y)*ny;
        Lu[q] = f_plus(x, y) - f_minus(x, y);
    }

    auto res = laplace_panel_cauchy_2d(iface, a, b, Lu);

    for (int q = 0; q < Nq; ++q)
        REQUIRE_THAT(res.C[q], WithinAbs(a[q], 1e-10));
}

// ─── 2. Single-panel regression — flat panel with known polynomial ────────────
//
// Take a flat horizontal panel centred at (0.5, 0.0) with outward normal (0,1).
// Let the true C be the quadratic C(x,y) = 3 + 2x + y + 0.5x² + 0.3y² + 0.7xy.
// -ΔC = -(0.5 + 0.3) = -0.8 (constant for quadratic).
// The shifted collocation data is interpolated from three panel nodes.  On a
// flat panel, a quadratic trace and linear normal derivative are represented
// exactly, so the recovered polynomial should still be exact.

TEST_CASE("LaplaceLocalSolver2D: regression — flat panel exact polynomial recovery",
          "[local_cauchy][2d][regression]")
{
    // 3 GL nodes on [0, 1] × {0}, mapped from [0.5 + 0.5*s]
    double xs[3], ys[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i)
        xs[i] = 0.5 + 0.5 * kGL_s[i];

    // True polynomial: C(x,y) = 3 + 2x + y + 0.5x² + 0.3y² + 0.7xy
    auto C_true   = [](double x, double y){ return 3.0 + 2*x + y + 0.5*x*x + 0.3*y*y + 0.7*x*y; };
    auto Cx_true  = [](double x, double y){ return 2.0 + x + 0.7*y; };
    auto Cy_true  = [](double x, double y){ return 1.0 + 0.6*y + 0.7*x; };
    auto Cxx_true = [](double, double)     { return 1.0; };
    auto Cyy_true = [](double, double)     { return 0.6; };
    auto Cxy_true = [](double, double)     { return 0.7; };
    // -ΔC = -(Cxx + Cyy) = -(1.0 + 0.6) = -1.6
    const double neg_laplacian = -1.6;

    // Build a 1-panel interface
    Eigen::MatrixX2d pts(3, 2), nml(3, 2);
    Eigen::VectorXd  wts(3);
    Eigen::VectorXi  comp(1); comp[0] = 0;
    for (int i = 0; i < 3; ++i) {
        pts(i, 0) = xs[i]; pts(i, 1) = 0.0;
        nml(i, 0) = 0.0;   nml(i, 1) = 1.0;  // outward normal pointing up
        wts[i] = kGL_w[i] * 0.5;              // arc-length weight on [0,1]
    }
    Interface2D iface(pts, nml, wts, 3, comp);

    Eigen::VectorXd a(3), b(3), Lu(3);
    for (int i = 0; i < 3; ++i) {
        double x = xs[i], y = 0.0;
        double nx = 0.0,  ny = 1.0;
        a[i]  = C_true(x, y);
        b[i]  = Cx_true(x,y)*nx + Cy_true(x,y)*ny;
        Lu[i] = neg_laplacian;
    }

    auto res = laplace_panel_cauchy_2d(iface, a, b, Lu);

    // Check all 3 Gauss points
    for (int i = 0; i < 3; ++i) {
        double x = xs[i], y = 0.0;
        REQUIRE_THAT(res.C  [i], WithinAbs(C_true (x,y), 1e-10));
        REQUIRE_THAT(res.Cx [i], WithinAbs(Cx_true(x,y), 1e-10));
        REQUIRE_THAT(res.Cy [i], WithinAbs(Cy_true(x,y), 1e-10));
        REQUIRE_THAT(res.Cxx[i], WithinAbs(Cxx_true(x,y), 1e-10));
        REQUIRE_THAT(res.Cyy[i], WithinAbs(Cyy_true(x,y), 1e-10));
        REQUIRE_THAT(res.Cxy[i], WithinAbs(Cxy_true(x,y), 1e-10));
    }
}

// ─── 2b. Lobatto-center regression — flat panel exact polynomial recovery ───

TEST_CASE("LaplaceLocalSolver2D: Lobatto centers recover flat-panel polynomial",
          "[local_cauchy][2d][lobatto][regression]")
{
    const double xs[3] = {0.0, 0.5, 1.0};

    auto C_true   = [](double x, double y){ return 3.0 + 2*x + y + 0.5*x*x + 0.3*y*y + 0.7*x*y; };
    auto Cx_true  = [](double x, double y){ return 2.0 + x + 0.7*y; };
    auto Cy_true  = [](double x, double y){ return 1.0 + 0.6*y + 0.7*x; };
    auto Cxx_true = [](double, double)     { return 1.0; };
    auto Cyy_true = [](double, double)     { return 0.6; };
    auto Cxy_true = [](double, double)     { return 0.7; };
    const double neg_laplacian = -1.6;

    Eigen::MatrixX2d pts(3, 2), nml(3, 2);
    Eigen::VectorXd  wts(3);
    Eigen::VectorXi  comp(1); comp[0] = 0;
    for (int i = 0; i < 3; ++i) {
        pts(i, 0) = xs[i]; pts(i, 1) = 0.0;
        nml(i, 0) = 0.0;   nml(i, 1) = 1.0;
        wts[i] = kLobatto_w[i] * 0.5;
    }
    Interface2D iface(pts, nml, wts, 3, comp, PanelNodeLayout2D::ChebyshevLobatto);

    Eigen::VectorXd a(3), b(3), Lu(3);
    for (int i = 0; i < 3; ++i) {
        const double x = xs[i];
        a[i] = C_true(x, 0.0);
        b[i] = Cy_true(x, 0.0);
        Lu[i] = neg_laplacian;
    }

    const auto res = laplace_panel_lobatto_center_cauchy_2d(iface, a, b, Lu);
    REQUIRE(res.centers.rows() == 4);

    const double expected_x[4] = {0.125, 0.375, 0.625, 0.875};
    for (int i = 0; i < 4; ++i) {
        const double x = expected_x[i];
        const double y = 0.0;
        REQUIRE_THAT(res.centers(i, 0), WithinAbs(x, 1e-14));
        REQUIRE_THAT(res.centers(i, 1), WithinAbs(y, 1e-14));
        REQUIRE(res.panel_index[i] == 0);
        REQUIRE_THAT(res.C  [i], WithinAbs(C_true  (x, y), 1e-10));
        REQUIRE_THAT(res.Cx [i], WithinAbs(Cx_true (x, y), 1e-10));
        REQUIRE_THAT(res.Cy [i], WithinAbs(Cy_true (x, y), 1e-10));
        REQUIRE_THAT(res.Cxx[i], WithinAbs(Cxx_true(x, y), 1e-10));
        REQUIRE_THAT(res.Cyy[i], WithinAbs(Cyy_true(x, y), 1e-10));
        REQUIRE_THAT(res.Cxy[i], WithinAbs(Cxy_true(x, y), 1e-10));
    }
}

// ─── 3. Gradient convergence — O(h²) for Cx and Cy ──────────────────────────
//
// As panels are refined, the first-order derivatives Cx and Cy converge.
// A quadratic polynomial fitted to 3 Dirichlet + 2 Neumann + 1 PDE constraints
// approximates C to O(h³) and its gradient to O(h²).

TEST_CASE("LaplaceLocalSolver2D: gradient convergence — star interface",
          "[local_cauchy][2d][convergence]")
{
    const int    Nps[]    = {16, 32, 64, 128};
    const int    n_levels = 4;
    double eCx[n_levels], eCy[n_levels];

    for (int l = 0; l < n_levels; ++l) {
        auto e   = panel_cauchy_errors(Nps[l]);
        eCx[l]   = e.eCx;
        eCy[l]   = e.eCy;
    }

    std::printf("\n  Panel Cauchy 2D — gradient convergence (star interface):\n");
    std::printf("  %8s  %12s  %8s  %12s  %8s\n",
                "N_panels", "eCx", "rate", "eCy", "rate");
    std::printf("  %8d  %12.4e  %8s  %12.4e  %8s\n",
                Nps[0], eCx[0], "—", eCy[0], "—");
    for (int l = 1; l < n_levels; ++l) {
        double rx = std::log2(eCx[l-1]/eCx[l]);
        double ry = std::log2(eCy[l-1]/eCy[l]);
        std::printf("  %8d  %12.4e  %8.3f  %12.4e  %8.3f\n",
                    Nps[l], eCx[l], rx, eCy[l], ry);
    }

    // The shifted collocation is pre-asymptotic on the coarsest refinement,
    // so assert the sustained rate over the resolved levels.
    double rx = std::log2(eCx[1] / eCx[3]) / 2.0;
    double ry = std::log2(eCy[1] / eCy[3]) / 2.0;
    REQUIRE(rx > 1.5);   // expect ~2nd order
    REQUIRE(ry > 1.5);
}

// ─── 4. Second-derivative convergence — O(h) for Cxx, Cyy, Cxy ──────────────
//
// Second derivatives of a quadratic interpolant converge at O(h):
// differentiating once reduces the approximation order by 1.

TEST_CASE("LaplaceLocalSolver2D: second-derivative convergence — star interface",
          "[local_cauchy][2d][convergence]")
{
    const int    Nps[]    = {16, 32, 64, 128};
    const int    n_levels = 4;
    double eCxx[n_levels], eCyy[n_levels], eCxy[n_levels];

    for (int l = 0; l < n_levels; ++l) {
        auto e     = panel_cauchy_errors(Nps[l]);
        eCxx[l]    = e.eCxx;
        eCyy[l]    = e.eCyy;
        eCxy[l]    = e.eCxy;
    }

    std::printf("\n  Panel Cauchy 2D — 2nd-deriv convergence (star interface):\n");
    std::printf("  %8s  %12s  %8s  %12s  %8s  %12s  %8s\n",
                "N_panels", "eCxx", "rate", "eCyy", "rate", "eCxy", "rate");
    std::printf("  %8d  %12.4e  %8s  %12.4e  %8s  %12.4e  %8s\n",
                Nps[0], eCxx[0], "—", eCyy[0], "—", eCxy[0], "—");
    for (int l = 1; l < n_levels; ++l) {
        double rxx = std::log2(eCxx[l-1]/eCxx[l]);
        double ryy = std::log2(eCyy[l-1]/eCyy[l]);
        double rxy = std::log2(eCxy[l-1]/eCxy[l]);
        std::printf("  %8d  %12.4e  %8.3f  %12.4e  %8.3f  %12.4e  %8.3f\n",
                    Nps[l], eCxx[l], rxx, eCyy[l], ryy, eCxy[l], rxy);
    }

    double rxx = std::log2(eCxx[1] / eCxx[3]) / 2.0;
    double ryy = std::log2(eCyy[1] / eCyy[3]) / 2.0;
    double rxy = std::log2(eCxy[1] / eCxy[3]) / 2.0;
    // Expect ~1st order once past the coarsest pre-asymptotic level.
    REQUIRE(rxx > 0.7);
    REQUIRE(ryy > 0.7);
    REQUIRE(rxy > 0.7);
}

// ─── 4b. Lobatto-center derivative convergence — star interface ─────────────

TEST_CASE("LaplaceLocalSolver2D: Lobatto-center derivative convergence — star interface",
          "[local_cauchy][2d][lobatto][convergence]")
{
    const int Nps[] = {16, 32, 64, 128};
    const int n_levels = 4;
    double eCx[n_levels], eCy[n_levels], eCxx[n_levels], eCyy[n_levels], eCxy[n_levels];

    for (int l = 0; l < n_levels; ++l) {
        auto e = lobatto_center_cauchy_errors(Nps[l]);
        eCx[l] = e.eCx;
        eCy[l] = e.eCy;
        eCxx[l] = e.eCxx;
        eCyy[l] = e.eCyy;
        eCxy[l] = e.eCxy;
    }

    std::printf("\n  Lobatto-center Cauchy 2D — derivative convergence (star interface):\n");
    std::printf("  %8s  %12s  %8s  %12s  %8s  %12s  %8s  %12s  %8s  %12s  %8s\n",
                "N_panels", "eCx", "rate", "eCy", "rate",
                "eCxx", "rate", "eCyy", "rate", "eCxy", "rate");
    std::printf("  %8d  %12.4e  %8s  %12.4e  %8s  %12.4e  %8s  %12.4e  %8s  %12.4e  %8s\n",
                Nps[0], eCx[0], "—", eCy[0], "—",
                eCxx[0], "—", eCyy[0], "—", eCxy[0], "—");
    for (int l = 1; l < n_levels; ++l) {
        const double rx = std::log2(eCx[l - 1] / eCx[l]);
        const double ry = std::log2(eCy[l - 1] / eCy[l]);
        const double rxx = std::log2(eCxx[l - 1] / eCxx[l]);
        const double ryy = std::log2(eCyy[l - 1] / eCyy[l]);
        const double rxy = std::log2(eCxy[l - 1] / eCxy[l]);
        std::printf("  %8d  %12.4e  %8.3f  %12.4e  %8.3f  %12.4e  %8.3f  %12.4e  %8.3f  %12.4e  %8.3f\n",
                    Nps[l], eCx[l], rx, eCy[l], ry,
                    eCxx[l], rxx, eCyy[l], ryy, eCxy[l], rxy);
    }

    const double rx = std::log2(eCx[1] / eCx[3]) / 2.0;
    const double ry = std::log2(eCy[1] / eCy[3]) / 2.0;
    const double rxx = std::log2(eCxx[1] / eCxx[3]) / 2.0;
    const double ryy = std::log2(eCyy[1] / eCyy[3]) / 2.0;
    const double rxy = std::log2(eCxy[1] / eCxy[3]) / 2.0;
    REQUIRE(rx > 1.5);
    REQUIRE(ry > 1.5);
    REQUIRE(rxx > 0.7);
    REQUIRE(ryy > 0.7);
    REQUIRE(rxy > 0.7);
}

// ─── 5. IIM plug-in: panel Cauchy solver feeds iim_correct_rhs_taylor ────────
//
// On a N=64 star grid, use laplace_panel_cauchy_2d to produce (C_q, Cx_q, …)
// and pass them into iim_correct_rhs_taylor.  The resulting max-norm solution
// error must be the same order as the analytical Taylor baseline at the same N.

TEST_CASE("LaplaceLocalSolver2D: IIM plug-in — panel Cauchy feeds Taylor correction",
          "[local_cauchy][2d][iim]")
{
    const int N = 64;
    const double h = 1.0 / N;

    // Grid
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];

    // Panel interface: use N panels × 3 pts = 3N quadrature nodes
    auto iface_panel = make_star_panels(0.5, 0.5, 0.28, 0.40, 5, N);

    // Plain interface (1 pt per panel) for GridPair2D + IIM correction
    // Reuse the same panel points but as a flat 1-per-panel interface
    // by wrapping the panel's Gauss points with points_per_panel=1:
    // GridPair2D doesn't care about panel structure — it uses all Nq points.
    // We pass iface_panel directly since Interface2D is just a point cloud for GridPair.
    GridPair2D gp(grid, iface_panel);

    // Domain labels
    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n) labels[n] = gp.domain_label(n);

    // f, u_exact at grid nodes
    Eigen::VectorXd f_arr(nx * ny), u_exact(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        double x = c[0], y = c[1];
        int lbl  = labels[n];
        u_exact[n] = (lbl == 1) ? u_plus(x, y) : u_minus(x, y);
        bool bdy = (n % nx == 0 || n % nx == nx-1 || n / nx == 0 || n / nx == ny-1);
        f_arr[n] = bdy ? 0.0 : ((lbl == 1) ? f_plus(x, y) : f_minus(x, y));
    }

    // ── Baseline: analytical Taylor C values (exact) ─────────────────────
    int Nq = iface_panel.num_points();
    Eigen::VectorXd C_q(Nq), Cx_q(Nq), Cy_q(Nq), Cxx_q(Nq), Cxy_q(Nq), Cyy_q(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface_panel.points()(q, 0);
        double y = iface_panel.points()(q, 1);
        C_q[q]   = C_fn  (x, y);
        Cx_q[q]  = Cx_fn (x, y);
        Cy_q[q]  = Cy_fn (x, y);
        Cxx_q[q] = Cxx_fn(x, y);
        Cxy_q[q] = Cxy_fn(x, y);
        Cyy_q[q] = Cyy_fn(x, y);
    }

    auto F_exact_taylor = iim_correct_rhs_taylor(grid, gp, iface_panel, f_arr,
                                                  C_q, Cx_q, Cy_q,
                                                  Cxx_q, Cxy_q, Cyy_q, labels);
    LaplaceFftBulkSolverZfft2D solver(grid, ZfftBcType::Dirichlet);
    Eigen::VectorXd u_exact_taylor;
    solver.solve(-F_exact_taylor, u_exact_taylor);

    double err_exact_taylor = 0.0;
    for (int j = 1; j < ny-1; ++j)
        for (int i = 1; i < nx-1; ++i)
            err_exact_taylor = std::max(err_exact_taylor,
                std::abs(u_exact_taylor[j*nx+i] - u_exact[j*nx+i]));

    // ── Panel Cauchy solver ───────────────────────────────────────────────
    Eigen::VectorXd a(Nq), b(Nq), Lu(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface_panel.points()(q, 0);
        double y = iface_panel.points()(q, 1);
        double nx_q = iface_panel.normals()(q, 0);
        double ny_q = iface_panel.normals()(q, 1);
        a[q]  = C_fn(x, y);
        b[q]  = Cx_fn(x,y)*nx_q + Cy_fn(x,y)*ny_q;
        Lu[q] = f_plus(x, y) - f_minus(x, y);
    }

    auto cauchy = laplace_panel_cauchy_2d(iface_panel, a, b, Lu);

    auto F_panel = iim_correct_rhs_taylor(grid, gp, iface_panel, f_arr,
                                          cauchy.C, cauchy.Cx, cauchy.Cy,
                                          cauchy.Cxx, cauchy.Cxy, cauchy.Cyy,
                                          labels);
    Eigen::VectorXd u_panel;
    solver.solve(-F_panel, u_panel);

    double err_panel = 0.0;
    for (int j = 1; j < ny-1; ++j)
        for (int i = 1; i < nx-1; ++i)
            err_panel = std::max(err_panel,
                std::abs(u_panel[j*nx+i] - u_exact[j*nx+i]));

    std::printf("\n  IIM plug-in (N=%d):\n", N);
    std::printf("    exact-Taylor err  = %.4e\n", err_exact_taylor);
    std::printf("    panel-Cauchy err  = %.4e  (ratio %.2f×)\n",
                err_panel, err_panel / err_exact_taylor);

    // Panel Cauchy result should stay in the same order as the analytical
    // Taylor baseline, though shifted collocation has a larger constant.
    REQUIRE(err_panel < 10.0 * err_exact_taylor);
    // Must also be well below the O(1) uncorrected baseline
    REQUIRE(err_panel < 1e-2);
}
