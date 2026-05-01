#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "core/grid/cartesian_grid_2d.hpp"
#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/local_cauchy/laplace_panel_solver_2d.hpp"
#include "core/solver/iim_laplace_2d.hpp"
#include "core/transfer/laplace_spread_2d.hpp"
#include "core/transfer/laplace_restrict_2d.hpp"

using namespace kfbim;
using Catch::Matchers::WithinAbs;

static constexpr double kPi = 3.14159265358979323846;

static const double kGL_s[3] = {
    -0.7745966692414834,
     0.0,
    +0.7745966692414834
};
static const double kGL_w[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};

static Interface2D make_star_panels(double cx, double cy,
                                    double R, double A, int K,
                                    int N_panels)
{
    const int Nq = 3 * N_panels;
    Eigen::MatrixX2d pts(Nq, 2), nml(Nq, 2);
    Eigen::VectorXd wts(Nq);
    Eigen::VectorXi comp = Eigen::VectorXi::Zero(N_panels);
    const double dth = 2.0 * kPi / N_panels;

    int q = 0;
    for (int p = 0; p < N_panels; ++p) {
        const double th_mid = (p + 0.5) * dth;
        const double half_dth = 0.5 * dth;
        for (int i = 0; i < 3; ++i) {
            const double th = th_mid + half_dth * kGL_s[i];
            const double r = R * (1.0 + A * std::cos(K * th));
            const double drdt = -R * A * K * std::sin(K * th);
            pts(q, 0) = cx + r * std::cos(th);
            pts(q, 1) = cy + r * std::sin(th);

            const double tx = drdt * std::cos(th) - r * std::sin(th);
            const double ty = drdt * std::sin(th) + r * std::cos(th);
            const double tlen = std::sqrt(tx * tx + ty * ty);
            nml(q, 0) = ty / tlen;
            nml(q, 1) = -tx / tlen;
            wts[q] = kGL_w[i] * half_dth * tlen;
            ++q;
        }
    }

    return {pts, nml, wts, 3, comp};
}

static double sol_u_int(double x, double y) {
    return std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
}

static double sol_u_ext(double x, double y) {
    return std::sin(kPi * x) * std::sin(kPi * y);
}

static double sol_f_int(double x, double y) {
    return 8.0 * kPi * kPi * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
}

static double sol_f_ext(double x, double y) {
    return 2.0 * kPi * kPi * std::sin(kPi * x) * std::sin(kPi * y);
}

static double C_fn(double x, double y) {
    return sol_u_int(x, y) - sol_u_ext(x, y);
}

static double Cx_fn(double x, double y) {
    return 2.0 * kPi * std::cos(2.0 * kPi * x) * std::sin(2.0 * kPi * y)
           - kPi * std::cos(kPi * x) * std::sin(kPi * y);
}

static double Cy_fn(double x, double y) {
    return 2.0 * kPi * std::sin(2.0 * kPi * x) * std::cos(2.0 * kPi * y)
           - kPi * std::sin(kPi * x) * std::cos(kPi * y);
}

struct Quad2D {
    double c0;
    double cx;
    double cy;
    double cxx;
    double cxy;
    double cyy;
};

static double quad_value(const Quad2D& q, double x, double y) {
    return q.c0 + q.cx * x + q.cy * y
           + 0.5 * q.cxx * x * x + q.cxy * x * y + 0.5 * q.cyy * y * y;
}

static LocalPoly2D quad_poly_at(const Quad2D& q, Eigen::Vector2d center) {
    LocalPoly2D poly;
    poly.center = center;
    poly.coeffs.resize(6);
    poly.coeffs << quad_value(q, center[0], center[1]),
                   q.cx + q.cxx * center[0] + q.cxy * center[1],
                   q.cy + q.cxy * center[0] + q.cyy * center[1],
                   q.cxx,
                   q.cxy,
                   q.cyy;
    return poly;
}

TEST_CASE("LaplacePanelSpread2D matches Taylor IIM RHS correction",
          "[transfer][spread][2d]")
{
    const int N = 32;
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto iface = make_star_panels(0.5, 0.5, 0.28, 0.40, 5, N);
    GridPair2D gp(grid, iface);

    const int Nq = iface.num_points();
    std::vector<LaplaceJumpData2D> jumps(Nq);
    Eigen::VectorXd a(Nq), b(Nq), Lu(Nq);
    for (int q = 0; q < Nq; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const double nx = iface.normals()(q, 0);
        const double ny = iface.normals()(q, 1);
        a[q] = C_fn(x, y);
        b[q] = Cx_fn(x, y) * nx + Cy_fn(x, y) * ny;
        Lu[q] = sol_f_int(x, y) - sol_f_ext(x, y);
        jumps[q].u_jump = a[q];
        jumps[q].un_jump = b[q];
        jumps[q].rhs_derivs = Eigen::VectorXd::Constant(1, Lu[q]);
    }

    LaplacePanelSpread2D spread(gp);
    Eigen::VectorXd rhs_correction = Eigen::VectorXd::Zero(grid.num_dofs());
    const auto polys = spread.apply(jumps, rhs_correction);

    const auto cauchy = laplace_panel_cauchy_2d(iface, a, b, Lu);
    std::vector<int> labels(grid.num_dofs());
    for (int n = 0; n < grid.num_dofs(); ++n)
        labels[n] = gp.domain_label(n);
    const Eigen::VectorXd zero_rhs = Eigen::VectorXd::Zero(grid.num_dofs());
    const Eigen::VectorXd expected =
        iim_correct_rhs_taylor(grid, gp, iface, zero_rhs,
                               cauchy.C, cauchy.Cx, cauchy.Cy,
                               cauchy.Cxx, cauchy.Cxy, cauchy.Cyy,
                               labels);

    REQUIRE(polys.size() == static_cast<size_t>(Nq));
    for (int q = 0; q < Nq; ++q) {
        REQUIRE(polys[q].coeffs.size() == 6);
        REQUIRE_THAT(polys[q].coeffs[4], WithinAbs(cauchy.Cxy[q], 1e-12));
        REQUIRE_THAT(polys[q].coeffs[5], WithinAbs(cauchy.Cyy[q], 1e-12));
    }
    for (int n = 0; n < grid.num_dofs(); ++n)
        REQUIRE_THAT(rhs_correction[n], WithinAbs(expected[n], 1e-10));
}

TEST_CASE("LaplaceQuadraticRestrict2D fits bulk polynomial and subtracts correction",
          "[transfer][restrict][2d]")
{
    const int N = 32;
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto iface = make_star_panels(0.5, 0.5, 0.24, 0.15, 3, 12);
    GridPair2D gp(grid, iface);

    const Quad2D physical{1.2, -0.7, 2.3, 0.5, -1.1, 0.8};
    const Quad2D correction{-0.4, 0.8, 0.2, 0.25, -0.3, 0.1};

    Eigen::VectorXd bulk(grid.num_dofs());
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const auto c = grid.coord(n);
        if (gp.domain_label(n) == 1) { // inside (interior)
            bulk[n] = quad_value(physical, c[0], c[1]);
        } else { // outside (exterior)
            bulk[n] = quad_value(physical, c[0], c[1])
                      - quad_value(correction, c[0], c[1]);
        }
    }

    std::vector<LocalPoly2D> correction_polys(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        Eigen::Vector2d center = iface.points().row(q).transpose();
        correction_polys[q] = quad_poly_at(correction, center);
    }

    LaplaceQuadraticRestrict2D restrict_op(gp);
    const auto result = restrict_op.apply(bulk, correction_polys);

    REQUIRE(result.size() == static_cast<size_t>(iface.num_points()));
    for (int q = 0; q < iface.num_points(); ++q) {
        Eigen::Vector2d center = iface.points().row(q).transpose();
        const LocalPoly2D expected = quad_poly_at(physical, center);
        for (int k = 0; k < 6; ++k)
            REQUIRE_THAT(result[q].coeffs[k], WithinAbs(expected.coeffs[k], 1e-9));
    }
}

// ── Piecewise smooth manufactured solution for Restrict convergence test ──────
//
//   u_out = sin(πx) sin(πy)       exterior (outside circle, label 0)
//   u_in  = sin(2πx) sin(2πy)     interior (inside  circle, label 1)
//
//   C  = u_out − u_in  (jump: exterior minus interior)
//   Cn = ∂C/∂n = Cx·nx + Cy·ny
//
//   bulk[n] = u_out  if label[n]=0,  u_in  if label[n]=1
//   correction_poly at each interface point: Taylor coeffs of C
//
//   Restrict(bulk, correction) → u_trace (interior trace u⁻)
//   u_avg  = u_trace + C/2     vs  (u_out + u_in)/2
//   un_avg = un_trace + Cn/2   vs  (∂u_out/∂n + ∂u_in/∂n)/2

static double u_int_ps(double x, double y) { return std::sin(kPi*x)*std::sin(kPi*y); }
static double u_ext_ps (double x, double y) { return std::sin(2*kPi*x)*std::sin(2*kPi*y); }

static double u_int_x(double x, double y) { return  kPi*std::cos(kPi*x)*std::sin(kPi*y); }
static double u_int_y(double x, double y) { return  kPi*std::sin(kPi*x)*std::cos(kPi*y); }
static double u_ext_x (double x, double y) { return 2*kPi*std::cos(2*kPi*x)*std::sin(2*kPi*y); }
static double u_ext_y (double x, double y) { return 2*kPi*std::sin(2*kPi*x)*std::cos(2*kPi*y); }

static double u_int_xx(double x, double y) { return -kPi*kPi*u_int_ps(x,y); }
static double u_int_yy(double x, double y) { return -kPi*kPi*u_int_ps(x,y); }
static double u_int_xy(double x, double y) { return  kPi*kPi*std::cos(kPi*x)*std::cos(kPi*y); }
static double u_ext_xx (double x, double y) { return -4*kPi*kPi*u_ext_ps(x,y); }
static double u_ext_yy (double x, double y) { return -4*kPi*kPi*u_ext_ps(x,y); }
static double u_ext_xy (double x, double y) { return  4*kPi*kPi*std::cos(2*kPi*x)*std::cos(2*kPi*y); }

static double C_ps (double x, double y) { return u_int_ps(x,y) - u_ext_ps(x,y); }
static double Cx_ps(double x, double y) { return u_int_x(x,y)  - u_ext_x(x,y); }
static double Cy_ps(double x, double y) { return u_int_y(x,y)  - u_ext_y(x,y); }
static double Cxx_ps(double x, double y){ return u_int_xx(x,y) - u_ext_xx(x,y); }
static double Cxy_ps(double x, double y){ return u_int_xy(x,y) - u_ext_xy(x,y); }
static double Cyy_ps(double x, double y){ return u_int_yy(x,y) - u_ext_yy(x,y); }

struct RestrictConvData {
    double u_avg_err;
    double un_avg_err;
};

static RestrictConvData run_restrict_convergence(int N)
{
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto iface = make_star_panels(0.5, 0.5, 0.3, 0.0, 0, N);  // circle, A=0
    GridPair2D gp(grid, iface);
    const int Nq = iface.num_points();

    // Piecewise smooth bulk field at grid nodes
    Eigen::VectorXd bulk(grid.num_dofs());
    for (int n = 0; n < grid.num_dofs(); ++n) {
        auto c = grid.coord(n);
        double x = c[0], y = c[1];
        bulk[n] = (gp.domain_label(n) == 1) ? u_int_ps(x, y) : u_ext_ps(x, y);
    }

    // Correction polynomials at interface points (exact Taylor coeffs of C)
    std::vector<LocalPoly2D> corr_polys(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0), y = iface.points()(q, 1);
        Eigen::Vector2d center(x, y);
        corr_polys[q].center = center;
        corr_polys[q].coeffs.resize(6);
        corr_polys[q].coeffs << C_ps(x,y), Cx_ps(x,y), Cy_ps(x,y),
                                Cxx_ps(x,y), Cxy_ps(x,y), Cyy_ps(x,y);
    }

    LaplaceQuadraticRestrict2D restrict_op(gp);
    auto result = restrict_op.apply(bulk, corr_polys);

    double u_avg_err  = 0.0;
    double un_avg_err = 0.0;
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0), y = iface.points()(q, 1);
        double nx = iface.normals()(q, 0), ny = iface.normals()(q, 1);

        double jump_u  = C_ps(x, y);
        double jump_un = Cx_ps(x,y)*nx + Cy_ps(x,y)*ny;

        double u_trace  = result[q].coeffs[0];
        double ux_trace = result[q].coeffs[1];
        double uy_trace = result[q].coeffs[2];
        double un_trace = ux_trace * nx + uy_trace * ny;

        double num_u_avg  = u_trace  - 0.5 * jump_u;
        double num_un_avg = un_trace - 0.5 * jump_un;

        double ex_u_avg  = 0.5 * (u_int_ps(x,y) + u_ext_ps(x,y));
        double ex_un_avg = 0.5 * ((u_int_x(x,y)*nx + u_int_y(x,y)*ny)
                                + (u_ext_x(x,y)*nx  + u_ext_y(x,y)*ny));

        u_avg_err  = std::max(u_avg_err,  std::abs(num_u_avg  - ex_u_avg));
        un_avg_err = std::max(un_avg_err, std::abs(num_un_avg - ex_un_avg));
    }

    return {u_avg_err, un_avg_err};
}

TEST_CASE("LaplaceQuadraticRestrict2D: piecewise smooth convergence — circle interface",
          "[transfer][restrict][convergence][piecewise][2d]")
{
    const int Ns[]     = {32, 64, 128, 256};
    const int n_levels = 4;
    RestrictConvData data[n_levels];

    std::printf("\n  LaplaceQuadraticRestrict2D: piecewise smooth convergence\n");
    std::printf("  u_int=sin(πx)sin(πy) (int), u_ext=sin(2πx)sin(2πy) (ext)\n");
    std::printf("  u_avg = u_trace - C/2,  un_avg = un_trace - Cn/2\n");
    std::printf("  %6s  %12s  %8s  %12s  %8s\n",
                "N", "u_avg_err", "rate", "un_avg_err", "rate");

    for (int l = 0; l < n_levels; ++l) {
        data[l] = run_restrict_convergence(Ns[l]);
        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s  %12.4e  %8s\n",
                        Ns[l], data[l].u_avg_err, "—",
                        data[l].un_avg_err, "—");
        } else {
            double rate_u  = std::log2(data[l-1].u_avg_err  / data[l].u_avg_err);
            double rate_un = std::log2(data[l-1].un_avg_err / data[l].un_avg_err);
            std::printf("  %6d  %12.4e  %8.3f  %12.4e  %8.3f\n",
                        Ns[l], data[l].u_avg_err, rate_u,
                        data[l].un_avg_err, rate_un);
        }
    }

    // Require O(h²) convergence for u_avg and at least O(h) for un_avg
    double rate_u_32_256 = std::log2(data[0].u_avg_err / data[3].u_avg_err) / 3.0;
    double rate_un_32_256 = std::log2(data[0].un_avg_err / data[3].un_avg_err) / 3.0;
    REQUIRE(rate_u_32_256  > 1.5);
    REQUIRE(rate_un_32_256 > 1.0);
}
