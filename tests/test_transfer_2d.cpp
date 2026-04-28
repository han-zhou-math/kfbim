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

static double u_plus(double x, double y) {
    return std::sin(kPi * x) * std::sin(kPi * y);
}

static double u_minus(double x, double y) {
    return std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
}

static double f_plus(double x, double y) {
    return 2.0 * kPi * kPi * std::sin(kPi * x) * std::sin(kPi * y);
}

static double f_minus(double x, double y) {
    return 8.0 * kPi * kPi * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
}

static double C_fn(double x, double y) {
    return u_plus(x, y) - u_minus(x, y);
}

static double Cx_fn(double x, double y) {
    return kPi * std::cos(kPi * x) * std::sin(kPi * y)
           - 2.0 * kPi * std::cos(2.0 * kPi * x) * std::sin(2.0 * kPi * y);
}

static double Cy_fn(double x, double y) {
    return kPi * std::sin(kPi * x) * std::cos(kPi * y)
           - 2.0 * kPi * std::sin(2.0 * kPi * x) * std::cos(2.0 * kPi * y);
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
        Lu[q] = f_plus(x, y) - f_minus(x, y);
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
        bulk[n] = quad_value(physical, c[0], c[1])
                  + quad_value(correction, c[0], c[1]);
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
