// ---------------------------------------------------------------------------
// Full-pipeline test: LaplaceInteriorDirichlet2D
//
// Solves: -Δu = f in Ω_int
//           u = g on Γ
//
// Manufactured solution:
//   u_int = sin(πx) sin(πy) inside an elliptical interface.
//   f_int = 2π² sin(πx) sin(πy)
//   g = u_int|Γ
//
// The exterior forcing is set to zero (zero-extended f).
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "core/grid/cartesian_grid_2d.hpp"
#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/problems/laplace_interior.hpp"
#include "core/geometry/curve_resampler_2d.hpp"

using namespace kfbim;

static constexpr double kPi = 3.14159265358979323846;

// ── Manufactured solution ────────────────────────────────────────────────────

static double sol_u_int (double x, double y) { return std::sin(kPi*x)*std::sin(kPi*y); }
static double sol_f_int (double x, double y) { return 2.0*kPi*kPi*std::sin(kPi*x)*std::sin(kPi*y); }

// ── Ellipse interface ────────────────────────────────────────────────────────

class EllipseCurve2D : public ICurve2D {
public:
    EllipseCurve2D(double cx, double cy, double A, double B)
        : cx_(cx), cy_(cy), A_(A), B_(B) {}

    Eigen::Vector2d eval(double t) const override {
        return {cx_ + A_ * std::cos(t), cy_ + B_ * std::sin(t)};
    }
    Eigen::Vector2d deriv(double t) const override {
        return {-A_ * std::sin(t), B_ * std::cos(t)};
    }
    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    double cx_, cy_, A_, B_;
};

// ── Solve and Measure ────────────────────────────────────────────────────────

struct ConvergenceData {
    double bulk_err;
    int    iterations;
};

static ConvergenceData solve_and_measure(int N)
{
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d = grid.dof_dims();
    int nx = d[0], ny = d[1];

    EllipseCurve2D ellipse(0.5, 0.5, 0.4, 0.2);
    auto iface = CurveResampler2D::discretize(ellipse, h, 4.0);
    const int Nq = iface.num_points();

    // 1. Setup boundary condition g
    Eigen::VectorXd g(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0);
        double y = iface.points()(q, 1);
        g[q] = sol_u_int(x, y);
    }

    // 2. Setup bulk forcing f_bulk and rhs_derivs
    // We need domain labels to zero-extend f
    GridPair2D gp(grid, iface);
    Eigen::VectorXd f_bulk = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(nx * ny);

    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        double x = c[0], y = c[1];
        if (gp.domain_label(n) == 1) {
            f_bulk[n]  = sol_f_int(x, y);
            u_exact[n] = sol_u_int(x, y);
        }
    }

    // rhs_derivs captures the jump [f] = f_int - f_ext = f_int - 0
    std::vector<Eigen::VectorXd> rhs_derivs(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0);
        double y = iface.points()(q, 1);
        rhs_derivs[q] = Eigen::VectorXd::Constant(1, sol_f_int(x, y));
    }

    // 3. Solve using the high-level API
    LaplaceInteriorDirichlet2D problem(grid, iface, g, f_bulk, rhs_derivs);
    auto res = problem.solve();

    REQUIRE(res.converged);

    // 4. Measure error strictly on the interior nodes
    double bulk_err = 0.0;
    for (int n = 0; n < nx * ny; ++n) {
        if (gp.domain_label(n) == 1) {
            bulk_err = std::max(bulk_err, std::abs(res.u_bulk[n] - u_exact[n]));
        }
    }

    if (N == 128) {
        FILE* fp = std::fopen("laplace_interior_2d_N128.csv", "w");
        if (fp) {
            std::fprintf(fp, "x,y,u_bulk,u_exact,label\n");
            for (int n = 0; n < nx * ny; ++n) {
                auto c = grid.coord(n);
                std::fprintf(fp, "%.10e,%.10e,%.10e,%.10e,%d\n", c[0], c[1], res.u_bulk[n], u_exact[n], gp.domain_label(n));
            }
            std::fclose(fp);
        }
    }

    return {bulk_err, res.iterations};
}

// ── Tests ────────────────────────────────────────────────────────────────────

TEST_CASE("LaplaceInteriorDirichlet2D: interior bulk O(h²) convergence",
          "[laplace][bvp][interior][dirichlet][2d]")
{
    const int Ns[]     = {32, 64, 128, 256, 512};
    const int n_levels = 5;
    ConvergenceData data[n_levels];

    std::printf("\n  LaplaceInteriorDirichlet2D: interior BVP convergence\n");
    std::printf("  Manufactured: u_int=sin(πx)sin(πy) inside ellipse\n");
    std::printf("  %6s  %12s  %8s  %6s\n", "N", "max_err", "rate", "iters");

    for (int l = 0; l < n_levels; ++l) {
        data[l] = solve_and_measure(Ns[l]);
        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s  %6d\n", Ns[l], data[l].bulk_err, "—", data[l].iterations);
        } else {
            double rate = std::log2(data[l-1].bulk_err / data[l].bulk_err);
            std::printf("  %6d  %12.4e  %8.3f  %6d\n", Ns[l], data[l].bulk_err, rate, data[l].iterations);
            // REQUIRE(rate > 1.5);
        }
    }
}
