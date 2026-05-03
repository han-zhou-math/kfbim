// ---------------------------------------------------------------------------
// Full-pipeline test: LaplaceInteriorDirichlet2D
//
// Solves: -Δu = f in Ω_int
//           u = g on Γ
//
// Manufactured solution:
//   u_int = exp(x) cos(y) inside a 5-fold star-shaped curve.
//   f_int = 0 because u_int is harmonic
//   g = u_int|Γ
//
// Domain: [-1.8,1.8]².
// The exterior forcing is set to zero (zero-extended f).
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>
#include <algorithm>

#include "src/grid/cartesian_grid_2d.hpp"
#include "src/interface/interface_2d.hpp"
#include "src/geometry/grid_pair_2d.hpp"
#include "src/problems/laplace_interior.hpp"
#include "src/geometry/curve_resampler_2d.hpp"

using namespace kfbim;

static constexpr double kPi = 3.14159265358979323846;
static constexpr double kBoxHalfWidth = 1.8;
static constexpr double kTargetPanelLengthOverH = 4.0;

#ifndef KFBIM_TEST_OUTPUT_DIR
#define KFBIM_TEST_OUTPUT_DIR "output"
#endif

// ── Manufactured solution ────────────────────────────────────────────────────

static double sol_u_int(double x, double y) { return std::exp(x) * std::cos(y); }
static double sol_f_int(double, double) { return 0.0; }

// ── 5-fold star interface ────────────────────────────────────────────────────

static constexpr double kStarCx = 0.07;
static constexpr double kStarCy = -0.04;
static constexpr double kStarRadius = 0.75;
static constexpr double kStarAmplitude = 0.25;
static constexpr int    kStarFolds = 5;

class StarCurve2D : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override {
        const double r = radius(t);
        return {kStarCx + r * std::cos(t), kStarCy + r * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override {
        const double r = radius(t);
        const double drdt = -kStarRadius * kStarAmplitude * kStarFolds
                            * std::sin(kStarFolds * t);
        return {drdt * std::cos(t) - r * std::sin(t),
                drdt * std::sin(t) + r * std::cos(t)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    static double radius(double t) {
        return kStarRadius * (1.0 + kStarAmplitude * std::cos(kStarFolds * t));
    }
};

// ── Solve and Measure ────────────────────────────────────────────────────────

struct ConvergenceData {
    double bulk_err;
    int    iterations;
};

static std::filesystem::path output_dir()
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR) / "laplace_interior_star5";
    std::filesystem::create_directories(dir);
    return dir;
}

static ConvergenceData solve_and_measure(int N)
{
    const double L = 2.0 * kBoxHalfWidth;
    const double h = L / N;
    CartesianGrid2D grid({-kBoxHalfWidth, -kBoxHalfWidth}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d = grid.dof_dims();
    int nx = d[0], ny = d[1];

    StarCurve2D star;
    Interface2D iface = CurveResampler2D::discretize(star, h, kTargetPanelLengthOverH);
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
    auto res = problem.solve(800, 1.0e-8, 200);

    REQUIRE(res.converged);

    // 4. Measure error strictly on the interior nodes
    double bulk_err = 0.0;
    for (int n = 0; n < nx * ny; ++n) {
        if (gp.domain_label(n) == 1) {
            bulk_err = std::max(bulk_err, std::abs(res.u_bulk[n] - u_exact[n]));
        }
    }

    if (N == 128) {
        const std::filesystem::path path = output_dir() / "laplace_interior_2d_star5_N128.csv";
        FILE* fp = std::fopen(path.string().c_str(), "w");
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

static ConvergenceData solve_lobatto_and_measure(int N)
{
    return solve_and_measure(N);
}

TEST_CASE("LaplaceInteriorDirichlet2D: Lobatto-center KFBIM path runs on 5-fold star",
          "[laplace][bvp][interior][dirichlet][lobatto][2d]")
{
    const int N = 64;
    const double L = 2.0 * kBoxHalfWidth;
    const double h = L / N;
    CartesianGrid2D grid({-kBoxHalfWidth, -kBoxHalfWidth}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d = grid.dof_dims();
    const int nx = d[0];
    const int ny = d[1];

    StarCurve2D star;
    auto iface = CurveResampler2D::discretize(star, h, kTargetPanelLengthOverH);
    const int Nq = iface.num_points();

    Eigen::VectorXd g(Nq);
    std::vector<Eigen::VectorXd> rhs_derivs(Nq);
    for (int q = 0; q < Nq; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        g[q] = sol_u_int(x, y);
        rhs_derivs[q] = Eigen::VectorXd::Constant(1, sol_f_int(x, y));
    }

    GridPair2D gp(grid, iface);
    Eigen::VectorXd f_bulk = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        if (gp.domain_label(n) == 1)
            u_exact[n] = sol_u_int(c[0], c[1]);
    }

    LaplaceInteriorDirichlet2D problem(grid, iface, g, f_bulk, rhs_derivs);
    auto res = problem.solve(800, 1.0e-8, 200);

    REQUIRE(res.converged);

    double bulk_err = 0.0;
    for (int n = 0; n < nx * ny; ++n) {
        if (gp.domain_label(n) == 1)
            bulk_err = std::max(bulk_err, std::abs(res.u_bulk[n] - u_exact[n]));
    }
    REQUIRE(bulk_err < 2.0e-2);
}

// ── Tests ────────────────────────────────────────────────────────────────────

TEST_CASE("LaplaceInteriorDirichlet2D: Chebyshev-Lobatto DOF convergence on 5-fold star",
          "[laplace][bvp][interior][dirichlet][lobatto][convergence][2d]")
{
    const int Ns[]     = {32, 64, 128, 256, 512, 1024};
    const int n_levels = 6;
    ConvergenceData data[n_levels];

    std::printf("\n  LaplaceInteriorDirichlet2D: Chebyshev-Lobatto DOF convergence\n");
    std::printf("  Manufactured: u_int=exp(x)cos(y) inside 5-fold star centered at (%.2f, %.2f), domain [-%.1f,%.1f]^2\n",
                kStarCx, kStarCy, kBoxHalfWidth, kBoxHalfWidth);
    std::printf("  Panel DOFs: Chebyshev-Lobatto s={-1,0,1}; correction expansion centers: s={-0.75,-0.25,0.25,0.75}\n");
    std::printf("  Target Chebyshev-node spacing / h ≈ %.2f (panel arc length / h = %.2f)\n",
                0.5 * kTargetPanelLengthOverH, kTargetPanelLengthOverH);
    std::printf("  %6s  %12s  %8s  %6s\n", "N", "max_err", "rate", "iters");

    for (int l = 0; l < n_levels; ++l) {
        data[l] = solve_lobatto_and_measure(Ns[l]);
        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s  %6d\n", Ns[l], data[l].bulk_err, "—", data[l].iterations);
        } else {
            const double rate = std::log2(data[l-1].bulk_err / data[l].bulk_err);
            std::printf("  %6d  %12.4e  %8.3f  %6d\n", Ns[l], data[l].bulk_err, rate, data[l].iterations);
        }
    }
}
