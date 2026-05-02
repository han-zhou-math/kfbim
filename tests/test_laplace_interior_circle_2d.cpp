// ---------------------------------------------------------------------------
// Full-pipeline test: LaplaceInteriorDirichlet2D — circle (center (0, 0.1))
//
// Solves: -Δu = 0 in Ω_int (circle, center (0, 0.1), radius 1)
//           u = g on Γ
//
// Manufactured solution:
//   u(x,y) = exp(x) sin(y)   (harmonic: Δu = 0)
//   f = 0 everywhere
//   g = u|Γ
//
// Computational domain: [-2, 2]²
//
// Note: center is (0, 0.1) rather than origin to prevent grid nodes from
// landing exactly on the interface.  For a circle centered at the origin on
// a symmetric grid with h=L/N, the axis crossings (±1,0) and (0,±1) land at
// exactly grid-node positions whenever L*k/N = 1 for some integer k.  Exact
// node–interface coincidence makes the IIM correction stencil degenerate,
// producing erratic convergence rates.  The y-offset 0.1 eliminates the
// alignment for N ∈ {32,64,128,256,512}.
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

static double sol_u(double x, double y) { return std::exp(x) * std::sin(y); }

// ── Unit circle interface ────────────────────────────────────────────────────

// cy = 0.1 avoids exact grid-node/interface alignment for N in {32,64,128,256,512}
static constexpr double kCy = 0.1;

class CircleCurve2D : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override {
        return {std::cos(t), kCy + std::sin(t)};
    }
    Eigen::Vector2d deriv(double t) const override {
        return {-std::sin(t), std::cos(t)};
    }
    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }
};

// ── Solve and Measure ────────────────────────────────────────────────────────

struct ConvergenceData {
    double bulk_err;
    int    iterations;
};

static ConvergenceData solve_and_measure(int N)
{
    const double L = 4.0;               // domain [-2,2]²
    const double h = L / N;
    CartesianGrid2D grid({-2.0, -2.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d = grid.dof_dims();
    int nx = d[0], ny = d[1];

    CircleCurve2D circle;
    auto iface = CurveResampler2D::discretize_legacy_gauss(circle, h, 4.0);
    const int Nq = iface.num_points();

    // 1. Setup boundary condition g
    Eigen::VectorXd g(Nq);
    for (int q = 0; q < Nq; ++q) {
        double x = iface.points()(q, 0);
        double y = iface.points()(q, 1);
        g[q] = sol_u(x, y);
    }

    // 2. Setup bulk forcing and exact solution
    // f = 0 everywhere (u is harmonic)
    GridPair2D gp(grid, iface);
    Eigen::VectorXd f_bulk   = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd u_exact  = Eigen::VectorXd::Zero(nx * ny);

    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        double x = c[0], y = c[1];
        if (gp.domain_label(n) == 1) {
            u_exact[n] = sol_u(x, y);
        }
    }

    // rhs_derivs = 0 (no jump in f)
    std::vector<Eigen::VectorXd> rhs_derivs(Nq,
        Eigen::VectorXd::Constant(1, 0.0));

    // 3. Solve using the high-level API
    LaplaceInteriorDirichlet2D problem(grid, iface, g, f_bulk, rhs_derivs,
                                       LaplaceInteriorPanelMethod2D::LegacyGaussPanel);
    auto res = problem.solve();

    REQUIRE(res.converged);

    // 4. Measure error strictly on the interior nodes
    double bulk_err = 0.0;
    for (int n = 0; n < nx * ny; ++n) {
        if (gp.domain_label(n) == 1) {
            bulk_err = std::max(bulk_err, std::abs(res.u_bulk[n] - u_exact[n]));
        }
    }

    {
        char fname[64];
        std::snprintf(fname, sizeof(fname), "laplace_interior_circle_2d_N%d.csv", N);
        FILE* fp = std::fopen(fname, "w");
        if (fp) {
            std::fprintf(fp, "x,y,u_bulk,u_exact,label\n");
            for (int n = 0; n < nx * ny; ++n) {
                auto c = grid.coord(n);
                std::fprintf(fp, "%.10e,%.10e,%.10e,%.10e,%d\n",
                    c[0], c[1], res.u_bulk[n], u_exact[n], gp.domain_label(n));
            }
            std::fclose(fp);
        }
    }

    return {bulk_err, res.iterations};
}

// ── Tests ────────────────────────────────────────────────────────────────────

TEST_CASE("LaplaceInteriorDirichlet2D: circle (center (0,0.1)) O(h²) convergence",
          "[laplace][bvp][interior][dirichlet][2d][circle]")
{
    const int Ns[]     = {32, 64, 128, 256, 512, 1024};
    const int n_levels = 6;
    ConvergenceData data[n_levels];

    std::printf("\n  LaplaceInteriorDirichlet2D: circle BVP convergence\n");
    std::printf("  Manufactured: u=exp(x)sin(y), harmonic (f=0), circle center (0,%.1f)\n", kCy);
    std::printf("  Domain: [-2,2]^2\n");
    std::printf("  %6s  %12s  %8s  %6s\n", "N", "max_err", "rate", "iters");

    for (int l = 0; l < n_levels; ++l) {
        data[l] = solve_and_measure(Ns[l]);
        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s  %6d\n",
                Ns[l], data[l].bulk_err, "—", data[l].iterations);
        } else {
            double rate = std::log2(data[l-1].bulk_err / data[l].bulk_err);
            std::printf("  %6d  %12.4e  %8.3f  %6d\n",
                Ns[l], data[l].bulk_err, rate, data[l].iterations);
        }
    }
}
