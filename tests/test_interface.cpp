#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <utility>
#include <vector>

#include "src/bulk_solvers/laplace_zfft_bulk_solver_2d.hpp"
#include "src/bulk_solvers/zfft_bc_type.hpp"
#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/geometry/grid_pair_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/local_cauchy/jump_data.hpp"
#include "src/potentials/laplace_potential.hpp"
#include "src/transfer/laplace_restrict_2d.hpp"
#include "src/transfer/laplace_spread_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEta = 1.1;
constexpr double kTargetNodeSpacingOverH = 1.5;
constexpr double kTargetPanelLengthOverH = 2.0 * kTargetNodeSpacingOverH;

constexpr double kStarCx = 0.07;
constexpr double kStarCy = -0.04;
constexpr double kStarRadius = 0.75;
constexpr double kStarAmplitude = 0.25;
constexpr int    kStarFolds = 3;
constexpr double kBoxMargin = 0.30;

#ifndef KFBIM_TEST_OUTPUT_DIR
#define KFBIM_TEST_OUTPUT_DIR "output"
#endif

class Star3Curve2D final : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override
    {
        const double r = radius(t);
        return {kStarCx + r * std::cos(t),
                kStarCy + r * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override
    {
        const double r = radius(t);
        const double drdt = -kStarRadius * kStarAmplitude * kStarFolds
                            * std::sin(kStarFolds * t);
        return {drdt * std::cos(t) - r * std::sin(t),
                drdt * std::sin(t) + r * std::cos(t)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    static double radius(double t)
    {
        return kStarRadius * (1.0 + kStarAmplitude * std::cos(kStarFolds * t));
    }
};

struct Box2D {
    std::array<double, 2> lower;
    double                side_length;
};

struct SolveData {
    double bulk_err = 0.0;
    int    num_panels = 0;
    int    num_interface_points = 0;
};

struct StageTimer {
    using Clock = std::chrono::steady_clock;

    Clock::time_point start = Clock::now();

    double lap()
    {
        const Clock::time_point now = Clock::now();
        const std::chrono::duration<double> elapsed = now - start;
        start = now;
        return elapsed.count();
    }
};

bool profile_enabled()
{
    return std::getenv("KFBIM_PROFILE_INTERFACE_2D") != nullptr;
}

int max_n_from_env()
{
    const char* value = std::getenv("KFBIM_INTERFACE_2D_MAX_N");
    if (value == nullptr)
        return 0;
    return std::max(0, std::atoi(value));
}

Box2D make_outer_box(const ICurve2D& curve)
{
    constexpr int kSamples = 8192;

    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < kSamples; ++i) {
        const double t = curve.t_min()
                         + (curve.t_max() - curve.t_min())
                         * static_cast<double>(i) / static_cast<double>(kSamples);
        const Eigen::Vector2d p = curve.eval(t);
        xmin = std::min(xmin, p[0]);
        xmax = std::max(xmax, p[0]);
        ymin = std::min(ymin, p[1]);
        ymax = std::max(ymax, p[1]);
    }

    const double cx = 0.5 * (xmin + xmax);
    const double cy = 0.5 * (ymin + ymax);
    const double span = std::max(xmax - xmin, ymax - ymin);
    const double side = span + 2.0 * kBoxMargin;
    return {{cx - 0.5 * side, cy - 0.5 * side}, side};
}

double sine_mode(const Box2D& box, double x, double y, int mx, int my)
{
    const double xi = x - box.lower[0];
    const double yi = y - box.lower[1];
    const double kx = mx * kPi / box.side_length;
    const double ky = my * kPi / box.side_length;
    return std::sin(kx * xi) * std::sin(ky * yi);
}

Eigen::Vector2d sine_mode_grad(const Box2D& box, double x, double y, int mx, int my)
{
    const double xi = x - box.lower[0];
    const double yi = y - box.lower[1];
    const double kx = mx * kPi / box.side_length;
    const double ky = my * kPi / box.side_length;
    return {kx * std::cos(kx * xi) * std::sin(ky * yi),
            ky * std::sin(kx * xi) * std::cos(ky * yi)};
}

double u_int(const Box2D& box, double x, double y)
{
    return 0.80 * sine_mode(box, x, y, 1, 2)
           + 0.25 * sine_mode(box, x, y, 3, 1);
}

Eigen::Vector2d grad_u_int(const Box2D& box, double x, double y)
{
    return 0.80 * sine_mode_grad(box, x, y, 1, 2)
           + 0.25 * sine_mode_grad(box, x, y, 3, 1);
}

double f_int(const Box2D& box, double x, double y)
{
    const double l2_12 = (1.0 * 1.0 + 2.0 * 2.0) * kPi * kPi
                         / (box.side_length * box.side_length);
    const double l2_31 = (3.0 * 3.0 + 1.0 * 1.0) * kPi * kPi
                         / (box.side_length * box.side_length);
    return 0.80 * (l2_12 + kEta) * sine_mode(box, x, y, 1, 2)
           + 0.25 * (l2_31 + kEta) * sine_mode(box, x, y, 3, 1);
}

double u_ext(const Box2D& box, double x, double y)
{
    return -0.50 * sine_mode(box, x, y, 2, 1)
           + 0.35 * sine_mode(box, x, y, 1, 3);
}

Eigen::Vector2d grad_u_ext(const Box2D& box, double x, double y)
{
    return -0.50 * sine_mode_grad(box, x, y, 2, 1)
           + 0.35 * sine_mode_grad(box, x, y, 1, 3);
}

double f_ext(const Box2D& box, double x, double y)
{
    const double l2_21 = (2.0 * 2.0 + 1.0 * 1.0) * kPi * kPi
                         / (box.side_length * box.side_length);
    const double l2_13 = (1.0 * 1.0 + 3.0 * 3.0) * kPi * kPi
                         / (box.side_length * box.side_length);
    return -0.50 * (l2_21 + kEta) * sine_mode(box, x, y, 2, 1)
           + 0.35 * (l2_13 + kEta) * sine_mode(box, x, y, 1, 3);
}

bool is_outer_boundary_node(int idx, int nx, int ny)
{
    const int i = idx % nx;
    const int j = idx / nx;
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

std::vector<LaplaceJumpData2D> make_jumps(const Interface2D& iface,
                                          const Box2D&       box)
{
    const int n_iface = iface.num_points();
    std::vector<LaplaceJumpData2D> jumps(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const Eigen::Vector2d normal(iface.normals()(q, 0),
                                     iface.normals()(q, 1));
        jumps[q].u_jump = u_int(box, x, y) - u_ext(box, x, y);
        jumps[q].un_jump = (grad_u_int(box, x, y) - grad_u_ext(box, x, y))
                               .dot(normal);
        jumps[q].rhs_derivs =
            Eigen::VectorXd::Constant(1, f_int(box, x, y) - f_ext(box, x, y));
    }
    return jumps;
}

std::filesystem::path output_dir()
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR) / "laplace_interface_star3";
    std::filesystem::create_directories(dir);
    return dir;
}

double tail_average_order(const std::vector<double>& rates, int count = 3)
{
    REQUIRE(rates.size() > 1);
    const int end = static_cast<int>(rates.size());
    const int begin = std::max(1, end - count);
    double sum = 0.0;
    int used = 0;
    for (int i = begin; i < end; ++i) {
        REQUIRE(std::isfinite(rates[i]));
        sum += rates[i];
        ++used;
    }
    REQUIRE(used > 0);
    return sum / static_cast<double>(used);
}

SolveData solve_and_measure(int N)
{
    StageTimer timer;
    const bool profile = profile_enabled();
    Star3Curve2D star;
    const Box2D box = make_outer_box(star);
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid2D grid(box.lower, {h, h}, {N, N}, DofLayout2D::Node);
    const double t_grid = timer.lap();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int n_dof = nx * ny;

    Interface2D iface =
        CurveResampler2D::discretize(star, h, kTargetPanelLengthOverH);
    const double t_interface = timer.lap();
    GridPair2D grid_pair(grid, iface);
    const double t_grid_pair = timer.lap();

    Eigen::VectorXd f_bulk = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        if (grid_pair.domain_label(n) > 0) {
            u_exact[n] = u_int(box, x, y);
            f_bulk[n] = is_outer_boundary_node(n, nx, ny) ? 0.0 : f_int(box, x, y);
        } else {
            u_exact[n] = u_ext(box, x, y);
            f_bulk[n] = is_outer_boundary_node(n, nx, ny) ? 0.0 : f_ext(box, x, y);
        }
    }
    const double t_rhs = timer.lap();

    LaplaceQuadraticPanelCenterSpread2D spread(grid_pair, kEta);
    LaplaceFftBulkSolverZfft2D bulk_solver(grid, ZfftBcType::Dirichlet, kEta);
    LaplaceQuadraticPanelCenterRestrict2D restrict_op(grid_pair);
    LaplacePotentialEval2D potentials(spread, bulk_solver, restrict_op);
    const double t_ops = timer.lap();

    const std::vector<LaplaceJumpData2D> jumps = make_jumps(iface, box);
    const double t_jumps = timer.lap();

    Eigen::VectorXd u_bulk;
    double t_spread = 0.0;
    double t_bulk_solve = 0.0;
    double t_restrict = 0.0;
    if (profile) {
        Eigen::VectorXd rhs = f_bulk;
        const std::vector<LocalPoly2D> correction_polys = spread.apply(jumps, rhs);
        t_spread = timer.lap();

        bulk_solver.solve(-rhs, u_bulk);
        t_bulk_solve = timer.lap();

        const std::vector<LocalPoly2D> solution_polys =
            restrict_op.apply(u_bulk, correction_polys);
        REQUIRE(solution_polys.size() == static_cast<std::size_t>(iface.num_points()));
        t_restrict = timer.lap();
    } else {
        auto result = potentials.evaluate(jumps, f_bulk);
        u_bulk = std::move(result.u_bulk);
    }

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n)
        bulk_err = std::max(bulk_err, std::abs(u_bulk[n] - u_exact[n]));
    const double t_error = timer.lap();

    if (profile) {
        std::printf("    profile N=%d panels=%d iface=%d grid=%d\n",
                    N, iface.num_panels(), iface.num_points(), n_dof);
        std::printf("      grid %.3fs interface %.3fs grid_pair %.3fs rhs %.3fs ops %.3fs jumps %.3fs\n",
                    t_grid, t_interface, t_grid_pair, t_rhs, t_ops, t_jumps);
        std::printf("      spread %.3fs bulk_solve %.3fs restrict %.3fs error %.3fs total %.3fs\n",
                    t_spread, t_bulk_solve, t_restrict, t_error,
                    t_grid + t_interface + t_grid_pair + t_rhs + t_ops + t_jumps
                        + t_spread + t_bulk_solve + t_restrict + t_error);
    }

    return {bulk_err, iface.num_panels(), iface.num_points()};
}

} // namespace

TEST_CASE("Constant-coefficient screened interface problem converges on 3-fold star",
          "[screened][laplace][interface][lobatto][convergence][2d]")
{
    std::vector<int> Ns{32, 64, 128, 256, 512};
    const int max_n = max_n_from_env();
    if (max_n > 0) {
        Ns.erase(std::remove_if(Ns.begin(), Ns.end(),
                                [max_n](int n) { return n > max_n; }),
                 Ns.end());
    }
    REQUIRE_FALSE(Ns.empty());

    std::vector<SolveData> data(Ns.size());
    std::vector<double> rates(Ns.size(), 0.0);

    const std::filesystem::path out_dir = output_dir();
    std::ofstream csv(out_dir / "convergence.csv");
    csv << "N,max_err,order,GMRES\n";

    std::printf("\n  Constant-coefficient screened interface problem on 3-fold star\n");
    std::printf("  Manufactured sine modes vanish on the outer Cartesian box; eta = %.2f\n",
                kEta);
    std::printf("  Panels: P2 quadratic; node_spacing/h = %.2f; panel_length/h = %.2f; output: %s\n",
                kTargetNodeSpacingOverH, kTargetPanelLengthOverH,
                out_dir.string().c_str());
    std::printf("  %6s  %12s  %8s  %6s\n", "N", "max_err", "order", "GMRES");

    for (std::size_t l = 0; l < Ns.size(); ++l) {
        data[l] = solve_and_measure(Ns[l]);
        REQUIRE(std::isfinite(data[l].bulk_err));
        REQUIRE(data[l].num_interface_points == 2 * data[l].num_panels);

        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s  %6d\n",
                        Ns[l], data[l].bulk_err, "-", 0);
            csv << Ns[l] << "," << std::setprecision(16) << data[l].bulk_err
                << ",," << 0 << "\n";
        } else {
            rates[l] = std::log2(data[l - 1].bulk_err / data[l].bulk_err);
            std::printf("  %6d  %12.4e  %8.3f  %6d\n",
                        Ns[l], data[l].bulk_err, rates[l], 0);
            csv << Ns[l] << "," << std::setprecision(16) << data[l].bulk_err
                << "," << rates[l] << "," << 0 << "\n";
        }
    }

    REQUIRE(data.back().bulk_err < data.front().bulk_err);
    REQUIRE(tail_average_order(rates) > 1.5);
    REQUIRE(data.back().bulk_err < 1.0e-2);
}
