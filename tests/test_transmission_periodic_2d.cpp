// ---------------------------------------------------------------------------
// 2D transmission convergence test with bi-periodic outer boundary conditions.
//
// Solves -div(beta grad u) + kappa^2 u = f on a fixed periodic unit square.
// The manufactured interior/exterior solutions are both periodic trigonometric
// fields, while the interface is an off-center P2 quadratic 3-fold star.
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "kfbim/geometry.hpp"
#include "kfbim/grid.hpp"
#include "kfbim/laplace.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kBoxLower = 0.0;
constexpr double kBoxSide = 1.0;
constexpr double kStarCx = 0.48;
constexpr double kStarCy = 0.51;
constexpr double kStarRadius = 0.23;
constexpr double kStarAmplitude = 0.18;
constexpr int    kStarFolds = 3;
constexpr double kTargetNodeSpacingOverH = 1.5;
constexpr double kTargetPanelLengthOverH = 2.0 * kTargetNodeSpacingOverH;
constexpr double kBetaInt = 2.0;
constexpr double kBetaExt = 1.0;
constexpr double kLambdaSq = 1.3;

#ifndef KFBIM_TEST_OUTPUT_DIR
#define KFBIM_TEST_OUTPUT_DIR "output"
#endif

class PeriodicStarCurve2D final : public ICurve2D {
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

struct ConvergenceData {
    double bulk_err = 0.0;
    int    iterations = 0;
};

double u_int(double x, double y)
{
    const double a = 2.0 * kPi * (x + y);
    const double b = 4.0 * kPi * x - 2.0 * kPi * y;
    return 0.4 + 0.25 * std::cos(a) + 0.18 * std::sin(b);
}

Eigen::Vector2d grad_u_int(double x, double y)
{
    const double a = 2.0 * kPi * (x + y);
    const double b = 4.0 * kPi * x - 2.0 * kPi * y;
    return {-0.5 * kPi * std::sin(a) + 0.72 * kPi * std::cos(b),
            -0.5 * kPi * std::sin(a) - 0.36 * kPi * std::cos(b)};
}

double q_int(double x, double y)
{
    const double a = 2.0 * kPi * (x + y);
    const double b = 4.0 * kPi * x - 2.0 * kPi * y;
    return 2.0 * kPi * kPi * std::cos(a)
           + 3.6 * kPi * kPi * std::sin(b)
           + kLambdaSq * u_int(x, y);
}

double u_ext(double x, double y)
{
    const double sx = std::sin(2.0 * kPi * x);
    const double cy = std::cos(2.0 * kPi * y);
    const double c = 4.0 * kPi * x + 2.0 * kPi * y;
    return -0.1 + 0.32 * sx * cy + 0.17 * std::cos(c);
}

Eigen::Vector2d grad_u_ext(double x, double y)
{
    const double sx = std::sin(2.0 * kPi * x);
    const double cx = std::cos(2.0 * kPi * x);
    const double sy = std::sin(2.0 * kPi * y);
    const double cy = std::cos(2.0 * kPi * y);
    const double c = 4.0 * kPi * x + 2.0 * kPi * y;
    return {0.64 * kPi * cx * cy - 0.68 * kPi * std::sin(c),
            -0.64 * kPi * sx * sy - 0.34 * kPi * std::sin(c)};
}

double q_ext(double x, double y)
{
    const double sx = std::sin(2.0 * kPi * x);
    const double cy = std::cos(2.0 * kPi * y);
    const double c = 4.0 * kPi * x + 2.0 * kPi * y;
    return 2.56 * kPi * kPi * sx * cy
           + 3.4 * kPi * kPi * std::cos(c)
           + kLambdaSq * u_ext(x, y);
}

std::filesystem::path output_dir()
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR)
        / "laplace_transmission_periodic_2d";
    std::filesystem::create_directories(dir);
    return dir;
}

void write_grid_csv(const std::filesystem::path& path,
                    const CartesianGrid2D&       grid,
                    const Eigen::VectorXd&       u_bulk,
                    const Eigen::VectorXd&       u_exact,
                    const Eigen::VectorXd&       abs_error,
                    const std::vector<int>&      labels)
{
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int step = 1;

    std::ofstream out(path);
    out << std::setprecision(16);
    out << "i,j,x,y,u_bulk,u_exact,abs_error,label\n";
    for (int j = 0; j < ny; j += step) {
        for (int i = 0; i < nx; i += step) {
            const int n = grid.index(i, j);
            const auto c = grid.coord(i, j);
            out << i << "," << j << ","
                << c[0] << "," << c[1] << ","
                << u_bulk[n] << "," << u_exact[n] << ","
                << abs_error[n] << "," << labels[n] << "\n";
        }
    }
}

ConvergenceData solve_and_measure(int N, const std::filesystem::path& out_dir)
{
    const double h = kBoxSide / static_cast<double>(N);
    CartesianGrid2D grid({kBoxLower, kBoxLower},
                         {h, h},
                         {N, N},
                         DofLayout2D::CellCenter);

    PeriodicStarCurve2D star;
    Interface2D iface =
        CurveResampler2D::discretize(star, h, kTargetPanelLengthOverH);
    GridPair2D grid_pair(grid, iface);

    const int n_dof = grid.num_dofs();
    const int n_iface = iface.num_points();

    Eigen::VectorXd u_jump(n_iface);
    Eigen::VectorXd beta_flux_jump(n_iface);
    LaplaceTransmissionRhsData2D rhs_data;
    rhs_data.reduced_rhs_bulk = Eigen::VectorXd::Zero(n_dof);
    rhs_data.reduced_rhs_int_derivs.resize(n_iface);
    rhs_data.reduced_rhs_ext_derivs.resize(n_iface);

    const auto& points = iface.points();
    const auto& normals = iface.normals();
    for (int q = 0; q < n_iface; ++q) {
        const double x = points(q, 0);
        const double y = points(q, 1);
        const Eigen::Vector2d normal(normals(q, 0), normals(q, 1));
        u_jump[q] = u_int(x, y) - u_ext(x, y);
        beta_flux_jump[q] = kBetaInt * grad_u_int(x, y).dot(normal)
                            - kBetaExt * grad_u_ext(x, y).dot(normal);
        rhs_data.reduced_rhs_int_derivs[q] =
            Eigen::VectorXd::Constant(1, q_int(x, y));
        rhs_data.reduced_rhs_ext_derivs[q] =
            Eigen::VectorXd::Constant(1, q_ext(x, y));
    }

    Eigen::VectorXd u_exact(n_dof);
    Eigen::VectorXd abs_error = Eigen::VectorXd::Zero(n_dof);
    std::vector<int> labels(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        labels[n] = grid_pair.domain_label(n);
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        if (labels[n] > 0) {
            rhs_data.reduced_rhs_bulk[n] = q_int(x, y);
            u_exact[n] = u_int(x, y);
        } else {
            rhs_data.reduced_rhs_bulk[n] = q_ext(x, y);
            u_exact[n] = u_ext(x, y);
        }
    }

    LaplaceTransmissionCoefficients2D coefficients{
        kBetaInt, kBetaExt, kBetaInt * kLambdaSq, kBetaExt * kLambdaSq};
    LaplaceTransmission2D problem(grid,
                                  iface,
                                  LaplaceTransmissionMode2D::CommonRatio,
                                  coefficients,
                                  ZfftBcType::Periodic);

    auto result = problem.solve(u_jump,
                                beta_flux_jump,
                                rhs_data,
                                Eigen::VectorXd(),
                                250,
                                1.0e-8,
                                100);
    REQUIRE(result.converged);

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n) {
        abs_error[n] = std::abs(result.u_bulk[n] - u_exact[n]);
        bulk_err = std::max(bulk_err, abs_error[n]);
    }

    if (N <= 256) {
        char fname[128];
        std::snprintf(fname, sizeof(fname), "periodic_transmission_N%04d.csv", N);
        write_grid_csv(out_dir / fname, grid, result.u_bulk, u_exact, abs_error, labels);
    }

    return {bulk_err, result.iterations};
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

} // namespace

TEST_CASE("Common-ratio transmission 2D: bi-periodic box convergence",
          "[transmission][laplace][interface][periodic][lobatto][convergence][2d]")
{
    const std::vector<int> levels = {32, 64, 128, 256, 512};
    const std::filesystem::path out_dir = output_dir();

    std::ofstream csv(out_dir / "convergence.csv");
    csv << "N,max_err,order,GMRES\n";

    std::vector<ConvergenceData> data(levels.size());
    std::vector<double> rates(levels.size(), 0.0);

    std::printf("\n  Common-ratio transmission 2D with bi-periodic box BC\n");
    std::printf("  beta_int=%.3f beta_ext=%.3f, lambda^2=%.3f\n",
                kBetaInt, kBetaExt, kLambdaSq);
    std::printf("  Grid: cell-centered periodic unit square; interface: off-center 3-fold star\n");
    std::printf("  Target P2-node spacing / h = %.2f (panel_length/h = %.2f)\n",
                kTargetNodeSpacingOverH, kTargetPanelLengthOverH);
    std::printf("  Output: %s\n", out_dir.string().c_str());
    std::printf("  %6s  %12s  %8s  %6s\n", "N", "max_err", "order", "GMRES");

    for (std::size_t l = 0; l < levels.size(); ++l) {
        data[l] = solve_and_measure(levels[l], out_dir);
        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s  %6d\n",
                        levels[l], data[l].bulk_err, "-", data[l].iterations);
            csv << levels[l] << "," << std::setprecision(16) << data[l].bulk_err
                << ",," << data[l].iterations << "\n";
        } else {
            rates[l] = std::log2(data[l - 1].bulk_err / data[l].bulk_err);
            std::printf("  %6d  %12.4e  %8.3f  %6d\n",
                        levels[l], data[l].bulk_err, rates[l], data[l].iterations);
            csv << levels[l] << "," << std::setprecision(16) << data[l].bulk_err
                << "," << rates[l] << "," << data[l].iterations << "\n";
        }

        REQUIRE(std::isfinite(data[l].bulk_err));
    }

    REQUIRE(data.back().bulk_err < data.front().bulk_err);
    REQUIRE(tail_average_order(rates) > 1.2);
    REQUIRE(data.back().bulk_err < 5.0e-3);
}
