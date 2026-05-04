// ---------------------------------------------------------------------------
// Full-pipeline tests for piecewise-constant coefficient interface PDEs.
//
// Solves: -div(beta grad u) + kappa^2 u = f in Omega_int/Omega_ext.
// The solver API takes q=f/beta, so each phase uses
//   -Delta u + lambda^2 u = q, lambda^2 = kappa^2 / beta.
//
// The manufactured solutions prescribe [u] and [beta du/dn] on a 3-fold star,
// with nonzero Dirichlet data on the outer Cartesian box.
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/geometry/grid_pair_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/operators/laplace_transmission_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
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

struct TransmissionCaseConfig {
    const char*                         label;
    const char*                         output_leaf;
    const char*                         file_prefix;
    LaplaceTransmissionMode2D           mode;
    LaplaceTransmissionCoefficients2D   coefficients;
    int                                 max_iter;
    double                              final_error_threshold;
};

struct ConvergenceData {
    double bulk_err;
    int    iterations;
};

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

bool is_outer_boundary_node(int idx, int nx, int ny)
{
    const int i = idx % nx;
    const int j = idx / nx;
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

double lambda_sq_int(const TransmissionCaseConfig& config)
{
    return config.coefficients.kappa_sq_int / config.coefficients.beta_int;
}

double lambda_sq_ext(const TransmissionCaseConfig& config)
{
    return config.coefficients.kappa_sq_ext / config.coefficients.beta_ext;
}

double u_int(double x, double y)
{
    return std::exp(0.35 * x + 0.15 * y)
           + 0.2 * std::cos(1.7 * x - 0.3 * y);
}

Eigen::Vector2d grad_u_int(double x, double y)
{
    const double e = std::exp(0.35 * x + 0.15 * y);
    const double p = 1.7 * x - 0.3 * y;
    return {0.35 * e - 0.34 * std::sin(p),
            0.15 * e + 0.06 * std::sin(p)};
}

double q_int(double x, double y, double lambda_sq)
{
    const double e = std::exp(0.35 * x + 0.15 * y);
    const double p = 1.7 * x - 0.3 * y;
    return (lambda_sq - 0.145) * e
           + 0.2 * (lambda_sq + 2.98) * std::cos(p);
}

double u_ext(double x, double y)
{
    return 0.6 * std::sin(0.8 * x + 0.4 * y)
           + 0.25 * x - 0.15 * y + 0.1 * x * y + 0.5;
}

Eigen::Vector2d grad_u_ext(double x, double y)
{
    const double c = std::cos(0.8 * x + 0.4 * y);
    return {0.48 * c + 0.25 + 0.1 * y,
            0.24 * c - 0.15 + 0.1 * x};
}

double q_ext(double x, double y, double lambda_sq)
{
    return 0.48 * std::sin(0.8 * x + 0.4 * y)
           + lambda_sq * u_ext(x, y);
}

std::filesystem::path output_dir(const char* leaf)
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR) / leaf;
    std::filesystem::create_directories(dir);
    return dir;
}

std::vector<int> sampled_axis_indices(int count)
{
    constexpr int kMaxIntervalsForPlot = 256;
    const int step = std::max(1, (count - 1 + kMaxIntervalsForPlot - 1)
                                 / kMaxIntervalsForPlot);

    std::vector<int> indices;
    for (int i = 0; i < count; i += step)
        indices.push_back(i);
    if (indices.empty() || indices.back() != count - 1)
        indices.push_back(count - 1);
    return indices;
}

void write_sampled_grid_csv(const std::filesystem::path& path,
                            const CartesianGrid2D&       grid,
                            const Eigen::VectorXd&       u_bulk,
                            const Eigen::VectorXd&       u_exact,
                            const Eigen::VectorXd&       abs_error,
                            const std::vector<int>&      labels)
{
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];

    const std::vector<int> sample_i = sampled_axis_indices(nx);
    const std::vector<int> sample_j = sampled_axis_indices(ny);

    std::ofstream out(path);
    out << std::setprecision(16);
    out << "i,j,x,y,u_bulk,u_exact,abs_error,label\n";
    for (int j : sample_j) {
        for (int i : sample_i) {
            const int n = grid.index(i, j);
            const auto c = grid.coord(i, j);
            out << i << "," << j << ","
                << c[0] << "," << c[1] << ","
                << u_bulk[n] << "," << u_exact[n] << ","
                << abs_error[n] << "," << labels[n] << "\n";
        }
    }
}

void write_interface_points_csv(const std::filesystem::path& path,
                                const Interface2D&           iface)
{
    std::ofstream out(path);
    out << std::setprecision(16);
    out << "q,x,y\n";
    for (int q = 0; q < iface.num_points(); ++q) {
        out << q << ","
            << iface.points()(q, 0) << ","
            << iface.points()(q, 1) << "\n";
    }
}

ConvergenceData solve_and_measure(int N,
                                  const std::filesystem::path& out_dir,
                                  const TransmissionCaseConfig& config)
{
    Star3Curve2D star;
    const Box2D box = make_outer_box(star);
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid2D grid(box.lower, {h, h}, {N, N}, DofLayout2D::Node);
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int n_dof = nx * ny;

    Interface2D iface =
        CurveResampler2D::discretize(star, h, kTargetPanelLengthOverH);
    const int n_iface = iface.num_points();

    const double lambda_int = lambda_sq_int(config);
    const double lambda_ext = lambda_sq_ext(config);

    Eigen::VectorXd mu(n_iface);
    Eigen::VectorXd sigma(n_iface);
    LaplaceTransmissionRhsData2D rhs_data;
    rhs_data.reduced_rhs_int_derivs.resize(n_iface);
    rhs_data.reduced_rhs_ext_derivs.resize(n_iface);

    const auto& points = iface.points();
    const auto& normals = iface.normals();
    for (int q = 0; q < n_iface; ++q) {
        const double x = points(q, 0);
        const double y = points(q, 1);
        const Eigen::Vector2d normal(normals(q, 0), normals(q, 1));
        mu[q] = u_int(x, y) - u_ext(x, y);
        sigma[q] = config.coefficients.beta_int * grad_u_int(x, y).dot(normal)
                   - config.coefficients.beta_ext * grad_u_ext(x, y).dot(normal);
        rhs_data.reduced_rhs_int_derivs[q] =
            Eigen::VectorXd::Constant(1, q_int(x, y, lambda_int));
        rhs_data.reduced_rhs_ext_derivs[q] =
            Eigen::VectorXd::Constant(1, q_ext(x, y, lambda_ext));
    }

    GridPair2D gp(grid, iface);
    std::vector<int> labels(n_dof);
    rhs_data.reduced_rhs_bulk = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd abs_error = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd outer_bc = Eigen::VectorXd::Zero(n_dof);

    for (int n = 0; n < n_dof; ++n) {
        labels[n] = gp.domain_label(n);
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];

        if (is_outer_boundary_node(n, nx, ny)) {
            outer_bc[n] = u_ext(x, y);
            u_exact[n] = outer_bc[n];
            continue;
        }

        if (labels[n] > 0) {
            rhs_data.reduced_rhs_bulk[n] = q_int(x, y, lambda_int);
            u_exact[n] = u_int(x, y);
        } else {
            rhs_data.reduced_rhs_bulk[n] = q_ext(x, y, lambda_ext);
            u_exact[n] = u_ext(x, y);
        }
    }

    LaplaceTransmission2D problem(grid, iface, config.mode, config.coefficients);

    auto result = problem.solve(mu,
                                sigma,
                                rhs_data,
                                outer_bc,
                                config.max_iter,
                                1.0e-8,
                                100);
    REQUIRE(result.converged);

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n) {
        abs_error[n] = std::abs(result.u_bulk[n] - u_exact[n]);
        bulk_err = std::max(bulk_err, abs_error[n]);
    }

    char fname[128];
    std::snprintf(fname, sizeof(fname), "%s_N%04d.csv", config.file_prefix, N);
    write_sampled_grid_csv(out_dir / fname, grid, result.u_bulk, u_exact, abs_error, labels);

    char iface_fname[128];
    std::snprintf(iface_fname, sizeof(iface_fname),
                  "%s_interface_N%04d.csv", config.file_prefix, N);
    write_interface_points_csv(out_dir / iface_fname, iface);

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

void run_convergence_case(const TransmissionCaseConfig& config,
                          const std::vector<int>&       levels)
{
    const std::filesystem::path out_dir = output_dir(config.output_leaf);
    std::ofstream csv(out_dir / "convergence.csv");
    csv << "N,max_err,order,GMRES\n";

    std::vector<ConvergenceData> data(levels.size());
    std::vector<double> rates(levels.size(), 0.0);

    std::printf("\n  %s\n", config.label);
    std::printf("  beta_int=%.3f beta_ext=%.3f, lambda_int^2=%.3f lambda_ext^2=%.3f\n",
                config.coefficients.beta_int,
                config.coefficients.beta_ext,
                lambda_sq_int(config),
                lambda_sq_ext(config));
    std::printf("  Interface: 3-fold star; panels=Chebyshev-Lobatto; nonzero box Dirichlet BC\n");
    std::printf("  Target Chebyshev-node spacing / h = %.2f (panel_length/h = %.2f)\n",
                kTargetNodeSpacingOverH, kTargetPanelLengthOverH);
    std::printf("  Output: %s\n", out_dir.string().c_str());
    std::printf("  %6s  %12s  %8s  %6s\n", "N", "max_err", "order", "GMRES");

    for (std::size_t l = 0; l < levels.size(); ++l) {
        data[l] = solve_and_measure(levels[l], out_dir, config);

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
    REQUIRE(tail_average_order(rates) > 1.5);
    REQUIRE(data.back().bulk_err < config.final_error_threshold);
}

TransmissionCaseConfig common_ratio_case()
{
    constexpr double beta_int = 2.0;
    constexpr double beta_ext = 1.0;
    constexpr double lambda_sq = 1.1;
    return {"Common-ratio transmission: -div(beta grad u)+kappa^2 u=f",
            "laplace_transmission_common_ratio_2d",
            "transmission_common_ratio",
            LaplaceTransmissionMode2D::CommonRatio,
            {beta_int, beta_ext, beta_int * lambda_sq, beta_ext * lambda_sq},
            250,
            5.0e-2};
}

TransmissionCaseConfig different_ratios_case()
{
    return {"Different-ratio transmission: generic two-density KFBI operator",
            "laplace_transmission_different_ratios_2d",
            "transmission_different_ratios",
            LaplaceTransmissionMode2D::DifferentRatios,
            {10.0, 1.0, 11.0, 0.7},
            400,
            5.0e-2};
}

} // namespace

TEST_CASE("Common-ratio transmission 2D: Chebyshev-Lobatto convergence on 3-fold star",
          "[transmission][laplace][interface][lobatto][convergence][2d]")
{
    run_convergence_case(common_ratio_case(), {32, 64, 128, 256, 512});
}

TEST_CASE("Different-ratio transmission 2D: Chebyshev-Lobatto convergence on 3-fold star",
          "[transmission][laplace][interface][different-ratio][lobatto][convergence][2d]")
{
    run_convergence_case(different_ratios_case(), {32, 64, 128, 256, 512});
}
