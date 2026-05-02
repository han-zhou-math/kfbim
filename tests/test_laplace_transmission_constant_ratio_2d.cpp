// ---------------------------------------------------------------------------
// Full-pipeline test: constant-ratio discontinuous-coefficient interface PDE
//
// Solves: -div(beta grad u) + kappa^2 u = f in Omega_int/Omega_ext
// with piecewise constant beta and kappa^2 / beta = lambda^2 on both sides.
//
// After division by beta, both subdomains use the same screened operator:
//   -Delta u + lambda^2 u = q, q=f/beta.
//
// The manufactured solution uses different smooth branches across a 3-fold
// 5-fold star.  This gives prescribed jumps [u] and [beta du/dn], and a nonzero
// Dirichlet condition on the outer Cartesian box.
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include "core/geometry/curve_2d.hpp"
#include "core/geometry/curve_resampler_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/grid/cartesian_grid_2d.hpp"
#include "core/problems/laplace_transmission_constant_ratio_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kBetaInt = 2.0;
constexpr double kBetaExt = 1.0;
constexpr double kLambdaSq = 1.1;
constexpr double kTargetPanelLengthOverH = 4.0;

constexpr double kStarCx = 0.07;
constexpr double kStarCy = -0.04;
constexpr double kStarRadius = 0.65;
constexpr double kStarAmplitude = 0.20;
constexpr int    kStarFolds = 5;
constexpr double kBoxMargin = 0.35;

#ifndef KFBIM_TEST_OUTPUT_DIR
#define KFBIM_TEST_OUTPUT_DIR "output"
#endif

#ifndef KFBIM_PYTHON_EXECUTABLE
#define KFBIM_PYTHON_EXECUTABLE "python3"
#endif

#ifndef KFBIM_TRANSMISSION_VIS_SCRIPT
#define KFBIM_TRANSMISSION_VIS_SCRIPT "scripts/visualize_transmission_constant_ratio_2d.py"
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

double q_int(double x, double y)
{
    const double e = std::exp(0.35 * x + 0.15 * y);
    const double p = 1.7 * x - 0.3 * y;
    return (kLambdaSq - 0.145) * e
           + 0.2 * (kLambdaSq + 2.98) * std::cos(p);
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

double q_ext(double x, double y)
{
    return 0.48 * std::sin(0.8 * x + 0.4 * y) + kLambdaSq * u_ext(x, y);
}

std::filesystem::path output_dir()
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR)
        / "laplace_transmission_constant_ratio_2d";
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

int run_python_visualization(const std::filesystem::path& out_dir)
{
    const std::string command =
        std::string("\"") + KFBIM_PYTHON_EXECUTABLE + "\" "
        + "\"" + KFBIM_TRANSMISSION_VIS_SCRIPT + "\" "
        + "\"" + out_dir.string() + "\"";
    return std::system(command.c_str());
}

struct ConvergenceData {
    double bulk_err;
    int    iterations;
};

ConvergenceData solve_and_measure(int N, const std::filesystem::path& out_dir)
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

    Eigen::VectorXd mu(n_iface);
    Eigen::VectorXd sigma(n_iface);
    std::vector<Eigen::VectorXd> rhs_derivs(n_iface);
    const auto& points = iface.points();
    const auto& normals = iface.normals();
    for (int q = 0; q < n_iface; ++q) {
        const double x = points(q, 0);
        const double y = points(q, 1);
        const Eigen::Vector2d normal(normals(q, 0), normals(q, 1));
        mu[q] = u_int(x, y) - u_ext(x, y);
        sigma[q] = kBetaInt * grad_u_int(x, y).dot(normal)
                   - kBetaExt * grad_u_ext(x, y).dot(normal);
        rhs_derivs[q] = Eigen::VectorXd::Constant(1, q_int(x, y) - q_ext(x, y));
    }

    GridPair2D gp(grid, iface);
    std::vector<int> labels(n_dof);
    Eigen::VectorXd q_bulk = Eigen::VectorXd::Zero(n_dof);
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

        if (labels[n] == 1) {
            q_bulk[n] = q_int(x, y);
            u_exact[n] = u_int(x, y);
        } else {
            q_bulk[n] = q_ext(x, y);
            u_exact[n] = u_ext(x, y);
        }
    }

    LaplaceTransmissionConstantRatio2D problem(
        grid, iface, kBetaInt, kBetaExt, kLambdaSq);

    auto result = problem.solve(mu, sigma, q_bulk, rhs_derivs, outer_bc,
                                250, 1.0e-8, 100);
    REQUIRE(result.converged);

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n) {
        abs_error[n] = std::abs(result.u_bulk[n] - u_exact[n]);
        bulk_err = std::max(bulk_err, abs_error[n]);
    }

    char fname[128];
    std::snprintf(fname, sizeof(fname), "transmission_constant_ratio_N%04d.csv", N);
    write_sampled_grid_csv(out_dir / fname, grid, result.u_bulk, u_exact, abs_error, labels);

    char iface_fname[128];
    std::snprintf(iface_fname, sizeof(iface_fname),
                  "transmission_constant_ratio_interface_N%04d.csv", N);
    write_interface_points_csv(out_dir / iface_fname, iface);

    return {bulk_err, result.iterations};
}

} // namespace

TEST_CASE("Constant-ratio transmission 2D: Chebyshev-Lobatto convergence on 5-fold star",
          "[transmission][laplace][interface][lobatto][convergence][2d]")
{
    const int Ns[] = {32, 64, 128, 256, 512, 1024};
    constexpr int n_levels = static_cast<int>(sizeof(Ns) / sizeof(Ns[0]));

    const std::filesystem::path out_dir = output_dir();
    std::ofstream csv(out_dir / "convergence.csv");
    csv << "N,max_err,rate,iters\n";

    ConvergenceData data[n_levels];

    std::printf("\n  Constant-ratio transmission: -div(beta grad u)+kappa^2 u=f\n");
    std::printf("  beta_int=%.3f beta_ext=%.3f, kappa^2/beta=lambda^2=%.3f\n",
                kBetaInt, kBetaExt, kLambdaSq);
    std::printf("  Interface: 5-fold star; panels=Chebyshev-Lobatto; nonzero box Dirichlet BC\n");
    std::printf("  Target Chebyshev-node spacing / h ≈ %.2f\n",
                0.5 * kTargetPanelLengthOverH);
    std::printf("  Output: %s\n", out_dir.string().c_str());
    std::printf("  %6s  %12s  %8s  %6s\n", "N", "max_err", "rate", "iters");

    for (int l = 0; l < n_levels; ++l) {
        data[l] = solve_and_measure(Ns[l], out_dir);

        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s  %6d\n",
                        Ns[l], data[l].bulk_err, "-", data[l].iterations);
            csv << Ns[l] << "," << std::setprecision(16) << data[l].bulk_err
                << ",," << data[l].iterations << "\n";
        } else {
            const double rate = std::log2(data[l - 1].bulk_err / data[l].bulk_err);
            std::printf("  %6d  %12.4e  %8.3f  %6d\n",
                        Ns[l], data[l].bulk_err, rate, data[l].iterations);
            csv << Ns[l] << "," << std::setprecision(16) << data[l].bulk_err
                << "," << rate << "," << data[l].iterations << "\n";
        }

        REQUIRE(std::isfinite(data[l].bulk_err));
    }

    REQUIRE(data[n_levels - 1].bulk_err < data[0].bulk_err);
    REQUIRE(data[n_levels - 1].bulk_err < 5.0e-2);

    csv.close();
    const int vis_rc = run_python_visualization(out_dir);
    REQUIRE(vis_rc == 0);
}
