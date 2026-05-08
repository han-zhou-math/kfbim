#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include "kfbim/geometry.hpp"
#include "kfbim/grid.hpp"
#include "kfbim/laplace.hpp"

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

enum class BvpCase {
    InteriorDirichlet,
    ExteriorDirichlet,
    InteriorNeumann,
    ExteriorNeumann
};

struct SolveData {
    double bulk_err = 0.0;
    double wall_time = 0.0;
    int    iterations = 0;
    int    num_panels = 0;
    int    num_interface_points = 0;
    int    density_size = 0;
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

double exact_u(double x, double y)
{
    const double a = 0.8 * x + 0.3 * y;
    const double b = 0.2 * x - 0.7 * y;
    const double c = 0.5 * x;
    const double d = 0.4 * y;
    return std::sin(a) + 0.4 * std::cos(b)
           + 0.2 * std::sin(c) * std::cos(d);
}

Eigen::Vector2d exact_grad(double x, double y)
{
    const double a = 0.8 * x + 0.3 * y;
    const double b = 0.2 * x - 0.7 * y;
    const double c = 0.5 * x;
    const double d = 0.4 * y;
    return {0.8 * std::cos(a) - 0.08 * std::sin(b)
            + 0.1 * std::cos(c) * std::cos(d),
            0.3 * std::cos(a) + 0.28 * std::sin(b)
            - 0.08 * std::sin(c) * std::sin(d)};
}

double exact_f(double x, double y)
{
    const double a = 0.8 * x + 0.3 * y;
    const double b = 0.2 * x - 0.7 * y;
    const double c = 0.5 * x;
    const double d = 0.4 * y;
    const double lap_u = -0.73 * std::sin(a) - 0.212 * std::cos(b)
                         - 0.082 * std::sin(c) * std::cos(d);
    return -lap_u + kEta * exact_u(x, y);
}

bool is_dirichlet(BvpCase bvp)
{
    return bvp == BvpCase::InteriorDirichlet
        || bvp == BvpCase::ExteriorDirichlet;
}

bool is_interior(BvpCase bvp)
{
    return bvp == BvpCase::InteriorDirichlet
        || bvp == BvpCase::InteriorNeumann;
}

const char* bvp_name(BvpCase bvp)
{
    switch (bvp) {
    case BvpCase::InteriorDirichlet: return "interior Dirichlet";
    case BvpCase::ExteriorDirichlet: return "exterior Dirichlet";
    case BvpCase::InteriorNeumann: return "interior Neumann";
    case BvpCase::ExteriorNeumann: return "exterior Neumann";
    }
    return "unknown";
}

const char* bvp_file_stem(BvpCase bvp)
{
    switch (bvp) {
    case BvpCase::InteriorDirichlet: return "interior_dirichlet";
    case BvpCase::ExteriorDirichlet: return "exterior_dirichlet";
    case BvpCase::InteriorNeumann: return "interior_neumann";
    case BvpCase::ExteriorNeumann: return "exterior_neumann";
    }
    return "unknown";
}

LaplaceBvpType2D bvp_type(BvpCase bvp)
{
    switch (bvp) {
    case BvpCase::InteriorDirichlet:
        return LaplaceBvpType2D::InteriorDirichlet;
    case BvpCase::ExteriorDirichlet:
        return LaplaceBvpType2D::ExteriorDirichlet;
    case BvpCase::InteriorNeumann:
        return LaplaceBvpType2D::InteriorNeumann;
    case BvpCase::ExteriorNeumann:
        return LaplaceBvpType2D::ExteriorNeumann;
    }
    return LaplaceBvpType2D::InteriorDirichlet;
}

double final_error_threshold(BvpCase bvp)
{
    switch (bvp) {
    case BvpCase::InteriorDirichlet: return 5.0e-3;
    case BvpCase::ExteriorDirichlet: return 5.0e-3;
    case BvpCase::InteriorNeumann: return 2.0e-2;
    case BvpCase::ExteriorNeumann: return 1.0e-2;
    }
    return 1.0e-2;
}

bool is_outer_boundary_node(int idx, int nx, int ny)
{
    const int i = idx % nx;
    const int j = idx / nx;
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

std::filesystem::path output_dir()
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR) / "laplace_bvp_star3";
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

SolveData solve_and_measure(BvpCase bvp, int N)
{
    const auto wall_start = std::chrono::steady_clock::now();
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
    const int n_panels = iface.num_panels();
    const int n_iface = iface.num_points();

    Eigen::VectorXd g(n_iface);
    std::vector<Eigen::VectorXd> rhs_derivs(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        if (is_dirichlet(bvp)) {
            g[q] = exact_u(x, y);
        } else {
            const Eigen::Vector2d normal(iface.normals()(q, 0),
                                         iface.normals()(q, 1));
            g[q] = exact_grad(x, y).dot(normal);
        }
        rhs_derivs[q] = Eigen::VectorXd::Constant(1, exact_f(x, y));
    }

    GridPair2D gp(grid, iface);
    Eigen::VectorXd f_bulk = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd outer_bc = Eigen::VectorXd::Zero(n_dof);

    const int physical_label = is_interior(bvp) ? 1 : 0;
    for (int n = 0; n < n_dof; ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        u_exact[n] = exact_u(x, y);
        outer_bc[n] = exact_u(x, y);

        if (gp.domain_label(n) == physical_label
            && !is_outer_boundary_node(n, nx, ny)) {
            f_bulk[n] = exact_f(x, y);
        }
    }

    LaplaceBvpOptions2D options;
    options.panel_method = LaplaceBvpPanelMethod2D::QuadraticPanelCenter;
    options.eta = kEta;
    if (!is_interior(bvp))
        options.outer_dirichlet_values = outer_bc;

    LaplaceBvp2D problem(grid, iface, bvp_type(bvp), options);
    auto result = problem.solve(g, f_bulk, rhs_derivs, 800, 1.0e-8, 200);
    REQUIRE(result.converged);

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n) {
        if (gp.domain_label(n) != physical_label)
            continue;
        if (!is_interior(bvp) && is_outer_boundary_node(n, nx, ny))
            continue;
        bulk_err = std::max(bulk_err, std::abs(result.u_bulk[n] - u_exact[n]));
    }

    if (!is_interior(bvp)) {
        double boundary_err = 0.0;
        for (int n = 0; n < n_dof; ++n)
            if (is_outer_boundary_node(n, nx, ny))
                boundary_err = std::max(boundary_err,
                                        std::abs(result.u_bulk[n] - outer_bc[n]));
        REQUIRE(boundary_err < 1.0e-12);
    }

    const double wall_time =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();

    return {bulk_err,
            wall_time,
            result.iterations,
            n_panels,
            n_iface,
            static_cast<int>(result.density.size())};
}

void check_convergence(BvpCase bvp)
{
    const std::vector<int> Ns{32, 64, 128, 256, 512};
    std::vector<SolveData> data(Ns.size());
    std::vector<double> rates(Ns.size(), 0.0);

    const std::filesystem::path out_dir = output_dir();
    const std::filesystem::path csv_path =
        out_dir / (std::string(bvp_file_stem(bvp)) + "_convergence.csv");
    std::ofstream csv(csv_path);
    csv << "N,panels,iface_pts,density,max_err,order,wall_s,GMRES\n";

    std::printf("\n  Screened %s BVP on 3-fold star: -Delta u + %.2f u = f\n",
                bvp_name(bvp), kEta);
    std::printf("  Manufactured: sin(0.8x+0.3y) + 0.4cos(0.2x-0.7y) + 0.2sin(0.5x)cos(0.4y)\n");
    std::printf("  Panels: P2 quadratic; node_spacing/h = %.2f; panel_length/h = %.2f; output: %s\n",
                kTargetNodeSpacingOverH, kTargetPanelLengthOverH,
                csv_path.string().c_str());
    std::printf("  %6s  %8s  %10s  %11s  %12s  %8s  %8s  %6s\n",
                "N", "panels", "iface_pts", "density", "max_err", "order",
                "wall_s", "GMRES");

    for (std::size_t l = 0; l < Ns.size(); ++l) {
        data[l] = solve_and_measure(bvp, Ns[l]);
        REQUIRE(std::isfinite(data[l].bulk_err));
        REQUIRE(data[l].num_interface_points == 2 * data[l].num_panels);
        REQUIRE(data[l].density_size == data[l].num_interface_points);

        if (l == 0) {
            std::printf("  %6d  %8d  %10d  %11d  %12.4e  %8s  %8.3f  %6d\n",
                        Ns[l], data[l].num_panels, data[l].num_interface_points,
                        data[l].density_size, data[l].bulk_err, "-",
                        data[l].wall_time, data[l].iterations);
            csv << Ns[l] << "," << data[l].num_panels << ","
                << data[l].num_interface_points << "," << data[l].density_size
                << "," << std::setprecision(16) << data[l].bulk_err
                << ",," << data[l].wall_time << "," << data[l].iterations << "\n";
        } else {
            rates[l] = std::log2(data[l - 1].bulk_err / data[l].bulk_err);
            std::printf("  %6d  %8d  %10d  %11d  %12.4e  %8.3f  %8.3f  %6d\n",
                        Ns[l], data[l].num_panels, data[l].num_interface_points,
                        data[l].density_size, data[l].bulk_err, rates[l],
                        data[l].wall_time, data[l].iterations);
            csv << Ns[l] << "," << data[l].num_panels << ","
                << data[l].num_interface_points << "," << data[l].density_size
                << "," << std::setprecision(16) << data[l].bulk_err
                << "," << rates[l] << "," << data[l].wall_time
                << "," << data[l].iterations << "\n";
        }
    }

    REQUIRE(data.back().bulk_err < data.front().bulk_err);
    REQUIRE(tail_average_order(rates) > 1.5);
    REQUIRE(data.back().bulk_err < final_error_threshold(bvp));
}

} // namespace

TEST_CASE("LaplaceBvp2D modes converge on 3-fold star",
          "[screened][laplace][bvp][lobatto][convergence][2d]")
{
    check_convergence(BvpCase::InteriorDirichlet);
    check_convergence(BvpCase::ExteriorDirichlet);
    check_convergence(BvpCase::InteriorNeumann);
    check_convergence(BvpCase::ExteriorNeumann);
}
