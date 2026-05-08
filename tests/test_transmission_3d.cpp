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
#include <string>
#include <vector>

#include "p2_sphere_fixture_3d.hpp"
#include "kfbim/geometry.hpp"
#include "kfbim/grid.hpp"
#include "kfbim/laplace.hpp"

using namespace kfbim;
using namespace kfbim_test_3d;

namespace {

#ifndef KFBIM_TEST_OUTPUT_DIR
#define KFBIM_TEST_OUTPUT_DIR "output"
#endif

struct TransmissionCaseConfig {
    const char*                       label;
    const char*                       output_leaf;
    LaplaceTransmissionMode3D         mode;
    LaplaceTransmissionCoefficients3D coefficients;
    int                               max_iter;
    int                               default_max_level;
    double                            final_error_threshold;
};

struct ConvergenceData {
    double bulk_err = 0.0;
    double wall_time = 0.0;
    int    iterations = 0;
    int    num_panels = 0;
    int    num_interface_points = 0;
    int    phi_size = 0;
    int    psi_size = 0;
};

double lambda_sq_int(const TransmissionCaseConfig& config)
{
    return config.coefficients.kappa_sq_int / config.coefficients.beta_int;
}

double lambda_sq_ext(const TransmissionCaseConfig& config)
{
    return config.coefficients.kappa_sq_ext / config.coefficients.beta_ext;
}

double u_int(double x, double y, double z)
{
    const double e = std::exp(0.25 * x + 0.15 * y - 0.10 * z);
    const double p = 1.1 * x - 0.3 * y + 0.4 * z;
    return e + 0.2 * std::cos(p);
}

Eigen::Vector3d grad_u_int(double x, double y, double z)
{
    const double e = std::exp(0.25 * x + 0.15 * y - 0.10 * z);
    const double p = 1.1 * x - 0.3 * y + 0.4 * z;
    return {0.25 * e - 0.22 * std::sin(p),
            0.15 * e + 0.06 * std::sin(p),
           -0.10 * e - 0.08 * std::sin(p)};
}

double q_int(double x, double y, double z, double lambda_sq)
{
    const double e = std::exp(0.25 * x + 0.15 * y - 0.10 * z);
    const double p = 1.1 * x - 0.3 * y + 0.4 * z;
    return (lambda_sq - 0.095) * e
           + 0.2 * (lambda_sq + 1.46) * std::cos(p);
}

double u_ext(double x, double y, double z)
{
    const double s = 0.8 * x + 0.4 * y - 0.2 * z;
    return 0.5 * std::sin(s)
           + 0.20 * x - 0.15 * y + 0.10 * z
           + 0.08 * x * y - 0.06 * x * z + 0.4;
}

Eigen::Vector3d grad_u_ext(double x, double y, double z)
{
    const double c = std::cos(0.8 * x + 0.4 * y - 0.2 * z);
    return {0.4 * c + 0.20 + 0.08 * y - 0.06 * z,
            0.2 * c - 0.15 + 0.08 * x,
           -0.1 * c + 0.10 - 0.06 * x};
}

double q_ext(double x, double y, double z, double lambda_sq)
{
    const double s = 0.8 * x + 0.4 * y - 0.2 * z;
    return 0.42 * std::sin(s) + lambda_sq * u_ext(x, y, z);
}

std::filesystem::path output_dir(const char* leaf)
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR) / leaf;
    std::filesystem::create_directories(dir);
    return dir;
}

std::vector<int> levels_for_config(const TransmissionCaseConfig& config)
{
    std::vector<int> levels{8, 16, 32};
    if (config.default_max_level >= 64)
        levels.push_back(64);
    if (std::getenv("KFBIM_HIGH_RES_3D") != nullptr) {
        if (levels.back() < 64)
            levels.push_back(64);
        levels.push_back(128);
    }
    return levels;
}

double tail_average_order(const std::vector<double>& rates, int count = 2)
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

ConvergenceData solve_and_measure(int N,
                                  const TransmissionCaseConfig& config)
{
    const auto wall_start = std::chrono::steady_clock::now();
    const Box3D box = standard_box();
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid3D grid(box.lower, {h, h, h}, {N, N, N}, DofLayout3D::Node);
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int n_dof = grid.num_dofs();

    Interface3D iface = make_p2_sphere(surface_subdivision_for_grid(N));
    const int n_iface = iface.num_points();

    LaplaceTransmission3D problem(grid, iface, config.mode, config.coefficients);
    const GridPair3D& gp = problem.grid_pair();

    const double lambda_int = lambda_sq_int(config);
    const double lambda_ext = lambda_sq_ext(config);

    Eigen::VectorXd mu(n_iface);
    Eigen::VectorXd sigma(n_iface);
    LaplaceTransmissionRhsData3D rhs_data;
    rhs_data.reduced_rhs_int_derivs.resize(n_iface);
    rhs_data.reduced_rhs_ext_derivs.resize(n_iface);

    const auto& points = iface.points();
    const auto& normals = iface.normals();
    for (int q = 0; q < n_iface; ++q) {
        const double x = points(q, 0);
        const double y = points(q, 1);
        const double z = points(q, 2);
        const Eigen::Vector3d normal(normals(q, 0),
                                     normals(q, 1),
                                     normals(q, 2));
        mu[q] = u_int(x, y, z) - u_ext(x, y, z);
        sigma[q] = config.coefficients.beta_int * grad_u_int(x, y, z).dot(normal)
                   - config.coefficients.beta_ext * grad_u_ext(x, y, z).dot(normal);
        rhs_data.reduced_rhs_int_derivs[q] =
            Eigen::VectorXd::Constant(1, q_int(x, y, z, lambda_int));
        rhs_data.reduced_rhs_ext_derivs[q] =
            Eigen::VectorXd::Constant(1, q_ext(x, y, z, lambda_ext));
    }

    rhs_data.reduced_rhs_bulk = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd outer_bc = Eigen::VectorXd::Zero(n_dof);

    for (int n = 0; n < n_dof; ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        const double z = c[2];

        if (is_outer_boundary_node(n, nx, ny, nz)) {
            outer_bc[n] = u_ext(x, y, z);
            u_exact[n] = outer_bc[n];
            continue;
        }

        if (gp.domain_label(n) > 0) {
            rhs_data.reduced_rhs_bulk[n] = q_int(x, y, z, lambda_int);
            u_exact[n] = u_int(x, y, z);
        } else {
            rhs_data.reduced_rhs_bulk[n] = q_ext(x, y, z, lambda_ext);
            u_exact[n] = u_ext(x, y, z);
        }
    }

    auto result = problem.solve(mu,
                                sigma,
                                rhs_data,
                                outer_bc,
                                config.max_iter,
                                1.0e-8,
                                120);
    REQUIRE(result.converged);

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n)
        bulk_err = std::max(bulk_err, std::abs(result.u_bulk[n] - u_exact[n]));

    const double wall_time =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();

    return {bulk_err,
            wall_time,
            result.iterations,
            iface.num_panels(),
            n_iface,
            static_cast<int>(result.phi_density.size()),
            static_cast<int>(result.psi_density.size())};
}

void run_convergence_case(const TransmissionCaseConfig& config)
{
    const std::vector<int> levels = levels_for_config(config);
    const std::filesystem::path out_dir = output_dir(config.output_leaf);
    std::ofstream csv(out_dir / "convergence.csv");
    csv << "N,panels,iface_pts,phi_density,psi_density,max_err,order,wall_s,GMRES\n";

    std::vector<ConvergenceData> data(levels.size());
    std::vector<double> rates(levels.size(), 0.0);

    std::printf("\n  %s\n", config.label);
    std::printf("  beta_int=%.3f beta_ext=%.3f, lambda_int^2=%.3f lambda_ext^2=%.3f\n",
                config.coefficients.beta_int,
                config.coefficients.beta_ext,
                lambda_sq_int(config),
                lambda_sq_ext(config));
    std::printf("  Interface: shared P2 sphere; nonzero cube Dirichlet BC\n");
    std::printf("  Target P2 node_spacing/h = %.2f; output: %s\n",
                kTargetNodeSpacingOverH, out_dir.string().c_str());
    std::printf("  Set KFBIM_HIGH_RES_3D=1 for high-resolution levels.\n");
    std::printf("  %6s  %8s  %10s  %11s  %11s  %12s  %8s  %8s  %6s\n",
                "N", "panels", "iface_pts", "phi", "psi", "max_err", "order",
                "wall_s", "GMRES");

    for (std::size_t l = 0; l < levels.size(); ++l) {
        data[l] = solve_and_measure(levels[l], config);
        REQUIRE(std::isfinite(data[l].bulk_err));
        REQUIRE(data[l].num_interface_points > 0);
        REQUIRE(data[l].psi_size == data[l].num_interface_points);
        REQUIRE(data[l].phi_size == data[l].num_interface_points);

        if (l == 0) {
            std::printf("  %6d  %8d  %10d  %11d  %11d  %12.4e  %8s  %8.3f  %6d\n",
                        levels[l], data[l].num_panels, data[l].num_interface_points,
                        data[l].phi_size, data[l].psi_size, data[l].bulk_err,
                        "-", data[l].wall_time, data[l].iterations);
            csv << levels[l] << "," << data[l].num_panels << ","
                << data[l].num_interface_points << "," << data[l].phi_size << ","
                << data[l].psi_size << "," << std::setprecision(16)
                << data[l].bulk_err << ",," << data[l].wall_time
                << "," << data[l].iterations << "\n";
        } else {
            rates[l] = std::log(static_cast<double>(data[l - 1].bulk_err)
                                / static_cast<double>(data[l].bulk_err))
                       / std::log(static_cast<double>(levels[l])
                                  / static_cast<double>(levels[l - 1]));
            std::printf("  %6d  %8d  %10d  %11d  %11d  %12.4e  %8.3f  %8.3f  %6d\n",
                        levels[l], data[l].num_panels, data[l].num_interface_points,
                        data[l].phi_size, data[l].psi_size, data[l].bulk_err,
                        rates[l], data[l].wall_time, data[l].iterations);
            csv << levels[l] << "," << data[l].num_panels << ","
                << data[l].num_interface_points << "," << data[l].phi_size << ","
                << data[l].psi_size << "," << std::setprecision(16)
                << data[l].bulk_err << "," << rates[l] << ","
                << data[l].wall_time << "," << data[l].iterations << "\n";
        }
    }

    REQUIRE(data.back().bulk_err < data.front().bulk_err);
    REQUIRE(tail_average_order(rates) > 0.6);
    REQUIRE(data.back().bulk_err < config.final_error_threshold);
}

TransmissionCaseConfig common_ratio_case()
{
    constexpr double beta_int = 2.0;
    constexpr double beta_ext = 1.0;
    constexpr double lambda_sq = 1.1;
    return {"Common-ratio transmission 3D: -div(beta grad u)+kappa^2 u=f",
            "laplace_transmission_common_ratio_3d",
            LaplaceTransmissionMode3D::CommonRatio,
            {beta_int, beta_ext, beta_int * lambda_sq, beta_ext * lambda_sq},
            300,
            64,
            8.0e-2};
}

TransmissionCaseConfig different_ratios_case()
{
    return {"Different-ratio transmission 3D: generic two-density KFBI operator",
            "laplace_transmission_different_ratios_3d",
            LaplaceTransmissionMode3D::DifferentRatios,
            {10.0, 1.0, 11.0, 0.7},
            500,
            32,
            1.2e-1};
}

} // namespace

TEST_CASE("Common-ratio transmission 3D: P2 sphere convergence",
          "[transmission][laplace][interface][p2][convergence][3d]")
{
    run_convergence_case(common_ratio_case());
}

TEST_CASE("Different-ratio transmission 3D: P2 sphere convergence",
          "[transmission][laplace][interface][different-ratio][p2][convergence][3d]")
{
    run_convergence_case(different_ratios_case());
}
