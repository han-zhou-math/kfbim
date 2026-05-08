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

#include "p2_sphere_fixture_3d.hpp"
#include "kfbim/geometry.hpp"
#include "kfbim/grid.hpp"
#include "kfbim/laplace.hpp"

using namespace kfbim;
using namespace kfbim_test_3d;

namespace {

constexpr double kEta = 1.1;

#ifndef KFBIM_TEST_OUTPUT_DIR
#define KFBIM_TEST_OUTPUT_DIR "output"
#endif

enum class BvpCase {
    InteriorDirichlet,
    ExteriorDirichlet,
    InteriorNeumann,
    ExteriorNeumann
};

struct SolveData {
    double bulk_err = 0.0;
    int    iterations = 0;
    int    num_panels = 0;
    int    num_interface_points = 0;
    int    density_size = 0;
};

double exact_u(double x, double y, double z)
{
    const double p = 0.7 * x + 0.2 * y - 0.3 * z;
    const double q = 0.4 * x - 0.6 * y + 0.5 * z;
    return std::sin(p) + 0.3 * std::cos(q)
           + 0.1 * x * y * z + 0.2 * x - 0.15 * y + 0.05 * z;
}

Eigen::Vector3d exact_grad(double x, double y, double z)
{
    const double p = 0.7 * x + 0.2 * y - 0.3 * z;
    const double q = 0.4 * x - 0.6 * y + 0.5 * z;
    return {0.7 * std::cos(p) - 0.12 * std::sin(q) + 0.1 * y * z + 0.2,
            0.2 * std::cos(p) + 0.18 * std::sin(q) + 0.1 * x * z - 0.15,
           -0.3 * std::cos(p) - 0.15 * std::sin(q) + 0.1 * x * y + 0.05};
}

double exact_f(double x, double y, double z)
{
    const double p = 0.7 * x + 0.2 * y - 0.3 * z;
    const double q = 0.4 * x - 0.6 * y + 0.5 * z;
    return 0.62 * std::sin(p) + 0.231 * std::cos(q)
           + kEta * exact_u(x, y, z);
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

LaplaceBvpType3D bvp_type(BvpCase bvp)
{
    switch (bvp) {
    case BvpCase::InteriorDirichlet:
        return LaplaceBvpType3D::InteriorDirichlet;
    case BvpCase::ExteriorDirichlet:
        return LaplaceBvpType3D::ExteriorDirichlet;
    case BvpCase::InteriorNeumann:
        return LaplaceBvpType3D::InteriorNeumann;
    case BvpCase::ExteriorNeumann:
        return LaplaceBvpType3D::ExteriorNeumann;
    }
    return LaplaceBvpType3D::InteriorDirichlet;
}

double final_error_threshold(BvpCase bvp)
{
    switch (bvp) {
    case BvpCase::InteriorDirichlet: return 5.0e-2;
    case BvpCase::ExteriorDirichlet: return 5.0e-2;
    case BvpCase::InteriorNeumann: return 1.0e-1;
    case BvpCase::ExteriorNeumann: return 1.0e-1;
    }
    return 1.0e-1;
}

std::filesystem::path output_dir()
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR) / "laplace_bvp_sphere3d";
    std::filesystem::create_directories(dir);
    return dir;
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

SolveData solve_and_measure(BvpCase bvp, int N)
{
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

    Eigen::VectorXd outer_bc = Eigen::VectorXd::Zero(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        const auto c = grid.coord(n);
        outer_bc[n] = exact_u(c[0], c[1], c[2]);
    }

    LaplaceBvpOptions3D options;
    options.eta = kEta;
    if (!is_interior(bvp))
        options.outer_dirichlet_values = outer_bc;

    LaplaceBvp3D problem(grid, iface, bvp_type(bvp), options);
    const GridPair3D& gp = problem.grid_pair();

    Eigen::VectorXd g(n_iface);
    std::vector<Eigen::VectorXd> rhs_derivs(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const double z = iface.points()(q, 2);
        if (is_dirichlet(bvp)) {
            g[q] = exact_u(x, y, z);
        } else {
            const Eigen::Vector3d normal(iface.normals()(q, 0),
                                         iface.normals()(q, 1),
                                         iface.normals()(q, 2));
            g[q] = exact_grad(x, y, z).dot(normal);
        }
        rhs_derivs[q] = Eigen::VectorXd::Constant(1, exact_f(x, y, z));
    }

    Eigen::VectorXd f_bulk = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(n_dof);
    const int physical_label = is_interior(bvp) ? 1 : 0;
    for (int n = 0; n < n_dof; ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        const double z = c[2];
        u_exact[n] = exact_u(x, y, z);

        if (gp.domain_label(n) == physical_label
            && !is_outer_boundary_node(n, nx, ny, nz)) {
            f_bulk[n] = exact_f(x, y, z);
        }
    }

    auto result = problem.solve(g, f_bulk, rhs_derivs, 400, 1.0e-8, 120);
    REQUIRE(result.converged);

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n) {
        if (gp.domain_label(n) != physical_label)
            continue;
        if (!is_interior(bvp) && is_outer_boundary_node(n, nx, ny, nz))
            continue;
        bulk_err = std::max(bulk_err, std::abs(result.u_bulk[n] - u_exact[n]));
    }

    if (!is_interior(bvp)) {
        double boundary_err = 0.0;
        for (int n = 0; n < n_dof; ++n) {
            if (is_outer_boundary_node(n, nx, ny, nz)) {
                boundary_err = std::max(boundary_err,
                                        std::abs(result.u_bulk[n] - outer_bc[n]));
            }
        }
        REQUIRE(boundary_err < 1.0e-12);
    }

    return {bulk_err,
            result.iterations,
            iface.num_panels(),
            n_iface,
            static_cast<int>(result.density.size())};
}

void check_convergence(BvpCase bvp)
{
    const std::vector<int> Ns = convergence_levels_3d();
    std::vector<SolveData> data(Ns.size());
    std::vector<double> rates(Ns.size(), 0.0);

    const std::filesystem::path out_dir = output_dir();
    const std::filesystem::path csv_path =
        out_dir / (std::string(bvp_file_stem(bvp)) + "_convergence.csv");
    std::ofstream csv(csv_path);
    csv << "N,panels,iface_pts,density,max_err,order,GMRES\n";

    std::printf("\n  Screened %s BVP on P2 sphere: -Delta u + %.2f u = f\n",
                bvp_name(bvp), kEta);
    std::printf("  Surface: shared P2 triangles; target P2 node_spacing/h = %.2f; output: %s\n",
                kTargetNodeSpacingOverH, csv_path.string().c_str());
    std::printf("  Set KFBIM_HIGH_RES_3D=1 to include N=128.\n");
    std::printf("  %6s  %8s  %10s  %11s  %12s  %8s  %6s\n",
                "N", "panels", "iface_pts", "density", "max_err", "order", "GMRES");

    for (std::size_t l = 0; l < Ns.size(); ++l) {
        data[l] = solve_and_measure(bvp, Ns[l]);
        REQUIRE(std::isfinite(data[l].bulk_err));
        REQUIRE(data[l].num_interface_points > 0);
        REQUIRE(data[l].density_size == data[l].num_interface_points);

        if (l == 0) {
            std::printf("  %6d  %8d  %10d  %11d  %12.4e  %8s  %6d\n",
                        Ns[l], data[l].num_panels, data[l].num_interface_points,
                        data[l].density_size, data[l].bulk_err, "-",
                        data[l].iterations);
            csv << Ns[l] << "," << data[l].num_panels << ","
                << data[l].num_interface_points << "," << data[l].density_size
                << "," << std::setprecision(16) << data[l].bulk_err
                << ",," << data[l].iterations << "\n";
        } else {
            rates[l] = std::log(static_cast<double>(data[l - 1].bulk_err)
                                / static_cast<double>(data[l].bulk_err))
                       / std::log(static_cast<double>(Ns[l])
                                  / static_cast<double>(Ns[l - 1]));
            std::printf("  %6d  %8d  %10d  %11d  %12.4e  %8.3f  %6d\n",
                        Ns[l], data[l].num_panels, data[l].num_interface_points,
                        data[l].density_size, data[l].bulk_err, rates[l],
                        data[l].iterations);
            csv << Ns[l] << "," << data[l].num_panels << ","
                << data[l].num_interface_points << "," << data[l].density_size
                << "," << std::setprecision(16) << data[l].bulk_err
                << "," << rates[l] << "," << data[l].iterations << "\n";
        }
    }

    REQUIRE(data.back().bulk_err < data.front().bulk_err);
    REQUIRE(tail_average_order(rates) > 0.6);
    REQUIRE(data.back().bulk_err < final_error_threshold(bvp));
}

} // namespace

TEST_CASE("LaplaceBvp3D modes converge on P2 sphere",
          "[screened][laplace][bvp][p2][convergence][3d]")
{
    check_convergence(BvpCase::InteriorDirichlet);
    check_convergence(BvpCase::ExteriorDirichlet);
    check_convergence(BvpCase::InteriorNeumann);
    check_convergence(BvpCase::ExteriorNeumann);
}
