#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "p2_torus_fixture_3d.hpp"
#include "kfbim/geometry.hpp"
#include "kfbim/grid.hpp"
#include "kfbim/laplace.hpp"

using namespace kfbim;
using namespace kfbim_test_torus_3d;

namespace {

#ifndef KFBIM_TEST_OUTPUT_DIR
#define KFBIM_TEST_OUTPUT_DIR "output"
#endif

struct ConvergenceData {
    double bulk_err = 0.0;
    double wall_time = 0.0;
    int    iterations = 0;
    int    num_panels = 0;
    int    num_interface_points = 0;
    int    phi_size = 0;
    int    psi_size = 0;
};

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

std::filesystem::path output_dir()
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR)
        / "laplace_transmission_common_ratio_torus_3d";
    std::filesystem::create_directories(dir);
    return dir;
}

int visualization_level()
{
    const char* env = std::getenv("KFBIM_TORUS_VIS_N");
    if (env == nullptr)
        return -1;
    return std::atoi(env);
}

void write_surface_points_csv(const std::filesystem::path& path,
                              const Interface3D&           iface)
{
    std::ofstream out(path);
    out << std::setprecision(16);
    out << "q,x,y,z,nx,ny,nz,u_int,u_ext,u_avg,u_jump\n";
    for (int q = 0; q < iface.num_points(); ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const double z = iface.points()(q, 2);
        const double ui = u_int(x, y, z);
        const double ue = u_ext(x, y, z);
        out << q << ","
            << x << "," << y << "," << z << ","
            << iface.normals()(q, 0) << ","
            << iface.normals()(q, 1) << ","
            << iface.normals()(q, 2) << ","
            << ui << "," << ue << ","
            << 0.5 * (ui + ue) << ","
            << ui - ue << "\n";
    }
}

void write_surface_panels_csv(const std::filesystem::path& path,
                              const Interface3D&           iface)
{
    std::ofstream out(path);
    out << "panel,v0,v1,v2,p0,p1,p2,p3,p4,p5\n";
    for (int p = 0; p < iface.num_panels(); ++p) {
        out << p << ","
            << iface.panels()(p, 0) << ","
            << iface.panels()(p, 1) << ","
            << iface.panels()(p, 2);
        for (int q = 0; q < iface.points_per_panel(); ++q)
            out << "," << iface.point_index(p, q);
        out << "\n";
    }
}

void write_z_slice_csv(const std::filesystem::path& path,
                       const CartesianGrid3D&       grid,
                       const GridPair3D&            grid_pair,
                       const Eigen::VectorXd&       u_bulk,
                       const Eigen::VectorXd&       u_exact)
{
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int k0 = std::max(
        0,
        std::min(nz - 1,
                 static_cast<int>(std::lround(
                     (kTorusCz - grid.origin()[2]) / grid.spacing()[2]))));

    std::ofstream out(path);
    out << std::setprecision(16);
    out << "i,j,k,x,y,z,u_bulk,u_exact,abs_error,label\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int n = grid.index(i, j, k0);
            const auto c = grid.coord(i, j, k0);
            out << i << "," << j << "," << k0 << ","
                << c[0] << "," << c[1] << "," << c[2] << ","
                << u_bulk[n] << "," << u_exact[n] << ","
                << std::abs(u_bulk[n] - u_exact[n]) << ","
                << grid_pair.domain_label(n) << "\n";
        }
    }
}

void write_visualization_outputs(const std::filesystem::path& out_dir,
                                 int                          N,
                                 const CartesianGrid3D&       grid,
                                 const Interface3D&           iface,
                                 const GridPair3D&            grid_pair,
                                 const Eigen::VectorXd&       u_bulk,
                                 const Eigen::VectorXd&       u_exact)
{
    char tag[32];
    std::snprintf(tag, sizeof(tag), "N%04d", N);

    write_surface_points_csv(out_dir / (std::string("torus_surface_points_")
                                        + tag + ".csv"),
                             iface);
    write_surface_panels_csv(out_dir / (std::string("torus_surface_panels_")
                                        + tag + ".csv"),
                             iface);
    write_z_slice_csv(out_dir / (std::string("torus_solution_zslice_")
                                 + tag + ".csv"),
                      grid,
                      grid_pair,
                      u_bulk,
                      u_exact);
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

ConvergenceData solve_and_measure(int N, const std::filesystem::path& out_dir)
{
    const auto wall_start = std::chrono::steady_clock::now();
    constexpr double beta_int = 2.0;
    constexpr double beta_ext = 1.0;
    constexpr double lambda_sq = 1.1;
    const LaplaceTransmissionCoefficients3D coefficients{
        beta_int, beta_ext, beta_int * lambda_sq, beta_ext * lambda_sq};

    const Box3D box = standard_box();
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid3D grid(box.lower, {h, h, h}, {N, N, N}, DofLayout3D::Node);
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int n_dof = grid.num_dofs();

    Interface3D iface = make_p2_torus_for_grid(N);
    const int n_iface = iface.num_points();

    LaplaceTransmission3D problem(grid,
                                  iface,
                                  LaplaceTransmissionMode3D::CommonRatio,
                                  coefficients);
    const GridPair3D& gp = problem.grid_pair();

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
        sigma[q] = beta_int * grad_u_int(x, y, z).dot(normal)
                   - beta_ext * grad_u_ext(x, y, z).dot(normal);
        rhs_data.reduced_rhs_int_derivs[q] =
            Eigen::VectorXd::Constant(1, q_int(x, y, z, lambda_sq));
        rhs_data.reduced_rhs_ext_derivs[q] =
            Eigen::VectorXd::Constant(1, q_ext(x, y, z, lambda_sq));
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
            rhs_data.reduced_rhs_bulk[n] = q_int(x, y, z, lambda_sq);
            u_exact[n] = u_int(x, y, z);
        } else {
            rhs_data.reduced_rhs_bulk[n] = q_ext(x, y, z, lambda_sq);
            u_exact[n] = u_ext(x, y, z);
        }
    }

    auto result = problem.solve(mu, sigma, rhs_data, outer_bc, 350, 1.0e-8, 120);
    REQUIRE(result.converged);

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n)
        bulk_err = std::max(bulk_err, std::abs(result.u_bulk[n] - u_exact[n]));

    if (visualization_level() == N) {
        write_visualization_outputs(out_dir,
                                    N,
                                    grid,
                                    iface,
                                    gp,
                                    result.u_bulk,
                                    u_exact);
    }

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

} // namespace

TEST_CASE("Common-ratio transmission 3D: P2 torus convergence",
          "[transmission][laplace][interface][torus][p2][convergence][3d]")
{
    const std::vector<int> levels = convergence_levels_3d();
    const std::filesystem::path out_dir = output_dir();
    std::ofstream csv(out_dir / "convergence.csv");
    csv << "N,panels,iface_pts,phi_density,psi_density,max_err,order,wall_s,GMRES\n";

    std::vector<ConvergenceData> data(levels.size());
    std::vector<double> rates(levels.size(), 0.0);

    std::printf("\n  Common-ratio transmission 3D on P2 torus\n");
    std::printf("  beta_int=2.000 beta_ext=1.000, lambda_int^2=1.100 lambda_ext^2=1.100\n");
    std::printf("  Interface: shared P2 torus; nonzero cube Dirichlet BC\n");
    std::printf("  Major radius=%.2f, minor radius=%.2f, center=(%.2f, %.2f, %.2f)\n",
                kTorusMajorRadius,
                kTorusMinorRadius,
                kTorusCx,
                kTorusCy,
                kTorusCz);
    std::printf("  Target P2 node_spacing/h = %.2f; output: %s\n",
                kTargetNodeSpacingOverH, out_dir.string().c_str());
    std::printf("  Set KFBIM_HIGH_RES_3D=1 to include N=64, N=128, and N=256.\n");
    std::printf("  Set KFBIM_TORUS_VIS_N=<N> to write torus visualization CSVs.\n");
    std::printf("  %6s  %8s  %10s  %11s  %11s  %12s  %8s  %8s  %6s\n",
                "N", "panels", "iface_pts", "phi", "psi", "max_err", "order",
                "wall_s", "GMRES");

    for (std::size_t l = 0; l < levels.size(); ++l) {
        data[l] = solve_and_measure(levels[l], out_dir);
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
    REQUIRE(tail_average_order(rates) > 0.4);
    REQUIRE(data.back().bulk_err < 1.5e-1);
}
