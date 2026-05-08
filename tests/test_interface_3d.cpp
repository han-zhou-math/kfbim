#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "src/bulk_solvers/laplace_zfft_bulk_solver_3d.hpp"
#include "src/bulk_solvers/zfft_bc_type.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/interface/interface_3d.hpp"
#include "src/local_cauchy/jump_data.hpp"
#include "src/potentials/laplace_potential.hpp"
#include "src/transfer/laplace_restrict_3d.hpp"
#include "src/transfer/laplace_spread_3d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEta = 1.1;
constexpr double kSphereCx = 0.0;
constexpr double kSphereCy = 0.0;
constexpr double kSphereCz = 0.0;
constexpr double kSphereRadius = 1.0;
constexpr double kBoxLower = -1.5;
constexpr double kBoxSide = 3.0;
constexpr double kTargetNodeSpacingOverH = 1.5;
constexpr double kOctahedronEdgeLengthOnUnitSphere = 1.41421356237309504880;

#ifndef KFBIM_TEST_OUTPUT_DIR
#define KFBIM_TEST_OUTPUT_DIR "output"
#endif

struct Box3D {
    std::array<double, 3> lower;
    double                side_length;
};

struct SolveData {
    double bulk_err = 0.0;
    double wall_time = 0.0;
    int    num_panels = 0;
    int    num_interface_points = 0;
};

LaplaceCorrectionMethod3D correction_method_from_env()
{
    const char* value = std::getenv("KFBIM_INTERFACE_3D_CORRECTION");
    if (value == nullptr)
        return LaplaceCorrectionMethod3D::NearestExpansionCenter;

    const std::string method(value);
    if (method == "nearest" || method == "NearestExpansionCenter")
        return LaplaceCorrectionMethod3D::NearestExpansionCenter;
    if (method == "projection" || method == "ProjectionPoint")
        return LaplaceCorrectionMethod3D::ProjectionPoint;

    throw std::invalid_argument(
        "KFBIM_INTERFACE_3D_CORRECTION must be 'projection' or 'nearest'");
}

const char* correction_method_name(LaplaceCorrectionMethod3D method)
{
    switch (method) {
    case LaplaceCorrectionMethod3D::NearestExpansionCenter:
        return "Nearest expansion-center correction";
    case LaplaceCorrectionMethod3D::ProjectionPoint:
        return "Projection-point P2 correction";
    }
    return "Unknown correction";
}

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

struct SphereMeshData {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<double>          weights;
    std::vector<std::array<int, 3>> panels;
    std::vector<std::array<int, 6>> panel_points;
    std::map<std::string, int> vertex_by_key;
    std::map<std::pair<int, int>, int> edge_midpoint;
};

std::string point_key(Eigen::Vector3d unit)
{
    unit.normalize();
    const long long x = std::llround(unit[0] * 1.0e12);
    const long long y = std::llround(unit[1] * 1.0e12);
    const long long z = std::llround(unit[2] * 1.0e12);
    return std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);
}

Eigen::Vector3d sphere_center()
{
    return {kSphereCx, kSphereCy, kSphereCz};
}

Eigen::Vector3d sphere_point(Eigen::Vector3d unit)
{
    unit.normalize();
    return sphere_center() + kSphereRadius * unit;
}

int add_sphere_node(SphereMeshData& mesh, Eigen::Vector3d unit)
{
    unit.normalize();
    const std::string key = point_key(unit);
    const auto it = mesh.vertex_by_key.find(key);
    if (it != mesh.vertex_by_key.end())
        return it->second;

    const int idx = static_cast<int>(mesh.points.size());
    mesh.vertex_by_key.emplace(key, idx);
    mesh.points.push_back(sphere_point(unit));
    mesh.normals.push_back(unit);
    mesh.weights.push_back(0.0);
    return idx;
}

int add_edge_midpoint(SphereMeshData& mesh, int a, int b)
{
    if (a > b)
        std::swap(a, b);
    const auto key = std::make_pair(a, b);
    const auto it = mesh.edge_midpoint.find(key);
    if (it != mesh.edge_midpoint.end())
        return it->second;

    Eigen::Vector3d unit = mesh.normals[a] + mesh.normals[b];
    unit.normalize();
    const int idx = static_cast<int>(mesh.points.size());
    mesh.edge_midpoint.emplace(key, idx);
    mesh.points.push_back(sphere_point(unit));
    mesh.normals.push_back(unit);
    mesh.weights.push_back(0.0);
    return idx;
}

double triangle_area(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c)
{
    return 0.5 * ((b - a).cross(c - a)).norm();
}

Interface3D make_p2_sphere(int subdivision)
{
    subdivision = std::max(1, subdivision);

    const std::array<Eigen::Vector3d, 6> base_vertices = {
        Eigen::Vector3d( 1.0,  0.0,  0.0),
        Eigen::Vector3d( 0.0,  1.0,  0.0),
        Eigen::Vector3d(-1.0,  0.0,  0.0),
        Eigen::Vector3d( 0.0, -1.0,  0.0),
        Eigen::Vector3d( 0.0,  0.0,  1.0),
        Eigen::Vector3d( 0.0,  0.0, -1.0)
    };

    constexpr int px = 0;
    constexpr int py = 1;
    constexpr int nx = 2;
    constexpr int ny = 3;
    constexpr int pz = 4;
    constexpr int nz = 5;

    const std::array<std::array<int, 3>, 8> base_faces = {{
        {{pz, px, py}},
        {{pz, py, nx}},
        {{pz, nx, ny}},
        {{pz, ny, px}},
        {{nz, py, px}},
        {{nz, nx, py}},
        {{nz, ny, nx}},
        {{nz, px, ny}}
    }};

    SphereMeshData mesh;

    auto lattice_unit = [&](const std::array<int, 3>& face, int i, int j) {
        const double l1 = static_cast<double>(i) / static_cast<double>(subdivision);
        const double l2 = static_cast<double>(j) / static_cast<double>(subdivision);
        const double l0 = 1.0 - l1 - l2;
        Eigen::Vector3d unit = l0 * base_vertices[face[0]]
                             + l1 * base_vertices[face[1]]
                             + l2 * base_vertices[face[2]];
        unit.normalize();
        return unit;
    };

    for (const auto& face : base_faces) {
        std::vector<std::vector<int>> ids(subdivision + 1);
        for (int i = 0; i <= subdivision; ++i) {
            ids[i].resize(subdivision + 1 - i, -1);
            for (int j = 0; j <= subdivision - i; ++j)
                ids[i][j] = add_sphere_node(mesh, lattice_unit(face, i, j));
        }

        auto add_panel = [&](int a, int b, int c) {
            const int eab = add_edge_midpoint(mesh, a, b);
            const int ebc = add_edge_midpoint(mesh, b, c);
            const int eca = add_edge_midpoint(mesh, c, a);
            mesh.panels.push_back({a, b, c});
            mesh.panel_points.push_back({a, b, c, eab, ebc, eca});

            const double area = triangle_area(mesh.points[a],
                                              mesh.points[b],
                                              mesh.points[c]);
            const double w = area / 6.0;
            mesh.weights[a] += w;
            mesh.weights[b] += w;
            mesh.weights[c] += w;
            mesh.weights[eab] += w;
            mesh.weights[ebc] += w;
            mesh.weights[eca] += w;
        };

        for (int i = 0; i < subdivision; ++i) {
            for (int j = 0; j < subdivision - i; ++j) {
                add_panel(ids[i][j], ids[i + 1][j], ids[i][j + 1]);
                if (i + j < subdivision - 1)
                    add_panel(ids[i + 1][j], ids[i + 1][j + 1], ids[i][j + 1]);
            }
        }
    }

    const int n_points = static_cast<int>(mesh.points.size());
    const int n_panels = static_cast<int>(mesh.panels.size());

    Eigen::MatrixX3d points(n_points, 3);
    Eigen::MatrixX3d normals(n_points, 3);
    Eigen::VectorXd weights(n_points);
    for (int i = 0; i < n_points; ++i) {
        points.row(i) = mesh.points[i].transpose();
        normals.row(i) = mesh.normals[i].transpose();
        weights[i] = mesh.weights[i];
    }

    Eigen::MatrixX3d vertices = points;
    Eigen::MatrixX3i panels(n_panels, 3);
    Eigen::MatrixXi panel_point_indices(n_panels, 6);
    for (int p = 0; p < n_panels; ++p) {
        panels(p, 0) = mesh.panels[p][0];
        panels(p, 1) = mesh.panels[p][1];
        panels(p, 2) = mesh.panels[p][2];
        for (int q = 0; q < 6; ++q)
            panel_point_indices(p, q) = mesh.panel_points[p][q];
    }

    return Interface3D(vertices,
                       panels,
                       points,
                       normals,
                       weights,
                       6,
                       panel_point_indices,
                       Eigen::VectorXi::Zero(n_panels),
                       PanelNodeLayout3D::QuadraticLagrange);
}

double sine_mode(const Box3D& box, double x, double y, double z,
                 int mx, int my, int mz)
{
    const double xi = x - box.lower[0];
    const double yi = y - box.lower[1];
    const double zi = z - box.lower[2];
    const double kx = mx * kPi / box.side_length;
    const double ky = my * kPi / box.side_length;
    const double kz = mz * kPi / box.side_length;
    return std::sin(kx * xi) * std::sin(ky * yi) * std::sin(kz * zi);
}

Eigen::Vector3d sine_mode_grad(const Box3D& box,
                               double x,
                               double y,
                               double z,
                               int mx,
                               int my,
                               int mz)
{
    const double xi = x - box.lower[0];
    const double yi = y - box.lower[1];
    const double zi = z - box.lower[2];
    const double kx = mx * kPi / box.side_length;
    const double ky = my * kPi / box.side_length;
    const double kz = mz * kPi / box.side_length;
    return {
        kx * std::cos(kx * xi) * std::sin(ky * yi) * std::sin(kz * zi),
        ky * std::sin(kx * xi) * std::cos(ky * yi) * std::sin(kz * zi),
        kz * std::sin(kx * xi) * std::sin(ky * yi) * std::cos(kz * zi)
    };
}

double laplace_eigenvalue(const Box3D& box, int mx, int my, int mz)
{
    return static_cast<double>(mx * mx + my * my + mz * mz)
           * kPi * kPi / (box.side_length * box.side_length);
}

double u_int(const Box3D& box, double x, double y, double z)
{
    return 0.70 * sine_mode(box, x, y, z, 1, 2, 1)
           + 0.25 * sine_mode(box, x, y, z, 2, 1, 3);
}

Eigen::Vector3d grad_u_int(const Box3D& box, double x, double y, double z)
{
    return 0.70 * sine_mode_grad(box, x, y, z, 1, 2, 1)
           + 0.25 * sine_mode_grad(box, x, y, z, 2, 1, 3);
}

double f_int(const Box3D& box, double x, double y, double z)
{
    return 0.70 * (laplace_eigenvalue(box, 1, 2, 1) + kEta)
           * sine_mode(box, x, y, z, 1, 2, 1)
           + 0.25 * (laplace_eigenvalue(box, 2, 1, 3) + kEta)
           * sine_mode(box, x, y, z, 2, 1, 3);
}

double u_ext(const Box3D& box, double x, double y, double z)
{
    return -0.45 * sine_mode(box, x, y, z, 2, 1, 1)
           + 0.30 * sine_mode(box, x, y, z, 1, 3, 2);
}

Eigen::Vector3d grad_u_ext(const Box3D& box, double x, double y, double z)
{
    return -0.45 * sine_mode_grad(box, x, y, z, 2, 1, 1)
           + 0.30 * sine_mode_grad(box, x, y, z, 1, 3, 2);
}

double f_ext(const Box3D& box, double x, double y, double z)
{
    return -0.45 * (laplace_eigenvalue(box, 2, 1, 1) + kEta)
           * sine_mode(box, x, y, z, 2, 1, 1)
           + 0.30 * (laplace_eigenvalue(box, 1, 3, 2) + kEta)
           * sine_mode(box, x, y, z, 1, 3, 2);
}

bool is_outer_boundary_node(int idx, int nx, int ny, int nz)
{
    const int nxy = nx * ny;
    const int k = idx / nxy;
    const int rem = idx % nxy;
    const int j = rem / nx;
    const int i = rem % nx;
    return i == 0 || i == nx - 1
        || j == 0 || j == ny - 1
        || k == 0 || k == nz - 1;
}

std::vector<LaplaceJumpData3D> make_jumps(const Interface3D& iface,
                                          const Box3D&       box)
{
    const int n_iface = iface.num_points();
    std::vector<LaplaceJumpData3D> jumps(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const double z = iface.points()(q, 2);
        const Eigen::Vector3d normal(iface.normals()(q, 0),
                                     iface.normals()(q, 1),
                                     iface.normals()(q, 2));
        jumps[q].u_jump = u_int(box, x, y, z) - u_ext(box, x, y, z);
        jumps[q].un_jump = (grad_u_int(box, x, y, z) - grad_u_ext(box, x, y, z))
                               .dot(normal);
        jumps[q].rhs_derivs =
            Eigen::VectorXd::Constant(1, f_int(box, x, y, z) - f_ext(box, x, y, z));
    }
    return jumps;
}

std::filesystem::path output_dir()
{
    std::filesystem::path dir =
        std::filesystem::path(KFBIM_TEST_OUTPUT_DIR) / "laplace_interface_sphere3d";
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

bool profile_enabled()
{
    return std::getenv("KFBIM_PROFILE_INTERFACE_3D") != nullptr;
}

int max_n_from_env()
{
    const char* value = std::getenv("KFBIM_INTERFACE_3D_MAX_N");
    if (value == nullptr)
        return 0;
    return std::max(0, std::atoi(value));
}

int surface_subdivision_for_grid(int N)
{
    const double approx_subdivision =
        kOctahedronEdgeLengthOnUnitSphere * kSphereRadius * static_cast<double>(N)
        / (2.0 * kTargetNodeSpacingOverH * kBoxSide);
    return std::max(2, static_cast<int>(std::lround(approx_subdivision)));
}

SolveData solve_and_measure(int N)
{
    const auto wall_start = StageTimer::Clock::now();
    StageTimer timer;
    const bool profile = profile_enabled();
    const LaplaceCorrectionMethod3D correction_method =
        correction_method_from_env();
    const Box3D box{{kBoxLower, kBoxLower, kBoxLower}, kBoxSide};
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid3D grid(box.lower, {h, h, h}, {N, N, N}, DofLayout3D::Node);
    const double t_grid = timer.lap();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int n_dof = grid.num_dofs();

    const int surface_subdivision = surface_subdivision_for_grid(N);
    Interface3D iface = make_p2_sphere(surface_subdivision);
    const double t_interface = timer.lap();
    GridPair3D grid_pair(grid, iface);
    const double t_grid_pair = timer.lap();

    Eigen::VectorXd f_bulk = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        const double z = c[2];
        if (grid_pair.domain_label(n) > 0) {
            u_exact[n] = u_int(box, x, y, z);
            f_bulk[n] = is_outer_boundary_node(n, nx, ny, nz) ? 0.0 : f_int(box, x, y, z);
        } else {
            u_exact[n] = u_ext(box, x, y, z);
            f_bulk[n] = is_outer_boundary_node(n, nx, ny, nz) ? 0.0 : f_ext(box, x, y, z);
        }
    }
    const double t_rhs = timer.lap();

    LaplaceQuadraticPatchCenterSpread3D spread(
        grid_pair, kEta, correction_method);
    LaplaceFftBulkSolverZfft3D bulk_solver(grid, ZfftBcType::Dirichlet, kEta);
    LaplaceQuadraticPatchCenterRestrict3D restrict_op(grid_pair);
    const double t_ops = timer.lap();

    const std::vector<LaplaceJumpData3D> jumps = make_jumps(iface, box);
    Eigen::VectorXd rhs = f_bulk;
    const double t_jumps = timer.lap();

    const LaplaceSpreadResult3D spread_result = spread.apply(jumps, rhs);
    const double t_spread = timer.lap();

    Eigen::VectorXd u_bulk;
    bulk_solver.solve(-rhs, u_bulk);
    const double t_bulk_solve = timer.lap();

    const std::vector<LocalPoly3D> solution_polys =
        restrict_op.apply(u_bulk, spread_result);
    REQUIRE(solution_polys.size() == static_cast<std::size_t>(iface.num_points()));
    const double t_restrict = timer.lap();

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

    const double wall_time =
        std::chrono::duration<double>(StageTimer::Clock::now() - wall_start)
            .count();
    return {bulk_err, wall_time, iface.num_panels(), iface.num_points()};
}

} // namespace

TEST_CASE("Constant-coefficient screened interface problem converges on P2 sphere",
          "[screened][laplace][interface][p2][convergence][3d]")
{
    std::vector<int> Ns{4, 8, 16, 32, 64, 128};
    const int max_n = max_n_from_env();
    if (max_n > 0) {
        Ns.erase(std::remove_if(Ns.begin(), Ns.end(),
                                [max_n](int n) { return n > max_n; }),
                 Ns.end());
        REQUIRE_FALSE(Ns.empty());
    }
    std::vector<SolveData> data(Ns.size());
    std::vector<double> rates(Ns.size(), 0.0);
    const LaplaceCorrectionMethod3D correction_method =
        correction_method_from_env();

    const std::filesystem::path out_dir = output_dir();
    std::ofstream csv(out_dir / "convergence.csv");
    csv << "N,max_err,order,wall_s,GMRES\n";

    std::printf("\n  Constant-coefficient screened interface problem on P2 sphere\n");
    std::printf("  Manufactured sine modes vanish on the outer Cartesian box; eta = %.2f\n",
                kEta);
    std::printf("  Surface: shared P2 triangles; target P2 node_spacing/h = %.2f\n",
                kTargetNodeSpacingOverH);
    std::printf("  Geometry: unit sphere centered at origin in (-1.5,1.5)^3\n");
    std::printf("  %s; output: %s\n",
                correction_method_name(correction_method),
                out_dir.string().c_str());
    std::printf("  %6s  %12s  %8s  %8s  %6s\n",
                "N", "max_err", "order", "wall_s", "GMRES");

    for (std::size_t l = 0; l < Ns.size(); ++l) {
        data[l] = solve_and_measure(Ns[l]);
        REQUIRE(std::isfinite(data[l].bulk_err));
        REQUIRE(data[l].num_interface_points > 0);
        REQUIRE(data[l].num_panels > 0);

        if (l == 0) {
            std::printf("  %6d  %12.4e  %8s  %8.3f  %6d\n",
                        Ns[l], data[l].bulk_err, "-", data[l].wall_time, 0);
            csv << Ns[l] << "," << std::setprecision(16) << data[l].bulk_err
                << ",," << data[l].wall_time << "," << 0 << "\n";
        } else {
            rates[l] = std::log(data[l - 1].bulk_err / data[l].bulk_err)
                       / std::log(static_cast<double>(Ns[l])
                                  / static_cast<double>(Ns[l - 1]));
            std::printf("  %6d  %12.4e  %8.3f  %8.3f  %6d\n",
                        Ns[l], data[l].bulk_err, rates[l],
                        data[l].wall_time, 0);
            csv << Ns[l] << "," << std::setprecision(16) << data[l].bulk_err
                << "," << rates[l] << "," << data[l].wall_time
                << "," << 0 << "\n";
        }
    }

    if (Ns.size() > 1) {
        REQUIRE(data.back().bulk_err < data.front().bulk_err);
        REQUIRE(tail_average_order(rates) > 0.8);
        REQUIRE(data.back().bulk_err < 5.0e-2);
    }
}
