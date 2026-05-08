#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "src/interface/interface_3d.hpp"

namespace kfbim_test_3d {

constexpr double kSphereCx = 0.07;
constexpr double kSphereCy = -0.04;
constexpr double kSphereCz = 0.03;
constexpr double kSphereRadius = 0.55;
constexpr double kBoxLower = -1.0;
constexpr double kBoxSide = 2.0;
constexpr double kTargetNodeSpacingOverH = 1.2;
constexpr double kOctahedronEdgeLengthOnUnitSphere = 1.41421356237309504880;

struct Box3D {
    std::array<double, 3> lower;
    double                side_length;
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

inline Box3D standard_box()
{
    return {{kBoxLower, kBoxLower, kBoxLower}, kBoxSide};
}

inline Eigen::Vector3d sphere_center()
{
    return {kSphereCx, kSphereCy, kSphereCz};
}

inline std::string point_key(Eigen::Vector3d unit)
{
    unit.normalize();
    const long long x = std::llround(unit[0] * 1.0e12);
    const long long y = std::llround(unit[1] * 1.0e12);
    const long long z = std::llround(unit[2] * 1.0e12);
    return std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);
}

inline Eigen::Vector3d sphere_point(Eigen::Vector3d unit)
{
    unit.normalize();
    return sphere_center() + kSphereRadius * unit;
}

inline int add_sphere_node(SphereMeshData& mesh, Eigen::Vector3d unit)
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

inline int add_edge_midpoint(SphereMeshData& mesh, int a, int b)
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

inline double triangle_area(Eigen::Vector3d a,
                            Eigen::Vector3d b,
                            Eigen::Vector3d c)
{
    return 0.5 * ((b - a).cross(c - a)).norm();
}

inline kfbim::Interface3D make_p2_sphere(int subdivision)
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

    return kfbim::Interface3D(vertices,
                              panels,
                              points,
                              normals,
                              weights,
                              6,
                              panel_point_indices,
                              Eigen::VectorXi::Zero(n_panels),
                              kfbim::PanelNodeLayout3D::QuadraticLagrange);
}

inline int surface_subdivision_for_grid(int N)
{
    const double approx_subdivision =
        kOctahedronEdgeLengthOnUnitSphere * kSphereRadius * static_cast<double>(N)
        / (2.0 * kTargetNodeSpacingOverH * kBoxSide);
    return std::max(2, static_cast<int>(std::lround(approx_subdivision)));
}

inline bool is_outer_boundary_node(int idx, int nx, int ny, int nz)
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

inline std::vector<int> convergence_levels_3d()
{
    std::vector<int> levels{8, 16, 32, 64};
    if (std::getenv("KFBIM_HIGH_RES_3D") != nullptr)
        levels.push_back(128);
    return levels;
}

} // namespace kfbim_test_3d
