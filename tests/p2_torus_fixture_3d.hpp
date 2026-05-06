#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "src/interface/interface_3d.hpp"

namespace kfbim_test_torus_3d {

constexpr double kTorusCx = 0.07;
constexpr double kTorusCy = -0.04;
constexpr double kTorusCz = 0.03;
constexpr double kTorusMajorRadius = 0.42;
constexpr double kTorusMinorRadius = 0.18;
constexpr double kBoxLower = -1.0;
constexpr double kBoxSide = 2.0;
constexpr double kTargetNodeSpacingOverH = 1.5;
constexpr double kPi = 3.14159265358979323846;

struct Box3D {
    std::array<double, 3> lower;
    double                side_length;
};

struct TorusMeshData {
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<double>          weights;
    std::vector<std::array<int, 3>> panels;
    std::vector<std::array<int, 6>> panel_points;
    std::map<std::pair<int, int>, int> node_by_key;
};

inline Box3D standard_box()
{
    return {{kBoxLower, kBoxLower, kBoxLower}, kBoxSide};
}

inline Eigen::Vector3d torus_center()
{
    return {kTorusCx, kTorusCy, kTorusCz};
}

inline int wrap_mod(int value, int modulus)
{
    value %= modulus;
    if (value < 0)
        value += modulus;
    return value;
}

inline Eigen::Vector3d torus_point_from_angles(double u, double v)
{
    const double cu = std::cos(u);
    const double su = std::sin(u);
    const double cv = std::cos(v);
    const double sv = std::sin(v);
    const double rho = kTorusMajorRadius + kTorusMinorRadius * cv;
    return {kTorusCx + rho * cu,
            kTorusCy + rho * su,
            kTorusCz + kTorusMinorRadius * sv};
}

inline Eigen::Vector3d torus_normal_from_angles(double u, double v)
{
    Eigen::Vector3d normal(std::cos(v) * std::cos(u),
                           std::cos(v) * std::sin(u),
                           std::sin(v));
    normal.normalize();
    return normal;
}

inline int add_torus_node(TorusMeshData& mesh,
                          int            u2,
                          int            v2,
                          int            nu,
                          int            nv)
{
    const int ukey = wrap_mod(u2, 2 * nu);
    const int vkey = wrap_mod(v2, 2 * nv);
    const auto key = std::make_pair(ukey, vkey);
    const auto it = mesh.node_by_key.find(key);
    if (it != mesh.node_by_key.end())
        return it->second;

    const double u = kPi * static_cast<double>(ukey) / static_cast<double>(nu);
    const double v = kPi * static_cast<double>(vkey) / static_cast<double>(nv);
    const int idx = static_cast<int>(mesh.points.size());
    mesh.node_by_key.emplace(key, idx);
    mesh.points.push_back(torus_point_from_angles(u, v));
    mesh.normals.push_back(torus_normal_from_angles(u, v));
    mesh.weights.push_back(0.0);
    return idx;
}

inline double triangle_area(Eigen::Vector3d a,
                            Eigen::Vector3d b,
                            Eigen::Vector3d c)
{
    return 0.5 * ((b - a).cross(c - a)).norm();
}

inline void add_torus_panel(TorusMeshData& mesh,
                            int            a,
                            int            b,
                            int            c,
                            int            eab,
                            int            ebc,
                            int            eca)
{
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
}

inline kfbim::Interface3D make_p2_torus(int nu, int nv)
{
    nu = std::max(4, nu);
    nv = std::max(4, nv);

    TorusMeshData mesh;

    for (int i = 0; i < nu; ++i) {
        for (int j = 0; j < nv; ++j) {
            const int ip = i + 1;
            const int jp = j + 1;

            const int a = add_torus_node(mesh, 2 * i, 2 * j, nu, nv);
            const int b = add_torus_node(mesh, 2 * ip, 2 * j, nu, nv);
            const int c = add_torus_node(mesh, 2 * i, 2 * jp, nu, nv);
            const int d = add_torus_node(mesh, 2 * ip, 2 * jp, nu, nv);

            const int eab = add_torus_node(mesh, 2 * i + 1, 2 * j, nu, nv);
            const int ebc = add_torus_node(mesh, 2 * i + 1, 2 * j + 1, nu, nv);
            const int eca = add_torus_node(mesh, 2 * i, 2 * j + 1, nu, nv);
            add_torus_panel(mesh, a, b, c, eab, ebc, eca);

            const int ebd = add_torus_node(mesh, 2 * ip, 2 * j + 1, nu, nv);
            const int edc = add_torus_node(mesh, 2 * i + 1, 2 * jp, nu, nv);
            add_torus_panel(mesh, b, d, c, ebd, edc, ebc);
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

inline int torus_major_subdivision_for_grid(int N)
{
    const double h = kBoxSide / static_cast<double>(N);
    const double target_edge = 2.0 * kTargetNodeSpacingOverH * h;
    const double circumference =
        2.0 * kPi * (kTorusMajorRadius + kTorusMinorRadius);
    return std::max(8, static_cast<int>(std::lround(circumference / target_edge)));
}

inline int torus_minor_subdivision_for_grid(int N)
{
    const double h = kBoxSide / static_cast<double>(N);
    const double target_edge = 2.0 * kTargetNodeSpacingOverH * h;
    const double circumference = 2.0 * kPi * kTorusMinorRadius;
    return std::max(6, static_cast<int>(std::lround(circumference / target_edge)));
}

inline kfbim::Interface3D make_p2_torus_for_grid(int N)
{
    return make_p2_torus(torus_major_subdivision_for_grid(N),
                         torus_minor_subdivision_for_grid(N));
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
    std::vector<int> levels{8, 16, 32};
    if (std::getenv("KFBIM_HIGH_RES_3D") != nullptr) {
        levels.push_back(64);
        levels.push_back(128);
        levels.push_back(256);
    }
    return levels;
}

} // namespace kfbim_test_torus_3d
