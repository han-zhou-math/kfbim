#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "../interface/interface_3d.hpp"

namespace kfbim::geometry3d {

using P2Shape = Eigen::Matrix<double, 6, 1>;

struct RefTriangle {
    std::array<Eigen::Vector3d, 3> bary;
};

inline Eigen::Vector3d barycentric_from_uv(Eigen::Vector2d uv)
{
    return {1.0 - uv[0] - uv[1], uv[0], uv[1]};
}

inline Eigen::Vector2d uv_from_barycentric(Eigen::Vector3d bary)
{
    return {bary[1], bary[2]};
}

inline P2Shape p2_shape(Eigen::Vector3d bary)
{
    const double l0 = bary[0];
    const double l1 = bary[1];
    const double l2 = bary[2];

    P2Shape N;
    N[0] = l0 * (2.0 * l0 - 1.0);
    N[1] = l1 * (2.0 * l1 - 1.0);
    N[2] = l2 * (2.0 * l2 - 1.0);
    N[3] = 4.0 * l0 * l1;
    N[4] = 4.0 * l1 * l2;
    N[5] = 4.0 * l2 * l0;
    return N;
}

inline P2Shape p2_shape_du(Eigen::Vector3d bary)
{
    const double l0 = bary[0];
    const double l1 = bary[1];
    const double l2 = bary[2];

    P2Shape N;
    N[0] = 1.0 - 4.0 * l0;
    N[1] = 4.0 * l1 - 1.0;
    N[2] = 0.0;
    N[3] = 4.0 * (l0 - l1);
    N[4] = 4.0 * l2;
    N[5] = -4.0 * l2;
    return N;
}

inline P2Shape p2_shape_dv(Eigen::Vector3d bary)
{
    const double l0 = bary[0];
    const double l1 = bary[1];
    const double l2 = bary[2];

    P2Shape N;
    N[0] = 1.0 - 4.0 * l0;
    N[1] = 0.0;
    N[2] = 4.0 * l2 - 1.0;
    N[3] = -4.0 * l1;
    N[4] = 4.0 * l1;
    N[5] = 4.0 * (l0 - l2);
    return N;
}

inline P2Shape p2_shape_duu()
{
    P2Shape N;
    N << 4.0, 4.0, 0.0, -8.0, 0.0, 0.0;
    return N;
}

inline P2Shape p2_shape_duv()
{
    P2Shape N;
    N << 4.0, 0.0, 0.0, -4.0, 4.0, -4.0;
    return N;
}

inline P2Shape p2_shape_dvv()
{
    P2Shape N;
    N << 4.0, 0.0, 4.0, 0.0, 0.0, -8.0;
    return N;
}

inline Eigen::Vector3d panel_combination(const Interface3D& iface,
                                         int                panel,
                                         const P2Shape&     coeffs)
{
    Eigen::Vector3d value = Eigen::Vector3d::Zero();
    for (int q = 0; q < 6; ++q)
        value += coeffs[q] * iface.points().row(iface.point_index(panel, q)).transpose();
    return value;
}

inline Eigen::Vector3d panel_point(const Interface3D& iface,
                                   int                panel,
                                   Eigen::Vector3d    bary)
{
    return panel_combination(iface, panel, p2_shape(bary));
}

inline Eigen::Vector3d panel_tangent_u(const Interface3D& iface,
                                       int                panel,
                                       Eigen::Vector3d    bary)
{
    return panel_combination(iface, panel, p2_shape_du(bary));
}

inline Eigen::Vector3d panel_tangent_v(const Interface3D& iface,
                                       int                panel,
                                       Eigen::Vector3d    bary)
{
    return panel_combination(iface, panel, p2_shape_dv(bary));
}

inline Eigen::Vector3d panel_second_uu(const Interface3D& iface, int panel)
{
    return panel_combination(iface, panel, p2_shape_duu());
}

inline Eigen::Vector3d panel_second_uv(const Interface3D& iface, int panel)
{
    return panel_combination(iface, panel, p2_shape_duv());
}

inline Eigen::Vector3d panel_second_vv(const Interface3D& iface, int panel)
{
    return panel_combination(iface, panel, p2_shape_dvv());
}

inline Eigen::Vector3d panel_interpolated_normal(const Interface3D& iface,
                                                 int                panel,
                                                 Eigen::Vector3d    bary)
{
    const P2Shape N = p2_shape(bary);
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    for (int q = 0; q < 6; ++q)
        normal += N[q] * iface.normals().row(iface.point_index(panel, q)).transpose();

    double len = normal.norm();
    if (len > 1.0e-14)
        return normal / len;

    normal = iface.normals().row(iface.point_index(panel, 0)).transpose();
    len = normal.norm();
    if (len > 1.0e-14)
        return normal / len;

    return Eigen::Vector3d::UnitX();
}

inline Eigen::Vector3d panel_oriented_normal(const Interface3D& iface,
                                             int                panel,
                                             Eigen::Vector3d    bary)
{
    Eigen::Vector3d normal =
        panel_tangent_u(iface, panel, bary).cross(panel_tangent_v(iface, panel, bary));
    const double len = normal.norm();
    if (len <= 1.0e-14)
        return panel_interpolated_normal(iface, panel, bary);

    normal /= len;
    const Eigen::Vector3d ref = panel_interpolated_normal(iface, panel, bary);
    if (normal.dot(ref) < 0.0)
        normal = -normal;
    return normal;
}

inline double panel_scalar(const Interface3D&     iface,
                           int                    panel,
                           const Eigen::VectorXd& values,
                           Eigen::Vector3d        bary)
{
    const P2Shape N = p2_shape(bary);
    double value = 0.0;
    for (int q = 0; q < 6; ++q)
        value += N[q] * values[iface.point_index(panel, q)];
    return value;
}

inline std::vector<RefTriangle> subdivided_reference_triangles(int n = 4)
{
    if (n <= 0)
        throw std::invalid_argument("subdivision count must be positive");

    auto bary = [n](int i, int j) {
        const double l1 = static_cast<double>(i) / static_cast<double>(n);
        const double l2 = static_cast<double>(j) / static_cast<double>(n);
        const double l0 = 1.0 - l1 - l2;
        return Eigen::Vector3d(l0, l1, l2);
    };

    std::vector<RefTriangle> tris;
    tris.reserve(static_cast<std::size_t>(n * n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n - i; ++j) {
            tris.push_back({{bary(i, j), bary(i + 1, j), bary(i, j + 1)}});
            if (i + j < n - 1) {
                tris.push_back({{bary(i + 1, j),
                                 bary(i + 1, j + 1),
                                 bary(i, j + 1)}});
            }
        }
    }
    return tris;
}

inline std::vector<Eigen::Vector3d> expansion_center_barycentrics(int n = 4)
{
    const std::vector<RefTriangle> tris = subdivided_reference_triangles(n);
    std::vector<Eigen::Vector3d> centers;
    centers.reserve(tris.size());
    for (const auto& tri : tris)
        centers.push_back((tri.bary[0] + tri.bary[1] + tri.bary[2]) / 3.0);
    return centers;
}

} // namespace kfbim::geometry3d
