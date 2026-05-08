#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Dense>

#include "../interface/interface_2d.hpp"

namespace kfbim::geometry2d {

inline constexpr std::array<double, 3> kP2NodeS = {-1.0, 0.0, 1.0};
inline constexpr std::array<double, 4> kP2CenterS = {
    -0.75, -0.25, 0.25, 0.75};

inline bool is_quadratic_lagrange_panel_layout(const Interface2D& iface)
{
    return iface.points_per_panel() == 3
        && iface.panel_node_layout() == PanelNodeLayout2D::QuadraticLagrange;
}

inline void p2_shape(double s, double N[3])
{
    const double l0 = 0.5 * s * (s - 1.0);
    const double l1 = 1.0 - s * s;
    const double l2 = 0.5 * s * (s + 1.0);
    N[0] = l0;
    N[1] = l1;
    N[2] = l2;
}

inline void p2_shape_deriv(double s, double dN[3])
{
    dN[0] = s - 0.5;
    dN[1] = -2.0 * s;
    dN[2] = s + 0.5;
}

inline void p2_shape_second_deriv(double, double ddN[3])
{
    ddN[0] = 1.0;
    ddN[1] = -2.0;
    ddN[2] = 1.0;
}

inline Eigen::Vector2d panel_point(const Interface2D& iface,
                                   int                panel,
                                   double             s)
{
    double N[3];
    p2_shape(s, N);
    Eigen::Vector2d pt = Eigen::Vector2d::Zero();
    for (int q = 0; q < 3; ++q)
        pt += N[q] * iface.points().row(iface.point_index(panel, q)).transpose();
    return pt;
}

inline Eigen::Vector2d panel_tangent(const Interface2D& iface,
                                     int                panel,
                                     double             s)
{
    double dN[3];
    p2_shape_deriv(s, dN);
    Eigen::Vector2d tangent = Eigen::Vector2d::Zero();
    for (int q = 0; q < 3; ++q)
        tangent += dN[q] * iface.points().row(iface.point_index(panel, q)).transpose();
    return tangent;
}

inline Eigen::Vector2d panel_second_derivative(const Interface2D& iface,
                                               int                panel,
                                               double             s)
{
    double ddN[3];
    p2_shape_second_deriv(s, ddN);
    Eigen::Vector2d second = Eigen::Vector2d::Zero();
    for (int q = 0; q < 3; ++q)
        second += ddN[q] * iface.points().row(iface.point_index(panel, q)).transpose();
    return second;
}

inline Eigen::Vector2d panel_normal(const Interface2D& iface,
                                    int                panel,
                                    double             s)
{
    const Eigen::Vector2d tangent = panel_tangent(iface, panel, s);
    const double tlen = tangent.norm();

    double N[3];
    p2_shape(s, N);
    Eigen::Vector2d ref = Eigen::Vector2d::Zero();
    for (int q = 0; q < 3; ++q)
        ref += N[q] * iface.normals().row(iface.point_index(panel, q)).transpose();
    const double ref_len = ref.norm();

    if (tlen <= 1.0e-14) {
        if (ref_len > 1.0e-14)
            return ref / ref_len;
        return {1.0, 0.0};
    }

    Eigen::Vector2d normal(tangent[1] / tlen, -tangent[0] / tlen);
    if (ref_len > 1.0e-14 && normal.dot(ref) < 0.0)
        normal = -normal;
    return normal;
}

inline double panel_scalar(const Interface2D&     iface,
                           int                    panel,
                           const Eigen::VectorXd& values,
                           double                 s)
{
    double N[3];
    p2_shape(s, N);
    double value = 0.0;
    for (int q = 0; q < 3; ++q)
        value += N[q] * values[iface.point_index(panel, q)];
    return value;
}

inline double clamp_to_panel(double s)
{
    return std::max(-1.0, std::min(1.0, s));
}

} // namespace kfbim::geometry2d
