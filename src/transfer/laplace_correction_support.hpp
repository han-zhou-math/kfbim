#pragma once

#include <algorithm>
#include <array>
#include <vector>

#include "../geometry/grid_pair_2d.hpp"
#include "../geometry/grid_pair_3d.hpp"
#include "../grid/structured_grid_ops.hpp"

namespace kfbim {

struct LaplaceCrossingCorrectionOp {
    int rhs_node = -1;
    int correction_node = -1;
    int neighbor_slot = -1;
    int side_delta = 0;
    double stencil_weight = 0.0;
};

struct LaplaceCorrectionSupport2D {
    std::vector<LaplaceCrossingCorrectionOp> crossing_ops;
    std::vector<std::array<int, 6>> restrict_stencils;
    std::vector<int> projection_nodes;
    int restrict_sample_visits = 0;
};

struct LaplaceCorrectionSupport3D {
    std::vector<LaplaceCrossingCorrectionOp> crossing_ops;
    std::vector<std::array<int, 10>> restrict_stencils;
    std::vector<int> projection_nodes;
    int restrict_sample_visits = 0;
};

namespace laplace_correction_support_detail {

inline int side_from_label(int label)
{
    return label == 0 ? 0 : 1;
}

inline Eigen::Vector2d interface_point(const Interface2D& iface, int q)
{
    return iface.points().row(q).transpose();
}

inline Eigen::Vector3d interface_point(const Interface3D& iface, int q)
{
    return iface.points().row(q).transpose();
}

inline void append_unique_projection_nodes(const std::vector<char>& mark,
                                           std::vector<int>&        nodes)
{
    nodes.clear();
    for (int idx = 0; idx < static_cast<int>(mark.size()); ++idx) {
        if (mark[idx])
            nodes.push_back(idx);
    }
}

} // namespace laplace_correction_support_detail

inline LaplaceCorrectionSupport2D build_laplace_correction_support_2d(
    const GridPair2D& grid_pair,
    const char*       context)
{
    const auto& grid = grid_pair.grid();
    const auto& iface = grid_pair.interface();
    const int n_grid = grid.num_dofs();
    std::vector<char> needs_c(n_grid, 0);

    LaplaceCorrectionSupport2D support;
    for (int n = 0; n < n_grid; ++n) {
        if (structured_grid::is_boundary_node(grid, n))
            continue;

        const int side_n =
            laplace_correction_support_detail::side_from_label(
                grid_pair.domain_label(n));
        const auto neighbors = grid.neighbors(n);
        for (int slot = 0; slot < 4; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || structured_grid::is_boundary_node(grid, nb))
                continue;

            const int side_nb =
                laplace_correction_support_detail::side_from_label(
                    grid_pair.domain_label(nb));
            if (side_nb == side_n)
                continue;

            needs_c[nb] = 1;
            support.crossing_ops.push_back(
                {n,
                 nb,
                 slot,
                 side_n - side_nb,
                 structured_grid::stencil_weight_for_neighbor(grid, slot)});
        }
    }

    support.restrict_stencils.reserve(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const int closest = grid_pair.closest_bulk_node(q);
        const auto stencil =
            structured_grid::quadratic_restrict_stencil_nodes_2d(
                context,
                grid,
                closest,
                laplace_correction_support_detail::interface_point(iface, q));
        support.restrict_stencils.push_back(stencil);
        for (int node : stencil) {
            needs_c[node] = 1;
            ++support.restrict_sample_visits;
        }
    }

    laplace_correction_support_detail::append_unique_projection_nodes(
        needs_c, support.projection_nodes);
    return support;
}

inline LaplaceCorrectionSupport3D build_laplace_correction_support_3d(
    const GridPair3D& grid_pair,
    const char*       context)
{
    const auto& grid = grid_pair.grid();
    const auto& iface = grid_pair.interface();
    const int n_grid = grid.num_dofs();
    std::vector<char> needs_c(n_grid, 0);

    LaplaceCorrectionSupport3D support;
    for (int n = 0; n < n_grid; ++n) {
        if (structured_grid::is_boundary_node(grid, n))
            continue;

        const int side_n =
            laplace_correction_support_detail::side_from_label(
                grid_pair.domain_label(n));
        const auto neighbors = grid.neighbors(n);
        for (int slot = 0; slot < 6; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || structured_grid::is_boundary_node(grid, nb))
                continue;

            const int side_nb =
                laplace_correction_support_detail::side_from_label(
                    grid_pair.domain_label(nb));
            if (side_nb == side_n)
                continue;

            needs_c[nb] = 1;
            support.crossing_ops.push_back(
                {n,
                 nb,
                 slot,
                 side_n - side_nb,
                 structured_grid::stencil_weight_for_neighbor(grid, slot)});
        }
    }

    support.restrict_stencils.reserve(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const int closest = grid_pair.closest_bulk_node(q);
        const auto stencil =
            structured_grid::quadratic_restrict_stencil_nodes_3d(
                context,
                grid,
                closest,
                laplace_correction_support_detail::interface_point(iface, q));
        support.restrict_stencils.push_back(stencil);
        for (int node : stencil) {
            needs_c[node] = 1;
            ++support.restrict_sample_visits;
        }
    }

    laplace_correction_support_detail::append_unique_projection_nodes(
        needs_c, support.projection_nodes);
    return support;
}

} // namespace kfbim
