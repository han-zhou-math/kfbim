#include "laplace_restrict_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "../grid/structured_grid_ops.hpp"

namespace kfbim {

namespace {

constexpr int kQuadraticExpansionCentersPerPanel = 4;

Eigen::Vector2d interface_point(const Interface2D& iface, int q) {
    return iface.points().row(q).transpose();
}

Eigen::Vector2d grid_point(const CartesianGrid2D& grid, int idx) {
    return structured_grid::point(grid, idx);
}

double max_grid_spacing(const CartesianGrid2D& grid) {
    return structured_grid::max_spacing(grid);
}

int nearest_poly_index(const std::vector<LocalPoly2D>& center_polys,
                       Eigen::Vector2d                 pt)
{
    int best_idx = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < static_cast<int>(center_polys.size()); ++i) {
        const double dist2 = (pt - center_polys[i].center).squaredNorm();
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_idx = i;
        }
    }
    return best_idx;
}

std::vector<int> build_nearest_center_map(const GridPair2D&                 grid_pair,
                                          const std::vector<LocalPoly2D>&   center_polys,
                                          double                            band_radius)
{
    const auto& grid = grid_pair.grid();
    std::vector<int> nearest(grid.num_dofs(), -1);
    for (int idx : grid_pair.near_interface_nodes(band_radius))
        nearest[idx] = nearest_poly_index(center_polys, grid_point(grid, idx));
    return nearest;
}

int nearest_center_for_grid_node(const CartesianGrid2D&           grid,
                                 const std::vector<LocalPoly2D>& center_polys,
                                 const std::vector<int>&         nearest_center_map,
                                 int                             idx)
{
    if (idx >= 0 && idx < static_cast<int>(nearest_center_map.size())
        && nearest_center_map[idx] >= 0) {
        return nearest_center_map[idx];
    }
    return nearest_poly_index(center_polys, grid_point(grid, idx));
}

} // namespace

LaplaceQuadraticPanelCenterRestrict2D::LaplaceQuadraticPanelCenterRestrict2D(
    const GridPair2D& grid_pair,
    int               stencil_radius)
    : grid_pair_(grid_pair)
    , stencil_radius_(stencil_radius)
{
    if (stencil_radius_ < 1)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D stencil_radius must be positive");
}

std::vector<LocalPoly2D> LaplaceQuadraticPanelCenterRestrict2D::apply(
    const Eigen::VectorXd&          bulk_solution,
    const std::vector<LocalPoly2D>& correction_polys) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();
    const int expected_centers = kQuadraticExpansionCentersPerPanel * iface.num_panels();

    if (bulk_solution.size() != grid.num_dofs())
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D bulk_solution size must equal grid DOF count");
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout() != PanelNodeLayout2D::QuadraticLagrange) {
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D requires P2 quadratic 3-point panels");
    }
    if (static_cast<int>(correction_polys.size()) != expected_centers) {
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D correction_polys size must equal 4*num_panels");
    }

    const double band_radius = (static_cast<double>(stencil_radius_) + 1.0)
                               * std::sqrt(2.0) * max_grid_spacing(grid);
    const std::vector<int> nearest_center_for_node =
        build_nearest_center_map(grid_pair_, correction_polys, band_radius);

    std::vector<LocalPoly2D> result(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        result[q] = fit_at_interface_point(bulk_solution, q, correction_polys,
                                           nearest_center_for_node);
    }

    return result;
}

LocalPoly2D LaplaceQuadraticPanelCenterRestrict2D::fit_at_interface_point(
    const Eigen::VectorXd&          bulk_solution,
    int                             q,
    const std::vector<LocalPoly2D>& center_polys,
    const std::vector<int>&         nearest_center_for_grid_node_map) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int closest = grid_pair_.closest_bulk_node(q);
    const Eigen::Vector2d center = interface_point(iface, q);
    const std::array<int, 6> stencil_nodes =
        structured_grid::quadratic_restrict_stencil_nodes_2d(
            "LaplaceQuadraticPanelCenterRestrict2D",
            grid,
            closest,
            center);

    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 1> rhs;
    for (int r = 0; r < static_cast<int>(stencil_nodes.size()); ++r) {
        const int idx = stencil_nodes[r];
        const Eigen::Vector2d pt = grid_point(grid, idx);
        const double dx = pt[0] - center[0];
        const double dy = pt[1] - center[1];
        A(r, 0) = 1.0;
        A(r, 1) = dx;
        A(r, 2) = dy;
        A(r, 3) = 0.5 * dx * dx;
        A(r, 4) = dx * dy;
        A(r, 5) = 0.5 * dy * dy;

        double val = bulk_solution[idx];
        const int center_idx =
            nearest_center_for_grid_node(grid, center_polys,
                                         nearest_center_for_grid_node_map,
                                         idx);
        const double correction =
            0.5 * evaluate_taylor_poly_2d(center_polys[center_idx], pt);
        if (grid_pair_.domain_label(idx) == 0)
            val += correction;
        else
            val -= correction;
        rhs[r] = val;
    }

    Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(A);
    if (!lu.isInvertible()) {
        throw std::runtime_error(
            "LaplaceQuadraticPanelCenterRestrict2D singular fixed quadratic interpolation stencil");
    }

    LocalPoly2D poly;
    poly.center = center;
    poly.coeffs = lu.solve(rhs);
    return poly;
}

} // namespace kfbim
