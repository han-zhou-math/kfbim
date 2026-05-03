#include "laplace_spread_2d.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include "../local_cauchy/laplace_panel_solver_2d.hpp"

namespace kfbim {

namespace {

bool is_outer_boundary_node(const CartesianGrid2D& grid, int idx) {
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int i = idx % nx;
    const int j = idx / nx;
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

int side_from_label(int label) {
    return label == 0 ? 0 : 1;
}

double stencil_weight_for_neighbor(const CartesianGrid2D& grid, int neighbor_slot) {
    const auto h = grid.spacing();
    if (neighbor_slot == 0 || neighbor_slot == 1)
        return 1.0 / (h[0] * h[0]);
    return 1.0 / (h[1] * h[1]);
}

Eigen::Vector2d node_coord(const CartesianGrid2D& grid, int idx) {
    const auto c = grid.coord(idx);
    return {c[0], c[1]};
}

double max_grid_spacing(const CartesianGrid2D& grid) {
    const auto h = grid.spacing();
    return std::max(h[0], h[1]);
}

LocalPoly2D center_poly(const PanelCenterCauchyResult2D& cauchy, int idx) {
    LocalPoly2D poly;
    poly.center = cauchy.centers.row(idx).transpose();
    poly.coeffs.resize(6);
    poly.coeffs << cauchy.C[idx],
                   cauchy.Cx[idx],
                   cauchy.Cy[idx],
                   cauchy.Cxx[idx],
                   cauchy.Cxy[idx],
                   cauchy.Cyy[idx];
    return poly;
}

int nearest_center_index(const PanelCenterCauchyResult2D& cauchy,
                         Eigen::Vector2d                 pt)
{
    int best_idx = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < cauchy.centers.rows(); ++i) {
        const double dx = pt[0] - cauchy.centers(i, 0);
        const double dy = pt[1] - cauchy.centers(i, 1);
        const double dist2 = dx * dx + dy * dy;
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best_idx = i;
        }
    }
    return best_idx;
}

std::vector<int> build_nearest_center_map(const GridPair2D&              grid_pair,
                                          const PanelCenterCauchyResult2D& cauchy,
                                          double                         band_radius)
{
    const auto& grid = grid_pair.grid();
    std::vector<int> nearest(grid.num_dofs(), -1);
    for (int idx : grid_pair.near_interface_nodes(band_radius))
        nearest[idx] = nearest_center_index(cauchy, node_coord(grid, idx));
    return nearest;
}

int nearest_center_for_grid_node(const CartesianGrid2D&            grid,
                                 const PanelCenterCauchyResult2D& cauchy,
                                 const std::vector<int>&          nearest_center_map,
                                 int                              idx)
{
    if (idx >= 0 && idx < static_cast<int>(nearest_center_map.size())
        && nearest_center_map[idx] >= 0) {
        return nearest_center_map[idx];
    }
    return nearest_center_index(cauchy, node_coord(grid, idx));
}

} // namespace

LaplacePanelSpread2D::LaplacePanelSpread2D(const GridPair2D& grid_pair,
                                           double            kappa)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
{}

std::vector<LocalPoly2D> LaplacePanelSpread2D::apply(
    const std::vector<LaplaceJumpData2D>& jumps,
    Eigen::VectorXd&                      rhs_correction) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_grid = grid.num_dofs();
    const int n_iface = iface.num_points();

    if (iface.points_per_panel() != 3)
        throw std::invalid_argument("LaplacePanelSpread2D requires 3 interface points per panel");
    if (iface.panel_node_layout() != PanelNodeLayout2D::LegacyGaussLegendre)
        throw std::invalid_argument("LaplacePanelSpread2D is legacy and requires Gauss-Legendre panel nodes");
    if (static_cast<int>(jumps.size()) != n_iface)
        throw std::invalid_argument("LaplacePanelSpread2D jumps size must equal interface point count");
    if (rhs_correction.size() != n_grid)
        throw std::invalid_argument("LaplacePanelSpread2D rhs_correction size must equal grid DOF count");

    Eigen::VectorXd u_jump(n_iface);
    Eigen::VectorXd un_jump(n_iface);
    Eigen::VectorXd rhs_jump(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        if (jumps[q].rhs_derivs.size() < 1)
            throw std::invalid_argument("LaplacePanelSpread2D requires rhs_derivs[0] at every interface point");
        u_jump[q] = jumps[q].u_jump;
        un_jump[q] = jumps[q].un_jump;
        rhs_jump[q] = jumps[q].rhs_derivs[0];
    }

    const PanelCauchyResult2D cauchy =
        laplace_panel_cauchy_2d(iface, u_jump, un_jump, rhs_jump, kappa_);

    std::vector<LocalPoly2D> polys(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        polys[q].center = iface.points().row(q).transpose();
        polys[q].coeffs.resize(6);
        polys[q].coeffs << cauchy.C[q],
                           cauchy.Cx[q],
                           cauchy.Cy[q],
                           cauchy.Cxx[q],
                           cauchy.Cxy[q],
                           cauchy.Cyy[q];
    }

    for (int n = 0; n < n_grid; ++n) {
        if (is_outer_boundary_node(grid, n))
            continue;

        const int side_n = side_from_label(grid_pair_.domain_label(n));
        const auto neighbors = grid.neighbors(n);

        for (int slot = 0; slot < 4; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || is_outer_boundary_node(grid, nb))
                continue;

            const int side_nb = side_from_label(grid_pair_.domain_label(nb));
            if (side_nb == side_n)
                continue;

            const int q = grid_pair_.closest_interface_point(nb);
            const double correction = evaluate_taylor_poly_2d(polys[q], node_coord(grid, nb));
            rhs_correction[n] += static_cast<double>(side_n - side_nb)
                                 * correction
                                 * stencil_weight_for_neighbor(grid, slot);
        }
    }

    return polys;
}

LaplaceLobattoCenterSpread2D::LaplaceLobattoCenterSpread2D(
    const GridPair2D& grid_pair,
    double            kappa)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
{}

std::vector<LocalPoly2D> LaplaceLobattoCenterSpread2D::apply(
    const std::vector<LaplaceJumpData2D>& jumps,
    Eigen::VectorXd&                      rhs_correction) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_grid = grid.num_dofs();
    const int n_iface = iface.num_points();

    if (iface.points_per_panel() != 3)
        throw std::invalid_argument("LaplaceLobattoCenterSpread2D requires 3 interface points per panel");
    if (iface.panel_node_layout() != PanelNodeLayout2D::ChebyshevLobatto)
        throw std::invalid_argument("LaplaceLobattoCenterSpread2D requires Chebyshev-Lobatto panel nodes");
    if (static_cast<int>(jumps.size()) != n_iface)
        throw std::invalid_argument("LaplaceLobattoCenterSpread2D jumps size must equal interface point count");
    if (rhs_correction.size() != n_grid)
        throw std::invalid_argument("LaplaceLobattoCenterSpread2D rhs_correction size must equal grid DOF count");

    Eigen::VectorXd u_jump(n_iface);
    Eigen::VectorXd un_jump(n_iface);
    Eigen::VectorXd rhs_jump(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        if (jumps[q].rhs_derivs.size() < 1)
            throw std::invalid_argument("LaplaceLobattoCenterSpread2D requires rhs_derivs[0] at every interface point");
        u_jump[q] = jumps[q].u_jump;
        un_jump[q] = jumps[q].un_jump;
        rhs_jump[q] = jumps[q].rhs_derivs[0];
    }

    const PanelCenterCauchyResult2D cauchy =
        laplace_panel_lobatto_center_cauchy_2d(iface, u_jump, un_jump, rhs_jump, kappa_);

    std::vector<LocalPoly2D> center_polys(cauchy.centers.rows());
    for (int i = 0; i < cauchy.centers.rows(); ++i)
        center_polys[i] = center_poly(cauchy, i);

    const double band_radius = 2.0 * std::sqrt(2.0) * max_grid_spacing(grid);
    const std::vector<int> nearest_center_for_node =
        build_nearest_center_map(grid_pair_, cauchy, band_radius);

    for (int n = 0; n < n_grid; ++n) {
        if (is_outer_boundary_node(grid, n))
            continue;

        const int side_n = side_from_label(grid_pair_.domain_label(n));
        const auto neighbors = grid.neighbors(n);

        for (int slot = 0; slot < 4; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || is_outer_boundary_node(grid, nb))
                continue;

            const int side_nb = side_from_label(grid_pair_.domain_label(nb));
            if (side_nb == side_n)
                continue;

            const Eigen::Vector2d pt = node_coord(grid, nb);
            const int center_idx =
                nearest_center_for_grid_node(grid, cauchy,
                                             nearest_center_for_node, nb);
            const double correction = evaluate_taylor_poly_2d(center_polys[center_idx], pt);
            rhs_correction[n] += static_cast<double>(side_n - side_nb)
                                 * correction
                                 * stencil_weight_for_neighbor(grid, slot);
        }
    }

    return center_polys;
}

} // namespace kfbim
