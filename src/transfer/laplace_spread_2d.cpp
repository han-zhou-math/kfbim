#include "laplace_spread_2d.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include "../geometry/p2_projection_2d.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "../local_cauchy/laplace_panel_solver_2d.hpp"
#include "laplace_projection_correction_2d.hpp"

namespace kfbim {

namespace {

bool is_outer_boundary_node(const CartesianGrid2D& grid, int idx) {
    return structured_grid::is_boundary_node(grid, idx);
}

int side_from_label(int label) {
    return label == 0 ? 0 : 1;
}

double stencil_weight_for_neighbor(const CartesianGrid2D& grid, int neighbor_slot) {
    return structured_grid::stencil_weight_for_neighbor(grid, neighbor_slot);
}

Eigen::Vector2d node_coord(const CartesianGrid2D& grid, int idx) {
    return structured_grid::point(grid, idx);
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

} // namespace

LaplacePanelSpread2D::LaplacePanelSpread2D(const GridPair2D& grid_pair,
                                           double            kappa)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
{}

LaplaceSpreadResult2D LaplacePanelSpread2D::apply(
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

    LaplaceSpreadResult2D result;
    result.correction_method = LaplaceCorrectionMethod2D::NearestExpansionCenter;
    result.correction_polys = std::move(polys);
    return result;
}

LaplaceQuadraticPanelCenterSpread2D::LaplaceQuadraticPanelCenterSpread2D(
    const GridPair2D& grid_pair,
    double            kappa,
    LaplaceCorrectionMethod2D correction_method,
    int               projection_restrict_stencil_radius)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
    , correction_method_(correction_method)
    , projection_restrict_stencil_radius_(projection_restrict_stencil_radius)
    , support_(build_laplace_correction_support_2d(
          grid_pair,
          "LaplaceQuadraticPanelCenterSpread2D"))
{
    if (projection_restrict_stencil_radius_ < 1) {
        throw std::invalid_argument(
            "LaplaceQuadraticPanelCenterSpread2D projection restrict stencil radius must be positive");
    }
    if (correction_method_ == LaplaceCorrectionMethod2D::ProjectionPoint) {
        projection_cache_ =
            project_p2_grid_nodes_to_interface_2d(grid_pair_, support_.projection_nodes);
    }
}

LaplaceSpreadResult2D LaplaceQuadraticPanelCenterSpread2D::apply(
    const std::vector<LaplaceJumpData2D>& jumps,
    Eigen::VectorXd&                      rhs_correction) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_grid = grid.num_dofs();
    const int n_iface = iface.num_points();

    if (iface.points_per_panel() != 3)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D requires 3 interface points per panel");
    if (iface.panel_node_layout() != PanelNodeLayout2D::QuadraticLagrange)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D requires P2 quadratic panel nodes");
    if (static_cast<int>(jumps.size()) != n_iface)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D jumps size must equal interface point count");
    if (rhs_correction.size() != n_grid)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D rhs_correction size must equal grid DOF count");

    Eigen::VectorXd u_jump(n_iface);
    Eigen::VectorXd un_jump(n_iface);
    Eigen::VectorXd rhs_jump(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        if (jumps[q].rhs_derivs.size() < 1)
            throw std::invalid_argument("LaplaceQuadraticPanelCenterSpread2D requires rhs_derivs[0] at every interface point");
        u_jump[q] = jumps[q].u_jump;
        un_jump[q] = jumps[q].un_jump;
        rhs_jump[q] = jumps[q].rhs_derivs[0];
    }

    LaplaceSpreadResult2D result;
    result.correction_method = correction_method_;
    result.u_jump = u_jump;
    result.un_jump = un_jump;
    result.rhs_jump = rhs_jump;
    result.alpha = kappa_;

    if (correction_method_ == LaplaceCorrectionMethod2D::NearestExpansionCenter) {
        const PanelCenterCauchyResult2D cauchy =
            laplace_panel_quadratic_center_cauchy_2d(iface, u_jump, un_jump, rhs_jump, kappa_);

        std::vector<LocalPoly2D> center_polys(cauchy.centers.rows());
        for (int i = 0; i < cauchy.centers.rows(); ++i)
            center_polys[i] = center_poly(cauchy, i);

        for (const LaplaceCrossingCorrectionOp& op : support_.crossing_ops) {
            const Eigen::Vector2d pt = node_coord(grid, op.correction_node);
            const int center_idx =
                grid_pair_.nearest_p2_expansion_center(op.correction_node);
            const double correction =
                evaluate_taylor_poly_2d(center_polys[center_idx], pt);
            rhs_correction[op.rhs_node] += static_cast<double>(op.side_delta)
                                         * correction
                                         * op.stencil_weight;
        }

        result.correction_polys = std::move(center_polys);
        return result;
    }

    if (correction_method_ == LaplaceCorrectionMethod2D::ProjectionPoint) {
        result.projection_cache = projection_cache_;
        for (const LaplaceCrossingCorrectionOp& op : support_.crossing_ops) {
            const double correction =
                evaluate_projection_point_correction_2d(grid_pair_,
                                                        result.projection_cache,
                                                        op.correction_node,
                                                        result);
            rhs_correction[op.rhs_node] += static_cast<double>(op.side_delta)
                                         * correction
                                         * op.stencil_weight;
        }
        return result;
    }

    throw std::invalid_argument(
        "LaplaceQuadraticPanelCenterSpread2D unsupported correction method");
}

} // namespace kfbim
