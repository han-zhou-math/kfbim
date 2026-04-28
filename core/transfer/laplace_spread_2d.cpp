#include "laplace_spread_2d.hpp"

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
            rhs_correction[n] += static_cast<double>(side_nb - side_n)
                                 * correction
                                 * stencil_weight_for_neighbor(grid, slot);
        }
    }

    return polys;
}

} // namespace kfbim
