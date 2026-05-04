#include "laplace_restrict_2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kfbim {

namespace {

constexpr int kLobattoExpansionCentersPerPanel = 4;

Eigen::Vector2d interface_point(const Interface2D& iface, int q) {
    return iface.points().row(q).transpose();
}

Eigen::Vector2d grid_point(const CartesianGrid2D& grid, int idx) {
    const auto c = grid.coord(idx);
    return {c[0], c[1]};
}

double max_grid_spacing(const CartesianGrid2D& grid) {
    const auto h = grid.spacing();
    return std::max(h[0], h[1]);
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

LaplaceLobattoCenterRestrict2D::LaplaceLobattoCenterRestrict2D(
    const GridPair2D& grid_pair,
    int               stencil_radius)
    : grid_pair_(grid_pair)
    , stencil_radius_(stencil_radius)
{
    if (stencil_radius_ < 1)
        throw std::invalid_argument("LaplaceLobattoCenterRestrict2D stencil_radius must be positive");
}

std::vector<LocalPoly2D> LaplaceLobattoCenterRestrict2D::apply(
    const Eigen::VectorXd&          bulk_solution,
    const std::vector<LocalPoly2D>& correction_polys) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();
    const int expected_centers = kLobattoExpansionCentersPerPanel * iface.num_panels();

    if (bulk_solution.size() != grid.num_dofs())
        throw std::invalid_argument("LaplaceLobattoCenterRestrict2D bulk_solution size must equal grid DOF count");
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout() != PanelNodeLayout2D::ChebyshevLobatto) {
        throw std::invalid_argument("LaplaceLobattoCenterRestrict2D requires Chebyshev-Lobatto 3-point panels");
    }
    if (static_cast<int>(correction_polys.size()) != expected_centers) {
        throw std::invalid_argument("LaplaceLobattoCenterRestrict2D correction_polys size must equal 4*num_panels");
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

LocalPoly2D LaplaceLobattoCenterRestrict2D::fit_at_interface_point(
    const Eigen::VectorXd&          bulk_solution,
    int                             q,
    const std::vector<LocalPoly2D>& center_polys,
    const std::vector<int>&         nearest_center_for_grid_node_map) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];

    const int closest = grid_pair_.closest_bulk_node(q);
    const int ic = closest % nx;
    const int jc = closest / nx;

    struct Sample {
        int idx;
        double dist2;
    };
    std::vector<Sample> samples;
    samples.reserve((2 * stencil_radius_ + 1) * (2 * stencil_radius_ + 1));

    const Eigen::Vector2d center = interface_point(iface, q);
    for (int j = std::max(0, jc - stencil_radius_);
         j <= std::min(ny - 1, jc + stencil_radius_);
         ++j) {
        for (int i = std::max(0, ic - stencil_radius_);
             i <= std::min(nx - 1, ic + stencil_radius_);
             ++i) {
            const int idx = grid.index(i, j);
            const Eigen::Vector2d pt = grid_point(grid, idx);
            samples.push_back({idx, (pt - center).squaredNorm()});
        }
    }

    if (samples.size() < 6)
        throw std::runtime_error("LaplaceLobattoCenterRestrict2D needs at least 6 stencil samples");

    std::sort(samples.begin(), samples.end(),
              [](const Sample& a, const Sample& b) {
                  if (a.dist2 == b.dist2)
                      return a.idx < b.idx;
                  return a.dist2 < b.dist2;
              });

    const int n_rows = static_cast<int>(samples.size());
    Eigen::MatrixXd A(n_rows, 6);
    Eigen::VectorXd rhs(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        const int idx = samples[r].idx;
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

    LocalPoly2D poly;
    poly.center = center;
    poly.coeffs = A.colPivHouseholderQr().solve(rhs);
    return poly;
}

} // namespace kfbim
