#include "laplace_spread_3d.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../local_cauchy/laplace_patch_solver_3d.hpp"

namespace kfbim {

namespace {

using K3 = CGAL::Exact_predicates_inexact_constructions_kernel;
using CPoint3 = K3::Point_3;
using CPointWithIndex3 = std::pair<CPoint3, int>;
using CSearchBaseTraits3 = CGAL::Search_traits_3<K3>;
using CSearchTraits3 =
    CGAL::Search_traits_adapter<CPointWithIndex3,
                                CGAL::First_of_pair_property_map<CPointWithIndex3>,
                                CSearchBaseTraits3>;
using CNeighborSearch3 = CGAL::Orthogonal_k_neighbor_search<CSearchTraits3>;
using CSearchTree3 = CNeighborSearch3::Tree;

bool is_outer_boundary_node(const CartesianGrid3D& grid, int idx)
{
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int nxy = nx * ny;
    const int k = idx / nxy;
    const int rem = idx % nxy;
    const int j = rem / nx;
    const int i = rem % nx;
    return i == 0 || i == nx - 1
        || j == 0 || j == ny - 1
        || k == 0 || k == nz - 1;
}

int side_from_label(int label)
{
    return label == 0 ? 0 : 1;
}

double stencil_weight_for_neighbor(const CartesianGrid3D& grid, int neighbor_slot)
{
    const auto h = grid.spacing();
    if (neighbor_slot == 0 || neighbor_slot == 1)
        return 1.0 / (h[0] * h[0]);
    if (neighbor_slot == 2 || neighbor_slot == 3)
        return 1.0 / (h[1] * h[1]);
    return 1.0 / (h[2] * h[2]);
}

Eigen::Vector3d node_coord(const CartesianGrid3D& grid, int idx)
{
    const auto c = grid.coord(idx);
    return {c[0], c[1], c[2]};
}

double max_grid_spacing(const CartesianGrid3D& grid)
{
    const auto h = grid.spacing();
    return std::max(h[0], std::max(h[1], h[2]));
}

LocalPoly3D center_poly(const PatchCenterCauchyResult3D& cauchy, int idx)
{
    LocalPoly3D poly;
    poly.center = cauchy.centers.row(idx).transpose();
    poly.coeffs = cauchy.coeffs.row(idx).transpose();
    return poly;
}

class NearestCenterFinder3D {
public:
    explicit NearestCenterFinder3D(const std::vector<LocalPoly3D>& center_polys)
    {
        if (center_polys.empty())
            throw std::invalid_argument("NearestCenterFinder3D requires centers");

        points_.reserve(center_polys.size());
        for (int i = 0; i < static_cast<int>(center_polys.size()); ++i) {
            const Eigen::Vector3d& c = center_polys[i].center;
            points_.emplace_back(CPoint3(c[0], c[1], c[2]), i);
        }
        tree_ = std::make_unique<CSearchTree3>(points_.begin(), points_.end());
    }

    int nearest(Eigen::Vector3d pt) const
    {
        CNeighborSearch3 search(*tree_, CPoint3(pt[0], pt[1], pt[2]), 1);
        return search.begin()->first.second;
    }

private:
    std::vector<CPointWithIndex3> points_;
    std::unique_ptr<CSearchTree3> tree_;
};

int nearest_center_index(const NearestCenterFinder3D& center_finder,
                         Eigen::Vector3d             pt)
{
    return center_finder.nearest(pt);
}

std::vector<int> build_nearest_center_map(const GridPair3D&               grid_pair,
                                          const NearestCenterFinder3D&    center_finder,
                                          double                          band_radius)
{
    const auto& grid = grid_pair.grid();
    std::vector<int> nearest(grid.num_dofs(), -1);
    for (int idx : grid_pair.near_interface_nodes(band_radius))
        nearest[idx] = nearest_center_index(center_finder, node_coord(grid, idx));
    return nearest;
}

int nearest_center_for_grid_node(const CartesianGrid3D&           grid,
                                 const NearestCenterFinder3D&    center_finder,
                                 const std::vector<int>&         nearest_center_map,
                                 int                             idx)
{
    if (idx >= 0 && idx < static_cast<int>(nearest_center_map.size())
        && nearest_center_map[idx] >= 0) {
        return nearest_center_map[idx];
    }
    return nearest_center_index(center_finder, node_coord(grid, idx));
}

} // namespace

LaplaceQuadraticPatchCenterSpread3D::LaplaceQuadraticPatchCenterSpread3D(
    const GridPair3D& grid_pair,
    double            kappa)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
{}

std::vector<LocalPoly3D> LaplaceQuadraticPatchCenterSpread3D::apply(
    const std::vector<LaplaceJumpData3D>& jumps,
    Eigen::VectorXd&                      rhs_correction) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_grid = grid.num_dofs();
    const int n_iface = iface.num_points();

    if (iface.points_per_panel() != 6
        || iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange) {
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterSpread3D requires P2 QuadraticLagrange panels");
    }
    if (static_cast<int>(jumps.size()) != n_iface)
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterSpread3D jumps size must equal interface point count");
    if (rhs_correction.size() != n_grid)
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterSpread3D rhs_correction size must equal grid DOF count");

    Eigen::VectorXd u_jump(n_iface);
    Eigen::VectorXd un_jump(n_iface);
    Eigen::VectorXd rhs_jump(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        if (jumps[q].rhs_derivs.size() < 1) {
            throw std::invalid_argument(
                "LaplaceQuadraticPatchCenterSpread3D requires rhs_derivs[0] at every interface point");
        }
        u_jump[q] = jumps[q].u_jump;
        un_jump[q] = jumps[q].un_jump;
        rhs_jump[q] = jumps[q].rhs_derivs[0];
    }

    const PatchCenterCauchyResult3D cauchy =
        laplace_p2_patch_center_cauchy_3d(iface, u_jump, un_jump, rhs_jump, kappa_);

    std::vector<LocalPoly3D> center_polys(cauchy.centers.rows());
    for (int i = 0; i < cauchy.centers.rows(); ++i)
        center_polys[i] = center_poly(cauchy, i);
    const NearestCenterFinder3D center_finder(center_polys);

    const double band_radius = 2.0 * std::sqrt(3.0) * max_grid_spacing(grid);
    const std::vector<int> nearest_center_for_node =
        build_nearest_center_map(grid_pair_, center_finder, band_radius);

    for (int n = 0; n < n_grid; ++n) {
        if (is_outer_boundary_node(grid, n))
            continue;

        const int side_n = side_from_label(grid_pair_.domain_label(n));
        const auto neighbors = grid.neighbors(n);

        for (int slot = 0; slot < 6; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || is_outer_boundary_node(grid, nb))
                continue;

            const int side_nb = side_from_label(grid_pair_.domain_label(nb));
            if (side_nb == side_n)
                continue;

            const Eigen::Vector3d pt = node_coord(grid, nb);
            const int center_idx =
                nearest_center_for_grid_node(grid,
                                             center_finder,
                                             nearest_center_for_node,
                                             nb);
            const double correction =
                evaluate_taylor_poly_3d(center_polys[center_idx], pt);
            rhs_correction[n] += static_cast<double>(side_n - side_nb)
                                 * correction
                                 * stencil_weight_for_neighbor(grid, slot);
        }
    }

    return center_polys;
}

} // namespace kfbim
