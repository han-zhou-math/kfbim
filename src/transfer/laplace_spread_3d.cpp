#include "laplace_spread_3d.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../geometry/p2_projection_3d.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "../local_cauchy/laplace_patch_solver_3d.hpp"
#include "laplace_projection_correction_3d.hpp"

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
    return structured_grid::is_boundary_node(grid, idx);
}

int side_from_label(int label)
{
    return label == 0 ? 0 : 1;
}

double stencil_weight_for_neighbor(const CartesianGrid3D& grid, int neighbor_slot)
{
    return structured_grid::stencil_weight_for_neighbor(grid, neighbor_slot);
}

Eigen::Vector3d node_coord(const CartesianGrid3D& grid, int idx)
{
    return structured_grid::point(grid, idx);
}

double max_grid_spacing(const CartesianGrid3D& grid)
{
    return structured_grid::max_spacing(grid);
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

struct ProjectionSupport3D {
    std::vector<int> nodes;
    int crossing_edges = 0;
    int restrict_sample_visits = 0;
};

void mark_projection_restrict_samples(const GridPair3D& grid_pair,
                                      int               stencil_radius,
                                      std::vector<char>& needs_c,
                                      int&              sample_visits)
{
    const auto& grid = grid_pair.grid();
    const auto& iface = grid_pair.interface();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int nxy = nx * ny;

    for (int q = 0; q < iface.num_points(); ++q) {
        const int closest = grid_pair.closest_bulk_node(q);
        const int kc = closest / nxy;
        const int rem = closest % nxy;
        const int jc = rem / nx;
        const int ic = rem % nx;

        for (int k = std::max(0, kc - stencil_radius);
             k <= std::min(nz - 1, kc + stencil_radius);
             ++k) {
            for (int j = std::max(0, jc - stencil_radius);
                 j <= std::min(ny - 1, jc + stencil_radius);
                 ++j) {
                for (int i = std::max(0, ic - stencil_radius);
                     i <= std::min(nx - 1, ic + stencil_radius);
                     ++i) {
                    needs_c[grid.index(i, j, k)] = 1;
                    ++sample_visits;
                }
            }
        }
    }
}

ProjectionSupport3D build_projection_support(const GridPair3D& grid_pair,
                                             int               restrict_stencil_radius)
{
    const auto& grid = grid_pair.grid();
    const int n_grid = grid.num_dofs();
    std::vector<char> needs_c(n_grid, 0);

    ProjectionSupport3D support;
    for (int n = 0; n < n_grid; ++n) {
        if (is_outer_boundary_node(grid, n))
            continue;

        const int side_n = side_from_label(grid_pair.domain_label(n));
        const auto neighbors = grid.neighbors(n);
        for (int slot = 0; slot < 6; ++slot) {
            const int nb = neighbors[slot];
            if (nb < 0 || is_outer_boundary_node(grid, nb))
                continue;

            const int side_nb = side_from_label(grid_pair.domain_label(nb));
            if (side_nb == side_n)
                continue;
            needs_c[nb] = 1;
            ++support.crossing_edges;
        }
    }

    mark_projection_restrict_samples(grid_pair,
                                     restrict_stencil_radius,
                                     needs_c,
                                     support.restrict_sample_visits);

    support.nodes.reserve(n_grid / 8);
    for (int idx = 0; idx < n_grid; ++idx) {
        if (needs_c[idx])
            support.nodes.push_back(idx);
    }
    return support;
}

bool profile_projection_transfer_3d()
{
    return std::getenv("KFBIM_PROFILE_INTERFACE_3D") != nullptr
        || std::getenv("KFBIM_PROFILE_TRANSFER_3D") != nullptr;
}

using ProfileClock = std::chrono::steady_clock;

double seconds_since(ProfileClock::time_point start)
{
    return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

} // namespace

LaplaceQuadraticPatchCenterSpread3D::LaplaceQuadraticPatchCenterSpread3D(
    const GridPair3D& grid_pair,
    double            kappa,
    LaplaceCorrectionMethod3D correction_method,
    int projection_restrict_stencil_radius)
    : grid_pair_(grid_pair)
    , kappa_(kappa)
    , correction_method_(correction_method)
    , projection_restrict_stencil_radius_(projection_restrict_stencil_radius)
{
    if (projection_restrict_stencil_radius_ < 1)
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterSpread3D projection restrict stencil radius must be positive");
}

LaplaceSpreadResult3D LaplaceQuadraticPatchCenterSpread3D::apply(
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

    LaplaceSpreadResult3D result;
    result.correction_method = correction_method_;
    result.u_jump = u_jump;
    result.un_jump = un_jump;
    result.rhs_jump = rhs_jump;
    result.alpha = kappa_;

    const double band_radius = 2.0 * std::sqrt(3.0) * max_grid_spacing(grid);

    if (correction_method_ == LaplaceCorrectionMethod3D::NearestExpansionCenter) {
        const PatchCenterCauchyResult3D cauchy =
            laplace_p2_patch_center_cauchy_3d(iface,
                                              u_jump,
                                              un_jump,
                                              rhs_jump,
                                              kappa_);

        std::vector<LocalPoly3D> center_polys(cauchy.centers.rows());
        for (int i = 0; i < cauchy.centers.rows(); ++i)
            center_polys[i] = center_poly(cauchy, i);
        const NearestCenterFinder3D center_finder(center_polys);

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

        result.correction_polys = std::move(center_polys);
        return result;
    }

    if (correction_method_ == LaplaceCorrectionMethod3D::ProjectionPoint) {
        const ProjectionSupport3D support =
            build_projection_support(grid_pair_,
                                     projection_restrict_stencil_radius_);
        const ProfileClock::time_point projection_start = ProfileClock::now();
        result.projection_cache =
            project_p2_grid_nodes_to_interface_3d(grid_pair_, support.nodes);
        const double t_projection = seconds_since(projection_start);

        const ProfileClock::time_point correction_start = ProfileClock::now();
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

                const double correction =
                    evaluate_projection_point_correction_3d(grid_pair_,
                                                            result.projection_cache,
                                                            nb,
                                                            result);
                rhs_correction[n] += static_cast<double>(side_n - side_nb)
                                     * correction
                                     * stencil_weight_for_neighbor(grid, slot);
            }
        }
        const double t_correction = seconds_since(correction_start);

        if (profile_projection_transfer_3d()) {
            std::printf("      projection support nodes=%zu crossing_edges=%d restrict_sample_visits=%d project %.3fs correction %.3fs\n",
                        result.projection_cache.nodes().size(),
                        support.crossing_edges,
                        support.restrict_sample_visits,
                        t_projection,
                        t_correction);
        }

        return result;
    }

    throw std::invalid_argument(
        "LaplaceQuadraticPatchCenterSpread3D unsupported correction method");
}

} // namespace kfbim
