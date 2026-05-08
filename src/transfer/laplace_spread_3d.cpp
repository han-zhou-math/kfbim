#include "laplace_spread_3d.hpp"

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

LocalPoly3D center_poly(const PatchCenterCauchyResult3D& cauchy, int idx)
{
    LocalPoly3D poly;
    poly.center = cauchy.centers.row(idx).transpose();
    poly.coeffs = cauchy.coeffs.row(idx).transpose();
    return poly;
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
                const int center_idx = grid_pair_.nearest_p2_expansion_center(nb);
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
