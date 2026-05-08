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
    , support_(build_laplace_correction_support_3d(
          grid_pair,
          "LaplaceQuadraticPatchCenterSpread3D"))
{
    if (projection_restrict_stencil_radius_ < 1)
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterSpread3D projection restrict stencil radius must be positive");
    if (correction_method_ == LaplaceCorrectionMethod3D::ProjectionPoint) {
        projection_cache_ =
            project_p2_grid_nodes_to_interface_3d(grid_pair_, support_.projection_nodes);
    }
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

        for (const LaplaceCrossingCorrectionOp& op : support_.crossing_ops) {
            const Eigen::Vector3d pt = node_coord(grid, op.correction_node);
            const int center_idx =
                grid_pair_.nearest_p2_expansion_center(op.correction_node);
            const double correction =
                evaluate_taylor_poly_3d(center_polys[center_idx], pt);
            rhs_correction[op.rhs_node] += static_cast<double>(op.side_delta)
                                         * correction
                                         * op.stencil_weight;
        }

        result.correction_polys = std::move(center_polys);
        return result;
    }

    if (correction_method_ == LaplaceCorrectionMethod3D::ProjectionPoint) {
        result.projection_cache = projection_cache_;

        const ProfileClock::time_point correction_start = ProfileClock::now();
        for (const LaplaceCrossingCorrectionOp& op : support_.crossing_ops) {
            const double correction =
                evaluate_projection_point_correction_3d(grid_pair_,
                                                        result.projection_cache,
                                                        op.correction_node,
                                                        result);
            rhs_correction[op.rhs_node] += static_cast<double>(op.side_delta)
                                         * correction
                                         * op.stencil_weight;
        }
        const double t_correction = seconds_since(correction_start);

        if (profile_projection_transfer_3d()) {
            std::printf("      projection support nodes=%zu crossing_edges=%zu restrict_sample_visits=%d correction %.3fs\n",
                        result.projection_cache.nodes().size(),
                        support_.crossing_ops.size(),
                        support_.restrict_sample_visits,
                        t_correction);
        }

        return result;
    }

    throw std::invalid_argument(
        "LaplaceQuadraticPatchCenterSpread3D unsupported correction method");
}

} // namespace kfbim
