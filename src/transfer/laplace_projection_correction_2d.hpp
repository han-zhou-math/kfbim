#pragma once

#include <stdexcept>

#include "i_spread.hpp"
#include "../geometry/p2_curve_2d.hpp"

namespace kfbim {

inline void require_projection_curve_data(const Interface2D&           iface,
                                          const LaplaceSpreadResult2D& spread_result)
{
    const int n_iface = iface.num_points();
    if (spread_result.u_jump.size() != n_iface
        || spread_result.un_jump.size() != n_iface
        || spread_result.rhs_jump.size() != n_iface) {
        throw std::invalid_argument(
            "projection-point correction requires curve jump data");
    }
}

inline double evaluate_projection_point_correction_2d(
    const Interface2D&         iface,
    const CurveProjection2D&   projection,
    const LaplaceSpreadResult2D& spread_result)
{
    require_projection_curve_data(iface, spread_result);
    if (projection.panel < 0 || projection.panel >= iface.num_panels())
        throw std::invalid_argument("projection-point correction has invalid panel");

    const double s = projection.local_s;
    const double c = geometry2d::panel_scalar(iface,
                                              projection.panel,
                                              spread_result.u_jump,
                                              s);
    const double cn = geometry2d::panel_scalar(iface,
                                               projection.panel,
                                               spread_result.un_jump,
                                               s);
    const double rhs_jump = geometry2d::panel_scalar(iface,
                                                     projection.panel,
                                                     spread_result.rhs_jump,
                                                     s);
    const double normal_divergence =
        geometry2d::panel_normal_divergence(iface, projection.panel, s);
    const double surface_laplace =
        geometry2d::panel_laplace_beltrami_scalar(iface,
                                                  projection.panel,
                                                  spread_result.u_jump,
                                                  s);
    const double cnn = spread_result.alpha * c
                     - rhs_jump
                     - normal_divergence * cn
                     - surface_laplace;

    const double d = projection.signed_distance;
    return c + d * cn + 0.5 * d * d * cnn;
}

inline double evaluate_projection_point_correction_2d(
    const GridPair2D&             grid_pair,
    const NarrowBandProjection2D& band,
    int                           grid_node,
    const LaplaceSpreadResult2D&  spread_result)
{
    if (!band.has_projection(grid_node)) {
        throw std::runtime_error(
            "projection-point correction missing narrow-band projection");
    }
    return evaluate_projection_point_correction_2d(grid_pair.interface(),
                                                   band.projection(grid_node),
                                                   spread_result);
}

} // namespace kfbim
