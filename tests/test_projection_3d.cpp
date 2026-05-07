#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "p2_sphere_fixture_3d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/p2_surface_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"

using namespace kfbim;
using namespace kfbim_test_3d;

namespace {

Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int idx)
{
    const auto c = grid.coord(idx);
    return {c[0], c[1], c[2]};
}

} // namespace

TEST_CASE("GridPair3D projection projects narrow-band nodes onto curved P2 panels",
          "[projection][3d]")
{
    const int N = 24;
    const Box3D box = standard_box();
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid3D grid(box.lower, {h, h, h}, {N, N, N}, DofLayout3D::Node);
    const Interface3D iface = make_p2_sphere(surface_subdivision_for_grid(N));
    GridPair3D gp(grid, iface);

    const double radius = 2.0 * std::sqrt(3.0) * h;
    const std::vector<int> baseline_nodes = gp.near_interface_nodes(radius);
    REQUIRE_FALSE(baseline_nodes.empty());

    std::vector<int> baseline_label(grid.num_dofs());
    std::vector<int> baseline_closest(grid.num_dofs());
    for (int n = 0; n < grid.num_dofs(); ++n) {
        baseline_label[n] = gp.domain_label(n);
        baseline_closest[n] = gp.closest_interface_point(n);
    }

    const NarrowBandProjection3D band = gp.project_near_interface_nodes(radius);

    REQUIRE(band.radius() == Catch::Approx(radius));
    REQUIRE(band.nodes() == baseline_nodes);
    REQUIRE(band.projections().size() == baseline_nodes.size());

    int outside_node = -1;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        if (!gp.is_near_interface(n, radius)) {
            outside_node = n;
            break;
        }
    }
    REQUIRE(outside_node >= 0);
    REQUIRE_FALSE(band.has_projection(outside_node));
    REQUIRE_THROWS_AS(band.projection(outside_node), std::out_of_range);

    double max_tangential_residual = 0.0;
    double max_edge_fallback_residual = 0.0;
    double max_point_mismatch = 0.0;
    double max_signed_distance_mismatch = 0.0;
    int converged_count = 0;
    int edge_fallback_count = 0;
    for (int node : baseline_nodes) {
        REQUIRE(band.has_projection(node));
        const SurfaceProjection3D& projection = band.projection(node);
        REQUIRE(projection.grid_node == node);
        REQUIRE(projection.panel >= 0);
        REQUIRE(projection.panel < iface.num_panels());
        REQUIRE(projection.component == 0);
        INFO("node=" << node
             << " panel=" << projection.panel
             << " bary=" << projection.barycentric.transpose()
             << " residual=" << projection.tangential_residual
             << " distance=" << projection.distance
             << " iterations=" << projection.iterations);

        const Eigen::Vector3d bary = projection.barycentric;
        REQUIRE(bary.sum() == Catch::Approx(1.0).margin(1.0e-12));
        REQUIRE(bary[0] >= -1.0e-10);
        REQUIRE(bary[1] >= -1.0e-10);
        REQUIRE(bary[2] >= -1.0e-10);

        const Eigen::Vector3d panel_point =
            geometry3d::panel_point(iface, projection.panel, bary);
        const Eigen::Vector3d panel_normal =
            geometry3d::panel_oriented_normal(iface, projection.panel, bary);
        const Eigen::Vector3d node_point = grid_point(grid, node);

        max_point_mismatch = std::max(max_point_mismatch,
                                      (panel_point - projection.point).norm());
        REQUIRE(projection.normal.norm() == Catch::Approx(1.0).epsilon(1.0e-10));
        REQUIRE(projection.normal.dot(panel_normal) == Catch::Approx(1.0).margin(1.0e-10));

        const double signed_distance =
            (node_point - projection.point).dot(projection.normal);
        max_signed_distance_mismatch = std::max(
            max_signed_distance_mismatch,
            std::abs(signed_distance - projection.signed_distance));
        REQUIRE(projection.distance == Catch::Approx((node_point - projection.point).norm())
                                            .margin(1.0e-12));
        if (projection.converged) {
            ++converged_count;
            max_tangential_residual =
                std::max(max_tangential_residual, projection.tangential_residual);
        } else {
            ++edge_fallback_count;
            max_edge_fallback_residual =
                std::max(max_edge_fallback_residual, projection.tangential_residual);
            REQUIRE(bary.minCoeff() == Catch::Approx(0.0).margin(1.0e-10));
        }
    }

    REQUIRE(max_point_mismatch < 1.0e-12);
    REQUIRE(max_signed_distance_mismatch < 1.0e-12);
    REQUIRE(max_tangential_residual < 1.0e-8);
    INFO("converged=" << converged_count
         << " edge_fallback=" << edge_fallback_count
         << " total=" << band.projections().size()
         << " max_edge_residual=" << max_edge_fallback_residual);
    REQUIRE(converged_count > static_cast<int>(0.9 * band.projections().size()));

    REQUIRE(gp.near_interface_nodes(radius) == baseline_nodes);
    for (int n = 0; n < grid.num_dofs(); ++n) {
        REQUIRE(gp.domain_label(n) == baseline_label[n]);
        REQUIRE(gp.closest_interface_point(n) == baseline_closest[n]);
    }
}
