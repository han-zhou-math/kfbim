#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "p2_sphere_fixture_3d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/geometry/p2_projection_3d.hpp"
#include "src/geometry/p2_surface_3d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/transfer/laplace_correction_support.hpp"
#include "src/transfer/laplace_projection_correction_3d.hpp"
#include "src/transfer/laplace_restrict_3d.hpp"
#include "src/transfer/laplace_spread_3d.hpp"

using namespace kfbim;
using namespace kfbim_test_3d;

namespace {

Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int idx)
{
    const auto c = grid.coord(idx);
    return {c[0], c[1], c[2]};
}

Interface3D make_flat_p2_triangle()
{
    Eigen::MatrixX3d points(6, 3);
    points << 0.0, 0.0, 0.0,
              1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.5, 0.0, 0.0,
              0.5, 0.5, 0.0,
              0.0, 0.5, 0.0;

    Eigen::MatrixX3d normals(6, 3);
    normals.rowwise() = Eigen::RowVector3d(0.0, 0.0, 1.0);

    Eigen::VectorXd weights = Eigen::VectorXd::Ones(6);
    Eigen::MatrixX3i panels(1, 3);
    panels << 0, 1, 2;
    Eigen::MatrixXi panel_point_indices(1, 6);
    panel_point_indices << 0, 1, 2, 3, 4, 5;

    return Interface3D(points,
                       panels,
                       points,
                       normals,
                       weights,
                       6,
                       panel_point_indices,
                       Eigen::VectorXi::Zero(1),
                       PanelNodeLayout3D::QuadraticLagrange);
}

Interface3D make_flat_raw_triangle()
{
    Eigen::MatrixX3d vertices(4, 3);
    vertices << 0.0, 0.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0;
    Eigen::MatrixX3i panels(4, 3);
    panels << 0, 2, 1,
              0, 1, 3,
              0, 3, 2,
              1, 2, 3;
    Eigen::MatrixX3d normals(4, 3);
    normals.rowwise() = Eigen::RowVector3d(0.0, 0.0, 1.0);
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(4);
    Eigen::MatrixXi panel_point_indices = panels;
    return Interface3D(vertices,
                       panels,
                       vertices,
                       normals,
                       weights,
                       3,
                       panel_point_indices,
                       Eigen::VectorXi::Zero(4),
                       PanelNodeLayout3D::Raw);
}

Eigen::VectorXd flat_phi_values(const Interface3D& iface)
{
    Eigen::VectorXd values(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        values[q] = x * x + y * y;
    }
    return values;
}

int brute_force_nearest_p2_center(const Interface3D& iface,
                                  const CartesianGrid3D& grid,
                                  int node)
{
    const Eigen::Vector3d pt = grid_point(grid, node);
    int best = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();
    int center_idx = 0;
    const std::vector<Eigen::Vector3d> bary =
        geometry3d::expansion_center_barycentrics();
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (const Eigen::Vector3d& b : bary) {
            const Eigen::Vector3d center = geometry3d::panel_point(iface, p, b);
            const double dist2 = (pt - center).squaredNorm();
            if (dist2 < best_dist2 - 1.0e-14
                || (std::abs(dist2 - best_dist2) <= 1.0e-14
                    && center_idx < best)) {
                best_dist2 = dist2;
                best = center_idx;
            }
            ++center_idx;
        }
    }
    return best;
}

double brute_force_nearest_p2_center_distance(const Interface3D& iface,
                                              const CartesianGrid3D& grid,
                                              int node)
{
    const Eigen::Vector3d pt = grid_point(grid, node);
    double best_dist2 = std::numeric_limits<double>::infinity();
    const std::vector<Eigen::Vector3d> bary =
        geometry3d::expansion_center_barycentrics();
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (const Eigen::Vector3d& b : bary) {
            const Eigen::Vector3d center = geometry3d::panel_point(iface, p, b);
            best_dist2 = std::min(best_dist2, (pt - center).squaredNorm());
        }
    }
    return std::sqrt(best_dist2);
}

std::vector<int> brute_force_p2_center_band_nodes(const Interface3D& iface,
                                                  const CartesianGrid3D& grid,
                                                  double radius)
{
    std::vector<int> nodes;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        if (brute_force_nearest_p2_center_distance(iface, grid, n) < radius)
            nodes.push_back(n);
    }
    return nodes;
}

int brute_force_nearest_interface_point(const Interface3D& iface,
                                        const CartesianGrid3D& grid,
                                        int node)
{
    const Eigen::Vector3d pt = grid_point(grid, node);
    int best = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (int q = 0; q < iface.num_points(); ++q) {
        const Eigen::Vector3d iface_pt = iface.points().row(q).transpose();
        const double dist2 = (pt - iface_pt).squaredNorm();
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best = q;
        }
    }
    return best;
}

void require_same_projection_band(const NarrowBandProjection3D& a,
                                  const NarrowBandProjection3D& b)
{
    REQUIRE(a.radius() == Catch::Approx(b.radius()));
    REQUIRE(a.nodes() == b.nodes());
    REQUIRE(a.projections().size() == b.projections().size());
    for (int node : a.nodes()) {
        REQUIRE(a.has_projection(node));
        REQUIRE(b.has_projection(node));
        const SurfaceProjection3D& pa = a.projection(node);
        const SurfaceProjection3D& pb = b.projection(node);
        REQUIRE(pa.grid_node == pb.grid_node);
        REQUIRE(pa.panel == pb.panel);
        REQUIRE(pa.component == pb.component);
        REQUIRE(pa.barycentric.isApprox(pb.barycentric, 1.0e-12));
        REQUIRE(pa.point.isApprox(pb.point, 1.0e-12));
        REQUIRE(pa.normal.isApprox(pb.normal, 1.0e-12));
        REQUIRE(pa.signed_distance == Catch::Approx(pb.signed_distance).margin(1.0e-12));
        REQUIRE(pa.distance == Catch::Approx(pb.distance).margin(1.0e-12));
        REQUIRE(pa.tangential_residual == Catch::Approx(pb.tangential_residual).margin(1.0e-12));
        REQUIRE(pa.iterations == pb.iterations);
        REQUIRE(pa.converged == pb.converged);
    }
}

} // namespace

TEST_CASE("P2 surface differential operators match a flat quadratic",
          "[projection][3d]")
{
    const Interface3D iface = make_flat_p2_triangle();
    const Eigen::Vector3d bary(0.2, 0.3, 0.5);
    const Eigen::VectorXd phi = flat_phi_values(iface);

    REQUIRE(geometry3d::panel_mean_curvature(iface, 0, bary)
            == Catch::Approx(0.0).margin(1.0e-14));
    REQUIRE(geometry3d::panel_laplace_beltrami_scalar(iface, 0, phi, bary)
            == Catch::Approx(4.0).margin(1.0e-12));
}

TEST_CASE("Projection-point correction uses normal Taylor surface formula",
          "[projection][3d]")
{
    const Interface3D iface = make_flat_p2_triangle();
    const Eigen::Vector3d bary(0.2, 0.3, 0.5);

    LaplaceSpreadResult3D spread_result;
    spread_result.correction_method = LaplaceCorrectionMethod3D::ProjectionPoint;
    spread_result.u_jump = flat_phi_values(iface);
    spread_result.un_jump = Eigen::VectorXd::Constant(iface.num_points(), 3.0);
    spread_result.rhs_jump = Eigen::VectorXd::Constant(iface.num_points(), 5.0);
    spread_result.alpha = 7.0;

    SurfaceProjection3D projection;
    projection.grid_node = 0;
    projection.panel = 0;
    projection.component = 0;
    projection.barycentric = bary;
    projection.point = geometry3d::panel_point(iface, 0, bary);
    projection.normal = Eigen::Vector3d(0.0, 0.0, 1.0);
    projection.signed_distance = 0.2;
    projection.distance = 0.2;
    projection.converged = true;

    const double correction =
        evaluate_projection_point_correction_3d(iface, projection, spread_result);

    const double c = 0.3 * 0.3 + 0.5 * 0.5;
    const double cnn = 7.0 * c - 5.0 - 4.0;
    const double expected = c + 0.2 * 3.0 + 0.5 * 0.2 * 0.2 * cnn;
    REQUIRE(correction == Catch::Approx(expected).margin(1.0e-12));
}

TEST_CASE("GridPair3D caches P2 expansion-center lookup and lazy interface DOF lookup",
          "[projection][3d][grid-pair]")
{
    const Interface3D iface = make_flat_p2_triangle();
    CartesianGrid3D grid({-0.5, -0.5, -0.5},
                         {0.1, 0.1, 0.1},
                         {20, 20, 20},
                         DofLayout3D::Node);
    GridPair3D gp(grid, iface);

    const std::vector<int> nodes = {
        grid.index(10, 10, 5),
        grid.index(0, 0, 0),
        grid.index(20, 20, 20),
        grid.index(12, 11, 8)};
    for (int node : nodes) {
        const int expected_center =
            brute_force_nearest_p2_center(iface, grid, node);
        REQUIRE(gp.nearest_p2_expansion_center(node) == expected_center);
        REQUIRE(gp.nearest_p2_expansion_center(node) == expected_center);

        const int expected_iface =
            brute_force_nearest_interface_point(iface, grid, node);
        REQUIRE(gp.closest_interface_point(node) == expected_iface);
        REQUIRE(gp.closest_interface_point(node) == expected_iface);
    }
}

TEST_CASE("GridPair3D center lookup and bands match brute force on a P2 sphere",
          "[projection][3d][grid-pair]")
{
    const int N = 12;
    const Box3D box = standard_box();
    const double h = box.side_length / static_cast<double>(N);
    CartesianGrid3D grid(box.lower, {h, h, h}, {N, N, N}, DofLayout3D::Node);
    const Interface3D iface = make_p2_sphere(surface_subdivision_for_grid(N));
    GridPair3D gp(grid, iface);

    const std::vector<int> nodes = {
        grid.index(6, 6, 6),
        grid.index(0, 0, 0),
        grid.index(12, 12, 12),
        grid.index(4, 7, 10),
        grid.index(9, 5, 3)};
    for (int node : nodes) {
        const int expected = brute_force_nearest_p2_center(iface, grid, node);
        REQUIRE(gp.nearest_p2_expansion_center(node) == expected);
        REQUIRE(gp.nearest_p2_expansion_center(node) == expected);
    }

    const double default_band = 3.0 * std::sqrt(3.0) * h;
    for (double radius : {1.5 * h, default_band, 4.5 * h}) {
        REQUIRE(gp.near_interface_nodes(radius)
                == brute_force_p2_center_band_nodes(iface, grid, radius));
        for (int node : nodes) {
            REQUIRE(gp.is_near_interface(node, radius)
                    == (brute_force_nearest_p2_center_distance(iface, grid, node)
                        < radius));
        }
    }
}

TEST_CASE("GridPair3D rejects P2 expansion-center lookup for non-P2 panels",
          "[projection][3d][grid-pair]")
{
    const Interface3D iface = make_flat_raw_triangle();
    CartesianGrid3D grid({-0.5, -0.5, -0.5},
                         {0.1, 0.1, 0.1},
                         {10, 10, 10},
                         DofLayout3D::Node);
    GridPair3D gp(grid, iface);
    REQUIRE_THROWS_AS(gp.nearest_p2_expansion_center(grid.index(5, 5, 5)),
                      std::runtime_error);
}

TEST_CASE("GridPair3D P2 BFS labels keep sphere inside/outside convention",
          "[projection][3d][grid-pair]")
{
    const int N = 16;
    const Box3D box = standard_box();
    const double h = box.side_length / static_cast<double>(N);
    CartesianGrid3D grid(box.lower, {h, h, h}, {N, N, N}, DofLayout3D::Node);
    const Interface3D iface = make_p2_sphere(surface_subdivision_for_grid(N));
    GridPair3D gp(grid, iface);

    REQUIRE(gp.domain_label(grid.index(N / 2, N / 2, N / 2)) == 1);
    REQUIRE(gp.domain_label(grid.index(0, 0, 0)) == 0);
    REQUIRE(gp.domain_label(grid.index(N, N / 2, N / 2)) == 0);
    REQUIRE_FALSE(gp.near_interface_nodes(2.0 * std::sqrt(3.0) * h).empty());
}

TEST_CASE("Projection-point spread option preserves zero correction",
          "[projection][3d]")
{
    const int N = 12;
    const Box3D box = standard_box();
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid3D grid(box.lower, {h, h, h}, {N, N, N}, DofLayout3D::Node);
    const Interface3D iface = make_p2_sphere(surface_subdivision_for_grid(N));
    GridPair3D gp(grid, iface);

    LaplaceQuadraticPatchCenterSpread3D spread(
        gp, 1.1, LaplaceCorrectionMethod3D::ProjectionPoint);
    std::vector<LaplaceJumpData3D> jumps(iface.num_points());
    for (auto& jump : jumps) {
        jump.u_jump = 0.0;
        jump.un_jump = 0.0;
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid.num_dofs());
    const LaplaceSpreadResult3D result = spread.apply(jumps, rhs);

    REQUIRE(result.correction_method == LaplaceCorrectionMethod3D::ProjectionPoint);
    REQUIRE(result.correction_polys.empty());
    REQUIRE(result.u_jump.size() == iface.num_points());
    REQUIRE(rhs.norm() == Catch::Approx(0.0).margin(1.0e-12));

    LaplaceQuadraticPatchCenterRestrict3D restrict_op(gp);
    const Eigen::VectorXd bulk = Eigen::VectorXd::Zero(grid.num_dofs());
    const std::vector<LocalPoly3D> polys = restrict_op.apply(bulk, result);
    REQUIRE(polys.size() == static_cast<std::size_t>(iface.num_points()));
    for (const LocalPoly3D& poly : polys)
        REQUIRE(poly.coeffs.norm() == Catch::Approx(0.0).margin(1.0e-12));
}

TEST_CASE("3D projection-point spread projects exact correction support",
          "[projection][3d]")
{
    const int N = 12;
    const Box3D box = standard_box();
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid3D grid(box.lower, {h, h, h}, {N, N, N}, DofLayout3D::Node);
    const Interface3D iface = make_p2_sphere(surface_subdivision_for_grid(N));
    GridPair3D gp(grid, iface);
    const LaplaceCorrectionSupport3D support =
        build_laplace_correction_support_3d(
            gp, "test 3D projection support");

    LaplaceQuadraticPatchCenterSpread3D spread(
        gp, 1.1, LaplaceCorrectionMethod3D::ProjectionPoint);
    std::vector<LaplaceJumpData3D> jumps(iface.num_points());
    for (auto& jump : jumps) {
        jump.u_jump = 0.0;
        jump.un_jump = 0.0;
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid.num_dofs());
    const LaplaceSpreadResult3D result = spread.apply(jumps, rhs);

    REQUIRE(result.correction_method == LaplaceCorrectionMethod3D::ProjectionPoint);
    REQUIRE(result.projection_cache.nodes() == support.projection_nodes);
    for (int node : support.projection_nodes)
        REQUIRE(result.projection_cache.has_projection(node));
    for (const auto& stencil : support.restrict_stencils) {
        for (int node : stencil)
            REQUIRE(result.projection_cache.has_projection(node));
    }
    for (const LaplaceCrossingCorrectionOp& op : support.crossing_ops)
        REQUIRE(result.projection_cache.has_projection(op.correction_node));
}

TEST_CASE("P2 projection service matches GridPair3D compatibility methods",
          "[projection][3d]")
{
    const int N = 12;
    const Box3D box = standard_box();
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid3D grid(box.lower, {h, h, h}, {N, N, N}, DofLayout3D::Node);
    const Interface3D iface = make_p2_sphere(surface_subdivision_for_grid(N));
    GridPair3D gp(grid, iface);

    const double radius = 2.0 * std::sqrt(3.0) * h;
    const NarrowBandProjection3D via_grid_pair =
        gp.project_near_interface_nodes(radius);
    const NarrowBandProjection3D via_service =
        project_p2_near_interface_nodes_3d(gp, radius);
    require_same_projection_band(via_grid_pair, via_service);

    std::vector<int> support;
    const auto nodes = gp.near_interface_nodes(radius);
    for (std::size_t i = 0; i < nodes.size() && i < 24; ++i)
        support.push_back(nodes[i]);
    REQUIRE_FALSE(support.empty());
    const NarrowBandProjection3D explicit_via_grid_pair =
        gp.project_grid_nodes_to_interface(support);
    const NarrowBandProjection3D explicit_via_service =
        project_p2_grid_nodes_to_interface_3d(gp, support);
    require_same_projection_band(explicit_via_grid_pair, explicit_via_service);
}

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
