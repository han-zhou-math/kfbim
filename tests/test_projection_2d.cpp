#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>
#include <vector>

#include "src/geometry/grid_pair_2d.hpp"
#include "src/geometry/p2_curve_2d.hpp"
#include "src/geometry/p2_projection_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/local_cauchy/jump_data.hpp"
#include "src/operators/laplace_bvp_2d.hpp"
#include "src/transfer/laplace_spread_2d.hpp"

using namespace kfbim;

namespace {

Interface2D make_flat_p2_panel()
{
    Eigen::MatrixX2d points(3, 2);
    points << -1.0, 0.0,
               0.0, 0.0,
               1.0, 0.0;

    Eigen::MatrixX2d normals(3, 2);
    normals.rowwise() = Eigen::RowVector2d(0.0, -1.0);

    Eigen::VectorXd weights = Eigen::VectorXd::Ones(3);
    Eigen::MatrixXi panel_point_indices(1, 3);
    panel_point_indices << 0, 1, 2;

    return Interface2D(points,
                       normals,
                       weights,
                       3,
                       panel_point_indices,
                       Eigen::VectorXi::Zero(1),
                       PanelNodeLayout2D::QuadraticLagrange);
}

void require_same_projection_band(const NarrowBandProjection2D& a,
                                  const NarrowBandProjection2D& b)
{
    REQUIRE(a.radius() == Catch::Approx(b.radius()));
    REQUIRE(a.nodes() == b.nodes());
    REQUIRE(a.projections().size() == b.projections().size());
    for (int node : a.nodes()) {
        REQUIRE(a.has_projection(node));
        REQUIRE(b.has_projection(node));
        const CurveProjection2D& pa = a.projection(node);
        const CurveProjection2D& pb = b.projection(node);
        REQUIRE(pa.grid_node == pb.grid_node);
        REQUIRE(pa.panel == pb.panel);
        REQUIRE(pa.component == pb.component);
        REQUIRE(pa.local_s == Catch::Approx(pb.local_s).margin(1.0e-12));
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

TEST_CASE("2D P2 names preserve legacy Lobatto aliases",
          "[projection][2d]")
{
    static_assert(PanelNodeLayout2D::ChebyshevLobatto
                      == PanelNodeLayout2D::QuadraticLagrange,
                  "2D active layout alias should remain compatible");
    static_assert(LaplaceBvpPanelMethod2D::ChebyshevLobattoCenter
                      == LaplaceBvpPanelMethod2D::QuadraticPanelCenter,
                  "2D BVP method alias should remain compatible");
    static_assert(std::is_same<LaplaceLobattoCenterSpread2D,
                               LaplaceQuadraticPanelCenterSpread2D>::value,
                  "2D spread alias should remain compatible");
    static_assert(std::is_same<LaplaceLobattoCenterRestrict2D,
                               LaplaceQuadraticPanelCenterRestrict2D>::value,
                  "2D restrict alias should remain compatible");
}

TEST_CASE("2D P2 projection service projects explicit grid-node support",
          "[projection][2d]")
{
    const Interface2D iface = make_flat_p2_panel();
    CartesianGrid2D grid({-1.0, -1.0}, {0.1, 0.1}, {20, 20}, DofLayout2D::Node);
    GridPair2D gp(grid, iface);

    const int node = grid.index(12, 13); // (0.2, 0.3)
    const NarrowBandProjection2D via_grid_pair =
        gp.project_grid_nodes_to_interface({node, node});
    const NarrowBandProjection2D via_service =
        project_p2_grid_nodes_to_interface_2d(gp, {node});
    require_same_projection_band(via_grid_pair, via_service);

    REQUIRE(via_grid_pair.nodes().size() == 1);
    REQUIRE(via_grid_pair.has_projection(node));
    const CurveProjection2D& projection = via_grid_pair.projection(node);
    REQUIRE(projection.grid_node == node);
    REQUIRE(projection.panel == 0);
    REQUIRE(projection.component == 0);
    REQUIRE(projection.converged);
    REQUIRE(projection.local_s == Catch::Approx(0.2).margin(1.0e-12));
    REQUIRE(projection.point[0] == Catch::Approx(0.2).margin(1.0e-12));
    REQUIRE(projection.point[1] == Catch::Approx(0.0).margin(1.0e-12));
    REQUIRE(projection.normal[0] == Catch::Approx(0.0).margin(1.0e-12));
    REQUIRE(projection.normal[1] == Catch::Approx(-1.0).margin(1.0e-12));
    REQUIRE(projection.signed_distance == Catch::Approx(-0.3).margin(1.0e-12));
    REQUIRE(projection.distance == Catch::Approx(0.3).margin(1.0e-12));
    REQUIRE(projection.tangential_residual == Catch::Approx(0.0).margin(1.0e-12));
}

TEST_CASE("2D P2 near-interface projection uses the GridPair compatibility method",
          "[projection][2d]")
{
    const Interface2D iface = make_flat_p2_panel();
    CartesianGrid2D grid({-1.0, -1.0}, {0.1, 0.1}, {20, 20}, DofLayout2D::Node);
    GridPair2D gp(grid, iface);

    const double radius = 0.16;
    const NarrowBandProjection2D via_grid_pair =
        gp.project_near_interface_nodes(radius);
    const NarrowBandProjection2D via_service =
        project_p2_near_interface_nodes_2d(gp, radius);
    require_same_projection_band(via_grid_pair, via_service);
    REQUIRE_FALSE(via_grid_pair.nodes().empty());
}

TEST_CASE("2D spread correction centers match shared P2 geometry centers",
          "[projection][2d]")
{
    const Interface2D iface = make_flat_p2_panel();
    CartesianGrid2D grid({-1.0, -1.0}, {0.1, 0.1}, {20, 20}, DofLayout2D::Node);
    GridPair2D gp(grid, iface);

    LaplaceQuadraticPanelCenterSpread2D spread(gp);
    std::vector<LaplaceJumpData2D> jumps(iface.num_points());
    for (auto& jump : jumps) {
        jump.u_jump = 0.0;
        jump.un_jump = 0.0;
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid.num_dofs());
    const std::vector<LocalPoly2D> polys = spread.apply(jumps, rhs);

    REQUIRE(polys.size() == geometry2d::kP2CenterS.size());
    for (int i = 0; i < static_cast<int>(geometry2d::kP2CenterS.size()); ++i) {
        const Eigen::Vector2d expected =
            geometry2d::panel_point(iface, 0, geometry2d::kP2CenterS[i]);
        REQUIRE(polys[i].center.isApprox(expected, 1.0e-12));
    }
}
