#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/geometry/grid_pair_2d.hpp"
#include "src/geometry/p2_curve_2d.hpp"
#include "src/geometry/p2_projection_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/local_cauchy/jump_data.hpp"
#include "src/operators/laplace_bvp_2d.hpp"
#include "src/transfer/laplace_correction_support.hpp"
#include "src/transfer/laplace_projection_correction_2d.hpp"
#include "src/transfer/laplace_restrict_2d.hpp"
#include "src/transfer/laplace_spread_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;

class CircleCurve2D final : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override
    {
        return {radius_ * std::cos(t), radius_ * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override
    {
        return {-radius_ * std::sin(t), radius_ * std::cos(t)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    double radius_ = 0.45;
};

class Star3Curve2D final : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override
    {
        const double r = radius(t);
        return {r * std::cos(t), r * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override
    {
        const double r = radius(t);
        const double drdt = -kRadius * kAmplitude * kFolds
                            * std::sin(kFolds * t);
        return {drdt * std::cos(t) - r * std::sin(t),
                drdt * std::sin(t) + r * std::cos(t)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }

private:
    static double radius(double t)
    {
        return kRadius * (1.0 + kAmplitude * std::cos(kFolds * t));
    }

    static constexpr double kRadius = 0.45;
    static constexpr double kAmplitude = 0.22;
    static constexpr int kFolds = 3;
};

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

Interface2D make_flat_legacy_panel()
{
    Eigen::MatrixX2d points(3, 2);
    points << -1.0, 0.0,
               0.0, 0.0,
               1.0, 0.0;

    Eigen::MatrixX2d normals(3, 2);
    normals.rowwise() = Eigen::RowVector2d(0.0, -1.0);

    Eigen::VectorXd weights = Eigen::VectorXd::Ones(3);
    Eigen::VectorXi components = Eigen::VectorXi::Zero(1);
    return Interface2D(points,
                       normals,
                       weights,
                       3,
                       components,
                       PanelNodeLayout2D::LegacyGaussLegendre);
}

int brute_force_nearest_p2_center(const Interface2D& iface,
                                  const CartesianGrid2D& grid,
                                  int node)
{
    const auto c = grid.coord(node);
    const Eigen::Vector2d pt(c[0], c[1]);
    int best = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();
    int center_idx = 0;
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (double s : geometry2d::kP2CenterS) {
            const Eigen::Vector2d center = geometry2d::panel_point(iface, p, s);
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

double brute_force_nearest_p2_center_distance(const Interface2D& iface,
                                              const CartesianGrid2D& grid,
                                              int node)
{
    const auto c = grid.coord(node);
    const Eigen::Vector2d pt(c[0], c[1]);
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (double s : geometry2d::kP2CenterS) {
            const Eigen::Vector2d center = geometry2d::panel_point(iface, p, s);
            best_dist2 = std::min(best_dist2, (pt - center).squaredNorm());
        }
    }
    return std::sqrt(best_dist2);
}

std::vector<int> brute_force_p2_center_band_nodes(const Interface2D& iface,
                                                  const CartesianGrid2D& grid,
                                                  double radius)
{
    std::vector<int> nodes;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        if (brute_force_nearest_p2_center_distance(iface, grid, n) < radius)
            nodes.push_back(n);
    }
    return nodes;
}

int brute_force_nearest_interface_point(const Interface2D& iface,
                                        const CartesianGrid2D& grid,
                                        int node)
{
    const auto c = grid.coord(node);
    const Eigen::Vector2d pt(c[0], c[1]);
    int best = 0;
    double best_dist2 = std::numeric_limits<double>::infinity();
    for (int q = 0; q < iface.num_points(); ++q) {
        const Eigen::Vector2d iface_pt = iface.points().row(q).transpose();
        const double dist2 = (pt - iface_pt).squaredNorm();
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best = q;
        }
    }
    return best;
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
    CartesianGrid2D grid({-1.5, -1.5}, {0.1, 0.1}, {30, 30}, DofLayout2D::Node);
    GridPair2D gp(grid, iface);

    const int node = grid.index(17, 18); // (0.2, 0.3)
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

TEST_CASE("2D GridPair caches P2 expansion-center lookup and lazy interface DOF lookup",
          "[projection][2d][grid-pair]")
{
    const Interface2D iface = make_flat_p2_panel();
    CartesianGrid2D grid({-1.5, -1.5}, {0.1, 0.1}, {30, 30}, DofLayout2D::Node);
    GridPair2D gp(grid, iface);

    const std::vector<int> nodes = {
        grid.index(12, 13),
        grid.index(4, 10),
        grid.index(16, 9),
        grid.index(10, 4)};
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

TEST_CASE("2D GridPair center lookup matches brute force on closed P2 curves",
          "[projection][2d][grid-pair]")
{
    CartesianGrid2D grid({-1.0, -1.0}, {0.1, 0.1}, {20, 20}, DofLayout2D::Node);
    CircleCurve2D circle;
    Star3Curve2D star;
    const std::vector<Interface2D> interfaces = {
        CurveResampler2D::discretize_quadratic_lagrange(circle, 0.1, 3.0),
        CurveResampler2D::discretize_quadratic_lagrange(star, 0.1, 3.0)};

    for (const Interface2D& iface : interfaces) {
        GridPair2D gp(grid, iface);
        const std::vector<int> nodes = {
            grid.index(10, 10),
            grid.index(0, 0),
            grid.index(20, 20),
            grid.index(3, 17),
            grid.index(17, 4)};
        for (int node : nodes) {
            const int expected =
                brute_force_nearest_p2_center(iface, grid, node);
            REQUIRE(gp.nearest_p2_expansion_center(node) == expected);
            REQUIRE(gp.nearest_p2_expansion_center(node) == expected);
        }
    }
}

TEST_CASE("2D GridPair center-distance bands match brute force center distances",
          "[projection][2d][grid-pair]")
{
    CircleCurve2D circle;
    CartesianGrid2D grid({-1.0, -1.0}, {0.1, 0.1}, {20, 20}, DofLayout2D::Node);
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(circle, 0.1, 3.0);
    GridPair2D gp(grid, iface);

    const double default_band = 3.0 * std::sqrt(2.0) * 0.1;
    for (double radius : {0.16, default_band, 0.65}) {
        REQUIRE(gp.near_interface_nodes(radius)
                == brute_force_p2_center_band_nodes(iface, grid, radius));
        for (int node : {grid.index(10, 10), grid.index(0, 0), grid.index(20, 10)}) {
            REQUIRE(gp.is_near_interface(node, radius)
                    == (brute_force_nearest_p2_center_distance(iface, grid, node)
                        < radius));
        }
    }
}

TEST_CASE("2D GridPair rejects P2 expansion-center lookup for non-P2 panels",
          "[projection][2d][grid-pair]")
{
    const Interface2D iface = make_flat_legacy_panel();
    CartesianGrid2D grid({-1.0, -1.0}, {0.1, 0.1}, {20, 20}, DofLayout2D::Node);
    GridPair2D gp(grid, iface);
    REQUIRE_THROWS_AS(gp.nearest_p2_expansion_center(grid.index(10, 10)),
                      std::runtime_error);
}

TEST_CASE("2D GridPair P2 BFS labels keep closed-curve inside/outside convention",
          "[projection][2d][grid-pair]")
{
    CircleCurve2D circle;
    Star3Curve2D star;
    CartesianGrid2D grid({-1.0, -1.0}, {0.1, 0.1}, {20, 20}, DofLayout2D::Node);
    const std::vector<Interface2D> interfaces = {
        CurveResampler2D::discretize_quadratic_lagrange(circle, 0.1, 3.0),
        CurveResampler2D::discretize_quadratic_lagrange(star, 0.1, 3.0)};

    for (const Interface2D& iface : interfaces) {
        GridPair2D gp(grid, iface);
        REQUIRE(gp.domain_label(grid.index(10, 10)) == 1);
        REQUIRE(gp.domain_label(grid.index(0, 0)) == 0);
        REQUIRE(gp.domain_label(grid.index(20, 10)) == 0);

        const std::vector<int> band = gp.near_interface_nodes(0.2);
        REQUIRE_FALSE(band.empty());
    }
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
    CartesianGrid2D grid({-1.5, -1.5}, {0.1, 0.1}, {30, 30}, DofLayout2D::Node);
    GridPair2D gp(grid, iface);

    LaplaceQuadraticPanelCenterSpread2D spread(gp);
    std::vector<LaplaceJumpData2D> jumps(iface.num_points());
    for (auto& jump : jumps) {
        jump.u_jump = 0.0;
        jump.un_jump = 0.0;
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid.num_dofs());
    const LaplaceSpreadResult2D result = spread.apply(jumps, rhs);
    const std::vector<LocalPoly2D>& polys = result.correction_polys;

    REQUIRE(polys.size() == geometry2d::kP2CenterS.size());
    for (int i = 0; i < static_cast<int>(geometry2d::kP2CenterS.size()); ++i) {
        const Eigen::Vector2d expected =
            geometry2d::panel_point(iface, 0, geometry2d::kP2CenterS[i]);
        REQUIRE(polys[i].center.isApprox(expected, 1.0e-12));
    }
}

TEST_CASE("2D projection-point correction uses normal Taylor curve formula",
          "[projection][2d]")
{
    const Interface2D iface = make_flat_p2_panel();

    LaplaceSpreadResult2D spread_result;
    spread_result.correction_method = LaplaceCorrectionMethod2D::ProjectionPoint;
    spread_result.u_jump = Eigen::VectorXd::Zero(iface.num_points());
    for (int q = 0; q < iface.num_points(); ++q) {
        const double x = iface.points()(q, 0);
        spread_result.u_jump[q] = x * x;
    }
    spread_result.un_jump = Eigen::VectorXd::Constant(iface.num_points(), 3.0);
    spread_result.rhs_jump = Eigen::VectorXd::Constant(iface.num_points(), 5.0);
    spread_result.alpha = 7.0;

    CurveProjection2D projection;
    projection.grid_node = 0;
    projection.panel = 0;
    projection.component = 0;
    projection.local_s = 0.3;
    projection.point = geometry2d::panel_point(iface, 0, projection.local_s);
    projection.normal = geometry2d::panel_normal(iface, 0, projection.local_s);
    projection.signed_distance = 0.2;
    projection.distance = 0.2;
    projection.converged = true;

    const double correction =
        evaluate_projection_point_correction_2d(iface, projection, spread_result);

    const double c = 0.3 * 0.3;
    const double cnn = 7.0 * c - 5.0 - 2.0;
    const double expected = c + 0.2 * 3.0 + 0.5 * 0.2 * 0.2 * cnn;
    REQUIRE(correction == Catch::Approx(expected).margin(1.0e-12));
}

TEST_CASE("2D projection-point spread projects exact correction support",
          "[projection][2d]")
{
    CircleCurve2D circle;
    CartesianGrid2D grid({-1.0, -1.0}, {0.1, 0.1}, {20, 20}, DofLayout2D::Node);
    const Interface2D iface =
        CurveResampler2D::discretize_quadratic_lagrange(circle, 0.1, 3.0);
    GridPair2D gp(grid, iface);
    const LaplaceCorrectionSupport2D support =
        build_laplace_correction_support_2d(
            gp, "test 2D projection support");

    LaplaceQuadraticPanelCenterSpread2D spread(
        gp, 1.1, LaplaceCorrectionMethod2D::ProjectionPoint);
    std::vector<LaplaceJumpData2D> jumps(iface.num_points());
    for (auto& jump : jumps) {
        jump.u_jump = 0.0;
        jump.un_jump = 0.0;
        jump.rhs_derivs = Eigen::VectorXd::Zero(1);
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(grid.num_dofs());
    const LaplaceSpreadResult2D result = spread.apply(jumps, rhs);

    REQUIRE(result.correction_method == LaplaceCorrectionMethod2D::ProjectionPoint);
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
