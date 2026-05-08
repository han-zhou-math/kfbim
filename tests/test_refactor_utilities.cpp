#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "src/geometry/grid_pair_2d.hpp"
#include "src/geometry/grid_pair_3d.hpp"
#include "src/grid/structured_grid_ops.hpp"
#include "src/operators/detail/laplace_operator_utils.hpp"
#include "src/transfer/laplace_restrict_2d.hpp"
#include "src/transfer/laplace_restrict_3d.hpp"
#include "src/transfer/i_spread.hpp"

using namespace kfbim;

namespace {

class IdentityOperator final : public IKFBIOperator {
public:
    explicit IdentityOperator(int size)
        : size_(size)
    {}

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        y = x;
    }

    int problem_size() const override { return size_; }

private:
    int size_;
};

Interface2D make_restrict_test_panel_2d()
{
    Eigen::MatrixX2d points(3, 2);
    points << -0.2, 0.0,
               0.0, 0.0,
               0.2, 0.0;

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

Interface3D make_restrict_test_triangle_3d()
{
    Eigen::MatrixX3d points(6, 3);
    points << -0.2, 0.0, 0.0,
               0.2, 0.0, 0.0,
               0.0, 0.3, 0.0,
               0.0, 0.0, 0.0,
               0.1, 0.15, 0.0,
              -0.1, 0.15, 0.0;

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

double quadratic_value_2d(Eigen::Vector2d p)
{
    return 1.2 + 0.7 * p[0] - 0.4 * p[1]
           + 0.5 * 2.1 * p[0] * p[0]
           - 0.8 * p[0] * p[1]
           + 0.5 * 1.3 * p[1] * p[1];
}

Eigen::VectorXd expected_quadratic_coeffs_2d(Eigen::Vector2d c)
{
    Eigen::VectorXd coeffs(6);
    coeffs << quadratic_value_2d(c),
              0.7 + 2.1 * c[0] - 0.8 * c[1],
             -0.4 - 0.8 * c[0] + 1.3 * c[1],
              2.1,
             -0.8,
              1.3;
    return coeffs;
}

double quadratic_value_3d(Eigen::Vector3d p)
{
    return -0.3 + 0.8 * p[0] - 0.5 * p[1] + 0.2 * p[2]
           + 0.5 * 1.7 * p[0] * p[0]
           - 0.9 * p[0] * p[1]
           + 0.6 * p[0] * p[2]
           + 0.5 * 1.4 * p[1] * p[1]
           - 0.7 * p[1] * p[2]
           + 0.5 * 1.1 * p[2] * p[2];
}

Eigen::VectorXd expected_quadratic_coeffs_3d(Eigen::Vector3d c)
{
    Eigen::VectorXd coeffs(10);
    coeffs << quadratic_value_3d(c),
              0.8 + 1.7 * c[0] - 0.9 * c[1] + 0.6 * c[2],
             -0.5 - 0.9 * c[0] + 1.4 * c[1] - 0.7 * c[2],
              0.2 + 0.6 * c[0] - 0.7 * c[1] + 1.1 * c[2],
              1.7,
             -0.9,
              0.6,
              1.4,
             -0.7,
              1.1;
    return coeffs;
}

} // namespace

TEST_CASE("structured grid Dirichlet lifting and restoration are shared helpers",
          "[utilities][grid]")
{
    CartesianGrid2D grid2({0.0, 0.0}, {1.0, 1.0}, {2, 2}, DofLayout2D::Node);
    Eigen::VectorXd rhs2 = Eigen::VectorXd::Zero(grid2.num_dofs());
    Eigen::VectorXd bc2 = Eigen::VectorXd::Ones(grid2.num_dofs());
    const Eigen::VectorXd lifted2 =
        structured_grid::apply_dirichlet_boundary_elimination(
            "test", grid2, rhs2, bc2);
    REQUIRE(lifted2[grid2.index(1, 1)] == Catch::Approx(4.0));
    REQUIRE(lifted2[grid2.index(0, 1)] == Catch::Approx(0.0));

    Eigen::VectorXd u2 = Eigen::VectorXd::Zero(grid2.num_dofs());
    structured_grid::restore_dirichlet_boundary("test", grid2, u2, bc2);
    REQUIRE(u2[grid2.index(0, 1)] == Catch::Approx(1.0));
    REQUIRE(u2[grid2.index(1, 1)] == Catch::Approx(0.0));

    CartesianGrid3D grid3({0.0, 0.0, 0.0},
                          {1.0, 1.0, 1.0},
                          {2, 2, 2},
                          DofLayout3D::Node);
    Eigen::VectorXd rhs3 = Eigen::VectorXd::Zero(grid3.num_dofs());
    Eigen::VectorXd bc3 = Eigen::VectorXd::Ones(grid3.num_dofs());
    const Eigen::VectorXd lifted3 =
        structured_grid::apply_dirichlet_boundary_elimination(
            "test", grid3, rhs3, bc3);
    REQUIRE(lifted3[grid3.index(1, 1, 1)] == Catch::Approx(6.0));
    REQUIRE(lifted3[grid3.index(0, 1, 1)] == Catch::Approx(0.0));
}

TEST_CASE("fixed quadratic restrict stencil helpers choose square stencils",
          "[utilities][grid][restrict]")
{
    REQUIRE(structured_grid::restrict_stencil_side(1.0, 1.0) == -1);
    REQUIRE(structured_grid::restrict_stencil_side(1.1, 1.0) == 1);
    REQUIRE(structured_grid::restrict_stencil_side(0.9, 1.0) == -1);

    const auto offsets2 =
        structured_grid::quadratic_restrict_stencil_offsets_2d(1, -1);
    REQUIRE(offsets2[0].i == 0);
    REQUIRE(offsets2[0].j == 0);
    REQUIRE(offsets2[1].i == 1);
    REQUIRE(offsets2[1].j == 0);
    REQUIRE(offsets2[2].i == 0);
    REQUIRE(offsets2[2].j == -1);
    REQUIRE(offsets2[3].i == 1);
    REQUIRE(offsets2[3].j == -1);
    REQUIRE(offsets2[4].i == -1);
    REQUIRE(offsets2[4].j == 0);
    REQUIRE(offsets2[5].i == 0);
    REQUIRE(offsets2[5].j == 1);

    CartesianGrid2D grid2({0.0, 0.0}, {1.0, 1.0}, {4, 4}, DofLayout2D::Node);
    const int base2 = grid2.index(2, 2);
    const auto nodes2 = structured_grid::quadratic_restrict_stencil_nodes_2d(
        "test", grid2, base2, Eigen::Vector2d(2.3, 1.7));
    REQUIRE(nodes2[0] == grid2.index(2, 2));
    REQUIRE(nodes2[1] == grid2.index(3, 2));
    REQUIRE(nodes2[2] == grid2.index(2, 1));
    REQUIRE(nodes2[3] == grid2.index(3, 1));
    REQUIRE(nodes2[4] == grid2.index(1, 2));
    REQUIRE(nodes2[5] == grid2.index(2, 3));

    Eigen::Matrix<double, 6, 6> A2;
    const Eigen::Vector2d center2(2.3, 1.7);
    for (int r = 0; r < 6; ++r) {
        const auto c = grid2.coord(nodes2[r]);
        const double dx = c[0] - center2[0];
        const double dy = c[1] - center2[1];
        A2(r, 0) = 1.0;
        A2(r, 1) = dx;
        A2(r, 2) = dy;
        A2(r, 3) = 0.5 * dx * dx;
        A2(r, 4) = dx * dy;
        A2(r, 5) = 0.5 * dy * dy;
    }
    REQUIRE(A2.fullPivLu().rank() == 6);
    REQUIRE_THROWS_AS(structured_grid::quadratic_restrict_stencil_nodes_2d(
                          "test", grid2, grid2.index(0, 0),
                          Eigen::Vector2d(0.0, 0.0)),
                      std::runtime_error);

    const auto offsets3 =
        structured_grid::quadratic_restrict_stencil_offsets_3d(1, -1, 1);
    REQUIRE(offsets3[0].i == 0);
    REQUIRE(offsets3[0].j == 0);
    REQUIRE(offsets3[0].k == 0);
    REQUIRE(offsets3[1].i == 0);
    REQUIRE(offsets3[1].j == -1);
    REQUIRE(offsets3[1].k == 0);
    REQUIRE(offsets3[2].i == 0);
    REQUIRE(offsets3[2].j == 0);
    REQUIRE(offsets3[2].k == 1);
    REQUIRE(offsets3[3].i == 1);
    REQUIRE(offsets3[3].j == 0);
    REQUIRE(offsets3[3].k == 0);
    REQUIRE(offsets3[4].i == 0);
    REQUIRE(offsets3[4].j == -1);
    REQUIRE(offsets3[4].k == 1);
    REQUIRE(offsets3[5].i == 0);
    REQUIRE(offsets3[5].j == 1);
    REQUIRE(offsets3[5].k == 0);
    REQUIRE(offsets3[6].i == 0);
    REQUIRE(offsets3[6].j == 0);
    REQUIRE(offsets3[6].k == -1);
    REQUIRE(offsets3[7].i == 1);
    REQUIRE(offsets3[7].j == -1);
    REQUIRE(offsets3[7].k == 0);
    REQUIRE(offsets3[8].i == 1);
    REQUIRE(offsets3[8].j == 0);
    REQUIRE(offsets3[8].k == 1);
    REQUIRE(offsets3[9].i == -1);
    REQUIRE(offsets3[9].j == 0);
    REQUIRE(offsets3[9].k == 0);

    CartesianGrid3D grid3({0.0, 0.0, 0.0},
                          {1.0, 1.0, 1.0},
                          {4, 4, 4},
                          DofLayout3D::Node);
    const int base3 = grid3.index(2, 2, 2);
    const auto nodes3 = structured_grid::quadratic_restrict_stencil_nodes_3d(
        "test", grid3, base3, Eigen::Vector3d(2.3, 1.7, 2.2));
    REQUIRE(nodes3[0] == grid3.index(2, 2, 2));
    REQUIRE(nodes3[1] == grid3.index(2, 1, 2));
    REQUIRE(nodes3[2] == grid3.index(2, 2, 3));
    REQUIRE(nodes3[3] == grid3.index(3, 2, 2));
    REQUIRE(nodes3[4] == grid3.index(2, 1, 3));
    REQUIRE(nodes3[5] == grid3.index(2, 3, 2));
    REQUIRE(nodes3[6] == grid3.index(2, 2, 1));
    REQUIRE(nodes3[7] == grid3.index(3, 1, 2));
    REQUIRE(nodes3[8] == grid3.index(3, 2, 3));
    REQUIRE(nodes3[9] == grid3.index(1, 2, 2));

    Eigen::Matrix<double, 10, 10> A3;
    const Eigen::Vector3d center3(2.3, 1.7, 2.2);
    for (int r = 0; r < 10; ++r) {
        const auto c = grid3.coord(nodes3[r]);
        const double dx = c[0] - center3[0];
        const double dy = c[1] - center3[1];
        const double dz = c[2] - center3[2];
        A3(r, 0) = 1.0;
        A3(r, 1) = dx;
        A3(r, 2) = dy;
        A3(r, 3) = dz;
        A3(r, 4) = 0.5 * dx * dx;
        A3(r, 5) = dx * dy;
        A3(r, 6) = dx * dz;
        A3(r, 7) = 0.5 * dy * dy;
        A3(r, 8) = dy * dz;
        A3(r, 9) = 0.5 * dz * dz;
    }
    REQUIRE(A3.fullPivLu().rank() == 10);
    REQUIRE_THROWS_AS(structured_grid::quadratic_restrict_stencil_nodes_3d(
                          "test", grid3, grid3.index(0, 0, 0),
                          Eigen::Vector3d(0.0, 0.0, 0.0)),
                      std::runtime_error);
}

TEST_CASE("mean projection utility removes the pure-Neumann constant mode",
          "[utilities][operators]")
{
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 4.0;
    operators_detail::project_mean_zero(v);
    REQUIRE(v.mean() == Catch::Approx(0.0).margin(1.0e-14));

    IdentityOperator identity(3);
    operators_detail::MeanProjectedOperator projected(identity);
    Eigen::VectorXd y;
    Eigen::VectorXd x(3);
    x << 3.0, 4.0, 8.0;
    projected.apply(x, y);
    REQUIRE(y.mean() == Catch::Approx(0.0).margin(1.0e-14));
    REQUIRE(y[0] == Catch::Approx(-2.0));
    REQUIRE(y[1] == Catch::Approx(-1.0));
    REQUIRE(y[2] == Catch::Approx(3.0));
}

TEST_CASE("Laplace jump packing validates sizes and RHS derivative entries",
          "[utilities][operators]")
{
    const int n_iface = 2;
    Eigen::VectorXd u_jump(n_iface);
    Eigen::VectorXd un_jump(n_iface);
    u_jump << 1.0, 2.0;
    un_jump << 3.0, 4.0;

    std::vector<Eigen::VectorXd> rhs_derivs(
        n_iface, Eigen::VectorXd::Ones(1));
    const auto jumps = operators_detail::make_laplace_jumps_2d(
        "test", n_iface, u_jump, un_jump, rhs_derivs);
    REQUIRE(jumps.size() == 2);
    REQUIRE(jumps[1].u_jump == Catch::Approx(2.0));
    REQUIRE(jumps[1].un_jump == Catch::Approx(4.0));
    REQUIRE(jumps[1].rhs_derivs[0] == Catch::Approx(1.0));

    std::vector<Eigen::VectorXd> empty_deriv(
        n_iface, Eigen::VectorXd());
    REQUIRE_THROWS_AS(operators_detail::make_laplace_jumps_2d(
                          "test", n_iface, u_jump, un_jump, empty_deriv),
                      std::invalid_argument);

    Eigen::VectorXd bad_size(1);
    bad_size << 1.0;
    REQUIRE_THROWS_AS(operators_detail::make_laplace_jumps_2d(
                          "test", n_iface, bad_size, un_jump, rhs_derivs),
                      std::invalid_argument);
}

TEST_CASE("Laplace restrict recovers quadratic data from fixed square stencils",
          "[utilities][transfer][restrict]")
{
    {
        CartesianGrid2D grid({-1.0, -1.0}, {0.1, 0.1}, {20, 20},
                             DofLayout2D::Node);
        const Interface2D iface = make_restrict_test_panel_2d();
        GridPair2D gp(grid, iface);

        Eigen::VectorXd bulk(grid.num_dofs());
        for (int idx = 0; idx < grid.num_dofs(); ++idx) {
            const auto c = grid.coord(idx);
            bulk[idx] = quadratic_value_2d(Eigen::Vector2d(c[0], c[1]));
        }

        std::vector<LocalPoly2D> correction_polys(4);
        for (LocalPoly2D& poly : correction_polys) {
            poly.center = Eigen::Vector2d::Zero();
            poly.coeffs = Eigen::VectorXd::Zero(6);
        }

        LaplaceQuadraticPanelCenterRestrict2D restrict_op(gp);
        const std::vector<LocalPoly2D> polys =
            restrict_op.apply(bulk, correction_polys);
        REQUIRE(polys.size() == static_cast<std::size_t>(iface.num_points()));
        for (int q = 0; q < iface.num_points(); ++q) {
            const Eigen::Vector2d center = iface.points().row(q).transpose();
            const Eigen::VectorXd expected =
                expected_quadratic_coeffs_2d(center);
            REQUIRE(polys[q].center.isApprox(center, 1.0e-14));
            REQUIRE(polys[q].coeffs.size() == expected.size());
            for (int c = 0; c < expected.size(); ++c)
                REQUIRE(polys[q].coeffs[c]
                        == Catch::Approx(expected[c]).margin(1.0e-10));
        }
    }

    {
        CartesianGrid3D grid({-1.0, -1.0, -1.0},
                             {0.1, 0.1, 0.1},
                             {20, 20, 20},
                             DofLayout3D::Node);
        const Interface3D iface = make_restrict_test_triangle_3d();
        GridPair3D gp(grid, iface);

        Eigen::VectorXd bulk(grid.num_dofs());
        for (int idx = 0; idx < grid.num_dofs(); ++idx) {
            const auto c = grid.coord(idx);
            bulk[idx] = quadratic_value_3d(
                Eigen::Vector3d(c[0], c[1], c[2]));
        }

        LaplaceSpreadResult3D spread_result;
        spread_result.correction_method =
            LaplaceCorrectionMethod3D::NearestExpansionCenter;
        spread_result.correction_polys.resize(16);
        for (LocalPoly3D& poly : spread_result.correction_polys) {
            poly.center = Eigen::Vector3d::Zero();
            poly.coeffs = Eigen::VectorXd::Zero(10);
        }

        LaplaceQuadraticPatchCenterRestrict3D restrict_op(gp);
        const std::vector<LocalPoly3D> polys =
            restrict_op.apply(bulk, spread_result);
        REQUIRE(polys.size() == static_cast<std::size_t>(iface.num_points()));
        for (int q = 0; q < iface.num_points(); ++q) {
            const Eigen::Vector3d center = iface.points().row(q).transpose();
            const Eigen::VectorXd expected =
                expected_quadratic_coeffs_3d(center);
            REQUIRE(polys[q].center.isApprox(center, 1.0e-14));
            REQUIRE(polys[q].coeffs.size() == expected.size());
            for (int c = 0; c < expected.size(); ++c)
                REQUIRE(polys[q].coeffs[c]
                        == Catch::Approx(expected[c]).margin(1.0e-10));
        }
    }
}

TEST_CASE("3D correction context keeps the spread-result compatibility name",
          "[utilities][transfer]")
{
    static_assert(std::is_same<LaplaceCorrectionContext3D,
                               LaplaceSpreadResult3D>::value,
                  "LaplaceSpreadResult3D should remain a compatibility alias");

    LaplaceCorrectionContext3D context;
    context.correction_method = LaplaceCorrectionMethod3D::ProjectionPoint;
    context.alpha = 1.25;

    LaplaceSpreadResult3D& legacy_name = context;
    REQUIRE(legacy_name.correction_method
            == LaplaceCorrectionMethod3D::ProjectionPoint);
    REQUIRE(legacy_name.alpha == Catch::Approx(1.25));
}
