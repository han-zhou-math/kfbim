#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "src/grid/structured_grid_ops.hpp"
#include "src/operators/detail/laplace_operator_utils.hpp"
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
