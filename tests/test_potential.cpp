#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/geometry/grid_pair_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/local_cauchy/jump_data.hpp"
#include "src/operator/laplace_potential.hpp"
#include "src/problems/laplace_interface_solver_2d.hpp"
#include "src/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "src/solver/zfft_bc_type.hpp"
#include "src/transfer/laplace_restrict_2d.hpp"
#include "src/transfer/laplace_spread_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;

class OffsetCircleCurve2D final : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override {
        return {0.07 + 0.8 * std::cos(t), -0.04 + 0.8 * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override {
        return {-0.8 * std::sin(t), 0.8 * std::cos(t)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }
};

double max_abs_diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    REQUIRE(a.size() == b.size());
    double err = 0.0;
    for (int i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

std::vector<LaplaceJumpData2D> make_jumps(const Interface2D&     iface,
                                          const Eigen::VectorXd& u_jump,
                                          const Eigen::VectorXd& un_jump,
                                          const Eigen::VectorXd& rhs_jump)
{
    const int Nq = iface.num_points();
    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int q = 0; q < Nq; ++q) {
        jumps[q].u_jump = u_jump[q];
        jumps[q].un_jump = un_jump[q];
        jumps[q].rhs_derivs = Eigen::VectorXd::Constant(1, rhs_jump[q]);
    }
    return jumps;
}

} // namespace

TEST_CASE("LaplacePotentialEval2D matches interface-solver jump relations",
          "[laplace][potential][lobatto][2d]")
{
    constexpr int N = 32;
    constexpr double half_width = 1.7;
    const double h = (2.0 * half_width) / static_cast<double>(N);

    CartesianGrid2D grid({-half_width, -half_width}, {h, h}, {N, N}, DofLayout2D::Node);
    OffsetCircleCurve2D curve;
    Interface2D iface = CurveResampler2D::discretize(curve, h, 4.0);
    GridPair2D grid_pair(grid, iface);

    LaplaceLobattoCenterSpread2D spread(grid_pair);
    LaplaceFftBulkSolverZfft2D bulk_solver(grid, ZfftBcType::Dirichlet);
    LaplaceLobattoCenterRestrict2D restrict_op(grid_pair);
    LaplaceInterfaceSolver2D interface_solver(spread, bulk_solver, restrict_op);
    LaplacePotentialEval2D potentials(spread, bulk_solver, restrict_op);

    const int Nq = iface.num_points();
    REQUIRE(potentials.problem_size() == Nq);

    Eigen::VectorXd phi(Nq);
    Eigen::VectorXd psi(Nq);
    Eigen::VectorXd q_jump(Nq);
    for (int i = 0; i < Nq; ++i) {
        const double x = iface.points()(i, 0);
        const double y = iface.points()(i, 1);
        phi[i] = 0.4 + 0.2 * x - 0.1 * y + 0.05 * std::sin(2.0 * x);
        psi[i] = -0.3 + 0.15 * x + 0.25 * y + 0.04 * std::cos(3.0 * y);
        q_jump[i] = 0.2 - 0.1 * x + 0.05 * y;
    }

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(Nq);
    const Eigen::VectorXd zero_rhs = Eigen::VectorXd::Zero(grid.num_dofs());

    SECTION("double-layer returns averaged trace and continuous normal derivative") {
        Eigen::VectorXd K_phi;
        Eigen::VectorXd H_phi;
        potentials.eval_double_layer(phi, K_phi, H_phi);

        auto jumps = make_jumps(iface, phi, zeros, zeros);
        auto expected = interface_solver.solve(jumps, zero_rhs);

        REQUIRE(max_abs_diff(K_phi, expected.u_avg) < 1.0e-10);
        REQUIRE(max_abs_diff(H_phi, expected.un_avg) < 1.0e-10);
        Eigen::VectorXd reconstructed_jump = (K_phi + 0.5 * phi) - (K_phi - 0.5 * phi);
        REQUIRE(max_abs_diff(reconstructed_jump, phi) < 1.0e-14);
    }

    SECTION("single-layer returns continuous trace and averaged normal derivative") {
        Eigen::VectorXd S_psi;
        Eigen::VectorXd Kt_psi;
        potentials.eval_single_layer(psi, S_psi, Kt_psi);

        auto jumps = make_jumps(iface, zeros, psi, zeros);
        auto expected = interface_solver.solve(jumps, zero_rhs);

        REQUIRE(max_abs_diff(S_psi, expected.u_avg) < 1.0e-10);
        REQUIRE(max_abs_diff(Kt_psi, expected.un_avg) < 1.0e-10);
        Eigen::VectorXd reconstructed_flux_jump =
            (Kt_psi + 0.5 * psi) - (Kt_psi - 0.5 * psi);
        REQUIRE(max_abs_diff(reconstructed_flux_jump, psi) < 1.0e-14);
    }

    SECTION("Newton interface forcing returns continuous trace and normal derivative") {
        Eigen::VectorXd N_q;
        Eigen::VectorXd Nn_q;
        potentials.eval_newton(q_jump, N_q, Nn_q);

        auto jumps = make_jumps(iface, zeros, zeros, q_jump);
        auto expected = interface_solver.solve(jumps, zero_rhs);

        REQUIRE(max_abs_diff(N_q, expected.u_avg) < 1.0e-10);
        REQUIRE(max_abs_diff(Nn_q, expected.un_avg) < 1.0e-10);
    }
}
