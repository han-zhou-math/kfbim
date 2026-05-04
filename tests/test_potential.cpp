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

double eval_global_quadratic(const Eigen::VectorXd& coeffs, const Eigen::Vector2d& pt)
{
    return coeffs[0]
         + coeffs[1] * pt[0]
         + coeffs[2] * pt[1]
         + 0.5 * coeffs[3] * pt[0] * pt[0]
         + coeffs[4] * pt[0] * pt[1]
         + 0.5 * coeffs[5] * pt[1] * pt[1];
}

Eigen::Vector2d grad_global_quadratic(const Eigen::VectorXd& coeffs,
                                      const Eigen::Vector2d& pt)
{
    return {coeffs[1] + coeffs[3] * pt[0] + coeffs[4] * pt[1],
            coeffs[2] + coeffs[4] * pt[0] + coeffs[5] * pt[1]};
}

LocalPoly2D make_global_quadratic_poly(const Eigen::VectorXd& coeffs,
                                       const Eigen::Vector2d& center)
{
    LocalPoly2D poly;
    poly.center = center;
    poly.coeffs.resize(6);
    const Eigen::Vector2d grad = grad_global_quadratic(coeffs, center);
    poly.coeffs << eval_global_quadratic(coeffs, center),
                   grad[0],
                   grad[1],
                   coeffs[3],
                   coeffs[4],
                   coeffs[5];
    return poly;
}

} // namespace

TEST_CASE("LaplaceLobattoCenterRestrict2D returns average trace and flux",
          "[laplace][restrict][lobatto][2d]")
{
    constexpr int N = 32;
    constexpr double half_width = 1.7;
    const double h = (2.0 * half_width) / static_cast<double>(N);

    CartesianGrid2D grid({-half_width, -half_width}, {h, h}, {N, N}, DofLayout2D::Node);
    OffsetCircleCurve2D curve;
    Interface2D iface = CurveResampler2D::discretize(curve, h, 4.0);
    GridPair2D grid_pair(grid, iface);
    LaplaceLobattoCenterRestrict2D restrict_op(grid_pair);

    Eigen::VectorXd avg_coeffs(6);
    avg_coeffs << 0.3, -0.2, 0.15, 0.07, -0.04, 0.09;

    Eigen::VectorXd jump_coeffs(6);
    jump_coeffs << -0.1, 0.25, -0.18, 0.05, 0.03, -0.06;

    std::vector<LocalPoly2D> correction_polys(4 * iface.num_panels());
    int cidx = 0;
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (int local = 0; local < 4; ++local) {
            const int q = iface.point_index(p, std::min(local, iface.points_per_panel() - 1));
            const Eigen::Vector2d center = iface.points().row(q).transpose();
            correction_polys[cidx++] = make_global_quadratic_poly(jump_coeffs, center);
        }
    }

    Eigen::VectorXd bulk_solution(grid.num_dofs());
    for (int idx = 0; idx < grid.num_dofs(); ++idx) {
        const auto c = grid.coord(idx);
        const Eigen::Vector2d pt{c[0], c[1]};
        const double avg = eval_global_quadratic(avg_coeffs, pt);
        const double jump = eval_global_quadratic(jump_coeffs, pt);
        bulk_solution[idx] = avg + (grid_pair.domain_label(idx) == 1 ? 0.5 : -0.5) * jump;
    }

    const auto polys = restrict_op.apply(bulk_solution, correction_polys);

    double trace_err = 0.0;
    double flux_err = 0.0;
    for (int q = 0; q < iface.num_points(); ++q) {
        const Eigen::Vector2d pt = iface.points().row(q).transpose();
        const Eigen::Vector2d normal = iface.normals().row(q).transpose();
        const Eigen::Vector2d grad = grad_global_quadratic(avg_coeffs, pt);

        trace_err = std::max(trace_err,
                             std::abs(polys[q].coeffs[0]
                                      - eval_global_quadratic(avg_coeffs, pt)));
        flux_err = std::max(flux_err,
                            std::abs(polys[q].coeffs[1] * normal[0]
                                     + polys[q].coeffs[2] * normal[1]
                                     - grad.dot(normal)));
    }

    REQUIRE(trace_err < 1.0e-9);
    REQUIRE(flux_err < 1.0e-9);
}

TEST_CASE("LaplacePotentialEval2D general and specialized paths agree",
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
    LaplacePotentialEval2D potentials(spread, bulk_solver, restrict_op);

    const int Nq = iface.num_points();
    REQUIRE(potentials.problem_size() == Nq);
    REQUIRE(potentials.arc_h_ratio() > 0.5);

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

    SECTION("general evaluate accepts arbitrary jumps and returns full data") {
        auto jumps = make_jumps(iface, phi, psi, q_jump);
        auto result = potentials.evaluate(jumps, zero_rhs);

        REQUIRE(result.u_bulk.size() == grid.num_dofs());
        REQUIRE(result.u_avg.size() == Nq);
        REQUIRE(result.un_avg.size() == Nq);
        for (int i = 0; i < Nq; ++i) {
            REQUIRE(std::isfinite(result.u_avg[i]));
            REQUIRE(std::isfinite(result.un_avg[i]));
        }
    }

    SECTION("double-layer returns averaged trace and continuous normal derivative") {
        Eigen::VectorXd K_phi;
        Eigen::VectorXd H_phi;
        potentials.eval_double_layer(phi, K_phi, H_phi);

        auto jumps = make_jumps(iface, phi, zeros, zeros);
        auto expected = potentials.evaluate(jumps, zero_rhs);

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
        auto expected = potentials.evaluate(jumps, zero_rhs);

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
        auto expected = potentials.evaluate(jumps, zero_rhs);

        REQUIRE(max_abs_diff(N_q, expected.u_avg) < 1.0e-10);
        REQUIRE(max_abs_diff(Nn_q, expected.un_avg) < 1.0e-10);
    }
}
