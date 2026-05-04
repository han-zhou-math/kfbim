#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "src/geometry/curve_2d.hpp"
#include "src/geometry/curve_resampler_2d.hpp"
#include "src/geometry/grid_pair_2d.hpp"
#include "src/grid/cartesian_grid_2d.hpp"
#include "src/operator/laplace_potential.hpp"
#include "src/problems/laplace_bvp_2d.hpp"
#include "src/problems/laplace_interior.hpp"
#include "src/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "src/solver/zfft_bc_type.hpp"
#include "src/transfer/laplace_restrict_2d.hpp"
#include "src/transfer/laplace_spread_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kKappaSq = 1.1;

constexpr double kBoxHalfWidth = 1.7;
constexpr double kCircleCx = 0.0;
constexpr double kCircleCy = 0.0;
constexpr double kCircleRadius = 1.0;
constexpr double kTargetInterfaceSpacingOverH = 1.5;

class UnitCircleCurve2D final : public ICurve2D {
public:
    Eigen::Vector2d eval(double t) const override
    {
        return {kCircleCx + kCircleRadius * std::cos(t),
                kCircleCy + kCircleRadius * std::sin(t)};
    }

    Eigen::Vector2d deriv(double t) const override
    {
        return {-kCircleRadius * std::sin(t),
                kCircleRadius * std::cos(t)};
    }

    double t_min() const override { return 0.0; }
    double t_max() const override { return 2.0 * kPi; }
};

struct Box2D {
    std::array<double, 2> lower;
    double                side_length;
};

enum class BvpCase {
    InteriorDirichlet,
    ExteriorDirichlet,
    InteriorNeumann,
    ExteriorNeumann
};

struct SolveData {
    double              bulk_err;
    int                 iterations;
    int                 num_panels;
    int                 num_interface_points;
    int                 density_size;
    std::vector<double> gmres_residuals;
};

double exact_u(double x, double y)
{
    const double a = 0.8 * x + 0.3 * y;
    const double b = 0.2 * x - 0.7 * y;
    const double c = 0.5 * x;
    const double d = 0.4 * y;
    return std::sin(a) + 0.4 * std::cos(b)
           + 0.2 * std::sin(c) * std::cos(d);
}

Eigen::Vector2d exact_grad(double x, double y)
{
    const double a = 0.8 * x + 0.3 * y;
    const double b = 0.2 * x - 0.7 * y;
    const double c = 0.5 * x;
    const double d = 0.4 * y;
    return {0.8 * std::cos(a) - 0.08 * std::sin(b)
            + 0.1 * std::cos(c) * std::cos(d),
            0.3 * std::cos(a) + 0.28 * std::sin(b)
            - 0.08 * std::sin(c) * std::sin(d)};
}

double exact_f(double x, double y, double eta)
{
    const double a = 0.8 * x + 0.3 * y;
    const double b = 0.2 * x - 0.7 * y;
    const double c = 0.5 * x;
    const double d = 0.4 * y;
    const double lap_u = -0.73 * std::sin(a) - 0.212 * std::cos(b)
                         - 0.082 * std::sin(c) * std::cos(d);
    return -lap_u + eta * exact_u(x, y);
}

bool is_dirichlet(BvpCase bvp)
{
    return bvp == BvpCase::InteriorDirichlet
        || bvp == BvpCase::ExteriorDirichlet;
}

bool is_interior(BvpCase bvp)
{
    return bvp == BvpCase::InteriorDirichlet
        || bvp == BvpCase::InteriorNeumann;
}

double eta_for_case(BvpCase bvp)
{
    (void)bvp;
    return kKappaSq;
}

bool solution_has_free_constant(BvpCase bvp)
{
    return bvp == BvpCase::InteriorNeumann && eta_for_case(bvp) == 0.0;
}

double target_panel_length_over_h(BvpCase bvp)
{
    (void)bvp;
    return 2.0 * kTargetInterfaceSpacingOverH;
}

const char* bvp_name(BvpCase bvp)
{
    switch (bvp) {
    case BvpCase::InteriorDirichlet: return "interior Dirichlet";
    case BvpCase::ExteriorDirichlet: return "exterior Dirichlet";
    case BvpCase::InteriorNeumann: return "interior Neumann";
    case BvpCase::ExteriorNeumann: return "exterior Neumann";
    }
    return "unknown";
}

bool is_outer_boundary_node(int idx, int nx, int ny)
{
    const int i = idx % nx;
    const int j = idx / nx;
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

Box2D make_outer_box()
{
    return {{-kBoxHalfWidth, -kBoxHalfWidth}, 2.0 * kBoxHalfWidth};
}

double max_abs_diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    REQUIRE(a.size() == b.size());
    double err = 0.0;
    for (int i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

std::vector<LaplaceJumpData2D> make_operator_test_jumps(
    const Eigen::VectorXd&             u_jump,
    const Eigen::VectorXd&             un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    REQUIRE(u_jump.size() == un_jump.size());
    REQUIRE(static_cast<Eigen::Index>(rhs_derivs.size()) == u_jump.size());
    std::vector<LaplaceJumpData2D> jumps(u_jump.size());
    for (int q = 0; q < u_jump.size(); ++q) {
        jumps[q].u_jump = u_jump[q];
        jumps[q].un_jump = un_jump[q];
        jumps[q].rhs_derivs = rhs_derivs[q];
    }
    return jumps;
}

void print_gmres_residuals(const SolveData& data)
{
    std::printf("          GMRES relres");
    for (std::size_t k = 0; k < data.gmres_residuals.size(); ++k) {
        if (k % 6 == 0)
            std::printf("\n            ");
        std::printf("%4zu:%9.2e", k, data.gmres_residuals[k]);
    }
    std::printf("\n");
}

SolveData solve_and_measure(BvpCase bvp, int N)
{
    UnitCircleCurve2D circle;
    const Box2D box = make_outer_box();
    const double h = box.side_length / static_cast<double>(N);

    CartesianGrid2D grid(box.lower, {h, h}, {N, N}, DofLayout2D::Node);
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int n_dof = nx * ny;

    Interface2D iface =
        CurveResampler2D::discretize(circle, h, target_panel_length_over_h(bvp));
    const int n_panels = iface.num_panels();
    const int n_iface = iface.num_points();
    const double eta = eta_for_case(bvp);

    Eigen::VectorXd g(n_iface);
    std::vector<Eigen::VectorXd> rhs_derivs(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        if (is_dirichlet(bvp)) {
            g[q] = exact_u(x, y);
        } else {
            const Eigen::Vector2d normal(iface.normals()(q, 0),
                                         iface.normals()(q, 1));
            g[q] = exact_grad(x, y).dot(normal);
        }
        rhs_derivs[q] = Eigen::VectorXd::Constant(1, exact_f(x, y, eta));
    }

    GridPair2D gp(grid, iface);
    Eigen::VectorXd f_bulk = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd u_exact = Eigen::VectorXd::Zero(n_dof);
    Eigen::VectorXd outer_bc = Eigen::VectorXd::Zero(n_dof);

    const int physical_label = is_interior(bvp) ? 1 : 0;
    for (int n = 0; n < n_dof; ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        u_exact[n] = exact_u(x, y);
        outer_bc[n] = exact_u(x, y);

        if (gp.domain_label(n) == physical_label
            && !is_outer_boundary_node(n, nx, ny)) {
            f_bulk[n] = exact_f(x, y, eta);
        }
    }

    LaplaceBvpOptions2D options;
    options.eta = eta;
    if (!is_interior(bvp))
        options.outer_dirichlet_values = outer_bc;

    Eigen::VectorXd u_bulk;
    int iterations = 0;
    int density_size = 0;
    bool converged = false;
    std::vector<double> gmres_residuals;

    switch (bvp) {
    case BvpCase::InteriorDirichlet: {
        LaplaceInteriorDirichlet2D problem(
            grid, iface, g, f_bulk, rhs_derivs,
            options.panel_method, options.eta);
        auto result = problem.solve(800, 1.0e-8, 200);
        u_bulk = std::move(result.u_bulk);
        iterations = result.iterations;
        density_size = static_cast<int>(result.density.size());
        converged = result.converged;
        gmres_residuals = std::move(result.residuals);
        break;
    }
    case BvpCase::ExteriorDirichlet: {
        LaplaceExteriorDirichlet2D problem(
            grid, iface, g, f_bulk, rhs_derivs, options);
        auto result = problem.solve(800, 1.0e-8, 200);
        u_bulk = std::move(result.u_bulk);
        iterations = result.iterations;
        density_size = static_cast<int>(result.density.size());
        converged = result.converged;
        gmres_residuals = std::move(result.residuals);
        break;
    }
    case BvpCase::InteriorNeumann: {
        LaplaceInteriorNeumann2D problem(
            grid, iface, g, f_bulk, rhs_derivs, options);
        auto result = problem.solve(800, 1.0e-8, 200);
        u_bulk = std::move(result.u_bulk);
        iterations = result.iterations;
        density_size = static_cast<int>(result.density.size());
        converged = result.converged;
        gmres_residuals = std::move(result.residuals);
        break;
    }
    case BvpCase::ExteriorNeumann: {
        LaplaceExteriorNeumann2D problem(
            grid, iface, g, f_bulk, rhs_derivs, options);
        auto result = problem.solve(800, 1.0e-8, 200);
        u_bulk = std::move(result.u_bulk);
        iterations = result.iterations;
        density_size = static_cast<int>(result.density.size());
        converged = result.converged;
        gmres_residuals = std::move(result.residuals);
        break;
    }
    }

    REQUIRE(converged);

    double constant_shift = 0.0;
    int measured_nodes = 0;
    if (solution_has_free_constant(bvp)) {
        for (int n = 0; n < n_dof; ++n) {
            if (gp.domain_label(n) != physical_label)
                continue;
            constant_shift += u_bulk[n] - u_exact[n];
            ++measured_nodes;
        }
        REQUIRE(measured_nodes > 0);
        constant_shift /= static_cast<double>(measured_nodes);
    }

    double bulk_err = 0.0;
    for (int n = 0; n < n_dof; ++n) {
        if (gp.domain_label(n) != physical_label)
            continue;
        if (!is_interior(bvp) && is_outer_boundary_node(n, nx, ny))
            continue;
        bulk_err = std::max(
            bulk_err,
            std::abs((u_bulk[n] - constant_shift) - u_exact[n]));
    }

    if (!is_interior(bvp)) {
        double boundary_err = 0.0;
        for (int n = 0; n < n_dof; ++n)
            if (is_outer_boundary_node(n, nx, ny))
                boundary_err = std::max(boundary_err, std::abs(u_bulk[n] - outer_bc[n]));
        REQUIRE(boundary_err < 1.0e-12);
    }

    return {bulk_err,
            iterations,
            n_panels,
            n_iface,
            density_size,
            std::move(gmres_residuals)};
}

void check_convergence(BvpCase bvp)
{
    const double eta = eta_for_case(bvp);
    const std::vector<int> Ns{32, 64, 128, 256, 512};
    const int n_levels = static_cast<int>(Ns.size());
    std::vector<SolveData> data(n_levels);
    std::vector<double> rates(n_levels, 0.0);

    if (eta == 0.0) {
        std::printf("\n  %s BVP on unit circle centered at origin: -Delta u = f\n",
                    bvp_name(bvp));
    } else {
        std::printf("\n  Screened %s BVP on unit circle centered at origin: -Delta u + %.2f u = f\n",
                    bvp_name(bvp), eta);
    }
    std::printf("  Manufactured: sin(0.8x+0.3y) + 0.4cos(0.2x-0.7y) + 0.2sin(0.5x)cos(0.4y)\n");
    std::printf("  Box: (-%.1f, %.1f)^2; panels: Chebyshev-Lobatto; target interface spacing/h = %.2f (panel_length/h = %.2f)\n",
                kBoxHalfWidth, kBoxHalfWidth,
                kTargetInterfaceSpacingOverH,
                target_panel_length_over_h(bvp));
    std::printf("  GMRES residual history: relative residual, index 0 is the initial residual\n");
    std::printf("  %6s  %8s  %10s  %11s  %12s  %8s  %6s\n",
                "N", "panels", "iface_pts", "density", "max_err", "order", "GMRES");

    for (int l = 0; l < n_levels; ++l) {
        data[l] = solve_and_measure(bvp, Ns[l]);
        if (l == 0) {
            std::printf("  %6d  %8d  %10d  %11d  %12.4e  %8s  %6d\n",
                        Ns[l], data[l].num_panels, data[l].num_interface_points,
                        data[l].density_size, data[l].bulk_err, "-",
                        data[l].iterations);
        } else {
            rates[l] = std::log2(data[l - 1].bulk_err / data[l].bulk_err);
            std::printf("  %6d  %8d  %10d  %11d  %12.4e  %8.3f  %6d\n",
                        Ns[l], data[l].num_panels, data[l].num_interface_points,
                        data[l].density_size, data[l].bulk_err, rates[l],
                        data[l].iterations);
        }
        print_gmres_residuals(data[l]);
        REQUIRE(std::isfinite(data[l].bulk_err));
        REQUIRE(data[l].num_interface_points == 2 * data[l].num_panels);
        REQUIRE(data[l].density_size == data[l].num_interface_points);
        REQUIRE(data[l].gmres_residuals.size()
                == static_cast<std::size_t>(data[l].iterations + 1));
    }

    const double final_rate_avg =
        (n_levels > 3)
        ? 0.5 * (rates[n_levels - 2] + rates[n_levels - 1])
        : 0.5 * (rates[1] + rates[2]);
    REQUIRE(data[n_levels - 1].bulk_err < data[0].bulk_err);
    REQUIRE(final_rate_avg > 1.5);
}

} // namespace

TEST_CASE("LaplaceKFBIOperator2D side modes match potential jump relations",
          "[laplace][bvp][operator][lobatto][2d]")
{
    constexpr int N = 32;
    constexpr double half_width = kBoxHalfWidth;
    const double h = (2.0 * half_width) / static_cast<double>(N);

    CartesianGrid2D grid({-half_width, -half_width}, {h, h}, {N, N}, DofLayout2D::Node);
    UnitCircleCurve2D curve;
    Interface2D iface =
        CurveResampler2D::discretize(
            curve, h, target_panel_length_over_h(BvpCase::InteriorDirichlet));
    GridPair2D grid_pair(grid, iface);

    LaplaceLobattoCenterSpread2D spread(grid_pair, kKappaSq);
    LaplaceFftBulkSolverZfft2D bulk_solver(grid, ZfftBcType::Dirichlet, kKappaSq);
    LaplaceLobattoCenterRestrict2D restrict_op(grid_pair);
    LaplacePotentialEval2D potentials(spread, bulk_solver, restrict_op);

    const int n_iface = iface.num_points();
    std::vector<Eigen::VectorXd> rhs_derivs(n_iface, Eigen::VectorXd::Zero(1));
    const Eigen::VectorXd zero_rhs = Eigen::VectorXd::Zero(grid.num_dofs());

    Eigen::VectorXd phi(n_iface);
    Eigen::VectorXd psi(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        phi[q] = 0.4 + 0.2 * x - 0.1 * y + 0.05 * std::sin(2.0 * x);
        psi[q] = -0.3 + 0.15 * x + 0.25 * y + 0.04 * std::cos(3.0 * y);
    }

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);
    const auto double_res =
        potentials.evaluate(make_operator_test_jumps(phi, zeros, rhs_derivs), zero_rhs);
    const auto single_res =
        potentials.evaluate(make_operator_test_jumps(zeros, psi, rhs_derivs), zero_rhs);

    Eigen::VectorXd actual;
    Eigen::VectorXd expected;

    LaplaceKFBIOperator2D interior_dirichlet(
        spread, bulk_solver, restrict_op, zero_rhs, rhs_derivs,
        LaplaceKFBIMode::InteriorDirichlet);
    interior_dirichlet.apply(phi, actual);
    expected = double_res.u_avg + 0.5 * phi;
    REQUIRE(max_abs_diff(actual, expected) < 1.0e-10);

    LaplaceKFBIOperator2D exterior_dirichlet(
        spread, bulk_solver, restrict_op, zero_rhs, rhs_derivs,
        LaplaceKFBIMode::ExteriorDirichlet);
    exterior_dirichlet.apply(phi, actual);
    expected = double_res.u_avg - 0.5 * phi;
    REQUIRE(max_abs_diff(actual, expected) < 1.0e-10);

    LaplaceKFBIOperator2D interior_neumann(
        spread, bulk_solver, restrict_op, zero_rhs, rhs_derivs,
        LaplaceKFBIMode::InteriorNeumann);
    interior_neumann.apply(psi, actual);
    expected = single_res.un_avg + 0.5 * psi;
    REQUIRE(max_abs_diff(actual, expected) < 1.0e-10);

    LaplaceKFBIOperator2D exterior_neumann(
        spread, bulk_solver, restrict_op, zero_rhs, rhs_derivs,
        LaplaceKFBIMode::ExteriorNeumann);
    exterior_neumann.apply(psi, actual);
    expected = single_res.un_avg - 0.5 * psi;
    REQUIRE(max_abs_diff(actual, expected) < 1.0e-10);
}

TEST_CASE("Laplace BVP wrappers converge on unit circle",
          "[screened][laplace][bvp][lobatto][convergence][2d]")
{
    check_convergence(BvpCase::InteriorDirichlet);
    check_convergence(BvpCase::ExteriorDirichlet);
    check_convergence(BvpCase::InteriorNeumann);
    check_convergence(BvpCase::ExteriorNeumann);
}
