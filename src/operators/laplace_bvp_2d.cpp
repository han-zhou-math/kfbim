#include "laplace_bvp_2d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "../gmres/gmres.hpp"
#include "../local_cauchy/jump_data.hpp"
#include "../bulk_solvers/zfft_bc_type.hpp"
#include "../grid/structured_grid_ops.hpp"
#include "detail/laplace_operator_utils.hpp"

namespace kfbim {

using operators_detail::MeanProjectedOperator;
using operators_detail::project_mean_zero;
using operators_detail::require_vector_size;
using operators_detail::signed_rhs_derivs;

namespace {

bool is_neumann_type(LaplaceBvpType2D type)
{
    return type == LaplaceBvpType2D::InteriorNeumann
        || type == LaplaceBvpType2D::ExteriorNeumann;
}

bool is_exterior_type(LaplaceBvpType2D type)
{
    return type == LaplaceBvpType2D::ExteriorDirichlet
        || type == LaplaceBvpType2D::ExteriorNeumann;
}

double side_jump_sign(LaplaceBvpType2D type)
{
    return is_exterior_type(type) ? -1.0 : 1.0;
}

bool uses_neumann_nullspace_projection(LaplaceBvpType2D type, double eta)
{
    return operators_detail::uses_interior_neumann_nullspace_projection(
        type == LaplaceBvpType2D::InteriorNeumann, eta);
}

bool profile_solve_2d()
{
    return std::getenv("KFBIM_PROFILE_SOLVE_2D") != nullptr;
}

struct StageTimer2D {
    using Clock = std::chrono::steady_clock;

    Clock::time_point start = Clock::now();
    Clock::time_point last = start;

    double lap()
    {
        const Clock::time_point now = Clock::now();
        const std::chrono::duration<double> elapsed = now - last;
        last = now;
        return elapsed.count();
    }

    double total() const
    {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }
};

double profile_total_without_potential(const LaplacePotentialEvalProfile2D& profile,
                                       double total)
{
    return std::max(0.0, total - profile.total_sec);
}

void print_potential_profile_2d(const char* label,
                                const LaplacePotentialEvalProfile2D& profile)
{
    if (profile.calls == 0)
        return;

    std::printf("      %-15s calls=%3d total %.3fs avg %.4fs | rhs %.3fs spread %.3fs bulk %.3fs restrict %.3fs avg_trace %.3fs\n",
                label,
                profile.calls,
                profile.total_sec,
                profile.total_sec / static_cast<double>(profile.calls),
                profile.rhs_copy_sec,
                profile.spread_sec,
                profile.bulk_solve_sec,
                profile.restrict_sec,
                profile.average_sec);
}

std::unique_ptr<ILaplaceSpread2D> make_laplace_spread_2d(
    LaplaceBvpPanelMethod2D method,
    const GridPair2D&       grid_pair,
    double                  eta)
{
    switch (method) {
    case LaplaceBvpPanelMethod2D::QuadraticPanelCenter:
        return std::make_unique<LaplaceQuadraticPanelCenterSpread2D>(grid_pair, eta);
    }
    throw std::invalid_argument("unsupported Laplace BVP panel method");
}

std::unique_ptr<ILaplaceRestrict2D> make_laplace_restrict_2d(
    LaplaceBvpPanelMethod2D method,
    const GridPair2D&       grid_pair)
{
    switch (method) {
    case LaplaceBvpPanelMethod2D::QuadraticPanelCenter:
        return std::make_unique<LaplaceQuadraticPanelCenterRestrict2D>(grid_pair);
    }
    throw std::invalid_argument("unsupported Laplace BVP panel method");
}

double rhs_deriv_sign_for_type(LaplaceBvpType2D type)
{
    switch (type) {
    case LaplaceBvpType2D::InteriorDirichlet:
    case LaplaceBvpType2D::InteriorNeumann:
        return 1.0;
    case LaplaceBvpType2D::ExteriorDirichlet:
    case LaplaceBvpType2D::ExteriorNeumann:
        return -1.0;
    }
    throw std::invalid_argument("unsupported Laplace BVP type");
}

std::vector<LaplaceJumpData2D> make_jumps(
    const Interface2D&                 iface,
    const Eigen::VectorXd&             density,
    LaplaceBvpType2D                   type,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    const int n_iface = iface.num_points();
    std::vector<LaplaceJumpData2D> jumps(n_iface);
    const bool neumann = is_neumann_type(type);
    for (int q = 0; q < n_iface; ++q) {
        jumps[q].u_jump = neumann ? 0.0 : density[q];
        jumps[q].un_jump = neumann ? density[q] : 0.0;
        jumps[q].rhs_derivs = rhs_derivs[q];
    }
    return jumps;
}

} // namespace

LaplaceBvp2D::LaplaceBvp2D(
    const CartesianGrid2D& grid,
    const Interface2D&     iface,
    LaplaceBvpType2D       type,
    LaplaceBvpOptions2D    options)
    : grid_pair_(grid, iface)
    , spread_(make_laplace_spread_2d(options.panel_method, grid_pair_, options.eta))
    , bulk_solver_(grid, ZfftBcType::Dirichlet, options.eta)
    , restrict_op_(make_laplace_restrict_2d(options.panel_method, grid_pair_))
    , potentials_(*spread_, bulk_solver_, *restrict_op_)
    , type_(type)
    , rhs_deriv_sign_(rhs_deriv_sign_for_type(type))
    , eta_(options.eta)
    , outer_dirichlet_values_(std::move(options.outer_dirichlet_values))
{
    if (outer_dirichlet_values_.size() != 0) {
        require_vector_size("LaplaceBvp2D",
                            "outer_dirichlet_values",
                            outer_dirichlet_values_.size(),
                            grid.num_dofs());
    }
}

void LaplaceBvp2D::apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const
{
    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("LaplaceBvp2D", "operator input", x.size(), n_iface);

    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(grid_pair_.grid().num_dofs());
    const std::vector<Eigen::VectorXd> zero_derivs(
        n_iface, Eigen::VectorXd::Zero(1));
    apply_with_rhs(x, zero_rhs, zero_derivs, y);
}

int LaplaceBvp2D::problem_size() const
{
    return grid_pair_.interface().num_points();
}

Eigen::VectorXd LaplaceBvp2D::apply_dirichlet_boundary_elimination(
    const Eigen::VectorXd& rhs) const
{
    return structured_grid::apply_dirichlet_boundary_elimination(
        "LaplaceBvp2D", grid_pair_.grid(), rhs, outer_dirichlet_values_);
}

void LaplaceBvp2D::restore_dirichlet_boundary(Eigen::VectorXd& u_bulk) const
{
    structured_grid::restore_dirichlet_boundary(
        "LaplaceBvp2D", grid_pair_.grid(), u_bulk, outer_dirichlet_values_);
}

void LaplaceBvp2D::apply_with_rhs(
    const Eigen::VectorXd&              density,
    const Eigen::VectorXd&              rhs,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    Eigen::VectorXd&                    y) const
{
    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("LaplaceBvp2D", "operator input", density.size(), n_iface);
    require_vector_size("LaplaceBvp2D", "rhs", rhs.size(), grid_pair_.grid().num_dofs());

    auto jumps = make_jumps(grid_pair_.interface(), density, type_, rhs_derivs);
    auto res = potentials_.evaluate(jumps, rhs);

    y.resize(n_iface);
    const double sign = side_jump_sign(type_);
    if (!is_neumann_type(type_)) {
        for (int q = 0; q < n_iface; ++q)
            y[q] = res.u_avg[q] + sign * 0.5 * jumps[q].u_jump;
    } else {
        for (int q = 0; q < n_iface; ++q)
            y[q] = res.un_avg[q] + sign * 0.5 * jumps[q].un_jump;
    }
}

LaplaceBvpSolveResult2D LaplaceBvp2D::solve(
    const Eigen::VectorXd&              boundary_data,
    const Eigen::VectorXd&              f_bulk,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    int max_iter,
    double tol,
    int restart) const
{
    StageTimer2D timer;
    const bool profile = profile_solve_2d();

    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("LaplaceBvp2D", "boundary_data", boundary_data.size(), n_iface);
    require_vector_size("LaplaceBvp2D", "f_bulk", f_bulk.size(), grid_pair_.grid().num_dofs());
    const double t_validate = timer.lap();

    const auto signed_derivs = signed_rhs_derivs(
        "LaplaceBvp2D", rhs_derivs, n_iface, rhs_deriv_sign_);
    const double t_rhs_derivs = timer.lap();

    const Eigen::VectorXd rhs = apply_dirichlet_boundary_elimination(f_bulk);
    const double t_boundary_elim = timer.lap();

    Eigen::VectorXd volume_gamma;
    if (profile)
        potentials_.reset_profile();
    apply_with_rhs(Eigen::VectorXd::Zero(n_iface), rhs, signed_derivs, volume_gamma);
    const LaplacePotentialEvalProfile2D volume_profile =
        profile ? potentials_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_volume_apply = timer.lap();

    Eigen::VectorXd b = boundary_data - volume_gamma;
    const double t_bie_rhs = timer.lap();

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd density = Eigen::VectorXd::Zero(n_iface);
    int iterations = 0;
    if (profile)
        potentials_.reset_profile();
    if (uses_neumann_nullspace_projection(type_, eta_)) {
        project_mean_zero(b);

        MeanProjectedOperator projected_op(*this);
        iterations = gmres.solve(projected_op, b, density);
        project_mean_zero(density);
    } else {
        iterations = gmres.solve(*this, b, density);
    }
    const LaplacePotentialEvalProfile2D gmres_profile =
        profile ? potentials_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_gmres = timer.lap();

    auto jumps = make_jumps(grid_pair_.interface(), density, type_, signed_derivs);
    const double t_final_jumps = timer.lap();

    if (profile)
        potentials_.reset_profile();
    auto full_res = potentials_.evaluate(jumps, rhs);
    const LaplacePotentialEvalProfile2D final_profile =
        profile ? potentials_.profile() : LaplacePotentialEvalProfile2D{};
    const double t_final_potential = timer.lap();

    restore_dirichlet_boundary(full_res.u_bulk);
    const double t_restore = timer.lap();

    auto residuals = gmres.residuals();
    const double t_result = timer.lap();
    const double t_total = timer.total();

    if (profile) {
        std::printf("    LaplaceBvp2D::solve profile iface=%d grid=%d iterations=%d\n",
                    n_iface, grid_pair_.grid().num_dofs(), iterations);
        std::printf("      setup validate %.3fs rhs_derivs %.3fs boundary_elim %.3fs volume_apply %.3fs bie_rhs %.3fs\n",
                    t_validate, t_rhs_derivs, t_boundary_elim,
                    t_volume_apply, t_bie_rhs);
        std::printf("      gmres %.3fs final_jumps %.3fs final_potential %.3fs restore %.3fs result %.3fs total %.3fs\n",
                    t_gmres, t_final_jumps, t_final_potential, t_restore,
                    t_result, t_total);
        print_potential_profile_2d("volume", volume_profile);
        print_potential_profile_2d("gmres", gmres_profile);
        print_potential_profile_2d("final", final_profile);
        std::printf("      gmres_nonpotential %.3fs\n",
                    profile_total_without_potential(gmres_profile, t_gmres));
    }

    return {std::move(full_res.u_bulk),
            std::move(density),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

} // namespace kfbim
