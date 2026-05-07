#include "laplace_transmission_3d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "../bulk_solvers/zfft_bc_type.hpp"
#include "../gmres/gmres.hpp"
#include "../local_cauchy/jump_data.hpp"

namespace kfbim {

namespace {

constexpr const char* kContext = "LaplaceTransmission3D";

void require_vector_size(const char* name, Eigen::Index actual, Eigen::Index expected)
{
    if (actual != expected) {
        throw std::invalid_argument(
            std::string(kContext) + ": " + name
            + " has size " + std::to_string(actual)
            + ", expected " + std::to_string(expected));
    }
}

bool is_boundary_node(int i, int j, int k, int nx, int ny, int nz)
{
    return i == 0 || i == nx - 1
        || j == 0 || j == ny - 1
        || k == 0 || k == nz - 1;
}

double checked_lambda_sq(const char* phase, double beta, double kappa_sq)
{
    if (beta <= 0.0) {
        throw std::invalid_argument(
            std::string(kContext) + ": " + phase + " beta must be positive");
    }
    if (kappa_sq < 0.0) {
        throw std::invalid_argument(
            std::string(kContext) + ": " + phase + " kappa_sq must be nonnegative");
    }
    return kappa_sq / beta;
}

void require_common_ratio(double lambda_sq_int, double lambda_sq_ext)
{
    const double scale = std::max({1.0, std::abs(lambda_sq_int), std::abs(lambda_sq_ext)});
    if (std::abs(lambda_sq_int - lambda_sq_ext) > 1.0e-12 * scale) {
        throw std::invalid_argument(
            std::string(kContext)
            + ": CommonRatio mode requires kappa_sq_int/beta_int == "
              "kappa_sq_ext/beta_ext");
    }
}

void validate_rhs_derivs(const char* name,
                         const std::vector<Eigen::VectorXd>& derivs,
                         int n_iface)
{
    if (static_cast<int>(derivs.size()) != n_iface) {
        throw std::invalid_argument(
            std::string(kContext) + ": " + name
            + " length does not match interface size");
    }
    for (int q = 0; q < n_iface; ++q) {
        if (derivs[q].size() == 0) {
            throw std::invalid_argument(
                std::string(kContext) + ": " + name
                + " entries must contain at least the value derivative");
        }
    }
}

void validate_rhs_data(const LaplaceTransmissionRhsData3D& rhs_data,
                       int n_iface,
                       int n_dof)
{
    require_vector_size("rhs_data.reduced_rhs_bulk",
                        rhs_data.reduced_rhs_bulk.size(),
                        n_dof);
    validate_rhs_derivs("rhs_data.reduced_rhs_int_derivs",
                        rhs_data.reduced_rhs_int_derivs,
                        n_iface);
    validate_rhs_derivs("rhs_data.reduced_rhs_ext_derivs",
                        rhs_data.reduced_rhs_ext_derivs,
                        n_iface);
}

std::vector<Eigen::VectorXd> combine_rhs_derivs(
    const std::vector<Eigen::VectorXd>& int_derivs,
    const std::vector<Eigen::VectorXd>& ext_derivs)
{
    const int n_iface = static_cast<int>(int_derivs.size());
    std::vector<Eigen::VectorXd> combined(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        if (int_derivs[q].size() != ext_derivs[q].size()) {
            throw std::invalid_argument(
                std::string(kContext)
                + ": interior and exterior rhs derivative entries must have matching sizes");
        }
        combined[q] = int_derivs[q] - ext_derivs[q];
    }
    return combined;
}

std::vector<Eigen::VectorXd> scaled_rhs_derivs(
    const std::vector<Eigen::VectorXd>& derivs,
    double scale)
{
    const int n_iface = static_cast<int>(derivs.size());
    std::vector<Eigen::VectorXd> scaled(n_iface);
    for (int q = 0; q < n_iface; ++q)
        scaled[q] = scale * derivs[q];
    return scaled;
}

std::vector<Eigen::VectorXd> zero_rhs_derivs(int n_iface)
{
    return std::vector<Eigen::VectorXd>(n_iface, Eigen::VectorXd::Zero(1));
}

std::vector<LaplaceJumpData3D> make_jumps(
    int                                n_iface,
    const Eigen::VectorXd&             u_jump,
    const Eigen::VectorXd&             un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    require_vector_size("u_jump", u_jump.size(), n_iface);
    require_vector_size("un_jump", un_jump.size(), n_iface);
    validate_rhs_derivs("rhs_derivs", rhs_derivs, n_iface);

    std::vector<LaplaceJumpData3D> jumps(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        jumps[q].u_jump = u_jump[q];
        jumps[q].un_jump = un_jump[q];
        jumps[q].rhs_derivs = rhs_derivs[q];
    }
    return jumps;
}

void split_phase_rhs(const GridPair3D&      grid_pair,
                     const Eigen::VectorXd& rhs_bulk,
                     Eigen::VectorXd&       rhs_int,
                     Eigen::VectorXd&       rhs_ext)
{
    const int n_dof = grid_pair.grid().num_dofs();
    require_vector_size("rhs_data.reduced_rhs_bulk", rhs_bulk.size(), n_dof);

    rhs_int = Eigen::VectorXd::Zero(n_dof);
    rhs_ext = Eigen::VectorXd::Zero(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        if (grid_pair.domain_label(n) > 0)
            rhs_int[n] = rhs_bulk[n];
        else
            rhs_ext[n] = rhs_bulk[n];
    }
}

} // namespace

LaplaceTransmission3D::LaplaceTransmission3D(
    const CartesianGrid3D&            grid,
    const Interface3D&                iface,
    LaplaceTransmissionMode3D         mode,
    LaplaceTransmissionCoefficients3D coefficients,
    LaplaceTransmissionOptions3D      options)
    : grid_pair_(grid, iface)
    , mode_(mode)
    , coefficients_(coefficients)
    , lambda_sq_int_(checked_lambda_sq("interior",
                                       coefficients.beta_int,
                                       coefficients.kappa_sq_int))
    , lambda_sq_ext_(checked_lambda_sq("exterior",
                                       coefficients.beta_ext,
                                       coefficients.kappa_sq_ext))
    , spread_int_(grid_pair_,
                  lambda_sq_int_,
                  options.correction_method,
                  options.restrict_stencil_radius)
    , spread_ext_(grid_pair_,
                  lambda_sq_ext_,
                  options.correction_method,
                  options.restrict_stencil_radius)
    , bulk_solver_int_(grid, ZfftBcType::Dirichlet, lambda_sq_int_)
    , bulk_solver_ext_(grid, ZfftBcType::Dirichlet, lambda_sq_ext_)
    , restrict_op_(grid_pair_, options.restrict_stencil_radius)
    , potentials_int_(spread_int_, bulk_solver_int_, restrict_op_)
    , potentials_ext_(spread_ext_, bulk_solver_ext_, restrict_op_)
{
    if (mode_ == LaplaceTransmissionMode3D::CommonRatio)
        require_common_ratio(lambda_sq_int_, lambda_sq_ext_);
}

void LaplaceTransmission3D::apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const
{
    switch (mode_) {
    case LaplaceTransmissionMode3D::CommonRatio:
        apply_common_ratio(x, y);
        return;
    case LaplaceTransmissionMode3D::DifferentRatios:
        apply_different_ratios(x, y);
        return;
    }

    throw std::invalid_argument(std::string(kContext) + ": unsupported transmission mode");
}

int LaplaceTransmission3D::problem_size() const
{
    const int n_iface = grid_pair_.interface().num_points();
    switch (mode_) {
    case LaplaceTransmissionMode3D::CommonRatio:
        return n_iface;
    case LaplaceTransmissionMode3D::DifferentRatios:
        return 2 * n_iface;
    }

    throw std::invalid_argument(std::string(kContext) + ": unsupported transmission mode");
}

Eigen::VectorXd LaplaceTransmission3D::apply_dirichlet_boundary_elimination(
    const Eigen::VectorXd& reduced_rhs_bulk,
    const Eigen::VectorXd& outer_dirichlet_values) const
{
    const auto& grid = grid_pair_.grid();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int n_dof = grid.num_dofs();

    require_vector_size("reduced_rhs_bulk", reduced_rhs_bulk.size(), n_dof);
    if (outer_dirichlet_values.size() == 0)
        return reduced_rhs_bulk;
    require_vector_size("outer_dirichlet_values", outer_dirichlet_values.size(), n_dof);

    Eigen::VectorXd modified = reduced_rhs_bulk;
    const double hx = grid.spacing()[0];
    const double hy = grid.spacing()[1];
    const double hz = grid.spacing()[2];
    const double inv_hx2 = 1.0 / (hx * hx);
    const double inv_hy2 = 1.0 / (hy * hy);
    const double inv_hz2 = 1.0 / (hz * hz);

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                const int n = grid.index(i, j, k);
                double boundary_lift = 0.0;
                if (i == 1)
                    boundary_lift += outer_dirichlet_values[grid.index(0, j, k)] * inv_hx2;
                if (i == nx - 2)
                    boundary_lift += outer_dirichlet_values[grid.index(nx - 1, j, k)] * inv_hx2;
                if (j == 1)
                    boundary_lift += outer_dirichlet_values[grid.index(i, 0, k)] * inv_hy2;
                if (j == ny - 2)
                    boundary_lift += outer_dirichlet_values[grid.index(i, ny - 1, k)] * inv_hy2;
                if (k == 1)
                    boundary_lift += outer_dirichlet_values[grid.index(i, j, 0)] * inv_hz2;
                if (k == nz - 2)
                    boundary_lift += outer_dirichlet_values[grid.index(i, j, nz - 1)] * inv_hz2;
                modified[n] += boundary_lift;
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (is_boundary_node(i, j, k, nx, ny, nz))
                    modified[grid.index(i, j, k)] = 0.0;
            }
        }
    }

    return modified;
}

void LaplaceTransmission3D::restore_dirichlet_boundary(
    Eigen::VectorXd&       u_bulk,
    const Eigen::VectorXd& outer_dirichlet_values) const
{
    if (outer_dirichlet_values.size() == 0)
        return;

    const auto& grid = grid_pair_.grid();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    require_vector_size("u_bulk", u_bulk.size(), grid.num_dofs());
    require_vector_size("outer_dirichlet_values", outer_dirichlet_values.size(), grid.num_dofs());

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (is_boundary_node(i, j, k, nx, ny, nz))
                    u_bulk[grid.index(i, j, k)] =
                        outer_dirichlet_values[grid.index(i, j, k)];
            }
        }
    }
}

void LaplaceTransmission3D::apply_common_ratio(
    const Eigen::VectorXd& x,
    Eigen::VectorXd&       y) const
{
    require_common_ratio(lambda_sq_int_, lambda_sq_ext_);

    const int n_iface = grid_pair_.interface().num_points();
    const int n_dof = grid_pair_.grid().num_dofs();
    require_vector_size("common-ratio operator input", x.size(), n_iface);

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);
    const Eigen::VectorXd zero_rhs = Eigen::VectorXd::Zero(n_dof);
    const auto zero_derivs = zero_rhs_derivs(n_iface);
    auto jumps = make_jumps(n_iface, zeros, x, zero_derivs);
    auto result = potentials_int_.evaluate(jumps, zero_rhs);

    const double beta_sum = coefficients_.beta_int + coefficients_.beta_ext;
    const double gamma =
        2.0 * (coefficients_.beta_int - coefficients_.beta_ext) / beta_sum;
    y = x + gamma * result.un_avg;
}

void LaplaceTransmission3D::apply_different_ratios(
    const Eigen::VectorXd& x,
    Eigen::VectorXd&       y) const
{
    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("different-ratio operator input", x.size(), 2 * n_iface);

    const Eigen::VectorXd phi = x.head(n_iface);
    const Eigen::VectorXd psi = x.tail(n_iface);

    const double beta_int = coefficients_.beta_int;
    const double beta_ext = coefficients_.beta_ext;
    const double beta_sum = beta_int + beta_ext;
    const double alpha_int = 2.0 * beta_ext / beta_sum;
    const double alpha_ext = 2.0 * beta_int / beta_sum;
    const double flux_row_scale = 2.0 / beta_sum;

    Eigen::VectorXd trace_int;
    Eigen::VectorXd normal_int;
    Eigen::VectorXd trace_ext;
    Eigen::VectorXd normal_ext;
    const Eigen::VectorXd phi_int = alpha_int * phi;
    const Eigen::VectorXd phi_ext = alpha_ext * phi;
    potentials_int_.eval_layer_combination(phi_int, psi, trace_int, normal_int);
    potentials_ext_.eval_layer_combination(phi_ext, psi, trace_ext, normal_ext);

    y.resize(2 * n_iface);
    y.head(n_iface) = phi + trace_int - trace_ext;
    y.tail(n_iface) =
        psi + flux_row_scale * (beta_int * normal_int - beta_ext * normal_ext);
}

LaplaceTransmissionSolveResult3D LaplaceTransmission3D::solve(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const LaplaceTransmissionRhsData3D& rhs_data,
    const Eigen::VectorXd&              outer_dirichlet_values,
    int                                max_iter,
    double                             tol,
    int                                restart) const
{
    const int n_iface = grid_pair_.interface().num_points();
    const int n_dof = grid_pair_.grid().num_dofs();

    require_vector_size("u_jump", u_jump.size(), n_iface);
    require_vector_size("beta_flux_jump", beta_flux_jump.size(), n_iface);
    validate_rhs_data(rhs_data, n_iface, n_dof);
    if (outer_dirichlet_values.size() != 0)
        require_vector_size("outer_dirichlet_values", outer_dirichlet_values.size(), n_dof);

    switch (mode_) {
    case LaplaceTransmissionMode3D::CommonRatio:
        return solve_common_ratio(u_jump,
                                  beta_flux_jump,
                                  rhs_data,
                                  outer_dirichlet_values,
                                  max_iter,
                                  tol,
                                  restart);
    case LaplaceTransmissionMode3D::DifferentRatios:
        return solve_different_ratios(u_jump,
                                      beta_flux_jump,
                                      rhs_data,
                                      outer_dirichlet_values,
                                      max_iter,
                                      tol,
                                      restart);
    }

    throw std::invalid_argument(std::string(kContext) + ": unsupported transmission mode");
}

LaplaceTransmissionSolveResult3D LaplaceTransmission3D::solve_common_ratio(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const LaplaceTransmissionRhsData3D& rhs_data,
    const Eigen::VectorXd&              outer_dirichlet_values,
    int                                max_iter,
    double                             tol,
    int                                restart) const
{
    require_common_ratio(lambda_sq_int_, lambda_sq_ext_);

    const int n_iface = grid_pair_.interface().num_points();
    const Eigen::VectorXd rhs = apply_dirichlet_boundary_elimination(
        rhs_data.reduced_rhs_bulk, outer_dirichlet_values);
    const auto rhs_derivs = combine_rhs_derivs(
        rhs_data.reduced_rhs_int_derivs,
        rhs_data.reduced_rhs_ext_derivs);

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);

    auto rhs_setup_jumps = make_jumps(n_iface, u_jump, zeros, rhs_derivs);
    auto rhs_setup_res = potentials_int_.evaluate(rhs_setup_jumps, rhs);

    const double beta_sum = coefficients_.beta_int + coefficients_.beta_ext;
    const double gamma =
        2.0 * (coefficients_.beta_int - coefficients_.beta_ext) / beta_sum;
    Eigen::VectorXd bie_rhs =
        (2.0 / beta_sum) * beta_flux_jump
        - gamma * rhs_setup_res.un_avg;

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd psi = Eigen::VectorXd::Zero(n_iface);
    const int iterations = gmres.solve(*this, bie_rhs, psi);

    auto jumps = make_jumps(n_iface, u_jump, psi, rhs_derivs);
    auto full_res = potentials_int_.evaluate(jumps, rhs);
    restore_dirichlet_boundary(full_res.u_bulk, outer_dirichlet_values);

    Eigen::VectorXd phi = u_jump;
    auto residuals = gmres.residuals();
    return {std::move(full_res.u_bulk),
            std::move(phi),
            std::move(psi),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

LaplaceTransmissionSolveResult3D LaplaceTransmission3D::solve_different_ratios(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const LaplaceTransmissionRhsData3D& rhs_data,
    const Eigen::VectorXd&              outer_dirichlet_values,
    int                                max_iter,
    double                             tol,
    int                                restart) const
{
    const int n_iface = grid_pair_.interface().num_points();
    const int n_dof = grid_pair_.grid().num_dofs();

    Eigen::VectorXd rhs_int_raw;
    Eigen::VectorXd rhs_ext_raw;
    split_phase_rhs(grid_pair_, rhs_data.reduced_rhs_bulk, rhs_int_raw, rhs_ext_raw);
    const Eigen::VectorXd rhs_int = rhs_int_raw;
    const Eigen::VectorXd rhs_ext =
        apply_dirichlet_boundary_elimination(rhs_ext_raw, outer_dirichlet_values);

    const auto rhs_int_derivs = rhs_data.reduced_rhs_int_derivs;
    const auto rhs_ext_derivs = scaled_rhs_derivs(
        rhs_data.reduced_rhs_ext_derivs, -1.0);

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);

    auto volume_int_jumps = make_jumps(n_iface, zeros, zeros, rhs_int_derivs);
    auto volume_ext_jumps = make_jumps(n_iface, zeros, zeros, rhs_ext_derivs);
    auto volume_int = potentials_int_.evaluate(volume_int_jumps, rhs_int);
    auto volume_ext = potentials_ext_.evaluate(volume_ext_jumps, rhs_ext);

    const double beta_int = coefficients_.beta_int;
    const double beta_ext = coefficients_.beta_ext;
    const double beta_sum = beta_int + beta_ext;

    Eigen::VectorXd rhs(2 * n_iface);
    rhs.head(n_iface) = u_jump - (volume_int.u_avg - volume_ext.u_avg);
    rhs.tail(n_iface) =
        (2.0 / beta_sum)
        * (beta_flux_jump - (beta_int * volume_int.un_avg
                             - beta_ext * volume_ext.un_avg));

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd density = Eigen::VectorXd::Zero(2 * n_iface);
    const int iterations = gmres.solve(*this, rhs, density);

    Eigen::VectorXd phi = density.head(n_iface);
    Eigen::VectorXd psi = density.tail(n_iface);

    const double alpha_int = 2.0 * beta_ext / beta_sum;
    const double alpha_ext = 2.0 * beta_int / beta_sum;

    auto full_int_jumps =
        make_jumps(n_iface, alpha_int * phi, psi, rhs_int_derivs);
    auto full_ext_jumps =
        make_jumps(n_iface, alpha_ext * phi, psi, rhs_ext_derivs);

    auto full_int = potentials_int_.evaluate(full_int_jumps, rhs_int);
    auto full_ext = potentials_ext_.evaluate(full_ext_jumps, rhs_ext);

    Eigen::VectorXd u_bulk(n_dof);
    for (int n = 0; n < n_dof; ++n) {
        u_bulk[n] = (grid_pair_.domain_label(n) > 0)
                        ? full_int.u_bulk[n]
                        : full_ext.u_bulk[n];
    }
    restore_dirichlet_boundary(u_bulk, outer_dirichlet_values);

    auto residuals = gmres.residuals();
    return {std::move(u_bulk),
            std::move(phi),
            std::move(psi),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

} // namespace kfbim
