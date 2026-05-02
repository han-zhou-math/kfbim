#include "laplace_transmission_constant_ratio_2d.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "../gmres/gmres.hpp"
#include "../local_cauchy/jump_data.hpp"
#include "../operator/i_kfbi_operator.hpp"
#include "../operator/laplace_potential.hpp"
#include "../solver/zfft_bc_type.hpp"
#include "laplace_interface_solver_2d.hpp"

namespace kfbim {

namespace {

void require_vector_size(const char* name, Eigen::Index actual, Eigen::Index expected)
{
    if (actual != expected) {
        throw std::invalid_argument(
            std::string("LaplaceTransmissionConstantRatio2D: ") + name
            + " has size " + std::to_string(actual)
            + ", expected " + std::to_string(expected));
    }
}

bool is_boundary_node(int i, int j, int nx, int ny)
{
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

std::vector<LaplaceJumpData2D> make_jumps(
    const Interface2D&                 iface,
    const Eigen::VectorXd&             u_jump,
    const Eigen::VectorXd&             un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    const int n_iface = iface.num_points();
    std::vector<LaplaceJumpData2D> jumps(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        jumps[q].u_jump = u_jump[q];
        jumps[q].un_jump = un_jump[q];
        jumps[q].rhs_derivs = rhs_derivs[q];
    }
    return jumps;
}

class ConstantRatioFluxOperator2D final : public IKFBIOperator {
public:
    ConstantRatioFluxOperator2D(const LaplacePotentialEval2D& potentials,
                                double                        gamma)
        : potentials_(potentials)
        , gamma_(gamma)
    {}

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        Eigen::VectorXd s_x;
        Eigen::VectorXd kt_x;
        potentials_.eval_single_layer(x, s_x, kt_x);
        y = x + gamma_ * kt_x;
    }

    int problem_size() const override
    {
        return potentials_.problem_size();
    }

private:
    const LaplacePotentialEval2D& potentials_;
    double                        gamma_;
};

} // namespace

LaplaceTransmissionConstantRatio2D::LaplaceTransmissionConstantRatio2D(
    const CartesianGrid2D& grid,
    const Interface2D&     iface,
    double                 beta_int,
    double                 beta_ext,
    double                 lambda_sq)
    : grid_pair_(grid, iface)
    , spread_(grid_pair_, lambda_sq)
    , bulk_solver_(grid, ZfftBcType::Dirichlet, lambda_sq)
    , restrict_op_(grid_pair_)
    , beta_int_(beta_int)
    , beta_ext_(beta_ext)
    , lambda_sq_(lambda_sq)
{
    if (beta_int_ <= 0.0 || beta_ext_ <= 0.0)
        throw std::invalid_argument("LaplaceTransmissionConstantRatio2D: beta values must be positive");
    if (lambda_sq_ < 0.0)
        throw std::invalid_argument("LaplaceTransmissionConstantRatio2D: lambda_sq must be nonnegative");
}

Eigen::VectorXd LaplaceTransmissionConstantRatio2D::apply_dirichlet_boundary_elimination(
    const Eigen::VectorXd& reduced_rhs_bulk,
    const Eigen::VectorXd& outer_dirichlet_values) const
{
    const auto& grid = grid_pair_.grid();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int n_dof = nx * ny;

    require_vector_size("reduced_rhs_bulk", reduced_rhs_bulk.size(), n_dof);
    if (outer_dirichlet_values.size() == 0)
        return reduced_rhs_bulk;
    require_vector_size("outer_dirichlet_values", outer_dirichlet_values.size(), n_dof);

    Eigen::VectorXd modified = reduced_rhs_bulk;
    const double hx = grid.spacing()[0];
    const double hy = grid.spacing()[1];
    const double inv_hx2 = 1.0 / (hx * hx);
    const double inv_hy2 = 1.0 / (hy * hy);

    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const int n = grid.index(i, j);
            double boundary_lift = 0.0;
            if (i == 1)
                boundary_lift += outer_dirichlet_values[grid.index(0, j)] * inv_hx2;
            if (i == nx - 2)
                boundary_lift += outer_dirichlet_values[grid.index(nx - 1, j)] * inv_hx2;
            if (j == 1)
                boundary_lift += outer_dirichlet_values[grid.index(i, 0)] * inv_hy2;
            if (j == ny - 2)
                boundary_lift += outer_dirichlet_values[grid.index(i, ny - 1)] * inv_hy2;
            modified[n] += boundary_lift;
        }
    }

    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            if (is_boundary_node(i, j, nx, ny))
                modified[grid.index(i, j)] = 0.0;

    return modified;
}

void LaplaceTransmissionConstantRatio2D::restore_dirichlet_boundary(
    Eigen::VectorXd&       u_bulk,
    const Eigen::VectorXd& outer_dirichlet_values) const
{
    if (outer_dirichlet_values.size() == 0)
        return;

    const auto& grid = grid_pair_.grid();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    require_vector_size("u_bulk", u_bulk.size(), nx * ny);
    require_vector_size("outer_dirichlet_values", outer_dirichlet_values.size(), nx * ny);

    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            if (is_boundary_node(i, j, nx, ny))
                u_bulk[grid.index(i, j)] = outer_dirichlet_values[grid.index(i, j)];
}

LaplaceTransmissionConstantRatioResult2D LaplaceTransmissionConstantRatio2D::solve(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              beta_flux_jump,
    const Eigen::VectorXd&              reduced_rhs_bulk,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    const Eigen::VectorXd&              outer_dirichlet_values,
    int                                max_iter,
    double                             tol,
    int                                restart) const
{
    const auto& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();
    const int n_dof = grid_pair_.grid().num_dofs();

    require_vector_size("u_jump", u_jump.size(), n_iface);
    require_vector_size("beta_flux_jump", beta_flux_jump.size(), n_iface);
    require_vector_size("reduced_rhs_bulk", reduced_rhs_bulk.size(), n_dof);
    if (static_cast<int>(rhs_derivs.size()) != n_iface) {
        throw std::invalid_argument(
            "LaplaceTransmissionConstantRatio2D: rhs_derivs length does not match interface size");
    }
    for (int q = 0; q < n_iface; ++q) {
        if (rhs_derivs[q].size() == 0) {
            throw std::invalid_argument(
                "LaplaceTransmissionConstantRatio2D: rhs_derivs entries must contain at least [q]");
        }
    }

    const Eigen::VectorXd rhs =
        apply_dirichlet_boundary_elimination(reduced_rhs_bulk, outer_dirichlet_values);

    LaplaceInterfaceSolver2D iface_solver(spread_, bulk_solver_, restrict_op_);
    LaplacePotentialEval2D potentials(spread_, bulk_solver_, restrict_op_);

    Eigen::VectorXd k_mu;
    Eigen::VectorXd h_mu;
    potentials.eval_double_layer(u_jump, k_mu, h_mu);

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(n_iface);
    auto volume_jumps = make_jumps(iface, zeros, zeros, rhs_derivs);
    auto volume_res = iface_solver.solve(volume_jumps, rhs);
    const Eigen::VectorXd& volume_normal = volume_res.un_avg;

    const double beta_sum = beta_int_ + beta_ext_;
    const double gamma = 2.0 * (beta_int_ - beta_ext_) / beta_sum;
    Eigen::VectorXd bie_rhs =
        (2.0 / beta_sum) * beta_flux_jump - gamma * (h_mu + volume_normal);

    ConstantRatioFluxOperator2D op(potentials, gamma);
    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd psi = Eigen::VectorXd::Zero(n_iface);
    const int iterations = gmres.solve(op, bie_rhs, psi);

    auto jumps = make_jumps(iface, u_jump, psi, rhs_derivs);
    auto full_res = iface_solver.solve(jumps, rhs);
    restore_dirichlet_boundary(full_res.u_bulk, outer_dirichlet_values);

    return {std::move(full_res.u_bulk),
            std::move(psi),
            gmres.residuals(),
            iterations,
            gmres.converged()};
}

} // namespace kfbim
