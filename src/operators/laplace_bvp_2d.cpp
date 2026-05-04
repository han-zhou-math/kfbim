#include "laplace_bvp_2d.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "../gmres/gmres.hpp"
#include "../local_cauchy/jump_data.hpp"
#include "../bulk_solvers/zfft_bc_type.hpp"

namespace kfbim {

namespace {

void require_vector_size(const char* context,
                         const char* name,
                         Eigen::Index actual,
                         Eigen::Index expected)
{
    if (actual != expected) {
        throw std::invalid_argument(
            std::string(context) + ": " + name
            + " has size " + std::to_string(actual)
            + ", expected " + std::to_string(expected));
    }
}

bool is_boundary_node(int i, int j, int nx, int ny)
{
    return i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
}

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
    return type == LaplaceBvpType2D::InteriorNeumann && std::abs(eta) < 1.0e-14;
}

void project_mean_zero(Eigen::VectorXd& v)
{
    if (v.size() == 0)
        throw std::invalid_argument("project_mean_zero: vector must be nonempty");
    v.array() -= v.mean();
}

class MeanProjectedOperator final : public IKFBIOperator {
public:
    explicit MeanProjectedOperator(const IKFBIOperator& op)
        : op_(op)
    {
        if (op_.problem_size() <= 0)
            throw std::invalid_argument(
                "MeanProjectedOperator: problem size must be positive");
    }

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override
    {
        Eigen::VectorXd x_projected = x;
        project_mean_zero(x_projected);

        op_.apply(x_projected, y);
        project_mean_zero(y);
    }

    int problem_size() const override { return op_.problem_size(); }

private:
    const IKFBIOperator& op_;
};

std::unique_ptr<ILaplaceSpread2D> make_laplace_spread_2d(
    LaplaceBvpPanelMethod2D method,
    const GridPair2D&       grid_pair,
    double                  eta)
{
    switch (method) {
    case LaplaceBvpPanelMethod2D::ChebyshevLobattoCenter:
        return std::make_unique<LaplaceLobattoCenterSpread2D>(grid_pair, eta);
    }
    throw std::invalid_argument("unsupported Laplace BVP panel method");
}

std::unique_ptr<ILaplaceRestrict2D> make_laplace_restrict_2d(
    LaplaceBvpPanelMethod2D method,
    const GridPair2D&       grid_pair)
{
    switch (method) {
    case LaplaceBvpPanelMethod2D::ChebyshevLobattoCenter:
        return std::make_unique<LaplaceLobattoCenterRestrict2D>(grid_pair);
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

std::vector<Eigen::VectorXd> signed_rhs_derivs(
    const char* context,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    int n_iface,
    double sign)
{
    if (static_cast<int>(rhs_derivs.size()) != n_iface) {
        throw std::invalid_argument(
            std::string(context) + ": rhs_derivs length does not match interface size");
    }

    std::vector<Eigen::VectorXd> signed_derivs(n_iface);
    for (int q = 0; q < n_iface; ++q)
        signed_derivs[q] = sign * rhs_derivs[q];
    return signed_derivs;
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
    if (outer_dirichlet_values_.size() == 0)
        return rhs;

    const auto& grid = grid_pair_.grid();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];

    Eigen::VectorXd modified = rhs;
    const double hx = grid.spacing()[0];
    const double hy = grid.spacing()[1];
    const double inv_hx2 = 1.0 / (hx * hx);
    const double inv_hy2 = 1.0 / (hy * hy);

    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const int n = grid.index(i, j);
            double boundary_lift = 0.0;
            if (i == 1)
                boundary_lift += outer_dirichlet_values_[grid.index(0, j)] * inv_hx2;
            if (i == nx - 2)
                boundary_lift += outer_dirichlet_values_[grid.index(nx - 1, j)] * inv_hx2;
            if (j == 1)
                boundary_lift += outer_dirichlet_values_[grid.index(i, 0)] * inv_hy2;
            if (j == ny - 2)
                boundary_lift += outer_dirichlet_values_[grid.index(i, ny - 1)] * inv_hy2;
            modified[n] += boundary_lift;
        }
    }

    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            if (is_boundary_node(i, j, nx, ny))
                modified[grid.index(i, j)] = 0.0;

    return modified;
}

void LaplaceBvp2D::restore_dirichlet_boundary(Eigen::VectorXd& u_bulk) const
{
    if (outer_dirichlet_values_.size() == 0)
        return;

    const auto& grid = grid_pair_.grid();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];

    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            if (is_boundary_node(i, j, nx, ny))
                u_bulk[grid.index(i, j)] = outer_dirichlet_values_[grid.index(i, j)];
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
    const int n_iface = grid_pair_.interface().num_points();
    require_vector_size("LaplaceBvp2D", "boundary_data", boundary_data.size(), n_iface);
    require_vector_size("LaplaceBvp2D", "f_bulk", f_bulk.size(), grid_pair_.grid().num_dofs());

    const auto signed_derivs = signed_rhs_derivs(
        "LaplaceBvp2D", rhs_derivs, n_iface, rhs_deriv_sign_);
    const Eigen::VectorXd rhs = apply_dirichlet_boundary_elimination(f_bulk);

    Eigen::VectorXd volume_gamma;
    apply_with_rhs(Eigen::VectorXd::Zero(n_iface), rhs, signed_derivs, volume_gamma);

    Eigen::VectorXd b = boundary_data - volume_gamma;

    GMRES gmres(max_iter, tol, restart);
    Eigen::VectorXd density = Eigen::VectorXd::Zero(n_iface);
    int iterations = 0;
    if (uses_neumann_nullspace_projection(type_, eta_)) {
        project_mean_zero(b);

        MeanProjectedOperator projected_op(*this);
        iterations = gmres.solve(projected_op, b, density);
        project_mean_zero(density);
    } else {
        iterations = gmres.solve(*this, b, density);
    }

    auto jumps = make_jumps(grid_pair_.interface(), density, type_, signed_derivs);
    auto full_res = potentials_.evaluate(jumps, rhs);
    restore_dirichlet_boundary(full_res.u_bulk);

    auto residuals = gmres.residuals();
    return {std::move(full_res.u_bulk),
            std::move(density),
            std::move(residuals),
            iterations,
            gmres.converged()};
}

} // namespace kfbim
