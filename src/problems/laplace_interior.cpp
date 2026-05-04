#include "laplace_interior.hpp"
#include <stdexcept>

#include "../gmres/gmres.hpp"
#include "../local_cauchy/jump_data.hpp"
#include "../operator/laplace_potential.hpp"

namespace kfbim {

namespace {

std::unique_ptr<ILaplaceSpread2D> make_laplace_spread_2d(
    LaplaceInteriorPanelMethod2D method,
    const GridPair2D&            grid_pair,
    double                       kappa)
{
    switch (method) {
    case LaplaceInteriorPanelMethod2D::ChebyshevLobattoCenter:
        return std::make_unique<LaplaceLobattoCenterSpread2D>(grid_pair, kappa);
    }
    throw std::invalid_argument("unsupported Laplace interior panel method");
}

std::unique_ptr<ILaplaceRestrict2D> make_laplace_restrict_2d(
    LaplaceInteriorPanelMethod2D method,
    const GridPair2D&            grid_pair)
{
    switch (method) {
    case LaplaceInteriorPanelMethod2D::ChebyshevLobattoCenter:
        return std::make_unique<LaplaceLobattoCenterRestrict2D>(grid_pair);
    }
    throw std::invalid_argument("unsupported Laplace interior panel method");
}

} // namespace

LaplaceInteriorDirichlet2D::LaplaceInteriorDirichlet2D(
    const CartesianGrid2D& grid,
    const Interface2D&     iface,
    const Eigen::VectorXd& g,
    const Eigen::VectorXd& f_bulk,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    LaplaceInteriorPanelMethod2D panel_method,
    double eta)
    : grid_pair_(grid, iface)
    , spread_(make_laplace_spread_2d(panel_method, grid_pair_, eta))
    , bulk_solver_(grid, ZfftBcType::Dirichlet, eta)
    , restrict_op_(make_laplace_restrict_2d(panel_method, grid_pair_))
    , g_(g)
    , f_bulk_(f_bulk)
    , rhs_derivs_(rhs_derivs)
{
}

LaplaceInteriorSolveResult2D LaplaceInteriorDirichlet2D::solve(int max_iter, double tol, int restart)
{
    const int Nq = grid_pair_.interface().num_points();

    // 1. Instantiate the operator
    LaplaceKFBIOperator2D op(*spread_, bulk_solver_, *restrict_op_, f_bulk_, rhs_derivs_, LaplaceKFBIMode::Dirichlet);

    // 2. Evaluate volume potential trace: V f|_Γ
    Eigen::VectorXd Vf_gamma;
    Eigen::VectorXd zero_phi = Eigen::VectorXd::Zero(Nq);
    op.apply_full(zero_phi, Vf_gamma);

    // 3. Form RHS for GMRES: b = g - V f|_Γ
    Eigen::VectorXd b = g_ - Vf_gamma;

    // 4. Solve using GMRES
    GMRES gmres_solver(max_iter, tol, restart);
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(Nq);
    int iterations = gmres_solver.solve(op, b, phi);

    // 5. Recover the bulk solution
    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        jumps[i].u_jump  = phi[i];
        jumps[i].un_jump = 0.0;
        jumps[i].rhs_derivs = rhs_derivs_[i];
    }

    LaplacePotentialEval2D potentials(*spread_, bulk_solver_, *restrict_op_);
    auto res = potentials.evaluate(jumps, f_bulk_);

    return {std::move(res.u_bulk),
            std::move(phi),
            iterations,
            gmres_solver.converged(),
            gmres_solver.residuals()};
}

} // namespace kfbim
