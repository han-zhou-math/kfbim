#include "laplace_interior.hpp"
#include "../gmres/gmres.hpp"
#include "../local_cauchy/jump_data.hpp"

namespace kfbim {

LaplaceInteriorDirichlet2D::LaplaceInteriorDirichlet2D(
    const CartesianGrid2D& grid,
    const Interface2D&     iface,
    const Eigen::VectorXd& g,
    const Eigen::VectorXd& f_bulk,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
    : grid_pair_(grid, iface)
    , spread_(grid_pair_)
    , bulk_solver_(grid, ZfftBcType::Dirichlet)
    , restrict_op_(grid_pair_)
    , g_(g)
    , f_bulk_(f_bulk)
    , rhs_derivs_(rhs_derivs)
{
}

LaplaceInteriorSolveResult2D LaplaceInteriorDirichlet2D::solve(int max_iter, double tol, int restart)
{
    const int Nq = grid_pair_.interface().num_points();

    // 1. Instantiate the operator
    LaplaceKFBIOperator2D op(spread_, bulk_solver_, restrict_op_, f_bulk_, rhs_derivs_, LaplaceKFBIMode::Dirichlet);

    // 2. Evaluate volume potential: V f|_Γ
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

    LaplaceInterfaceSolver2D iface_solver(spread_, bulk_solver_, restrict_op_);
    auto res = iface_solver.solve(jumps, f_bulk_);

    return {std::move(res.u_bulk), std::move(phi), iterations, gmres_solver.converged()};
}

} // namespace kfbim
