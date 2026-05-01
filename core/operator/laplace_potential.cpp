#include "laplace_potential.hpp"

#include "../transfer/i_spread.hpp"
#include "../transfer/i_restrict.hpp"
#include "../solver/i_bulk_solver.hpp"
#include "../local_cauchy/jump_data.hpp"
#include "../interface/interface_2d.hpp"
#include "../interface/interface_3d.hpp"

namespace kfbim {

// ============================================================================
// 2D
// ============================================================================

LaplacePotentialEval2D::LaplacePotentialEval2D(
    const ILaplaceSpread2D&     spread,
    const ILaplaceBulkSolver2D& bulk_solver,
    const ILaplaceRestrict2D&   restrict_op)
    : spread_(spread)
    , bulk_solver_(bulk_solver)
    , restrict_op_(restrict_op)
{}

int LaplacePotentialEval2D::problem_size() const
{
    return spread_.grid_pair().interface().num_points();
}

void LaplacePotentialEval2D::run_pipeline(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    Eigen::VectorXd&                    trace_ext,
    Eigen::VectorXd&                    flux_ext) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();
    const int   n_dof = spread_.grid_pair().grid().num_dofs();

    // 1. Build jump data
    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        jumps[i].u_jump     = u_jump[i];
        jumps[i].un_jump    = un_jump[i];
        jumps[i].rhs_derivs = rhs_derivs[i];
    }

    // 2. Spread: correction polynomials + corrected RHS
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_dof);
    auto correction_polys = spread_.apply(jumps, rhs);

    // 3. Bulk solve
    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);

    // 4. Restrict: fits bulk to interface, subtracts correction
    auto solution_polys = restrict_op_.apply(u_bulk, correction_polys);

    // 5. Extract exterior trace and exterior normal derivative
    const auto& normals = iface.normals();
    trace_ext.resize(Nq);
    flux_ext.resize(Nq);
    for (int i = 0; i < Nq; ++i) {
        trace_ext[i] = solution_polys[i].coeffs[0];
        flux_ext[i]  = solution_polys[i].coeffs[1] * normals(i, 0)
                     + solution_polys[i].coeffs[2] * normals(i, 1);
    }
}

// D[φ]: [u]=φ, [∂ₙu]=0, f=0
// u⁻ = trace_ext,  K[φ] = u⁻ + φ/2,  H[φ] = flux_ext (continuous)
void LaplacePotentialEval2D::eval_double_layer(
    const Eigen::VectorXd& phi,
    Eigen::VectorXd&       K_phi,
    Eigen::VectorXd&       H_phi) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));

    Eigen::VectorXd trace_ext, flux_ext;
    run_pipeline(phi, Eigen::VectorXd::Zero(Nq), rhs_derivs,
                 trace_ext, flux_ext);

    K_phi.resize(Nq);
    H_phi.resize(Nq);
    for (int i = 0; i < Nq; ++i) {
        K_phi[i] = trace_ext[i] + 0.5 * phi[i];
        H_phi[i] = flux_ext[i];
    }
}

// S[ψ]: [u]=0, [∂ₙu]=ψ, f=0
// S[ψ] = trace_ext (continuous),  K'[ψ] = flux_ext + ψ/2
void LaplacePotentialEval2D::eval_single_layer(
    const Eigen::VectorXd& psi,
    Eigen::VectorXd&       S_psi,
    Eigen::VectorXd&       Kt_psi) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));

    Eigen::VectorXd trace_ext, flux_ext;
    run_pipeline(Eigen::VectorXd::Zero(Nq), psi, rhs_derivs,
                 trace_ext, flux_ext);

    S_psi.resize(Nq);
    Kt_psi.resize(Nq);
    for (int i = 0; i < Nq; ++i) {
        S_psi[i]  = trace_ext[i];
        Kt_psi[i] = flux_ext[i] + 0.5 * psi[i];
    }
}

// N[q]: [u]=0, [∂ₙu]=0, [f]=q
// N[q] = trace_ext (continuous),  ∂ₙN[q] = flux_ext (continuous)
void LaplacePotentialEval2D::eval_newton(
    const Eigen::VectorXd& q,
    Eigen::VectorXd&       N_q,
    Eigen::VectorXd&       Nn_q) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq);
    for (int i = 0; i < Nq; ++i) {
        rhs_derivs[i] = Eigen::VectorXd::Zero(1);
        rhs_derivs[i][0] = q[i];
    }

    Eigen::VectorXd trace_ext, flux_ext;
    run_pipeline(Eigen::VectorXd::Zero(Nq), Eigen::VectorXd::Zero(Nq), rhs_derivs,
                 trace_ext, flux_ext);

    N_q.resize(Nq);
    Nn_q.resize(Nq);
    for (int i = 0; i < Nq; ++i) {
        N_q[i]  = trace_ext[i];
        Nn_q[i] = flux_ext[i];
    }
}

// ============================================================================
// 3D
// ============================================================================

LaplacePotentialEval3D::LaplacePotentialEval3D(
    const ILaplaceSpread3D&     spread,
    const ILaplaceBulkSolver3D& bulk_solver,
    const ILaplaceRestrict3D&   restrict_op)
    : spread_(spread)
    , bulk_solver_(bulk_solver)
    , restrict_op_(restrict_op)
{}

int LaplacePotentialEval3D::problem_size() const
{
    return spread_.grid_pair().interface().num_points();
}

void LaplacePotentialEval3D::run_pipeline(
    const Eigen::VectorXd&              u_jump,
    const Eigen::VectorXd&              un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs,
    Eigen::VectorXd&                    trace_ext,
    Eigen::VectorXd&                    flux_ext) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();
    const int   n_dof = spread_.grid_pair().grid().num_dofs();

    std::vector<LaplaceJumpData3D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        jumps[i].u_jump     = u_jump[i];
        jumps[i].un_jump    = un_jump[i];
        jumps[i].rhs_derivs = rhs_derivs[i];
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_dof);
    auto correction_polys = spread_.apply(jumps, rhs);

    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);

    auto solution_polys = restrict_op_.apply(u_bulk, correction_polys);

    const auto& normals = iface.normals();
    trace_ext.resize(Nq);
    flux_ext.resize(Nq);
    for (int i = 0; i < Nq; ++i) {
        trace_ext[i] = solution_polys[i].coeffs[0];
        flux_ext[i]  = solution_polys[i].coeffs[1] * normals(i, 0)
                     + solution_polys[i].coeffs[2] * normals(i, 1)
                     + solution_polys[i].coeffs[3] * normals(i, 2);
    }
}

void LaplacePotentialEval3D::eval_double_layer(
    const Eigen::VectorXd& phi,
    Eigen::VectorXd&       K_phi,
    Eigen::VectorXd&       H_phi) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));

    Eigen::VectorXd trace_ext, flux_ext;
    run_pipeline(phi, Eigen::VectorXd::Zero(Nq), rhs_derivs,
                 trace_ext, flux_ext);

    K_phi.resize(Nq);
    H_phi.resize(Nq);
    for (int i = 0; i < Nq; ++i) {
        K_phi[i] = trace_ext[i] + 0.5 * phi[i];
        H_phi[i] = flux_ext[i];
    }
}

void LaplacePotentialEval3D::eval_single_layer(
    const Eigen::VectorXd& psi,
    Eigen::VectorXd&       S_psi,
    Eigen::VectorXd&       Kt_psi) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));

    Eigen::VectorXd trace_ext, flux_ext;
    run_pipeline(Eigen::VectorXd::Zero(Nq), psi, rhs_derivs,
                 trace_ext, flux_ext);

    S_psi.resize(Nq);
    Kt_psi.resize(Nq);
    for (int i = 0; i < Nq; ++i) {
        S_psi[i]  = trace_ext[i];
        Kt_psi[i] = flux_ext[i] + 0.5 * psi[i];
    }
}

void LaplacePotentialEval3D::eval_newton(
    const Eigen::VectorXd& q,
    Eigen::VectorXd&       N_q,
    Eigen::VectorXd&       Nn_q) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq);
    for (int i = 0; i < Nq; ++i) {
        rhs_derivs[i] = Eigen::VectorXd::Zero(1);
        rhs_derivs[i][0] = q[i];
    }

    Eigen::VectorXd trace_ext, flux_ext;
    run_pipeline(Eigen::VectorXd::Zero(Nq), Eigen::VectorXd::Zero(Nq), rhs_derivs,
                 trace_ext, flux_ext);

    N_q.resize(Nq);
    Nn_q.resize(Nq);
    for (int i = 0; i < Nq; ++i) {
        N_q[i]  = trace_ext[i];
        Nn_q[i] = flux_ext[i];
    }
}

} // namespace kfbim
