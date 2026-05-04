#include "laplace_potential.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include "../transfer/i_spread.hpp"
#include "../transfer/i_restrict.hpp"
#include "../bulk_solvers/i_bulk_solver.hpp"
#include "../interface/interface_2d.hpp"
#include "../interface/interface_3d.hpp"

namespace kfbim {

// ============================================================================
// 2D
// ============================================================================

namespace {

double compute_arc_h_ratio_2d(const ILaplaceSpread2D& spread)
{
    const auto& grid = spread.grid_pair().grid();
    const double h = std::max(grid.spacing()[0], grid.spacing()[1]);

    const auto& iface = spread.grid_pair().interface();
    const auto& pts = iface.points();
    double min_arc = std::numeric_limits<double>::max();
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (int local_q = 0; local_q < iface.points_per_panel() - 1; ++local_q) {
            const int q0 = iface.point_index(p, local_q);
            const int q1 = iface.point_index(p, local_q + 1);
            const double dx = pts(q0, 0) - pts(q1, 0);
            const double dy = pts(q0, 1) - pts(q1, 1);
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d > 1e-14 && d < min_arc)
                min_arc = d;
        }
    }

    return (min_arc == std::numeric_limits<double>::max()) ? 0.0 : min_arc / h;
}

std::vector<LaplaceJumpData2D> make_jumps_2d(
    int                                n_iface,
    const Eigen::VectorXd&             u_jump,
    const Eigen::VectorXd&             un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    if (u_jump.size() != n_iface || un_jump.size() != n_iface
        || static_cast<int>(rhs_derivs.size()) != n_iface) {
        throw std::invalid_argument(
            "LaplacePotentialEval2D: specialized jump inputs must match interface size");
    }

    std::vector<LaplaceJumpData2D> jumps(n_iface);
    for (int i = 0; i < n_iface; ++i) {
        jumps[i].u_jump = u_jump[i];
        jumps[i].un_jump = un_jump[i];
        jumps[i].rhs_derivs = rhs_derivs[i];
    }
    return jumps;
}

} // namespace

LaplacePotentialEval2D::LaplacePotentialEval2D(
    const ILaplaceSpread2D&     spread,
    const ILaplaceBulkSolver2D& bulk_solver,
    const ILaplaceRestrict2D&   restrict_op)
    : spread_(spread)
    , bulk_solver_(bulk_solver)
    , restrict_op_(restrict_op)
    , arc_h_ratio_(compute_arc_h_ratio_2d(spread))
{
    if (arc_h_ratio_ < 0.5) {
        std::cerr << "LaplacePotentialEval2D: arc_h_ratio = "
                  << arc_h_ratio_ << " < 0.5 -- interface may be under-resolved\n";
    }
}

int LaplacePotentialEval2D::problem_size() const
{
    return spread_.grid_pair().interface().num_points();
}

double LaplacePotentialEval2D::arc_h_ratio() const
{
    return arc_h_ratio_;
}

LaplacePotentialEvalResult2D LaplacePotentialEval2D::evaluate(
    const std::vector<LaplaceJumpData2D>& jumps,
    const Eigen::VectorXd&                f_bulk) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();
    const int   n_dof = spread_.grid_pair().grid().num_dofs();

    if (static_cast<int>(jumps.size()) != Nq) {
        throw std::invalid_argument(
            "LaplacePotentialEval2D::evaluate jumps size must match interface size");
    }
    if (f_bulk.size() != n_dof) {
        throw std::invalid_argument(
            "LaplacePotentialEval2D::evaluate f_bulk size must match grid DOF count");
    }

    Eigen::VectorXd rhs = f_bulk;
    auto correction_polys = spread_.apply(jumps, rhs);

    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);

    auto solution_polys = restrict_op_.apply(u_bulk, correction_polys);

    Eigen::VectorXd u_avg(Nq);
    Eigen::VectorXd un_avg(Nq);
    const auto& normals = iface.normals();
    for (int i = 0; i < Nq; ++i) {
        u_avg[i] = solution_polys[i].coeffs[0];
        un_avg[i] = solution_polys[i].coeffs[1] * normals(i, 0)
                  + solution_polys[i].coeffs[2] * normals(i, 1);
    }

    return {std::move(u_bulk), std::move(u_avg), std::move(un_avg)};
}

// D[φ]: [u]=φ, [∂ₙu]=0, f=0
// K[φ] = averaged trace, H[φ] = averaged normal derivative.
void LaplacePotentialEval2D::eval_double_layer(
    const Eigen::VectorXd& phi,
    Eigen::VectorXd&       K_phi,
    Eigen::VectorXd&       H_phi) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));
    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(Nq);
    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(spread_.grid_pair().grid().num_dofs());

    auto result = evaluate(make_jumps_2d(Nq, phi, zeros, rhs_derivs), zero_rhs);
    K_phi = std::move(result.u_avg);
    H_phi = std::move(result.un_avg);
}

// S[ψ]: [u]=0, [∂ₙu]=ψ, f=0
// S[ψ] = averaged trace, K'[ψ] = averaged normal derivative.
void LaplacePotentialEval2D::eval_single_layer(
    const Eigen::VectorXd& psi,
    Eigen::VectorXd&       S_psi,
    Eigen::VectorXd&       Kt_psi) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));
    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(Nq);
    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(spread_.grid_pair().grid().num_dofs());

    auto result = evaluate(make_jumps_2d(Nq, zeros, psi, rhs_derivs), zero_rhs);
    S_psi = std::move(result.u_avg);
    Kt_psi = std::move(result.un_avg);
}

// N[q]: [u]=0, [∂ₙu]=0, [f]=q
// N[q] = averaged trace, ∂ₙN[q] = averaged normal derivative.
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

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(Nq);
    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(spread_.grid_pair().grid().num_dofs());

    auto result = evaluate(make_jumps_2d(Nq, zeros, zeros, rhs_derivs), zero_rhs);
    N_q = std::move(result.u_avg);
    Nn_q = std::move(result.un_avg);
}

// ============================================================================
// 3D
// ============================================================================

namespace {

std::vector<LaplaceJumpData3D> make_jumps_3d(
    int                                n_iface,
    const Eigen::VectorXd&             u_jump,
    const Eigen::VectorXd&             un_jump,
    const std::vector<Eigen::VectorXd>& rhs_derivs)
{
    if (u_jump.size() != n_iface || un_jump.size() != n_iface
        || static_cast<int>(rhs_derivs.size()) != n_iface) {
        throw std::invalid_argument(
            "LaplacePotentialEval3D: specialized jump inputs must match interface size");
    }

    std::vector<LaplaceJumpData3D> jumps(n_iface);
    for (int i = 0; i < n_iface; ++i) {
        jumps[i].u_jump = u_jump[i];
        jumps[i].un_jump = un_jump[i];
        jumps[i].rhs_derivs = rhs_derivs[i];
    }
    return jumps;
}

} // namespace

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

LaplacePotentialEvalResult3D LaplacePotentialEval3D::evaluate(
    const std::vector<LaplaceJumpData3D>& jumps,
    const Eigen::VectorXd&                f_bulk) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();
    const int   n_dof = spread_.grid_pair().grid().num_dofs();

    if (static_cast<int>(jumps.size()) != Nq) {
        throw std::invalid_argument(
            "LaplacePotentialEval3D::evaluate jumps size must match interface size");
    }
    if (f_bulk.size() != n_dof) {
        throw std::invalid_argument(
            "LaplacePotentialEval3D::evaluate f_bulk size must match grid DOF count");
    }

    Eigen::VectorXd rhs = f_bulk;
    auto correction_polys = spread_.apply(jumps, rhs);

    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);

    auto solution_polys = restrict_op_.apply(u_bulk, correction_polys);

    const auto& normals = iface.normals();
    Eigen::VectorXd u_avg(Nq);
    Eigen::VectorXd un_avg(Nq);
    for (int i = 0; i < Nq; ++i) {
        u_avg[i] = solution_polys[i].coeffs[0];
        un_avg[i] = solution_polys[i].coeffs[1] * normals(i, 0)
                  + solution_polys[i].coeffs[2] * normals(i, 1)
                  + solution_polys[i].coeffs[3] * normals(i, 2);
    }

    return {std::move(u_bulk), std::move(u_avg), std::move(un_avg)};
}

void LaplacePotentialEval3D::eval_double_layer(
    const Eigen::VectorXd& phi,
    Eigen::VectorXd&       K_phi,
    Eigen::VectorXd&       H_phi) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));
    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(Nq);
    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(spread_.grid_pair().grid().num_dofs());

    auto result = evaluate(make_jumps_3d(Nq, phi, zeros, rhs_derivs), zero_rhs);
    K_phi = std::move(result.u_avg);
    H_phi = std::move(result.un_avg);
}

void LaplacePotentialEval3D::eval_single_layer(
    const Eigen::VectorXd& psi,
    Eigen::VectorXd&       S_psi,
    Eigen::VectorXd&       Kt_psi) const
{
    const int Nq = problem_size();
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));
    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(Nq);
    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(spread_.grid_pair().grid().num_dofs());

    auto result = evaluate(make_jumps_3d(Nq, zeros, psi, rhs_derivs), zero_rhs);
    S_psi = std::move(result.u_avg);
    Kt_psi = std::move(result.un_avg);
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

    const Eigen::VectorXd zeros = Eigen::VectorXd::Zero(Nq);
    const Eigen::VectorXd zero_rhs =
        Eigen::VectorXd::Zero(spread_.grid_pair().grid().num_dofs());

    auto result = evaluate(make_jumps_3d(Nq, zeros, zeros, rhs_derivs), zero_rhs);
    N_q = std::move(result.u_avg);
    Nn_q = std::move(result.un_avg);
}

} // namespace kfbim
