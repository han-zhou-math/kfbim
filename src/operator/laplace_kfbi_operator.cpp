#include "laplace_kfbi_operator.hpp"

#include "../local_cauchy/jump_data.hpp"
#include "../solver/i_bulk_solver.hpp"
#include "../transfer/i_restrict.hpp"
#include "../transfer/i_spread.hpp"

namespace kfbim {

namespace {

bool is_neumann_mode(LaplaceKFBIMode mode)
{
    return mode == LaplaceKFBIMode::InteriorNeumann
        || mode == LaplaceKFBIMode::ExteriorNeumann;
}

bool is_exterior_mode(LaplaceKFBIMode mode)
{
    return mode == LaplaceKFBIMode::ExteriorDirichlet
        || mode == LaplaceKFBIMode::ExteriorNeumann;
}

double side_jump_sign(LaplaceKFBIMode mode)
{
    return is_exterior_mode(mode) ? -1.0 : 1.0;
}

} // namespace

// ---------------------------------------------------------------------------
// 2D
// ---------------------------------------------------------------------------

LaplaceKFBIOperator2D::LaplaceKFBIOperator2D(
    const ILaplaceSpread2D&      spread,
    const ILaplaceBulkSolver2D&  bulk_solver,
    const ILaplaceRestrict2D&    restrict_op,
    Eigen::VectorXd              base_rhs,
    std::vector<Eigen::VectorXd> rhs_derivs,
    LaplaceKFBIMode              mode)
    : potentials_(spread, bulk_solver, restrict_op)
    , base_rhs_(std::move(base_rhs))
    , rhs_derivs_(std::move(rhs_derivs))
    , mode_(mode)
{}

void LaplaceKFBIOperator2D::apply(const Eigen::VectorXd& x,
                                   Eigen::VectorXd&       y) const
{
    const int Nq = potentials_.problem_size();

    // 1. Unpack x → LaplaceJumpData
    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        if (is_neumann_mode(mode_)) {
            // Single-layer: [u]=0, [un]=phi
            jumps[i].u_jump  = 0.0;
            jumps[i].un_jump = x[i];
        } else {
            // Double-layer: [u]=phi, [un]=0
            jumps[i].u_jump  = x[i];
            jumps[i].un_jump = 0.0;
        }
        jumps[i].rhs_derivs = Eigen::VectorXd::Zero(rhs_derivs_[i].size());
    }

    // 2. Run full pipeline via solver
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(base_rhs_.size());
    auto res = potentials_.evaluate(jumps, rhs);

    // 3. Pack y from the appropriate trace
    y.resize(Nq);
    const double sign = side_jump_sign(mode_);
    if (!is_neumann_mode(mode_)) {
        // Side trace: u_side = u_avg +/- [u]/2
        for (int i = 0; i < Nq; ++i)
            y[i] = res.u_avg[i] + sign * 0.5 * jumps[i].u_jump;
    } else {
        // Side normal derivative: un_side = un_avg +/- [un]/2
        for (int i = 0; i < Nq; ++i)
            y[i] = res.un_avg[i] + sign * 0.5 * jumps[i].un_jump;
    }
}

void LaplaceKFBIOperator2D::apply_full(const Eigen::VectorXd& x,
                                        Eigen::VectorXd&       y) const
{
    const int Nq = potentials_.problem_size();

    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        if (is_neumann_mode(mode_)) {
            jumps[i].u_jump  = 0.0;
            jumps[i].un_jump = x[i];
        } else {
            jumps[i].u_jump  = x[i];
            jumps[i].un_jump = 0.0;
        }
        jumps[i].rhs_derivs = rhs_derivs_[i];
    }

    auto res = potentials_.evaluate(jumps, base_rhs_);

    y.resize(Nq);
    const double sign = side_jump_sign(mode_);
    if (!is_neumann_mode(mode_)) {
        for (int i = 0; i < Nq; ++i)
            y[i] = res.u_avg[i] + sign * 0.5 * jumps[i].u_jump;
    } else {
        for (int i = 0; i < Nq; ++i)
            y[i] = res.un_avg[i] + sign * 0.5 * jumps[i].un_jump;
    }
}

int LaplaceKFBIOperator2D::problem_size() const
{
    return potentials_.problem_size();
}

// ---------------------------------------------------------------------------
// 3D
// ---------------------------------------------------------------------------

LaplaceKFBIOperator3D::LaplaceKFBIOperator3D(
    const ILaplaceSpread3D&      spread,
    const ILaplaceBulkSolver3D&  bulk_solver,
    const ILaplaceRestrict3D&    restrict_op,
    Eigen::VectorXd              base_rhs,
    std::vector<Eigen::VectorXd> rhs_derivs,
    LaplaceKFBIMode              mode)
    : spread_(spread)
    , bulk_solver_(bulk_solver)
    , restrict_op_(restrict_op)
    , base_rhs_(std::move(base_rhs))
    , rhs_derivs_(std::move(rhs_derivs))
    , mode_(mode)
{}

void LaplaceKFBIOperator3D::apply(const Eigen::VectorXd& x,
                                   Eigen::VectorXd&       y) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();

    std::vector<LaplaceJumpData3D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        if (is_neumann_mode(mode_)) {
            jumps[i].u_jump  = 0.0;
            jumps[i].un_jump = x[i];
        } else {
            jumps[i].u_jump  = x[i];
            jumps[i].un_jump = 0.0;
        }
        jumps[i].rhs_derivs = Eigen::VectorXd::Zero(rhs_derivs_[i].size());
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(base_rhs_.size());
    auto correction_polys = spread_.apply(jumps, rhs);

    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);

    auto solution_polys = restrict_op_.apply(u_bulk, correction_polys);

    y.resize(Nq);
    if (!is_neumann_mode(mode_)) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0]
                 - (is_exterior_mode(mode_) ? jumps[i].u_jump : 0.0);
    } else {
        const auto& normals = iface.normals();
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[1] * normals(i, 0)
                 + solution_polys[i].coeffs[2] * normals(i, 1)
                 + solution_polys[i].coeffs[3] * normals(i, 2)
                 - (is_exterior_mode(mode_) ? jumps[i].un_jump : 0.0);
    }
}

void LaplaceKFBIOperator3D::apply_full(const Eigen::VectorXd& x,
                                        Eigen::VectorXd&       y) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();

    std::vector<LaplaceJumpData3D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        if (is_neumann_mode(mode_)) {
            jumps[i].u_jump  = 0.0;
            jumps[i].un_jump = x[i];
        } else {
            jumps[i].u_jump  = x[i];
            jumps[i].un_jump = 0.0;
        }
        jumps[i].rhs_derivs = rhs_derivs_[i];
    }

    Eigen::VectorXd rhs = base_rhs_;
    auto correction_polys = spread_.apply(jumps, rhs);

    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);

    auto solution_polys = restrict_op_.apply(u_bulk, correction_polys);

    y.resize(Nq);
    if (!is_neumann_mode(mode_)) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0]
                 - (is_exterior_mode(mode_) ? jumps[i].u_jump : 0.0);
    } else {
        const auto& normals = iface.normals();
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[1] * normals(i, 0)
                 + solution_polys[i].coeffs[2] * normals(i, 1)
                 + solution_polys[i].coeffs[3] * normals(i, 2)
                 - (is_exterior_mode(mode_) ? jumps[i].un_jump : 0.0);
    }
}

int LaplaceKFBIOperator3D::problem_size() const
{
    return spread_.grid_pair().interface().num_points();
}

} // namespace kfbim
