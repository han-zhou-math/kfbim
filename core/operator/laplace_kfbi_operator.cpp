#include "laplace_kfbi_operator.hpp"

#include "../local_cauchy/jump_data.hpp"

namespace kfbim {

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
    : spread_(spread)
    , bulk_solver_(bulk_solver)
    , restrict_op_(restrict_op)
    , base_rhs_(std::move(base_rhs))
    , rhs_derivs_(std::move(rhs_derivs))
    , mode_(mode)
{}

void LaplaceKFBIOperator2D::apply(const Eigen::VectorXd& x,
                                   Eigen::VectorXd&       y) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();

    // 1. Unpack x → LaplaceJumpData (Linear part: ignore RHS derivatives)
    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        if (mode_ == LaplaceKFBIMode::Dirichlet) {
            jumps[i].u_jump  = 0.0;
            jumps[i].un_jump = x[i];
        } else {
            jumps[i].u_jump  = x[i];
            jumps[i].un_jump = 0.0;
        }
        jumps[i].rhs_derivs = Eigen::VectorXd::Zero(rhs_derivs_[i].size());
    }

    // 2. rhs = Spread correction (Linear part: ignore base_rhs_)
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(base_rhs_.size());
    auto correction_polys = spread_.apply(jumps, rhs);

    // 3. Bulk solve: (−Δ_h) u = rhs
    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);

    // 4. Restrict
    auto solution_polys = restrict_op_.apply(u_bulk, correction_polys);

    // 5. Pack y
    y.resize(Nq);
    if (mode_ == LaplaceKFBIMode::Dirichlet || mode_ == LaplaceKFBIMode::DirichletDouble) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0];
    } else if (mode_ == LaplaceKFBIMode::ExteriorDirichletDouble) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0] + jumps[i].u_jump;
    } else {
        const auto& normals = iface.normals();
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[1] * normals(i, 0)
                 + solution_polys[i].coeffs[2] * normals(i, 1);
    }
}

void LaplaceKFBIOperator2D::apply_full(const Eigen::VectorXd& x,
                                        Eigen::VectorXd&       y) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();

    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        if (mode_ == LaplaceKFBIMode::Dirichlet) {
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
    if (mode_ == LaplaceKFBIMode::Dirichlet || mode_ == LaplaceKFBIMode::DirichletDouble) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0];
    } else if (mode_ == LaplaceKFBIMode::ExteriorDirichletDouble) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0] + jumps[i].u_jump;
    } else {
        const auto& normals = iface.normals();
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[1] * normals(i, 0)
                 + solution_polys[i].coeffs[2] * normals(i, 1);
    }
}

int LaplaceKFBIOperator2D::problem_size() const
{
    return spread_.grid_pair().interface().num_points();
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
        if (mode_ == LaplaceKFBIMode::Dirichlet) {
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
    if (mode_ == LaplaceKFBIMode::Dirichlet || mode_ == LaplaceKFBIMode::DirichletDouble) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0];
    } else if (mode_ == LaplaceKFBIMode::ExteriorDirichletDouble) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0] + jumps[i].u_jump;
    } else {
        const auto& normals = iface.normals();
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[1] * normals(i, 0)
                 + solution_polys[i].coeffs[2] * normals(i, 1)
                 + solution_polys[i].coeffs[3] * normals(i, 2);
    }
}

void LaplaceKFBIOperator3D::apply_full(const Eigen::VectorXd& x,
                                        Eigen::VectorXd&       y) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();

    std::vector<LaplaceJumpData3D> jumps(Nq);
    for (int i = 0; i < Nq; ++i) {
        if (mode_ == LaplaceKFBIMode::Dirichlet) {
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
    if (mode_ == LaplaceKFBIMode::Dirichlet || mode_ == LaplaceKFBIMode::DirichletDouble) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0];
    } else if (mode_ == LaplaceKFBIMode::ExteriorDirichletDouble) {
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[0] + jumps[i].u_jump;
    } else {
        const auto& normals = iface.normals();
        for (int i = 0; i < Nq; ++i)
            y[i] = solution_polys[i].coeffs[1] * normals(i, 0)
                 + solution_polys[i].coeffs[2] * normals(i, 1)
                 + solution_polys[i].coeffs[3] * normals(i, 2);
    }
}

int LaplaceKFBIOperator3D::problem_size() const
{
    return spread_.grid_pair().interface().num_points();
}

} // namespace kfbim
