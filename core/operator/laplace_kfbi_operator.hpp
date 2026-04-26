#pragma once

#include <vector>
#include <Eigen/Dense>
#include "i_kfbi_operator.hpp"
#include "../transfer/i_spread.hpp"
#include "../transfer/i_restrict.hpp"
#include "../solver/i_bulk_solver.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// BIE mode for Laplace BVPs
//
//   Dirichlet  GMRES iterates on [∂u/∂n] = σ; comparison quantity = trace
//              y[i] = poly.coeffs[0]  (u|_Γ)
//
//   Neumann    GMRES iterates on [u] = μ; comparison quantity = normal flux
//              y[i] = poly.coeffs[1]*n_x + poly.coeffs[2]*n_y  (∂u/∂n|_Γ)
//
// The normals n are taken from the Interface stored inside the Restrict's
// GridPair (Interface::normals().row(i)).
// ---------------------------------------------------------------------------

enum class LaplaceKFBIMode { Dirichlet, Neumann };

// ---------------------------------------------------------------------------
// Concrete Laplace KFBIM operators (Layer 3)
//
// Dirichlet mode packing (problem_size = num_points):
//   x[i] = [∂u/∂n] at point i,   u_jump = 0 (given)
//   y[i] = corrected trace u|_Γ
//
// Neumann mode packing (problem_size = num_points):
//   x[i] = [u] at point i,   un_jump = 0 (given)
//   y[i] = corrected normal flux ∂u/∂n|_Γ
//
// apply() pipeline:
//   1. Unpack x → LaplaceJumpData (mode determines which field)
//   2. rhs = base_rhs_ + Spread correction → local polys
//   3. BulkSolve(rhs) → bulk_solution
//   4. Restrict(bulk_solution, polys) → solution polys at interface
//   5. Extract trace or flux from poly → pack into y
//
// base_rhs_:   fixed bulk grid RHS from forcing f (no interface correction)
// rhs_derivs_: fixed derivative jumps of f at each interface point
//
// Component references are borrowed; they must outlive this object.
// ---------------------------------------------------------------------------

class LaplaceKFBIOperator2D : public IKFBIOperator {
public:
    LaplaceKFBIOperator2D(const ILaplaceSpread2D&      spread,
                           const ILaplaceBulkSolver2D&  bulk_solver,
                           const ILaplaceRestrict2D&    restrict_op,
                           Eigen::VectorXd              base_rhs,
                           std::vector<Eigen::VectorXd> rhs_derivs,
                           LaplaceKFBIMode              mode);

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;

    // = Interface2D::num_points()
    int problem_size() const override;

    LaplaceKFBIMode mode() const { return mode_; }

private:
    const ILaplaceSpread2D&     spread_;
    const ILaplaceBulkSolver2D& bulk_solver_;
    const ILaplaceRestrict2D&   restrict_op_;
    Eigen::VectorXd              base_rhs_;
    std::vector<Eigen::VectorXd> rhs_derivs_;
    LaplaceKFBIMode              mode_;
};

class LaplaceKFBIOperator3D : public IKFBIOperator {
public:
    LaplaceKFBIOperator3D(const ILaplaceSpread3D&      spread,
                           const ILaplaceBulkSolver3D&  bulk_solver,
                           const ILaplaceRestrict3D&    restrict_op,
                           Eigen::VectorXd              base_rhs,
                           std::vector<Eigen::VectorXd> rhs_derivs,
                           LaplaceKFBIMode              mode);

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;
    int  problem_size() const override;
    LaplaceKFBIMode mode() const { return mode_; }

private:
    const ILaplaceSpread3D&     spread_;
    const ILaplaceBulkSolver3D& bulk_solver_;
    const ILaplaceRestrict3D&   restrict_op_;
    Eigen::VectorXd              base_rhs_;
    std::vector<Eigen::VectorXd> rhs_derivs_;
    LaplaceKFBIMode              mode_;
};

} // namespace kfbim
