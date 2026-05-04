#pragma once

#include <vector>
#include <Eigen/Dense>
#include "i_kfbi_operator.hpp"
#include "laplace_potential.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// BIE mode for Laplace BVPs
//
//   InteriorDirichlet
//              Double-layer jump primitive, [u]=phi, [un]=0.
//              Returns the interior trace K[phi] + 1/2 phi.
//
//   ExteriorDirichlet
//              Double-layer jump primitive, [u]=phi, [un]=0.
//              Returns the exterior trace K[phi] - 1/2 phi.
//
//   InteriorNeumann
//              Single-layer jump primitive, [u]=0, [un]=psi.
//              Returns the interior normal derivative K'[psi] + 1/2 psi.
//
//   ExteriorNeumann
//              Single-layer jump primitive, [u]=0, [un]=psi.
//              Returns the exterior normal derivative K'[psi] - 1/2 psi.
// ---------------------------------------------------------------------------

enum class LaplaceKFBIMode {
    InteriorDirichlet,
    ExteriorDirichlet,
    InteriorNeumann,
    ExteriorNeumann,

    // Backward-compatible aliases for older call sites.
    Dirichlet = InteriorDirichlet,
    Neumann = InteriorNeumann
};

// ---------------------------------------------------------------------------
// Concrete Laplace KFBIM operators (Layer 3)
//
// Dirichlet mode packing (problem_size = num_points):
//   x[i] = [u] at point i,   [∂u/∂n] = 0
//   y[i] = selected-side trace u±|_Γ
//
// Neumann mode packing (problem_size = num_points):
//   x[i] = [∂u/∂n] at point i,   [u] = 0
//   y[i] = selected-side normal flux ∂u±/∂n|_Γ
//
// apply() pipeline:
//   1. Unpack x → LaplaceJumpData (mode determines which field)
//   2. rhs = base_rhs_ + Spread correction → local polys
//   3. BulkSolve(rhs) → bulk_solution
//   4. Restrict(bulk_solution, polys) → averaged solution polys at interface
//   5. Reconstruct selected-side trace or flux → pack into y
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

    // Linear part: y = A * x  (ignores base_rhs_ and rhs_derivs_)
    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;

    // Full affine map: y = A * x + b (includes base_rhs_ and rhs_derivs_)
    void apply_full(const Eigen::VectorXd& x, Eigen::VectorXd& y) const;

    // = Interface2D::num_points()
    int problem_size() const override;

    LaplaceKFBIMode mode() const { return mode_; }

private:
    LaplacePotentialEval2D      potentials_;
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

    // Linear part
    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;

    // Full affine map
    void apply_full(const Eigen::VectorXd& x, Eigen::VectorXd& y) const;

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
