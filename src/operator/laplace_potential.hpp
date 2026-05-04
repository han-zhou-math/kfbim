#pragma once

#include <vector>
#include <Eigen/Dense>
#include "../local_cauchy/jump_data.hpp"

namespace kfbim {

// Forward declarations
class ILaplaceSpread2D;
class ILaplaceBulkSolver2D;
class ILaplaceRestrict2D;
class ILaplaceSpread3D;
class ILaplaceBulkSolver3D;
class ILaplaceRestrict3D;

struct LaplacePotentialEvalResult2D {
    Eigen::VectorXd u_bulk;    // solution at all grid nodes
    Eigen::VectorXd u_avg;     // averaged trace (u_int + u_ext)/2
    Eigen::VectorXd un_avg;    // averaged normal derivative
};

// ============================================================================
// LaplacePotentialEval2D — general 2D KFBI pipeline and potential operators.
//
// evaluate() runs one Spread -> BulkSolve -> Restrict pass for arbitrary jumps
// and bulk RHS. Restrict returns averaged trace/flux values directly.
//
// The specialized potential helpers are thin wrappers around evaluate():
//   D[φ]: [u]=φ, [∂ₙu]=0, f=0
//   S[ψ]: [u]=0, [∂ₙu]=ψ, f=0
//   N[q]: [u]=0, [∂ₙu]=0, [f]=q
// ============================================================================

class LaplacePotentialEval2D {
public:
    LaplacePotentialEval2D(const ILaplaceSpread2D&     spread,
                           const ILaplaceBulkSolver2D& bulk_solver,
                           const ILaplaceRestrict2D&   restrict_op);

    int problem_size() const;
    double arc_h_ratio() const;

    LaplacePotentialEvalResult2D evaluate(
        const std::vector<LaplaceJumpData2D>& jumps,
        const Eigen::VectorXd&                f_bulk) const;

    // ── Double-layer potential D[φ] ──────────────────────────────────────
    // [u]=phi, [∂ₙu]=0, f=0
    void eval_double_layer(const Eigen::VectorXd& phi,
                           Eigen::VectorXd&       K_phi,
                           Eigen::VectorXd&       H_phi) const;

    // ── Single-layer potential S[ψ] ──────────────────────────────────────
    // [u]=0, [∂ₙu]=psi, f=0
    void eval_single_layer(const Eigen::VectorXd& psi,
                           Eigen::VectorXd&       S_psi,
                           Eigen::VectorXd&       Kt_psi) const;

    // ── Newton potential N[q] ────────────────────────────────────────────
    // [u]=0, [∂ₙu]=0, [f]=q
    void eval_newton(const Eigen::VectorXd& q,
                     Eigen::VectorXd&       N_q,
                     Eigen::VectorXd&       Nn_q) const;

private:
    const ILaplaceSpread2D&     spread_;
    const ILaplaceBulkSolver2D& bulk_solver_;
    const ILaplaceRestrict2D&   restrict_op_;
    double                      arc_h_ratio_;
};

// ============================================================================
// LaplacePotentialEval3D — same interface for 3D
// ============================================================================

class LaplacePotentialEval3D {
public:
    LaplacePotentialEval3D(const ILaplaceSpread3D&     spread,
                           const ILaplaceBulkSolver3D& bulk_solver,
                           const ILaplaceRestrict3D&   restrict_op);

    int problem_size() const;

    void eval_double_layer(const Eigen::VectorXd& phi,
                           Eigen::VectorXd&       K_phi,
                           Eigen::VectorXd&       H_phi) const;

    void eval_single_layer(const Eigen::VectorXd& psi,
                           Eigen::VectorXd&       S_psi,
                           Eigen::VectorXd&       Kt_psi) const;

    void eval_newton(const Eigen::VectorXd& q,
                     Eigen::VectorXd&       N_q,
                     Eigen::VectorXd&       Nn_q) const;

private:
    void run_pipeline(const Eigen::VectorXd&              u_jump,
                      const Eigen::VectorXd&              un_jump,
                      const std::vector<Eigen::VectorXd>& rhs_derivs,
                      Eigen::VectorXd&                    trace_int,
                      Eigen::VectorXd&                    flux_int) const;

    const ILaplaceSpread3D&     spread_;
    const ILaplaceBulkSolver3D& bulk_solver_;
    const ILaplaceRestrict3D&   restrict_op_;
};

} // namespace kfbim
