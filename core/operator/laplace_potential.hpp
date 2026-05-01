#pragma once

#include <vector>
#include <Eigen/Dense>

namespace kfbim {

// Forward declarations
class ILaplaceSpread2D;
class ILaplaceBulkSolver2D;
class ILaplaceRestrict2D;
class ILaplaceSpread3D;
class ILaplaceBulkSolver3D;
class ILaplaceRestrict3D;

// ============================================================================
// LaplacePotentialEval2D — modular boundary integral operator evaluation (2D)
//
// Evaluates the three fundamental potentials for the constant-coefficient
// interface problem  -Δu + κ u = f,  [u] = a,  [∂ₙu] = b:
//
//   D[φ]  Double-layer:  f=0,  a=φ,  b=0
//   S[ψ]  Single-layer:  f=0,  a=0,  b=ψ
//   N[q]  Newton:        f=q,  a=0,  b=0   (q = [f] on interface)
//
// Convention: u⁺ = interior limit, u⁻ = exterior limit, [u] = u⁺ − u⁻.
//
// Derived operators (returned alongside each potential):
//   K[φ]   = principal-value trace of D          = ½(D⁺ + D⁻)[φ]
//   H[φ]   = normal derivative of D (continuous) = ∂ₙD[φ]
//   S[ψ]   = trace of S (continuous)             = S⁺[ψ] = S⁻[ψ]
//   K'[ψ]  = averaged normal derivative of S     = ½(∂ₙS⁺ + ∂ₙS⁻)[ψ]
//   N[q]   = trace of N (continuous)
//   ∂ₙN[q] = normal derivative of N (continuous)
//
// Internal: one KFBI pipeline run (Spread → BulkSolve → Restrict) per call.
// The restrict step outputs the *exterior* trace/gradient (u⁻, ∂ₙu⁻);
// interior-side quantities follow from the jump relations [u] = u⁺ − u⁻ etc.
// ============================================================================

class LaplacePotentialEval2D {
public:
    LaplacePotentialEval2D(const ILaplaceSpread2D&     spread,
                           const ILaplaceBulkSolver2D& bulk_solver,
                           const ILaplaceRestrict2D&   restrict_op);

    int problem_size() const;

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
    // Run pipeline with given jumps and f-jumps; returns exterior trace + flux.
    void run_pipeline(const Eigen::VectorXd&              u_jump,
                      const Eigen::VectorXd&              un_jump,
                      const std::vector<Eigen::VectorXd>& rhs_derivs,
                      Eigen::VectorXd&                    trace_ext,
                      Eigen::VectorXd&                    flux_ext) const;

    const ILaplaceSpread2D&     spread_;
    const ILaplaceBulkSolver2D& bulk_solver_;
    const ILaplaceRestrict2D&   restrict_op_;
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
                      Eigen::VectorXd&                    trace_ext,
                      Eigen::VectorXd&                    flux_ext) const;

    const ILaplaceSpread3D&     spread_;
    const ILaplaceBulkSolver3D& bulk_solver_;
    const ILaplaceRestrict3D&   restrict_op_;
};

} // namespace kfbim
