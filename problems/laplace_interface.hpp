#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "../core/geometry/grid_pair_2d.hpp"
#include "../core/geometry/grid_pair_3d.hpp"
#include "../core/local_cauchy/i_local_cauchy_solver.hpp"
#include "../core/local_cauchy/jump_data.hpp"
#include "../core/solver/i_bulk_solver.hpp"
#include "../core/transfer/i_spread.hpp"
#include "../core/transfer/i_restrict.hpp"
#include "../core/operator/laplace_kfbi_operator.hpp"
#include "../core/gmres/i_outer_solver.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Laplace interface / transmission problems (Layer 5)
//
// The Cartesian box contains the full domain Ω = Ω+ ∪ Γ ∪ Ω-, split by the
// embedded interface Γ.  The outer box BC (Dirichlet, Neumann, or Periodic)
// is encoded in the BulkSolver.
//
// Two fundamentally different cases arise:
//
//   Case 1 — Smooth / prescribed jump  (LaplaceInterfaceProblem):
//     Δu = f±  in Ω±,  with fully prescribed jump data on Γ:
//       [u]     = α  (given)
//       [∂u/∂n] = γ  (given)
//     Both jump quantities are given ⇒ direct solve: one Spread + BulkSolve.
//     No iterative solve on the interface is needed.
//
//   Case 2 — Discontinuous coefficients  (LaplaceTransmissionProblem):
//     −∇·(β∇u) = f±  in Ω±  with piecewise-constant β (β+ ≠ β− in general),
//     physical transmission conditions:
//       [u]       = 0    (continuity)
//       [β ∂u/∂n] = 0   (continuity of normal flux)
//     The individual normal derivatives [∂u/∂n] need NOT be zero when β+ ≠ β−.
//     An iterative BIE solve is required (GMRES on σ = [∂u/∂n]).
// ---------------------------------------------------------------------------


// ===========================================================================
// Case 1 — Prescribed jump / smooth coefficients: DIRECT SOLVE
// ===========================================================================

struct LaplaceInterfaceResult2D {
    Eigen::VectorXd bulk_solution;  // u on the full grid, length = grid.num_dofs()
};

struct LaplaceInterfaceResult3D {
    Eigen::VectorXd bulk_solution;
};

// Pipeline: Spread(jumps) → base_rhs + correction → BulkSolve → bulk_solution.
// No Restrict, no outer solver.

class LaplaceInterfaceProblem2D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver2D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver2D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread2D>       spread;
    };

    LaplaceInterfaceProblem2D(const GridPair2D& grid_pair, Components components);

    // base_rhs: bulk grid RHS from f+ in Ω+ and f- in Ω-,  length = grid.num_dofs()
    // jumps[i]: fully prescribed jump data at interface quadrature point i
    //           (u_jump = α, un_jump = γ, rhs_derivs from f)
    LaplaceInterfaceResult2D solve(const Eigen::VectorXd&                base_rhs,
                                   const std::vector<LaplaceJumpData2D>& jumps) const;

private:
    const GridPair2D& grid_pair_;
    Components        components_;
};

class LaplaceInterfaceProblem3D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver3D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver3D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread3D>       spread;
    };

    LaplaceInterfaceProblem3D(const GridPair3D& grid_pair, Components components);

    LaplaceInterfaceResult3D solve(const Eigen::VectorXd&                base_rhs,
                                   const std::vector<LaplaceJumpData3D>& jumps) const;

private:
    const GridPair3D& grid_pair_;
    Components        components_;
};


// ===========================================================================
// Case 2 — Discontinuous coefficients: ITERATIVE SOLVE (GMRES)
// ===========================================================================
//
// BIE formulation
// ~~~~~~~~~~~~~~~
// Impose [u] = 0 and iterate on σ = [∂u/∂n] (which is non-zero when β+ ≠ β−).
// The physical condition [β ∂u/∂n] = 0 reads
//
//   β+ (∂u+/∂n) = β− (∂u−/∂n)
//   β+ (σ/2 + T σ)_+ = β− (−σ/2 + T σ)_−      (KFBIM jump relations)
//
// which reduces to a GMRES system A σ = b where A and b depend on β±.
//
// Requirements:
//   - The LocalCauchySolver must handle the variable-coefficient local problem
//     (local PDE: −∇·(β∇v) = 0 near Γ, with β piecewise constant).
//   - The BulkSolver must handle the variable-coefficient global problem.
//   - JumpData.un_jump stores [∂u/∂n] (NOT [β ∂u/∂n]); the coefficient
//     β± is known to the LocalCauchySolver and BulkSolver at construction.
//
// GMRES unknown: σ = [∂u/∂n],  length = num_interface_points.
// KFBIOperator mode: Neumann (restrict returns the flux ∂u/∂n at Γ).

struct LaplaceTransmissionResult2D {
    Eigen::VectorXd     bulk_solution;   // u on the full grid
    Eigen::VectorXd     un_jump;         // converged σ = [∂u/∂n] at each point
    std::vector<double> residuals;
    int                 num_iterations;
    bool                converged;
};

struct LaplaceTransmissionResult3D {
    Eigen::VectorXd     bulk_solution;
    Eigen::VectorXd     un_jump;
    std::vector<double> residuals;
    int                 num_iterations;
    bool                converged;
};

class LaplaceTransmissionProblem2D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver2D> local_solver;  // must support variable β
        std::unique_ptr<ILaplaceBulkSolver2D>  bulk_solver;   // must support variable β
        std::unique_ptr<ILaplaceSpread2D>       spread;
        std::unique_ptr<ILaplaceRestrict2D>     restrict_op;
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceTransmissionProblem2D(const GridPair2D& grid_pair, Components components);

    // base_rhs:   bulk grid RHS from f+, f− (with β± applied),  length = grid.num_dofs()
    // rhs_derivs: rhs_derivs[i] = derivative jumps of f at interface point i
    // gmres_rhs:  right-hand side of the BIE system (assembled from β±, f±, outer BC)
    // x0:         initial guess for σ = [∂u/∂n]; zero vector used if empty
    LaplaceTransmissionResult2D solve(const Eigen::VectorXd&              base_rhs,
                                      const std::vector<Eigen::VectorXd>& rhs_derivs,
                                      const Eigen::VectorXd&              gmres_rhs,
                                      const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair2D& grid_pair_;
    Components        components_;
};

class LaplaceTransmissionProblem3D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver3D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver3D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread3D>       spread;
        std::unique_ptr<ILaplaceRestrict3D>     restrict_op;
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceTransmissionProblem3D(const GridPair3D& grid_pair, Components components);

    LaplaceTransmissionResult3D solve(const Eigen::VectorXd&              base_rhs,
                                      const std::vector<Eigen::VectorXd>& rhs_derivs,
                                      const Eigen::VectorXd&              gmres_rhs,
                                      const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair3D& grid_pair_;
    Components        components_;
};

} // namespace kfbim
