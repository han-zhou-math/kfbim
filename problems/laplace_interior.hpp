#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "../core/geometry/grid_pair_2d.hpp"
#include "../core/geometry/grid_pair_3d.hpp"
#include "../core/local_cauchy/i_local_cauchy_solver.hpp"
#include "../core/solver/i_bulk_solver.hpp"
#include "../core/transfer/i_spread.hpp"
#include "../core/transfer/i_restrict.hpp"
#include "../core/operator/laplace_kfbi_operator.hpp"
#include "../core/gmres/i_outer_solver.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Interior Laplace BVPs (Layer 5)
//
// Computational domain: Ω_int = interior of the embedded interface Γ.
// The Cartesian grid covers a box that contains Ω_int.  The bulk solver
// operates on the full box with homogeneous outer-box BCs (commonly u = 0,
// ∂u/∂n = 0, or periodic); the outer-box BC type is encoded in the
// BulkSolver.  The base_rhs passed to solve() must already incorporate
// any non-zero forcing in the interior and be zero outside Ω_int.
//
// The Restrict operator (ILaplaceRestrict2D) returns the corrected solution
// polynomial at each interface point.  Dirichlet problems compare the trace
// (poly.coeffs[0]); Neumann problems compare the normal flux
// (poly.coeffs[1]*n_x + poly.coeffs[2]*n_y).  The same restrict
// implementation serves both — the KFBIOperator mode selects the component.
//
// Shared result type:
//   bulk_solution    DOF values on the Cartesian grid, length = grid.num_dofs()
//   interface_data   converged GMRES unknown at each quadrature point:
//                      Dirichlet → σ = [∂u/∂n]
//                      Neumann   → μ = [u]
//   residuals        relative residual after each GMRES iteration
//   converged        true if tol reached within max_iter
// ---------------------------------------------------------------------------

struct LaplaceInteriorResult2D {
    Eigen::VectorXd     bulk_solution;
    Eigen::VectorXd     interface_data;
    std::vector<double> residuals;
    int                 num_iterations;
    bool                converged;
};

struct LaplaceInteriorResult3D {
    Eigen::VectorXd     bulk_solution;
    Eigen::VectorXd     interface_data;
    std::vector<double> residuals;
    int                 num_iterations;
    bool                converged;
};

// ---------------------------------------------------------------------------
// Interior Dirichlet: Δu = f in Ω_int,  u = g on Γ
//
// BIE
// ~~~
// Extend u by zero in Ω_ext:  [u] = 0,  [∂u/∂n] = σ  (unknown).
//
//   Tσ = Restrict( BulkSolve( base_rhs + Spread(un_jump=σ, u_jump=0) ) ).trace
//
//   GMRES system:  (½I + T) σ = g_eff
//
// g_eff = g minus the trace contribution from the forcing f on the grid
// (accounted for in base_rhs; the caller pre-computes and passes gmres_rhs).
// GMRES unknown: σ = [∂u/∂n],  length = num_interface_points.
// KFBIOperator mode: Dirichlet.
// ---------------------------------------------------------------------------

class LaplaceInteriorDirichlet2D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver2D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver2D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread2D>       spread;
        std::unique_ptr<ILaplaceRestrict2D>     restrict_op;
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceInteriorDirichlet2D(const GridPair2D& grid_pair, Components components);

    // base_rhs:    bulk grid RHS from f (zeros outside Ω_int), length = grid.num_dofs()
    // rhs_derivs:  rhs_derivs[i] = derivative jumps of f at interface point i
    // gmres_rhs:   g_eff,  length = num_interface_points
    // x0:          initial guess for σ; zero vector used if empty
    LaplaceInteriorResult2D solve(const Eigen::VectorXd&              base_rhs,
                                  const std::vector<Eigen::VectorXd>& rhs_derivs,
                                  const Eigen::VectorXd&              gmres_rhs,
                                  const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair2D& grid_pair_;
    Components        components_;
};

class LaplaceInteriorDirichlet3D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver3D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver3D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread3D>       spread;
        std::unique_ptr<ILaplaceRestrict3D>     restrict_op;
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceInteriorDirichlet3D(const GridPair3D& grid_pair, Components components);

    LaplaceInteriorResult3D solve(const Eigen::VectorXd&              base_rhs,
                                  const std::vector<Eigen::VectorXd>& rhs_derivs,
                                  const Eigen::VectorXd&              gmres_rhs,
                                  const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair3D& grid_pair_;
    Components        components_;
};

// ---------------------------------------------------------------------------
// Interior Neumann: Δu = f in Ω_int,  ∂u/∂n = h on Γ
//
// BIE
// ~~~
// Extend u by zero in Ω_ext:  [u] = μ  (unknown),  [∂u/∂n] = 0.
//
//   T_n μ = Restrict( BulkSolve( base_rhs + Spread(u_jump=μ, un_jump=0) ) ).flux
//
//   GMRES system:  (-½I + T_n) μ = h_eff
//
// GMRES unknown: μ = [u],  length = num_interface_points.
// KFBIOperator mode: Neumann.
//
// Compatibility: ∫_Γ h dΓ = ∫_{Ω_int} f dΩ  (caller must verify).
// Unique up to a constant; pin the solution mean via post-processing if needed.
// ---------------------------------------------------------------------------

class LaplaceInteriorNeumann2D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver2D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver2D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread2D>       spread;
        std::unique_ptr<ILaplaceRestrict2D>     restrict_op;  // same type as Dirichlet
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceInteriorNeumann2D(const GridPair2D& grid_pair, Components components);

    // gmres_rhs: h_eff (effective Neumann data), length = num_interface_points
    LaplaceInteriorResult2D solve(const Eigen::VectorXd&              base_rhs,
                                  const std::vector<Eigen::VectorXd>& rhs_derivs,
                                  const Eigen::VectorXd&              gmres_rhs,
                                  const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair2D& grid_pair_;
    Components        components_;
};

class LaplaceInteriorNeumann3D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver3D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver3D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread3D>       spread;
        std::unique_ptr<ILaplaceRestrict3D>     restrict_op;
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceInteriorNeumann3D(const GridPair3D& grid_pair, Components components);

    LaplaceInteriorResult3D solve(const Eigen::VectorXd&              base_rhs,
                                  const std::vector<Eigen::VectorXd>& rhs_derivs,
                                  const Eigen::VectorXd&              gmres_rhs,
                                  const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair3D& grid_pair_;
    Components        components_;
};

} // namespace kfbim
