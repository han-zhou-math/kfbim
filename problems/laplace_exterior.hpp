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
// Exterior Laplace BVPs (Layer 5)
//
// Computational domain: Ω_ext = box \ Ω_int, the region between the
// embedded interface Γ and the outer Cartesian box boundary ∂Ω_box.
//
// Outer box boundary conditions (Dirichlet, Neumann, Periodic, or Mixed)
// are encoded entirely in the BulkSolver implementation.  The problem class
// only needs to know that the base_rhs already includes contributions from
// the outer box BCs (e.g. a Poisson solve on the box with homogeneous BC).
//
// Result fields are identical to LaplaceInteriorResult (see laplace_interior.hpp):
//   bulk_solution   DOF values on the full Cartesian grid
//   interface_data  converged GMRES unknown (σ or μ) at each interface point
//   residuals, num_iterations, converged
// ---------------------------------------------------------------------------

struct LaplaceExteriorResult2D {
    Eigen::VectorXd     bulk_solution;
    Eigen::VectorXd     interface_data;
    std::vector<double> residuals;
    int                 num_iterations;
    bool                converged;
};

struct LaplaceExteriorResult3D {
    Eigen::VectorXd     bulk_solution;
    Eigen::VectorXd     interface_data;
    std::vector<double> residuals;
    int                 num_iterations;
    bool                converged;
};

// ---------------------------------------------------------------------------
// Exterior Dirichlet: Δu = f in Ω_ext,  u = g on Γ,  BC on ∂Ω_box
//
// BIE formulation
// ~~~~~~~~~~~~~~~
// Embed u by zero in Ω_int:  [u] = 0,  [∂u/∂n] = -σ  (note exterior sign).
// The KFBIM operator (with outward normal pointing INTO Ω_int) gives
//
//   Tσ = Restrict( BulkSolve( base_rhs + Spread(u_jump=0, un_jump=-σ, f) ) )|_Γ
//
// BIE (exterior jump relation):
//
//   (-½I + T) σ = g_eff
//
// Sign differs from interior Dirichlet due to the orientation of the normal.
// GMRES unknown: σ ≈ ∂u/∂n|_{Ω_ext},  length = num_interface_points.
// ---------------------------------------------------------------------------

class LaplaceExteriorDirichlet2D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver2D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver2D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread2D>       spread;
        std::unique_ptr<ILaplaceRestrict2D>     restrict_op;   // value restrict
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceExteriorDirichlet2D(const GridPair2D& grid_pair, Components components);

    // base_rhs: bulk grid RHS incorporating forcing f AND the outer box BC
    //           (e.g. the homogeneous-box bulk solution is subtracted and its
    //           contribution folded into gmres_rhs externally)
    LaplaceExteriorResult2D solve(const Eigen::VectorXd&              base_rhs,
                                   const std::vector<Eigen::VectorXd>& rhs_derivs,
                                   const Eigen::VectorXd&              gmres_rhs,
                                   const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair2D& grid_pair_;
    Components        components_;
};

class LaplaceExteriorDirichlet3D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver3D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver3D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread3D>       spread;
        std::unique_ptr<ILaplaceRestrict3D>     restrict_op;
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceExteriorDirichlet3D(const GridPair3D& grid_pair, Components components);

    LaplaceExteriorResult3D solve(const Eigen::VectorXd&              base_rhs,
                                   const std::vector<Eigen::VectorXd>& rhs_derivs,
                                   const Eigen::VectorXd&              gmres_rhs,
                                   const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair3D& grid_pair_;
    Components        components_;
};

// ---------------------------------------------------------------------------
// Exterior Neumann: Δu = f in Ω_ext,  ∂u/∂n = h on Γ (exterior normal),
//                  BC on ∂Ω_box
//
// BIE formulation
// ~~~~~~~~~~~~~~~
// Embed u by zero in Ω_int:  [u] = μ  (unknown),  [∂u/∂n] = 0.
//
//   T_n μ = FluxRestrict( BulkSolve( base_rhs + Spread(u_jump=μ, un_jump=0, f) ) )|_Γ
//
// BIE (exterior flux jump relation):
//
//   (½I + T_n) μ = h_eff
//
// Sign differs from interior Neumann.
// GMRES unknown: μ = [u],  length = num_interface_points.
// ---------------------------------------------------------------------------

class LaplaceExteriorNeumann2D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver2D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver2D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread2D>       spread;
        std::unique_ptr<ILaplaceRestrict2D>     restrict_op;  // same type; flux extracted by KFBIOperator
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceExteriorNeumann2D(const GridPair2D& grid_pair, Components components);

    LaplaceExteriorResult2D solve(const Eigen::VectorXd&              base_rhs,
                                   const std::vector<Eigen::VectorXd>& rhs_derivs,
                                   const Eigen::VectorXd&              gmres_rhs,
                                   const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair2D& grid_pair_;
    Components        components_;
};

class LaplaceExteriorNeumann3D {
public:
    struct Components {
        std::unique_ptr<ILaplaceLocalSolver3D> local_solver;
        std::unique_ptr<ILaplaceBulkSolver3D>  bulk_solver;
        std::unique_ptr<ILaplaceSpread3D>       spread;
        std::unique_ptr<ILaplaceRestrict3D>     restrict_op;
        std::unique_ptr<IOuterSolver>           outer_solver;
    };

    LaplaceExteriorNeumann3D(const GridPair3D& grid_pair, Components components);

    LaplaceExteriorResult3D solve(const Eigen::VectorXd&              base_rhs,
                                   const std::vector<Eigen::VectorXd>& rhs_derivs,
                                   const Eigen::VectorXd&              gmres_rhs,
                                   const Eigen::VectorXd&              x0 = {}) const;

private:
    const GridPair3D& grid_pair_;
    Components        components_;
};

} // namespace kfbim
