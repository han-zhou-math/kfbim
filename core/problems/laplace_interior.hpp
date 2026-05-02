#pragma once

#include <vector>
#include <Eigen/Dense>
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"
#include "../geometry/grid_pair_2d.hpp"
#include "../transfer/laplace_spread_2d.hpp"
#include "../transfer/laplace_restrict_2d.hpp"
#include "../solver/laplace_zfft_bulk_solver_2d.hpp"
#include "../operator/laplace_kfbi_operator.hpp"

namespace kfbim {

struct LaplaceInteriorSolveResult2D {
    Eigen::VectorXd u_bulk;       // Bulk solution
    Eigen::VectorXd density;      // Solved density phi
    int             iterations;   // GMRES inner iterations taken
    bool            converged;    // Whether GMRES converged
};

// ---------------------------------------------------------------------------
// LaplaceInteriorDirichlet2D
//
// Solves the interior Dirichlet BVP:
//   -Δu = f   in Ω_int
//     u = g   on Γ
//
// Uses the KFBIM 2nd-kind boundary integral formulation:
//   (1/2 I - K) φ = g - V f|_Γ
// ---------------------------------------------------------------------------
class LaplaceInteriorDirichlet2D {
public:
    // Setup the solver with the grid, interface, boundary values g, and interior forcing f_int.
    // Both f_int and g are evaluated at the interface points/grid nodes directly.
    LaplaceInteriorDirichlet2D(const CartesianGrid2D& grid,
                               const Interface2D&     iface,
                               const Eigen::VectorXd& g,
                               const Eigen::VectorXd& f_bulk,
                               const std::vector<Eigen::VectorXd>& rhs_derivs);

    // Solve for the density phi using GMRES and recover the bulk solution.
    LaplaceInteriorSolveResult2D solve(int max_iter = 100, double tol = 1e-6, int restart = 50);

    const GridPair2D& grid_pair() const { return grid_pair_; }

private:
    GridPair2D                 grid_pair_;
    LaplacePanelSpread2D       spread_;
    LaplaceFftBulkSolverZfft2D bulk_solver_;
    LaplaceQuadraticRestrict2D restrict_op_;

    Eigen::VectorXd              g_;
    Eigen::VectorXd              f_bulk_;
    std::vector<Eigen::VectorXd> rhs_derivs_;
};

} // namespace kfbim
