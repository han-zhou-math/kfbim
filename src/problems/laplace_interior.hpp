#pragma once

#include <memory>
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

enum class LaplaceInteriorPanelMethod2D {
    ChebyshevLobattoCenter,

    // Backward-compatible alias for older Lobatto call sites.
    LobattoCenter = ChebyshevLobattoCenter
};

struct LaplaceInteriorSolveResult2D {
    Eigen::VectorXd u_bulk;       // Bulk solution
    Eigen::VectorXd density;      // Solved density phi
    int             iterations;   // GMRES inner iterations taken
    bool            converged;    // Whether GMRES converged
    std::vector<double> residuals; // Relative GMRES residual history
};

// ---------------------------------------------------------------------------
// LaplaceInteriorDirichlet2D
//
// Solves the interior Dirichlet BVP:
//   -Δu + eta*u = f   in Ω_int
//     u = g   on Γ
//
// Uses the KFBIM 2nd-kind boundary integral formulation:
//   (K + 1/2 I) φ = g - V f|_Γ
//
// Here K is the code-convention averaged trace for the jump primitive
// [u]=φ, [∂ₙu]=0.
// ---------------------------------------------------------------------------
class LaplaceInteriorDirichlet2D {
public:
    // Setup the solver with the grid, interface, boundary values g, and interior forcing f_int.
    // Both f_int and g are evaluated at the interface points/grid nodes directly.
    LaplaceInteriorDirichlet2D(const CartesianGrid2D& grid,
                               const Interface2D&     iface,
                               const Eigen::VectorXd& g,
                               const Eigen::VectorXd& f_bulk,
                               const std::vector<Eigen::VectorXd>& rhs_derivs,
                               LaplaceInteriorPanelMethod2D panel_method =
                                   LaplaceInteriorPanelMethod2D::ChebyshevLobattoCenter,
                               double eta = 0.0);

    // Solve for the density phi using GMRES and recover the bulk solution.
    LaplaceInteriorSolveResult2D solve(int max_iter = 100, double tol = 1e-8, int restart = 50);

    const GridPair2D& grid_pair() const { return grid_pair_; }

private:
    GridPair2D                 grid_pair_;
    std::unique_ptr<ILaplaceSpread2D> spread_;
    LaplaceFftBulkSolverZfft2D bulk_solver_;
    std::unique_ptr<ILaplaceRestrict2D> restrict_op_;

    Eigen::VectorXd              g_;
    Eigen::VectorXd              f_bulk_;
    std::vector<Eigen::VectorXd> rhs_derivs_;
};

} // namespace kfbim
