#pragma once

#include <vector>
#include <Eigen/Dense>
#include "../geometry/grid_pair_2d.hpp"
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"
#include "../solver/laplace_zfft_bulk_solver_2d.hpp"
#include "../transfer/laplace_restrict_2d.hpp"
#include "../transfer/laplace_spread_2d.hpp"

namespace kfbim {

struct LaplaceTransmissionConstantRatioResult2D {
    Eigen::VectorXd     u_bulk;
    Eigen::VectorXd     un_jump;
    std::vector<double> residuals;
    int                 iterations;
    bool                converged;
};

// ---------------------------------------------------------------------------
// LaplaceTransmissionConstantRatio2D
//
// Solves the 2D piecewise-constant coefficient interface problem
//
//   -div(beta grad u) + kappa_sq u = f
//
// in the special constant-ratio case
//
//   kappa_sq_int / beta_int = kappa_sq_ext / beta_ext = lambda_sq.
//
// Dividing by beta reduces both sides to the same screened operator
//
//   -Delta u + lambda_sq u = q,      q = f / beta,
//
// with jumps [u] = mu and [beta du/dn] = sigma. The solver iterates on the
// ordinary normal-derivative jump psi = [du/dn].
// ---------------------------------------------------------------------------
class LaplaceTransmissionConstantRatio2D {
public:
    LaplaceTransmissionConstantRatio2D(const CartesianGrid2D& grid,
                                       const Interface2D&     iface,
                                       double                 beta_int,
                                       double                 beta_ext,
                                       double                 lambda_sq);

    // reduced_rhs_bulk stores q=f/beta on grid nodes before outer-boundary
    // Dirichlet elimination. rhs_derivs[i][0] stores [q] at interface point i.
    //
    // beta_flux_jump is [beta du/dn] = beta_int*du_int/dn -
    // beta_ext*du_ext/dn. outer_dirichlet_values is optional; if provided it
    // must have grid.num_dofs() entries and is used on the Cartesian box
    // boundary only.
    LaplaceTransmissionConstantRatioResult2D solve(
        const Eigen::VectorXd&              u_jump,
        const Eigen::VectorXd&              beta_flux_jump,
        const Eigen::VectorXd&              reduced_rhs_bulk,
        const std::vector<Eigen::VectorXd>& rhs_derivs,
        const Eigen::VectorXd&              outer_dirichlet_values = Eigen::VectorXd(),
        int                                max_iter = 200,
        double                             tol = 1e-8,
        int                                restart = 80) const;

    const GridPair2D& grid_pair() const { return grid_pair_; }
    double beta_int() const { return beta_int_; }
    double beta_ext() const { return beta_ext_; }
    double lambda_sq() const { return lambda_sq_; }

private:
    Eigen::VectorXd apply_dirichlet_boundary_elimination(
        const Eigen::VectorXd& reduced_rhs_bulk,
        const Eigen::VectorXd& outer_dirichlet_values) const;

    void restore_dirichlet_boundary(Eigen::VectorXd&       u_bulk,
                                    const Eigen::VectorXd& outer_dirichlet_values) const;

    GridPair2D                    grid_pair_;
    LaplaceLobattoCenterSpread2D  spread_;
    LaplaceFftBulkSolverZfft2D    bulk_solver_;
    LaplaceLobattoCenterRestrict2D restrict_op_;
    double                        beta_int_;
    double                        beta_ext_;
    double                        lambda_sq_;
};

} // namespace kfbim
