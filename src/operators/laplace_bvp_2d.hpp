#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "../geometry/grid_pair_2d.hpp"
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"
#include "i_kfbi_operator.hpp"
#include "../potentials/laplace_potential.hpp"
#include "../bulk_solvers/laplace_zfft_bulk_solver_2d.hpp"
#include "../transfer/laplace_restrict_2d.hpp"
#include "../transfer/laplace_spread_2d.hpp"

namespace kfbim {

enum class LaplaceBvpType2D {
    InteriorDirichlet,
    ExteriorDirichlet,
    InteriorNeumann,
    ExteriorNeumann
};

enum class LaplaceBvpPanelMethod2D {
    QuadraticPanelCenter,

    // Backward-compatible aliases for older active 2D call sites.
    ChebyshevLobattoCenter = QuadraticPanelCenter,
    LobattoCenter = QuadraticPanelCenter
};

struct LaplaceBvpOptions2D {
    LaplaceBvpPanelMethod2D panel_method =
        LaplaceBvpPanelMethod2D::QuadraticPanelCenter;
    double eta = 0.0;
    Eigen::VectorXd outer_dirichlet_values;
};

struct LaplaceBvpSolveResult2D {
    Eigen::VectorXd     u_bulk;
    Eigen::VectorXd     density;
    std::vector<double> residuals;
    int                 iterations;
    bool                converged;
};

// ---------------------------------------------------------------------------
// LaplaceBvp2D
//
// Solves 2D interior/exterior Dirichlet and Neumann BVPs:
//   -Delta u + eta*u = f
//
// boundary_data stores Dirichlet values for Dirichlet modes and normal
// derivative values for Neumann modes.
// ---------------------------------------------------------------------------
class LaplaceBvp2D : public IKFBIOperator {
public:
    LaplaceBvp2D(const CartesianGrid2D& grid,
                 const Interface2D&     iface,
                 LaplaceBvpType2D       type,
                 LaplaceBvpOptions2D    options = {});

    void apply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override;
    int problem_size() const override;

    LaplaceBvpSolveResult2D solve(const Eigen::VectorXd&              boundary_data,
                                  const Eigen::VectorXd&              f_bulk,
                                  const std::vector<Eigen::VectorXd>& rhs_derivs,
                                  int max_iter = 100,
                                  double tol = 1e-8,
                                  int restart = 50) const;

    const GridPair2D& grid_pair() const { return grid_pair_; }
    LaplaceBvpType2D type() const { return type_; }
    double eta() const { return eta_; }

private:
    Eigen::VectorXd apply_dirichlet_boundary_elimination(
        const Eigen::VectorXd& rhs) const;

    void restore_dirichlet_boundary(Eigen::VectorXd& u_bulk) const;

    void apply_with_rhs(const Eigen::VectorXd&              density,
                        const Eigen::VectorXd&              rhs,
                        const std::vector<Eigen::VectorXd>& rhs_derivs,
                        Eigen::VectorXd&                    y) const;

    GridPair2D                       grid_pair_;
    std::unique_ptr<ILaplaceSpread2D> spread_;
    LaplaceFftBulkSolverZfft2D       bulk_solver_;
    std::unique_ptr<ILaplaceRestrict2D> restrict_op_;
    LaplacePotentialEval2D           potentials_;

    LaplaceBvpType2D             type_;
    double                       rhs_deriv_sign_;
    double                       eta_;
    Eigen::VectorXd              outer_dirichlet_values_;
};

} // namespace kfbim
