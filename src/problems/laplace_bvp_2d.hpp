#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "../geometry/grid_pair_2d.hpp"
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"
#include "../operator/laplace_kfbi_operator.hpp"
#include "../solver/laplace_zfft_bulk_solver_2d.hpp"
#include "../transfer/laplace_restrict_2d.hpp"
#include "../transfer/laplace_spread_2d.hpp"
#include "laplace_interior.hpp"

namespace kfbim {

struct LaplaceBvpOptions2D {
    LaplaceInteriorPanelMethod2D panel_method =
        LaplaceInteriorPanelMethod2D::ChebyshevLobattoCenter;
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

// Shared implementation behind the public 2D Laplace BVP wrappers.
class LaplaceBvpPipeline2D {
public:
    LaplaceBvpPipeline2D(const CartesianGrid2D&              grid,
                         const Interface2D&                  iface,
                         const Eigen::VectorXd&              g,
                         const Eigen::VectorXd&              f_bulk,
                         const std::vector<Eigen::VectorXd>& rhs_derivs,
                         LaplaceKFBIMode                     mode,
                         double                              rhs_deriv_sign,
                         LaplaceBvpOptions2D                 options);

    LaplaceBvpSolveResult2D solve(int max_iter = 100,
                                  double tol = 1e-8,
                                  int restart = 50) const;

    const GridPair2D& grid_pair() const { return grid_pair_; }

private:
    Eigen::VectorXd apply_dirichlet_boundary_elimination(
        const Eigen::VectorXd& rhs) const;

    void restore_dirichlet_boundary(Eigen::VectorXd& u_bulk) const;

    GridPair2D                       grid_pair_;
    std::unique_ptr<ILaplaceSpread2D> spread_;
    LaplaceFftBulkSolverZfft2D       bulk_solver_;
    std::unique_ptr<ILaplaceRestrict2D> restrict_op_;

    LaplaceKFBIMode              mode_;
    double                       eta_;
    Eigen::VectorXd              g_;
    Eigen::VectorXd              f_bulk_;
    std::vector<Eigen::VectorXd> rhs_derivs_;
    Eigen::VectorXd              outer_dirichlet_values_;
};

class LaplaceExteriorDirichlet2D {
public:
    LaplaceExteriorDirichlet2D(const CartesianGrid2D&              grid,
                               const Interface2D&                  iface,
                               const Eigen::VectorXd&              g,
                               const Eigen::VectorXd&              f_bulk,
                               const std::vector<Eigen::VectorXd>& rhs_derivs,
                               LaplaceBvpOptions2D                 options = {});

    LaplaceBvpSolveResult2D solve(int max_iter = 100,
                                  double tol = 1e-8,
                                  int restart = 50) const;

    const GridPair2D& grid_pair() const { return pipeline_.grid_pair(); }

private:
    LaplaceBvpPipeline2D pipeline_;
};

class LaplaceInteriorNeumann2D {
public:
    LaplaceInteriorNeumann2D(const CartesianGrid2D&              grid,
                             const Interface2D&                  iface,
                             const Eigen::VectorXd&              g,
                             const Eigen::VectorXd&              f_bulk,
                             const std::vector<Eigen::VectorXd>& rhs_derivs,
                             LaplaceBvpOptions2D                 options = {});

    LaplaceBvpSolveResult2D solve(int max_iter = 100,
                                  double tol = 1e-8,
                                  int restart = 50) const;

    const GridPair2D& grid_pair() const { return pipeline_.grid_pair(); }

private:
    LaplaceBvpPipeline2D pipeline_;
};

class LaplaceExteriorNeumann2D {
public:
    LaplaceExteriorNeumann2D(const CartesianGrid2D&              grid,
                             const Interface2D&                  iface,
                             const Eigen::VectorXd&              g,
                             const Eigen::VectorXd&              f_bulk,
                             const std::vector<Eigen::VectorXd>& rhs_derivs,
                             LaplaceBvpOptions2D                 options = {});

    LaplaceBvpSolveResult2D solve(int max_iter = 100,
                                  double tol = 1e-8,
                                  int restart = 50) const;

    const GridPair2D& grid_pair() const { return pipeline_.grid_pair(); }

private:
    LaplaceBvpPipeline2D pipeline_;
};

} // namespace kfbim
