#pragma once

#include <vector>
#include <Eigen/Dense>
#include "i_outer_solver.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Restarted GMRES (Layer 4)
//
// Implements GMRES(m) with Arnoldi factorization and Givens rotations.
// Setting restart = 0 (default) runs full (non-restarted) GMRES up to max_iter.
//
// Convergence criterion: ||r_k|| / ||r_0|| < tol
//
// residuals() returns the relative residual norm at the end of each outer
// iteration (i.e., each restart cycle), not each Arnoldi step.
// ---------------------------------------------------------------------------

class GMRES : public IOuterSolver {
public:
    // restart: number of Arnoldi vectors per cycle; 0 = full (non-restarted) GMRES
    explicit GMRES(int max_iter, double tol, int restart = 50);

    // Returns total number of inner iterations (Arnoldi steps) taken.
    int solve(const IKFBIOperator& op,
              const Eigen::VectorXd& rhs,
              Eigen::VectorXd&       x) override;

    const std::vector<double>& residuals() const override { return residuals_; }

    bool converged() const override { return converged_; }

private:
    int    max_iter_;
    double tol_;
    int    restart_;

    std::vector<double> residuals_;
    bool                converged_ = false;
};

} // namespace kfbim
