#pragma once

#include <vector>
#include <Eigen/Dense>
#include "../transfer/i_spread.hpp"
#include "../transfer/i_restrict.hpp"
#include "../solver/i_bulk_solver.hpp"
#include "../local_cauchy/jump_data.hpp"

namespace kfbim {

struct LaplaceInterfaceSolveResult2D {
    Eigen::VectorXd u_bulk;    // solution at all grid nodes
    Eigen::VectorXd u_avg;     // averaged trace (u_int + u_ext)/2 at each quad point
    Eigen::VectorXd un_avg;    // averaged normal derivative (∂ₙu_int + ∂ₙu_ext)/2
};

// Encapsulates the Spread → BulkSolve → Restrict pipeline.
// u_trace stores poly.coeffs[0] (interior limit u⁻); callers add [u] to get u⁺.
// un_trace stores ∂ₙu⁻ = coeffs[1]*n_x + coeffs[2]*n_y.
class LaplaceInterfaceSolver2D {
public:
    LaplaceInterfaceSolver2D(const ILaplaceSpread2D&     spread,
                              const ILaplaceBulkSolver2D& bulk_solver,
                              const ILaplaceRestrict2D&   restrict_op);

    // Runs the full pipeline: Spread(jumps, rhs) → BulkSolve(−rhs) → Restrict.
    // rhs is a copy of f_bulk; Spread accumulates into it, leaving f_bulk unchanged.
    LaplaceInterfaceSolveResult2D solve(
        const std::vector<LaplaceJumpData2D>& jumps,
        const Eigen::VectorXd&                f_bulk) const;

    int    num_points()   const;
    double arc_h_ratio()  const;

private:
    const ILaplaceSpread2D&     spread_;
    const ILaplaceBulkSolver2D& bulk_solver_;
    const ILaplaceRestrict2D&   restrict_op_;
    double                      arc_h_ratio_;
};

} // namespace kfbim
