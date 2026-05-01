#include "laplace_interface_solver_2d.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace kfbim {

LaplaceInterfaceSolver2D::LaplaceInterfaceSolver2D(
    const ILaplaceSpread2D&     spread,
    const ILaplaceBulkSolver2D& bulk_solver,
    const ILaplaceRestrict2D&   restrict_op)
    : spread_(spread)
    , bulk_solver_(bulk_solver)
    , restrict_op_(restrict_op)
{
    const auto& grid = spread_.grid_pair().grid();
    double h = std::max(grid.spacing()[0], grid.spacing()[1]);

    const auto& pts = spread_.grid_pair().interface().points();
    double min_arc = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(pts.rows()) - 1; ++i) {
        double dx = pts(i, 0) - pts(i + 1, 0);
        double dy = pts(i, 1) - pts(i + 1, 1);
        double d = std::sqrt(dx * dx + dy * dy);
        if (d < min_arc)
            min_arc = d;
    }

    arc_h_ratio_ = min_arc / h;
    if (arc_h_ratio_ < 0.5) {
        std::cerr << "LaplaceInterfaceSolver2D: arc_h_ratio = "
                  << arc_h_ratio_ << " < 0.5 — interface may be under-resolved\n";
    }
}

LaplaceInterfaceSolveResult2D LaplaceInterfaceSolver2D::solve(
    const std::vector<LaplaceJumpData2D>& jumps,
    const Eigen::VectorXd&                f_bulk) const
{
    const auto& iface = spread_.grid_pair().interface();
    const int   Nq    = iface.num_points();

    // 1. Spread: correct bulk RHS (accumulates into a copy of f_bulk)
    Eigen::VectorXd rhs = f_bulk;
    auto correction_polys = spread_.apply(jumps, rhs);

    // 2. Bulk solve: −Δh u = −rhs  →  u_bulk
    Eigen::VectorXd u_bulk;
    bulk_solver_.solve(-rhs, u_bulk);

    // 3. Restrict: interpolate + jump correct
    auto solution_polys = restrict_op_.apply(u_bulk, correction_polys);

    // 4. Extract and average traces
    // Restrict recovers interior trace: result[q].coeffs[0] = u_int
    // Average trace: u_avg = (u_int + u_ext)/2 = u_int - [u]/2
    // Average normal derivative: un_avg = (un_int + un_ext)/2 = un_int - [un]/2
    Eigen::VectorXd u_avg(Nq);
    Eigen::VectorXd un_avg(Nq);
    const auto& normals = iface.normals();
    for (int i = 0; i < Nq; ++i) {
        const double u_int  = solution_polys[i].coeffs[0];
        const double un_int = solution_polys[i].coeffs[1] * normals(i, 0)
                            + solution_polys[i].coeffs[2] * normals(i, 1);

        u_avg[i]  = u_int  - 0.5 * jumps[i].u_jump;
        un_avg[i] = un_int - 0.5 * jumps[i].un_jump;
    }

    return {std::move(u_bulk), std::move(u_avg), std::move(un_avg)};
}

int LaplaceInterfaceSolver2D::num_points() const
{
    return spread_.grid_pair().interface().num_points();
}

double LaplaceInterfaceSolver2D::arc_h_ratio() const
{
    return arc_h_ratio_;
}

} // namespace kfbim
