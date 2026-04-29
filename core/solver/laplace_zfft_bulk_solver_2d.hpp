#pragma once

#include "i_bulk_solver.hpp"
#include "zfft_bc_type.hpp"
#include "../grid/cartesian_grid_2d.hpp"
#include <Eigen/Dense>

namespace kfbim {

// ---------------------------------------------------------------------------
// LaplaceFftBulkSolverZfft2D
//
// Solves  (eta * I  -  Delta_h) u = f  on a uniform 2D Cartesian grid using
// Han Zhou's zfft library (FastDiffusionSolver2d).
//
// Boundary conditions and grid DOF conventions:
//   Periodic  — pass a CellCenter grid with {nx, ny} DOFs; flat index j*nx+i.
//   Dirichlet — pass a Node grid with {n+1, n+1} DOFs for n intervals;
//               flat index j*(n+1)+i.  Boundary values are implicitly zero.
//   Neumann   — same DOF layout as Dirichlet.
//
// Grid spacing:
//   Requires a SQUARE-CELL grid (dx == dy == h).  zfft uses a dimensionless
//   stencil; the solver scales the RHS by -h^2 internally, and scales the
//   eta parameter as eta_zfft = h^2 * eta.
//
// Order: ZFFT_TWO (2) or ZFFT_FOUR (4) — controls the FD stencil accuracy.
// ---------------------------------------------------------------------------

class LaplaceFftBulkSolverZfft2D : public ILaplaceBulkSolver2D {
public:
    LaplaceFftBulkSolverZfft2D(CartesianGrid2D grid,
                                ZfftBcType      bc,
                                double          eta   = 0.0,
                                int             order = 2);

    void solve(const Eigen::VectorXd& rhs,
               Eigen::VectorXd&       solution) const override;

    const ICartesianGrid2D& grid() const override { return grid_; }

private:
    CartesianGrid2D grid_;
    ZfftBcType      bc_;
    double          eta_;
    int             order_;   // 2 or 4
};

} // namespace kfbim
