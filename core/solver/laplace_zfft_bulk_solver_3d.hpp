#pragma once

#include "i_bulk_solver.hpp"
#include "zfft_bc_type.hpp"
#include "../grid/cartesian_grid_3d.hpp"
#include <Eigen/Dense>

namespace kfbim {

// ---------------------------------------------------------------------------
// LaplaceFftBulkSolverZfft3D
//
// Solves  (eta * I  -  Delta_h) u = f  on a uniform 3D Cartesian grid using
// Han Zhou's zfft library (FastDiffusionSolver3d).
//
// Boundary conditions and grid DOF conventions:
//   Periodic  — CellCenter grid with {nx, ny, nz} DOFs;
//               flat index k*(nx*ny)+j*nx+i.
//   Dirichlet — Node grid with {nx+1, ny+1, nz+1} DOFs for n-cell grids;
//               flat index k*((nx+1)*(ny+1))+j*(nx+1)+i.
//               Boundary values are implicitly zero.
//   Neumann   — same DOF layout as Dirichlet.
//
// Grid spacing:
//   Requires a SQUARE-CELL grid (dx == dy == dz == h).
//   Solver scales: f_zfft = -h^2 * f_physical, eta_zfft = h^2 * eta.
//
// Order: 2 or 4 — controls the FD stencil accuracy.
// ---------------------------------------------------------------------------

class LaplaceFftBulkSolverZfft3D : public ILaplaceBulkSolver3D {
public:
    LaplaceFftBulkSolverZfft3D(CartesianGrid3D grid,
                                ZfftBcType      bc,
                                double          eta   = 0.0,
                                int             order = 2);

    void solve(const Eigen::VectorXd& rhs,
               Eigen::VectorXd&       solution) override;

    const ICartesianGrid3D& grid() const override { return grid_; }

private:
    CartesianGrid3D grid_;
    ZfftBcType      bc_;
    double          eta_;
    int             order_;
};

} // namespace kfbim
