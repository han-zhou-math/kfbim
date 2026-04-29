#include "laplace_zfft_bulk_solver_2d.hpp"
#include "zfft.h"
#include <stdexcept>
#include <cmath>

namespace kfbim {

namespace {
    int zfft_bc(ZfftBcType bc) {
        switch (bc) {
            case ZfftBcType::Periodic:  return ZFFT_PERIODIC;
            case ZfftBcType::Dirichlet: return ZFFT_DIRICHLET;
            case ZfftBcType::Neumann:   return ZFFT_NEUMANN;
        }
        return ZFFT_PERIODIC;
    }
}

LaplaceFftBulkSolverZfft2D::LaplaceFftBulkSolverZfft2D(
    CartesianGrid2D grid, ZfftBcType bc, double eta, int order)
    : grid_(std::move(grid)), bc_(bc), eta_(eta), order_(order)
{
    if (order_ != 2 && order_ != 4)
        throw std::invalid_argument("order must be 2 or 4");

    double dx = grid_.spacing()[0], dy = grid_.spacing()[1];
    if (std::fabs(dx - dy) > 1e-12 * dx)
        throw std::invalid_argument(
            "LaplaceFftBulkSolverZfft2D requires square cells (dx == dy)");
}

// ---------------------------------------------------------------------------
// solve() — dispatch based on BC type.
//
// Sign convention: zfft solves (-stencil + eta_z) u = f_z, where
//   stencil u[i,j] = u[i+1]+u[i-1]+u[i,j+1]+u[i,j-1] - 4u[i,j]  (positive)
// The project interface solves (Delta_h - eta) u = -rhs, i.e.,
//   (-stencil/h^2 - eta) u = -rhs
//   (-stencil + eta_z) u = -h^2*rhs   where  eta_z = h^2 * eta
// So:  f_z = -h^2 * rhs_phys.
//
// Grid conventions:
//   Periodic  — dof_dims = {nx, ny}  (cells).  MatrixXd size (ny+1, nx+1).
//   Dirichlet — dof_dims = {nx, ny}  (nodes including boundary, nx=n+1 for n cells).
//               MatrixXd size (ny, nx):  rows()-1 = ny-1 = n intervals.
//   Neumann   — same as Dirichlet.
// ---------------------------------------------------------------------------
void LaplaceFftBulkSolverZfft2D::solve(const Eigen::VectorXd& rhs,
                                        Eigen::VectorXd&       solution) const
{
    auto d  = grid_.dof_dims();
    int  nx = d[0];
    int  ny = d[1];
    double h  = grid_.spacing()[0];   // dx == dy enforced in ctor
    double h2 = h * h;
    double eta_z = h2 * eta_;

    if (bc_ == ZfftBcType::Periodic) {
        // Grid: (nx, ny) cell-center DOFs, flat = j*nx+i.
        // MatrixXd(ny+1, nx+1): n1=ny intervals, n2=nx intervals.
        // Periodic data: mat[j][i] for j=0..ny-1, i=0..nx-1.
        MatrixXd mat(ny + 1, nx + 1);
        mat.fill(0.0);

        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                mat[j][i] = -h2 * rhs[j * nx + i];

        zfft::FastDiffusionSolver2d(mat, eta_z, ZFFT_PERIODIC, order_);

        solution.resize(nx * ny);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                solution[j * nx + i] = mat[j][i];

    } else {
        // Grid: (nx, ny) node DOFs (includes boundary), flat = j*nx+i.
        // nx = n+1 for n intervals in x.  ny = n+1 for n intervals in y.
        // MatrixXd(ny, nx): rows()=ny → n1=ny-1 intervals, n2=nx-1 intervals.
        // Interior: mat[j][i] for j=1..ny-2, i=1..nx-2.
        // Boundary nodes (j=0, j=ny-1, i=0, i=nx-1) stay 0 (homogeneous BC).
        MatrixXd mat(ny, nx);
        mat.fill(0.0);

        if (bc_ == ZfftBcType::Dirichlet) {
            for (int j = 1; j < ny - 1; j++)
                for (int i = 1; i < nx - 1; i++)
                    mat[j][i] = -h2 * rhs[j * nx + i];
        } else {  // Neumann — full domain including boundary
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    mat[j][i] = -h2 * rhs[j * nx + i];
        }

        zfft::FastDiffusionSolver2d(mat, eta_z, zfft_bc(bc_), order_);

        solution.resize(nx * ny);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                solution[j * nx + i] = mat[j][i];
    }
}

} // namespace kfbim
