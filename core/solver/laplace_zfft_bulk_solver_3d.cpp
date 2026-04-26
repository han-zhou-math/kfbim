#include "laplace_zfft_bulk_solver_3d.hpp"
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

LaplaceFftBulkSolverZfft3D::LaplaceFftBulkSolverZfft3D(
    CartesianGrid3D grid, ZfftBcType bc, double eta, int order)
    : grid_(std::move(grid)), bc_(bc), eta_(eta), order_(order)
{
    if (order_ != 2 && order_ != 4)
        throw std::invalid_argument("order must be 2 or 4");

    double dx = grid_.spacing()[0], dy = grid_.spacing()[1], dz = grid_.spacing()[2];
    if (std::fabs(dx - dy) > 1e-12 * dx || std::fabs(dx - dz) > 1e-12 * dx)
        throw std::invalid_argument(
            "LaplaceFftBulkSolverZfft3D requires square cells (dx == dy == dz)");
}

// ---------------------------------------------------------------------------
// solve() — dispatch based on BC type.
//
// Sign/scaling convention (same as 2D):
//   zfft solves  (-stencil + eta_z) u = f_z
//   Physical:    (-Delta_h - eta) u = -rhs_phys  →  (-stencil/h^2 - eta) u = -rhs_phys
//   So:  f_z = -h^2 * rhs_phys,  eta_z = h^2 * eta
//
// TensorXd layout: t[k][j][i] — k=z (slowest), j=y, i=x (fastest).
//   dim1 = nz-size, dim2 = ny-size, dim3 = nx-size.
//   FastDiffusionSolver3d uses n1 = dim[0]-1 intervals in z, etc.
//
// Flat index (matches CartesianGrid3D::index):  k*(ny_d*nx_d) + j*nx_d + i
//
// Periodic  — CellCenter DOFs: dof_dims = {nx, ny, nz}.
//             TensorXd(nz+1, ny+1, nx+1); data at t[k][j][i] for k<nz,j<ny,i<nx.
//
// Dirichlet — Node DOFs: dof_dims = {nx+1, ny+1, nz+1} for nx cells.
//             TensorXd(nz+1, ny+1, nx+1) where each +1 is the node count.
//             Interior: t[k][j][i] for k=1..nz-1, j=1..ny-1, i=1..nx-1.
//
// Neumann   — same layout as Dirichlet; all nodes filled.
// ---------------------------------------------------------------------------
void LaplaceFftBulkSolverZfft3D::solve(const Eigen::VectorXd& rhs,
                                        Eigen::VectorXd&       solution)
{
    auto d   = grid_.dof_dims();
    int  nx  = d[0], ny = d[1], nz = d[2];
    int  nxy = nx * ny;
    double h   = grid_.spacing()[0];
    double h2  = h * h;
    double eta_z = h2 * eta_;

    if (bc_ == ZfftBcType::Periodic) {
        // dof_dims = {nx_cells, ny_cells, nz_cells}
        // TensorXd(nz+1, ny+1, nx+1): n1=nz, n2=ny, n3=nx intervals
        TensorXd mat(nz + 1, ny + 1, nx + 1);
        mat.fill(0.0);

        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    mat[k][j][i] = -h2 * rhs[k * nxy + j * nx + i];

        zfft::FastDiffusionSolver3d(mat, eta_z, ZFFT_PERIODIC, order_);

        solution.resize(nx * ny * nz);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    solution[k * nxy + j * nx + i] = mat[k][j][i];

    } else {
        // dof_dims = {nx_nodes, ny_nodes, nz_nodes}  (includes boundary)
        // TensorXd(nz, ny, nx): n1=nz-1, n2=ny-1, n3=nx-1 intervals
        TensorXd mat(nz, ny, nx);
        mat.fill(0.0);

        if (bc_ == ZfftBcType::Dirichlet) {
            for (int k = 1; k < nz - 1; ++k)
                for (int j = 1; j < ny - 1; ++j)
                    for (int i = 1; i < nx - 1; ++i)
                        mat[k][j][i] = -h2 * rhs[k * nxy + j * nx + i];
        } else {  // Neumann — all nodes
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                        mat[k][j][i] = -h2 * rhs[k * nxy + j * nx + i];
        }

        zfft::FastDiffusionSolver3d(mat, eta_z, zfft_bc(bc_), order_);

        solution.resize(nx * ny * nz);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    solution[k * nxy + j * nx + i] = mat[k][j][i];
    }
}

} // namespace kfbim
