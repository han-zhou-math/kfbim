#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/solver/laplace_fft_bulk_solver_2d.hpp"
#include "core/solver/zfft_engine_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_3d.hpp"

#ifdef KFBIM_HAS_FFTW3
#include "core/solver/fftw3_engine.hpp"
#endif

using namespace kfbim;
using Catch::Matchers::WithinAbs;

// ============================================================
// Helpers
// ============================================================

static void apply_laplacian_periodic(
    const Eigen::VectorXd& u, Eigen::VectorXd& rhs,
    int nx, int ny, double dx, double dy)
{
    rhs.resize(nx * ny);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int ip = (i + 1) % nx, im = (i - 1 + nx) % nx;
            int jp = (j + 1) % ny, jm = (j - 1 + ny) % ny;
            double uxx = (u[j*nx+ip] - 2.0*u[j*nx+i] + u[j*nx+im]) / (dx*dx);
            double uyy = (u[jp*nx+i] - 2.0*u[j*nx+i] + u[jm*nx+i]) / (dy*dy);
            rhs[j*nx+i] = uxx + uyy;
        }
    }
}

// ============================================================
// FFT engine tests (only compiled when FFTW3 is present)
// ============================================================

#ifdef KFBIM_HAS_FFTW3

TEST_CASE("FFTW3Engine2D — forward/backward round-trip", "[solver][fftw3]") {
    int nx = 16, ny = 12;
    FFTW3Engine2D engine(nx, ny);

    std::vector<double> in(nx * ny), out(nx * ny);
    for (int k = 0; k < nx * ny; ++k)
        in[k] = std::sin(2.0 * M_PI * k / (nx * ny));

    std::vector<std::complex<double>> freq(nx * (ny / 2 + 1));
    engine.forward(in.data(), freq.data());
    engine.backward(freq.data(), out.data());

    double norm = 1.0 / (nx * ny);
    for (int k = 0; k < nx * ny; ++k)
        REQUIRE_THAT(out[k] * norm, WithinAbs(in[k], 1e-12));
}

TEST_CASE("LaplaceFftBulkSolver2D — inverts discrete Laplacian (periodic)", "[solver][fftw3]") {
    int    nx = 32, ny = 24;
    double dx = 1.0 / nx, dy = 1.0 / ny;

    CartesianGrid2D g({0.0, 0.0}, {dx, dy}, {nx, ny}, DofLayout2D::CellCenter);
    auto engine = std::make_unique<FFTW3Engine2D>(nx, ny);
    LaplaceFftBulkSolver2D solver(g, std::move(engine));

    int kx_val = 2, ky_val = 3;
    Eigen::VectorXd u_exact(nx * ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            u_exact[j*nx+i] = std::cos(2.0*M_PI*kx_val*i/nx)
                             * std::cos(2.0*M_PI*ky_val*j/ny);

    Eigen::VectorXd rhs;
    apply_laplacian_periodic(u_exact, rhs, nx, ny, dx, dy);

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    REQUIRE(u_sol.size() == nx * ny);
    for (int k = 0; k < nx * ny; ++k)
        REQUIRE_THAT(u_sol[k], WithinAbs(u_exact[k], 1e-10));
}

TEST_CASE("LaplaceFftBulkSolver2D — sine×sine mode, non-square grid", "[solver][fftw3]") {
    int    nx = 20, ny = 16;
    double Lx = 2.0, Ly = 1.5;
    double dx = Lx / nx, dy = Ly / ny;

    CartesianGrid2D g({0.0, 0.0}, {dx, dy}, {nx, ny}, DofLayout2D::CellCenter);
    auto engine = std::make_unique<FFTW3Engine2D>(nx, ny);
    LaplaceFftBulkSolver2D solver(g, std::move(engine));

    int kx_val = 1, ky_val = 2;
    Eigen::VectorXd u_exact(nx * ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            u_exact[j*nx+i] = std::sin(2.0*M_PI*kx_val*i/nx)
                             * std::sin(2.0*M_PI*ky_val*j/ny);

    Eigen::VectorXd rhs;
    apply_laplacian_periodic(u_exact, rhs, nx, ny, dx, dy);

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    for (int k = 0; k < nx * ny; ++k)
        REQUIRE_THAT(u_sol[k], WithinAbs(u_exact[k], 1e-10));
}

TEST_CASE("LaplaceFftBulkSolver2D — zero rhs gives zero solution", "[solver][fftw3]") {
    int nx = 16, ny = 16;
    double h = 0.1;
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {nx, ny}, DofLayout2D::CellCenter);
    auto engine = std::make_unique<FFTW3Engine2D>(nx, ny);
    LaplaceFftBulkSolver2D solver(g, std::move(engine));

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nx * ny);
    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    for (int k = 0; k < nx * ny; ++k)
        REQUIRE_THAT(u_sol[k], WithinAbs(0.0, 1e-14));
}

#endif // KFBIM_HAS_FFTW3

// ============================================================
// ZfftEngine2D round-trip
// ============================================================

TEST_CASE("ZfftEngine2D — forward/backward round-trip", "[solver][zfft]") {
    int nx = 16, ny = 16;
    ZfftEngine2D engine(nx, ny);

    std::vector<double> in(nx * ny), out(nx * ny);
    for (int k = 0; k < nx * ny; ++k)
        in[k] = std::sin(2.0 * M_PI * k / (nx * ny));

    std::vector<std::complex<double>> freq(ny * (nx / 2 + 1));
    engine.forward(in.data(), freq.data());
    engine.backward(freq.data(), out.data());

    double norm = 1.0 / (nx * ny);
    for (int k = 0; k < nx * ny; ++k)
        REQUIRE_THAT(out[k] * norm, WithinAbs(in[k], 1e-12));
}

// ============================================================
// 2D convergence helpers
// ============================================================

// ---------- 2D Periodic: u = sin(2πx)cos(2πy), Δu = -8π²u ----------
static double err_2d_periodic(int n)
{
    // CellCenter DOFs, nx=ny=n (must be power of 2 for ZfftEngine)
    double h = 1.0 / n;
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {n, n}, DofLayout2D::CellCenter);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Periodic);

    double pi = M_PI;
    Eigen::VectorXd u_exact(n * n), rhs(n * n);
    for (int j = 0; j < n; ++j) {
        double y = (j + 0.5) * h;
        for (int i = 0; i < n; ++i) {
            double x = (i + 0.5) * h;
            u_exact[j*n+i] = std::sin(2*pi*x) * std::cos(2*pi*y);
            rhs[j*n+i]     = -8.0 * pi*pi * u_exact[j*n+i];
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// ---------- 2D Dirichlet homogeneous: u = sin(πx)sin(πy), Δu = -2π²u ----------
static double err_2d_dirichlet_homo(int n)
{
    double h = 1.0 / n;
    int npx = n + 1, npy = n + 1;
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {n, n}, DofLayout2D::Node);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Dirichlet);

    double pi = M_PI;
    double lambda = -2.0 * pi*pi;

    Eigen::VectorXd u_exact(npx * npy), rhs(npx * npy);
    rhs.setZero();
    for (int j = 0; j < npy; ++j) {
        double y = j * h;
        for (int i = 0; i < npx; ++i) {
            double x = i * h;
            u_exact[j*npx+i] = std::sin(pi*x) * std::sin(pi*y);
            bool bdy = (i == 0 || i == n || j == 0 || j == n);
            if (!bdy) rhs[j*npx+i] = lambda * u_exact[j*npx+i];
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    // Check interior only
    double err = 0.0;
    for (int j = 1; j < npy-1; ++j)
        for (int i = 1; i < npx-1; ++i)
            err = std::max(err, std::abs(u_sol[j*npx+i] - u_exact[j*npx+i]));
    return err;
}

// ---------- 2D Dirichlet non-homogeneous: u = sin(πx/2)cos(πy/2) ----------
// Non-zero BCs at x=1 and y=0.
static double err_2d_dirichlet_nonhomo(int n)
{
    double h = 1.0 / n;
    int nx = n + 1, ny = n + 1;
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {n, n}, DofLayout2D::Node);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Dirichlet);

    double pi = M_PI;
    double kx = pi / 2.0, ky = pi / 2.0;
    double lambda = -(kx*kx + ky*ky);  // Δu = lambda * u

    Eigen::VectorXd u_exact(nx * ny);
    for (int j = 0; j < ny; ++j) {
        double y = j * h;
        for (int i = 0; i < nx; ++i) {
            double x = i * h;
            u_exact[j*nx+i] = std::sin(kx*x) * std::cos(ky*y);
        }
    }

    // Build corrected RHS: interior nodes only; subtract boundary contributions.
    double h2 = h * h;
    Eigen::VectorXd rhs(nx * ny);
    rhs.setZero();
    for (int j = 1; j < ny-1; ++j) {
        for (int i = 1; i < nx-1; ++i) {
            double f = lambda * u_exact[j*nx+i];
            if (i == 1)    f -= u_exact[j*nx + 0]      / h2;
            if (i == nx-2) f -= u_exact[j*nx + (nx-1)] / h2;
            if (j == 1)    f -= u_exact[0*nx + i]      / h2;
            if (j == ny-2) f -= u_exact[(ny-1)*nx + i] / h2;
            rhs[j*nx+i] = f;
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    // Restore boundary values and check all nodes.
    for (int j = 0; j < ny; ++j) {
        u_sol[j*nx + 0]      = u_exact[j*nx + 0];
        u_sol[j*nx + (nx-1)] = u_exact[j*nx + (nx-1)];
    }
    for (int i = 0; i < nx; ++i) {
        u_sol[0*nx + i]      = u_exact[0*nx + i];
        u_sol[(ny-1)*nx + i] = u_exact[(ny-1)*nx + i];
    }

    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// ---------- 2D Neumann homogeneous: u = cos(πx)cos(πy), Δu = -2π²u ----------
// ∂u/∂n = 0 on all boundaries; mean(u) = 0.
static double err_2d_neumann_homo(int n)
{
    double h = 1.0 / n;
    int nx = n + 1, ny = n + 1;
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {n, n}, DofLayout2D::Node);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Neumann);

    double pi = M_PI;
    double lambda = -2.0 * pi*pi;

    Eigen::VectorXd u_exact(nx * ny), rhs(nx * ny);
    for (int j = 0; j < ny; ++j) {
        double y = j * h;
        for (int i = 0; i < nx; ++i) {
            double x = i * h;
            u_exact[j*nx+i] = std::cos(pi*x) * std::cos(pi*y);
            rhs[j*nx+i]     = lambda * u_exact[j*nx+i];
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    // Fix additive constant: shift by mean difference.
    double shift = u_sol.mean() - u_exact.mean();
    u_sol.array() -= shift;

    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// ---------- 2D Neumann non-homogeneous: u = cos(πx)cos(πy) + x - 1/2 ----------
// ∂u/∂n = -1 at x=0, +1 at x=1, 0 at y=0 and y=1.
// Solvability: ∮ ∂u/∂n = -1+1 = 0. Mean of u = 0.
static double err_2d_neumann_nonhomo(int n)
{
    double h = 1.0 / n;
    int nx = n + 1, ny = n + 1;
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {n, n}, DofLayout2D::Node);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Neumann);

    double pi = M_PI;

    Eigen::VectorXd u_exact(nx * ny), rhs(nx * ny);
    for (int j = 0; j < ny; ++j) {
        double y = j * h;
        for (int i = 0; i < nx; ++i) {
            double x = i * h;
            u_exact[j*nx+i] = std::cos(pi*x) * std::cos(pi*y) + x - 0.5;
            // Δu = -2π²cos(πx)cos(πy);  Δ(x-1/2) = 0
            double f = -2.0*pi*pi * std::cos(pi*x) * std::cos(pi*y);
            // Ghost-node correction for non-homo Neumann:
            //   At x=0: ∂u/∂n = -∂u/∂x = -1  → f_corr = f - 2*(-1)/h = f + 2/h
            //   At x=1: ∂u/∂n = +∂u/∂x =  1  → f_corr = f - 2*(1)/h  = f - 2/h
            if (i == 0)    f += 2.0 / h;
            if (i == nx-1) f -= 2.0 / h;
            rhs[j*nx+i] = f;
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    double shift = u_sol.mean() - u_exact.mean();
    u_sol.array() -= shift;

    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// ============================================================
// 3D convergence helpers
// ============================================================

// ---------- 3D Periodic: u = sin(2πx)cos(2πy)cos(2πz), Δu = -12π²u ----------
static double err_3d_periodic(int n)
{
    double h = 1.0 / n;
    CartesianGrid3D g({0.0,0.0,0.0}, {h,h,h}, {n,n,n}, DofLayout3D::CellCenter);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Periodic);

    double pi = M_PI;
    int N = n * n * n;
    Eigen::VectorXd u_exact(N), rhs(N);
    for (int k = 0; k < n; ++k) {
        double z = (k + 0.5) * h;
        for (int j = 0; j < n; ++j) {
            double y = (j + 0.5) * h;
            for (int i = 0; i < n; ++i) {
                double x = (i + 0.5) * h;
                int idx = k*(n*n) + j*n + i;
                u_exact[idx] = std::sin(2*pi*x) * std::cos(2*pi*y) * std::cos(2*pi*z);
                rhs[idx]     = -12.0*pi*pi * u_exact[idx];
            }
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// ---------- 3D Dirichlet homogeneous: u = sin(πx)sin(πy)sin(πz), Δu = -3π²u ----------
static double err_3d_dirichlet_homo(int n)
{
    double h = 1.0 / n;
    int nx = n + 1, ny = n + 1, nz = n + 1;
    CartesianGrid3D g({0.0,0.0,0.0}, {h,h,h}, {n,n,n}, DofLayout3D::Node);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Dirichlet);

    double pi = M_PI;
    double lambda = -3.0 * pi*pi;
    int nxy = nx * ny;

    Eigen::VectorXd u_exact(nx*ny*nz), rhs(nx*ny*nz);
    rhs.setZero();
    for (int k = 0; k < nz; ++k) {
        double z = k * h;
        for (int j = 0; j < ny; ++j) {
            double y = j * h;
            for (int i = 0; i < nx; ++i) {
                double x = i * h;
                int idx = k*nxy + j*nx + i;
                u_exact[idx] = std::sin(pi*x) * std::sin(pi*y) * std::sin(pi*z);
                bool bdy = (i==0||i==n||j==0||j==n||k==0||k==n);
                if (!bdy) rhs[idx] = lambda * u_exact[idx];
            }
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    double err = 0.0;
    for (int k = 1; k < nz-1; ++k)
        for (int j = 1; j < ny-1; ++j)
            for (int i = 1; i < nx-1; ++i)
                err = std::max(err, std::abs(u_sol[k*nxy+j*nx+i] - u_exact[k*nxy+j*nx+i]));
    return err;
}

// ---------- 3D Dirichlet non-homo: u = sin(πx/2)cos(πy/2)cos(πz/2) ----------
// Non-zero BCs at x=1, y=0, z=0.
static double err_3d_dirichlet_nonhomo(int n)
{
    double h = 1.0 / n;
    int nx = n + 1, ny = n + 1, nz = n + 1;
    CartesianGrid3D g({0.0,0.0,0.0}, {h,h,h}, {n,n,n}, DofLayout3D::Node);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Dirichlet);

    double pi = M_PI;
    double kk = pi / 2.0;
    double lambda = -3.0 * kk * kk;  // Δu = lambda * u
    double h2 = h * h;
    int nxy = nx * ny;

    Eigen::VectorXd u_exact(nx*ny*nz);
    for (int k = 0; k < nz; ++k) {
        double z = k * h;
        for (int j = 0; j < ny; ++j) {
            double y = j * h;
            for (int i = 0; i < nx; ++i) {
                double x = i * h;
                u_exact[k*nxy+j*nx+i] = std::sin(kk*x) * std::cos(kk*y) * std::cos(kk*z);
            }
        }
    }

    // Corrected RHS: interior nodes only; subtract boundary contributions / h².
    Eigen::VectorXd rhs(nx*ny*nz);
    rhs.setZero();
    for (int k = 1; k < nz-1; ++k) {
        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                double f = lambda * u_exact[k*nxy+j*nx+i];
                if (i == 1)    f -= u_exact[k*nxy+j*nx + 0]      / h2;
                if (i == nx-2) f -= u_exact[k*nxy+j*nx + (nx-1)] / h2;
                if (j == 1)    f -= u_exact[k*nxy+0*nx+i]        / h2;
                if (j == ny-2) f -= u_exact[k*nxy+(ny-1)*nx+i]   / h2;
                if (k == 1)    f -= u_exact[0*nxy+j*nx+i]        / h2;
                if (k == nz-2) f -= u_exact[(nz-1)*nxy+j*nx+i]   / h2;
                rhs[k*nxy+j*nx+i] = f;
            }
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    // Restore boundary values.
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            u_sol[k*nxy+j*nx + 0]      = u_exact[k*nxy+j*nx + 0];
            u_sol[k*nxy+j*nx + (nx-1)] = u_exact[k*nxy+j*nx + (nx-1)];
        }
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i) {
            u_sol[k*nxy+0*nx+i]        = u_exact[k*nxy+0*nx+i];
            u_sol[k*nxy+(ny-1)*nx+i]   = u_exact[k*nxy+(ny-1)*nx+i];
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            u_sol[0*nxy+j*nx+i]        = u_exact[0*nxy+j*nx+i];
            u_sol[(nz-1)*nxy+j*nx+i]   = u_exact[(nz-1)*nxy+j*nx+i];
        }

    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// ---------- 3D Neumann homogeneous: u = cos(πx)cos(πy)cos(πz), Δu = -3π²u ----------
static double err_3d_neumann_homo(int n)
{
    double h = 1.0 / n;
    int nx = n + 1, ny = n + 1, nz = n + 1;
    CartesianGrid3D g({0.0,0.0,0.0}, {h,h,h}, {n,n,n}, DofLayout3D::Node);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Neumann);

    double pi = M_PI;
    double lambda = -3.0 * pi*pi;
    int nxy = nx * ny;

    Eigen::VectorXd u_exact(nx*ny*nz), rhs(nx*ny*nz);
    for (int k = 0; k < nz; ++k) {
        double z = k * h;
        for (int j = 0; j < ny; ++j) {
            double y = j * h;
            for (int i = 0; i < nx; ++i) {
                double x = i * h;
                int idx = k*nxy+j*nx+i;
                u_exact[idx] = std::cos(pi*x) * std::cos(pi*y) * std::cos(pi*z);
                rhs[idx]     = lambda * u_exact[idx];
            }
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    double shift = u_sol.mean() - u_exact.mean();
    u_sol.array() -= shift;

    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// ---------- 3D Neumann non-homo: u = cos(πx)cos(πy)cos(πz) + x - 1/2 ----------
// ∂u/∂n = -1 at x=0, +1 at x=1, 0 elsewhere. Mean(u) = 0.
static double err_3d_neumann_nonhomo(int n)
{
    double h = 1.0 / n;
    int nx = n + 1, ny = n + 1, nz = n + 1;
    CartesianGrid3D g({0.0,0.0,0.0}, {h,h,h}, {n,n,n}, DofLayout3D::Node);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Neumann);

    double pi = M_PI;
    int nxy = nx * ny;

    Eigen::VectorXd u_exact(nx*ny*nz), rhs(nx*ny*nz);
    for (int k = 0; k < nz; ++k) {
        double z = k * h;
        for (int j = 0; j < ny; ++j) {
            double y = j * h;
            for (int i = 0; i < nx; ++i) {
                double x = i * h;
                int idx = k*nxy+j*nx+i;
                u_exact[idx] = std::cos(pi*x)*std::cos(pi*y)*std::cos(pi*z) + x - 0.5;
                // Δu = -3π²cos(πx)cos(πy)cos(πz);  Δ(x-1/2) = 0
                double f = -3.0*pi*pi * std::cos(pi*x)*std::cos(pi*y)*std::cos(pi*z);
                // Ghost-node correction:
                //   x=0: ∂u/∂n=-∂u/∂x=-1 → f_corr = f + 2/h
                //   x=1: ∂u/∂n=+∂u/∂x= 1 → f_corr = f - 2/h
                if (i == 0)    f += 2.0 / h;
                if (i == nx-1) f -= 2.0 / h;
                rhs[idx] = f;
            }
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    double shift = u_sol.mean() - u_exact.mean();
    u_sol.array() -= shift;

    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// ============================================================
// Helper: print and verify convergence table
// ============================================================

static void check_convergence(
    const char* label,
    const std::vector<int>& ns,
    std::vector<double>(*err_fn)(const std::vector<int>&))
{
    // Not used — inlined in each TEST_CASE for clarity.
    (void)label; (void)ns; (void)err_fn;
}

// ============================================================
// 2D convergence tests
// ============================================================

TEST_CASE("2D Periodic Poisson — convergence", "[solver][zfft][convergence][2d]")
{
    // Periodic requires power-of-2 grids for ZfftEngine.
    const std::vector<int> ns = {16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D Periodic: u=sin(2πx)cos(2πy), Δu=-8π²u\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_periodic(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

TEST_CASE("2D Dirichlet homo — convergence", "[solver][zfft][convergence][2d]")
{
    const std::vector<int> ns = {8, 16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D Dirichlet (homo): u=sin(πx)sin(πy), Δu=-2π²u\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_dirichlet_homo(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

TEST_CASE("2D Dirichlet non-homo — convergence", "[solver][zfft][convergence][2d]")
{
    const std::vector<int> ns = {8, 16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D Dirichlet (non-homo): u=sin(πx/2)cos(πy/2)\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_dirichlet_nonhomo(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

TEST_CASE("2D Neumann homo — convergence", "[solver][zfft][convergence][2d]")
{
    const std::vector<int> ns = {8, 16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D Neumann (homo): u=cos(πx)cos(πy), Δu=-2π²u\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_neumann_homo(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

TEST_CASE("2D Neumann non-homo — convergence", "[solver][zfft][convergence][2d]")
{
    const std::vector<int> ns = {8, 16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D Neumann (non-homo): u=cos(πx)cos(πy)+x-1/2\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_neumann_nonhomo(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

// ============================================================
// 3D convergence tests (n=8,16,32 for speed)
// ============================================================

TEST_CASE("3D Periodic Poisson — convergence", "[solver][zfft][convergence][3d]")
{
    const std::vector<int> ns = {16, 32, 64};
    std::vector<double> errs;

    std::printf("\n  3D Periodic: u=sin(2πx)cos(2πy)cos(2πz), Δu=-12π²u\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_periodic(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

TEST_CASE("3D Dirichlet homo — convergence", "[solver][zfft][convergence][3d]")
{
    const std::vector<int> ns = {8, 16, 32};
    std::vector<double> errs;

    std::printf("\n  3D Dirichlet (homo): u=sin(πx)sin(πy)sin(πz), Δu=-3π²u\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_dirichlet_homo(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

TEST_CASE("3D Dirichlet non-homo — convergence", "[solver][zfft][convergence][3d]")
{
    const std::vector<int> ns = {8, 16, 32};
    std::vector<double> errs;

    std::printf("\n  3D Dirichlet (non-homo): u=sin(πx/2)cos(πy/2)cos(πz/2)\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_dirichlet_nonhomo(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

TEST_CASE("3D Neumann homo — convergence", "[solver][zfft][convergence][3d]")
{
    const std::vector<int> ns = {8, 16, 32};
    std::vector<double> errs;

    std::printf("\n  3D Neumann (homo): u=cos(πx)cos(πy)cos(πz), Δu=-3π²u\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_neumann_homo(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

TEST_CASE("3D Neumann non-homo — convergence", "[solver][zfft][convergence][3d]")
{
    const std::vector<int> ns = {8, 16, 32};
    std::vector<double> errs;

    std::printf("\n  3D Neumann (non-homo): u=cos(πx)cos(πy)cos(πz)+x-1/2\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_neumann_nonhomo(n);
        errs.push_back(e);
        if (errs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, e);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, e,
                        std::log2(errs[errs.size()-2] / e));
    }
    for (std::size_t k = 1; k < errs.size(); ++k) {
        double rate = std::log2(errs[k-1] / errs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}
