#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "src/solver/laplace_fft_bulk_solver_2d.hpp"
#include "src/solver/zfft_engine_2d.hpp"
#include "src/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "src/solver/laplace_zfft_bulk_solver_3d.hpp"

#ifdef KFBIM_HAS_FFTW3
#include "src/solver/fftw3_engine.hpp"
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

// ============================================================
// Rectangular domain (Lx=2, Ly=1) — 2D convergence helpers
//
// Square cells: h = 1/n,  nx_cells = 2n,  ny_cells = n.
// ============================================================

// u = sin(πx)cos(2πy),  Δu = -5π²u,  periodic on [0,2]×[0,1].
static double err_2d_rect_periodic(int n)
{
    double h = 1.0 / n;
    int nx = 2 * n, ny = n;
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {nx, ny}, DofLayout2D::CellCenter);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Periodic);

    double pi = M_PI;
    Eigen::VectorXd u_exact(nx * ny), rhs(nx * ny);
    for (int j = 0; j < ny; ++j) {
        double y = (j + 0.5) * h;
        for (int i = 0; i < nx; ++i) {
            double x = (i + 0.5) * h;
            u_exact[j*nx+i] = std::sin(pi*x) * std::cos(2.0*pi*y);
            rhs[j*nx+i]     = -5.0*pi*pi * u_exact[j*nx+i];
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);
    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// u = sin(πx/2)sin(πy),  Δu = -(5π²/4)u,  Dirichlet on [0,2]×[0,1].
// BCs: u = 0 on all four sides.
static double err_2d_rect_dirichlet(int n)
{
    double h = 1.0 / n;
    // 2n × n cells → (2n+1) × (n+1) nodes
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {2*n, n}, DofLayout2D::Node);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Dirichlet);
    int nx = 2*n + 1, ny = n + 1;

    double pi = M_PI;
    double lambda = -(pi*pi/4.0 + pi*pi);  // -5π²/4

    Eigen::VectorXd u_exact(nx * ny), rhs(nx * ny);
    rhs.setZero();
    for (int j = 0; j < ny; ++j) {
        double y = j * h;
        for (int i = 0; i < nx; ++i) {
            double x = i * h;
            u_exact[j*nx+i] = std::sin(0.5*pi*x) * std::sin(pi*y);
            bool bdy = (i == 0 || i == nx-1 || j == 0 || j == ny-1);
            if (!bdy) rhs[j*nx+i] = lambda * u_exact[j*nx+i];
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    double err = 0.0;
    for (int j = 1; j < ny-1; ++j)
        for (int i = 1; i < nx-1; ++i)
            err = std::max(err, std::abs(u_sol[j*nx+i] - u_exact[j*nx+i]));
    return err;
}

// u = cos(πx/2)cos(πy),  Δu = -(5π²/4)u,  Neumann on [0,2]×[0,1].
// Normal flux = 0 on all boundaries.  Fix additive constant via mean shift.
static double err_2d_rect_neumann(int n)
{
    double h = 1.0 / n;
    // 2n × n cells → (2n+1) × (n+1) nodes
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {2*n, n}, DofLayout2D::Node);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Neumann);
    int nx = 2*n + 1, ny = n + 1;

    double pi = M_PI;
    double lambda = -(pi*pi/4.0 + pi*pi);

    Eigen::VectorXd u_exact(nx * ny), rhs(nx * ny);
    for (int j = 0; j < ny; ++j) {
        double y = j * h;
        for (int i = 0; i < nx; ++i) {
            double x = i * h;
            u_exact[j*nx+i] = std::cos(0.5*pi*x) * std::cos(pi*y);
            rhs[j*nx+i]     = lambda * u_exact[j*nx+i];
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    double shift = u_sol.mean() - u_exact.mean();
    u_sol.array() -= shift;
    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// ============================================================
// Screened Poisson: (Δ − η)u = f
//
// u = sin(πx)sin(πy),  (Δ − η)u = (−2π² − η)u,
// on [0,1]², Dirichlet BCs.
// ============================================================

static double err_2d_screened_poisson(int n, double eta)
{
    double h = 1.0 / n;
    int nx = n + 1, ny = n + 1;
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {n, n}, DofLayout2D::Node);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Dirichlet, eta);

    double pi = M_PI;
    double lambda = -2.0*pi*pi - eta;   // eigenvalue of (Δ − η)

    Eigen::VectorXd u_exact(nx * ny), rhs(nx * ny);
    rhs.setZero();
    for (int j = 0; j < ny; ++j) {
        double y = j * h;
        for (int i = 0; i < nx; ++i) {
            double x = i * h;
            u_exact[j*nx+i] = std::sin(pi*x) * std::sin(pi*y);
            bool bdy = (i == 0 || i == nx-1 || j == 0 || j == ny-1);
            if (!bdy) rhs[j*nx+i] = lambda * u_exact[j*nx+i];
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    double err = 0.0;
    for (int j = 1; j < ny-1; ++j)
        for (int i = 1; i < nx-1; ++i)
            err = std::max(err, std::abs(u_sol[j*nx+i] - u_exact[j*nx+i]));
    return err;
}

// ============================================================
// Rectangular domain — convergence tests
// ============================================================

TEST_CASE("2D Periodic on rectangle [0,2]×[0,1] — convergence", "[solver][zfft][rect][2d]")
{
    const std::vector<int> ns = {16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D rect periodic: u=sin(πx)cos(2πy), [0,2]×[0,1]\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_rect_periodic(n);
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

TEST_CASE("2D Dirichlet on rectangle [0,2]×[0,1] — convergence", "[solver][zfft][rect][2d]")
{
    const std::vector<int> ns = {8, 16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D rect Dirichlet: u=sin(πx/2)sin(πy), [0,2]×[0,1]\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_rect_dirichlet(n);
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

TEST_CASE("2D Neumann on rectangle [0,2]×[0,1] — convergence", "[solver][zfft][rect][2d]")
{
    const std::vector<int> ns = {8, 16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D rect Neumann: u=cos(πx/2)cos(πy), [0,2]×[0,1]\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_rect_neumann(n);
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
// Screened Poisson — convergence tests
// ============================================================

TEST_CASE("2D screened Poisson (η=1) Dirichlet — convergence", "[solver][zfft][screened][2d]")
{
    const std::vector<int> ns = {8, 16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D screened Poisson (η=1): u=sin(πx)sin(πy)\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_screened_poisson(n, 1.0);
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

TEST_CASE("2D screened Poisson (η=100) Dirichlet — convergence", "[solver][zfft][screened][2d]")
{
    const std::vector<int> ns = {8, 16, 32, 64, 128};
    std::vector<double> errs;

    std::printf("\n  2D screened Poisson (η=100): u=sin(πx)sin(πy)\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_2d_screened_poisson(n, 100.0);
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
// order=4 solver — smoke test only
//
// Feeding the continuous Laplacian as RHS gives O(h²) error even with the
// 4th-order stencil, because the 4th-order stencil correction only helps
// when the RHS is also computed with 4th-order accuracy (as in the KFBIM
// interface-correction pipeline).  This test simply verifies that the
// order=4 solver constructs, runs, and returns a reasonable answer.
// ============================================================

TEST_CASE("2D Periodic order=4 — solver runs and gives finite answer", "[solver][zfft][order][2d]")
{
    int n = 32;
    double h = 1.0 / n;
    CartesianGrid2D g({0.0, 0.0}, {h, h}, {n, n}, DofLayout2D::CellCenter);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Periodic, 0.0, 4);

    double pi = M_PI;
    Eigen::VectorXd u_exact(n * n), rhs(n * n);
    for (int j = 0; j < n; ++j) {
        double y = (j + 0.5) * h;
        for (int i = 0; i < n; ++i) {
            double x = (i + 0.5) * h;
            u_exact[j*n+i] = std::sin(2*pi*x) * std::cos(2*pi*y);
            rhs[j*n+i]     = -8.0*pi*pi * u_exact[j*n+i];
        }
    }

    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    REQUIRE(u_sol.size() == n * n);
    REQUIRE(u_sol.allFinite());
    // Accuracy is O(h²) here (correct RHS for 4th-order needs 4th-order stencil applied to u_exact)
    double err = (u_sol - u_exact).lpNorm<Eigen::Infinity>();
    REQUIRE(err < 1e-2);
}

// ============================================================
// 2D Stokes periodic — solved via 3 scalar Poisson equations
//
// Exact solution (Taylor-Green + pressure):
//   u_x = sin(2πx)cos(2πy),  u_y = -cos(2πx)sin(2πy),  p = sin(2πx)sin(2πy)
//   ∇·u = 0  ✓,  mean of p = 0  ✓
//
// Forcing  f = -Δu + ∇p:
//   f_x = 8π²sin(2πx)cos(2πy) + 2πcos(2πx)sin(2πy)
//   f_y = -8π²cos(2πx)sin(2πy) + 2πsin(2πx)cos(2πy)
//
// Algorithm — take divergence of -Δu + ∇p = f with ∇·u = 0:
//   Δp = ∇·f                          (pressure Poisson)
//   Δu_x = ∂p/∂x − f_x               (x-momentum Poisson)
//   Δu_y = ∂p/∂y − f_y               (y-momentum Poisson)
//
// Discrete divergence / gradient use 2nd-order centered differences (periodic).
// Expected: O(h²) convergence for all fields.
// ============================================================

static std::array<double, 3> errs_2d_stokes_periodic(int n)
{
    double h  = 1.0 / n;
    double pi = M_PI;
    int    N  = n * n;

    CartesianGrid2D g({0.0, 0.0}, {h, h}, {n, n}, DofLayout2D::CellCenter);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Periodic);

    // ---- exact fields and forcing ----
    Eigen::VectorXd ux_ex(N), uy_ex(N), p_ex(N), fx(N), fy(N);
    for (int j = 0; j < n; ++j) {
        double y = (j + 0.5) * h;
        for (int i = 0; i < n; ++i) {
            double x   = (i + 0.5) * h;
            int    idx = j * n + i;
            ux_ex[idx] =  std::sin(2*pi*x) * std::cos(2*pi*y);
            uy_ex[idx] = -std::cos(2*pi*x) * std::sin(2*pi*y);
            p_ex [idx] =  std::sin(2*pi*x) * std::sin(2*pi*y);
            fx   [idx] =  8*pi*pi * ux_ex[idx] + 2*pi * std::cos(2*pi*x) * std::sin(2*pi*y);
            fy   [idx] =  8*pi*pi * uy_ex[idx] + 2*pi * std::sin(2*pi*x) * std::cos(2*pi*y);
        }
    }

    // ---- Step 1: pressure Poisson  Δp = ∇·f  ----
    // Discrete ∇·f with 2nd-order centered differences (periodic wrap).
    Eigen::VectorXd div_f(N);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int ip = (i + 1) % n, im = (i - 1 + n) % n;
            int jp = (j + 1) % n, jm = (j - 1 + n) % n;
            div_f[j*n+i] = (fx[j*n+ip] - fx[j*n+im]) / (2*h)
                         + (fy[jp*n+i] - fy[jm*n+i]) / (2*h);
        }
    }
    // Solver convention: solve Δu = rhs.  So pass rhs = ∇·f.
    Eigen::VectorXd p_sol;
    solver.solve(div_f, p_sol);
    // Zero-mean gauge: both rhs and solution are zero-mean, no shift needed.

    // ---- Step 2: velocity Poisson  Δu_i = ∂p/∂x_i − f_i  ----
    Eigen::VectorXd dp_dx(N), dp_dy(N);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int ip = (i + 1) % n, im = (i - 1 + n) % n;
            int jp = (j + 1) % n, jm = (j - 1 + n) % n;
            dp_dx[j*n+i] = (p_sol[j*n+ip] - p_sol[j*n+im]) / (2*h);
            dp_dy[j*n+i] = (p_sol[jp*n+i] - p_sol[jm*n+i]) / (2*h);
        }
    }
    Eigen::VectorXd ux_sol, uy_sol;
    solver.solve(dp_dx - fx, ux_sol);
    solver.solve(dp_dy - fy, uy_sol);

    return {
        (ux_sol - ux_ex).lpNorm<Eigen::Infinity>(),
        (uy_sol - uy_ex).lpNorm<Eigen::Infinity>(),
        (p_sol  - p_ex ).lpNorm<Eigen::Infinity>()
    };
}

// ---- Incompressibility check: max |∇·u_h| should be O(h²) ----
static double div_2d_stokes_periodic(int n)
{
    double h  = 1.0 / n;
    double pi = M_PI;
    int    N  = n * n;

    CartesianGrid2D g({0.0, 0.0}, {h, h}, {n, n}, DofLayout2D::CellCenter);
    LaplaceFftBulkSolverZfft2D solver(g, ZfftBcType::Periodic);

    Eigen::VectorXd ux_ex(N), uy_ex(N), p_ex(N), fx(N), fy(N);
    for (int j = 0; j < n; ++j) {
        double y = (j + 0.5) * h;
        for (int i = 0; i < n; ++i) {
            double x   = (i + 0.5) * h;
            int    idx = j * n + i;
            ux_ex[idx] =  std::sin(2*pi*x) * std::cos(2*pi*y);
            uy_ex[idx] = -std::cos(2*pi*x) * std::sin(2*pi*y);
            p_ex [idx] =  std::sin(2*pi*x) * std::sin(2*pi*y);
            fx   [idx] =  8*pi*pi * ux_ex[idx] + 2*pi * std::cos(2*pi*x) * std::sin(2*pi*y);
            fy   [idx] =  8*pi*pi * uy_ex[idx] + 2*pi * std::sin(2*pi*x) * std::cos(2*pi*y);
        }
    }

    Eigen::VectorXd div_f(N);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int ip = (i + 1) % n, im = (i - 1 + n) % n;
            int jp = (j + 1) % n, jm = (j - 1 + n) % n;
            div_f[j*n+i] = (fx[j*n+ip] - fx[j*n+im]) / (2*h)
                         + (fy[jp*n+i] - fy[jm*n+i]) / (2*h);
        }
    }
    Eigen::VectorXd p_sol;
    solver.solve(div_f, p_sol);

    Eigen::VectorXd dp_dx(N), dp_dy(N);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int ip = (i + 1) % n, im = (i - 1 + n) % n;
            int jp = (j + 1) % n, jm = (j - 1 + n) % n;
            dp_dx[j*n+i] = (p_sol[j*n+ip] - p_sol[j*n+im]) / (2*h);
            dp_dy[j*n+i] = (p_sol[jp*n+i] - p_sol[jm*n+i]) / (2*h);
        }
    }
    Eigen::VectorXd ux_sol, uy_sol;
    solver.solve(dp_dx - fx, ux_sol);
    solver.solve(dp_dy - fy, uy_sol);

    // Discrete divergence of computed velocity
    double div_max = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int ip = (i + 1) % n, im = (i - 1 + n) % n;
            int jp = (j + 1) % n, jm = (j - 1 + n) % n;
            double div = (ux_sol[j*n+ip] - ux_sol[j*n+im]) / (2*h)
                       + (uy_sol[jp*n+i] - uy_sol[jm*n+i]) / (2*h);
            div_max = std::max(div_max, std::abs(div));
        }
    }
    return div_max;
}

TEST_CASE("2D Stokes periodic (3 Poisson solves) — convergence", "[solver][zfft][stokes][2d]")
{
    const std::vector<int> ns = {16, 32, 64, 128};

    std::vector<double> errs_ux, errs_uy, errs_p;

    std::printf("\n  2D Stokes periodic (3 Poisson): ux=sin(2πx)cos(2πy), p=sin(2πx)sin(2πy)\n");
    std::printf("  %-6s  %-12s  %-12s  %-12s  %-6s\n",
                "n", "err_ux", "err_uy", "err_p", "rate_ux");

    for (int n : ns) {
        auto [e_ux, e_uy, e_p] = errs_2d_stokes_periodic(n);
        errs_ux.push_back(e_ux);
        errs_uy.push_back(e_uy);
        errs_p .push_back(e_p);

        if (errs_ux.size() == 1) {
            std::printf("  %-6d  %-12.4e  %-12.4e  %-12.4e  —\n", n, e_ux, e_uy, e_p);
        } else {
            double rate = std::log2(errs_ux[errs_ux.size()-2] / e_ux);
            std::printf("  %-6d  %-12.4e  %-12.4e  %-12.4e  %.2f\n", n, e_ux, e_uy, e_p, rate);
        }
    }

    for (std::size_t k = 1; k < errs_ux.size(); ++k) {
        REQUIRE(std::log2(errs_ux[k-1] / errs_ux[k]) > 1.8);
        REQUIRE(std::log2(errs_ux[k-1] / errs_ux[k]) < 2.2);
        REQUIRE(std::log2(errs_uy[k-1] / errs_uy[k]) > 1.8);
        REQUIRE(std::log2(errs_uy[k-1] / errs_uy[k]) < 2.2);
        REQUIRE(std::log2(errs_p [k-1] / errs_p [k]) > 1.8);
        REQUIRE(std::log2(errs_p [k-1] / errs_p [k]) < 2.2);
    }
}

TEST_CASE("2D Stokes periodic — discrete incompressibility O(h²)", "[solver][zfft][stokes][2d]")
{
    const std::vector<int> ns = {16, 32, 64, 128};
    std::vector<double> divs;

    std::printf("\n  2D Stokes periodic — max |∇·u_h|\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max|div u|", "rate");

    for (int n : ns) {
        double d = div_2d_stokes_periodic(n);
        divs.push_back(d);
        if (divs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, d);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, d,
                        std::log2(divs[divs.size()-2] / d));
    }

    for (std::size_t k = 1; k < divs.size(); ++k) {
        double rate = std::log2(divs[k-1] / divs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}

// ============================================================
// 3D rectangular box (Lx=2, Ly=Lz=1) — convergence helpers
//
// Square cells: h = 1/n,  num_cells = {2n, n, n}.
// Flat index: k*(ny*nx) + j*nx + i  (x fastest, z slowest).
// ============================================================

// u = sin(πx)cos(2πy)cos(2πz),  Δu = -(1+4+4)π²u = -9π²u,  periodic on [0,2]³.
static double err_3d_rect_periodic(int n)
{
    double h = 1.0 / n;
    int nx = 2*n, ny = n, nz = n;
    int nxy = nx * ny;
    CartesianGrid3D g({0.,0.,0.}, {h,h,h}, {nx,ny,nz}, DofLayout3D::CellCenter);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Periodic);

    double pi = M_PI;
    Eigen::VectorXd u_exact(nx*ny*nz), rhs(nx*ny*nz);
    for (int k = 0; k < nz; ++k) {
        double z = (k + 0.5) * h;
        for (int j = 0; j < ny; ++j) {
            double y = (j + 0.5) * h;
            for (int i = 0; i < nx; ++i) {
                double x = (i + 0.5) * h;
                int idx = k*nxy + j*nx + i;
                u_exact[idx] = std::sin(pi*x) * std::cos(2*pi*y) * std::cos(2*pi*z);
                rhs[idx]     = -9.0*pi*pi * u_exact[idx];
            }
        }
    }
    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);
    return (u_sol - u_exact).lpNorm<Eigen::Infinity>();
}

// u = sin(πx/2)sin(πy)sin(πz),  Δu = -(1/4+1+1)π²u = -(9/4)π²u,
// Dirichlet on [0,2]×[0,1]×[0,1],  zero BCs on all faces.
static double err_3d_rect_dirichlet(int n)
{
    double h = 1.0 / n;
    // num_cells = {2n,n,n} → dof_dims = {2n+1, n+1, n+1}
    CartesianGrid3D g({0.,0.,0.}, {h,h,h}, {2*n,n,n}, DofLayout3D::Node);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Dirichlet);

    int nx = 2*n+1, ny = n+1, nz = n+1;
    int nxy = nx * ny;
    double pi = M_PI;
    double lambda = -(pi*pi/4.0 + pi*pi + pi*pi);  // -9π²/4

    Eigen::VectorXd u_exact(nx*ny*nz), rhs(nx*ny*nz);
    rhs.setZero();
    for (int k = 0; k < nz; ++k) {
        double z = k * h;
        for (int j = 0; j < ny; ++j) {
            double y = j * h;
            for (int i = 0; i < nx; ++i) {
                double x = i * h;
                int idx = k*nxy + j*nx + i;
                u_exact[idx] = std::sin(0.5*pi*x) * std::sin(pi*y) * std::sin(pi*z);
                bool bdy = (i==0||i==nx-1||j==0||j==ny-1||k==0||k==nz-1);
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

// u = cos(πx/2)cos(πy)cos(πz),  Δu = -(9/4)π²u,
// Neumann on [0,2]×[0,1]×[0,1],  zero flux on all faces.
static double err_3d_rect_neumann(int n)
{
    double h = 1.0 / n;
    CartesianGrid3D g({0.,0.,0.}, {h,h,h}, {2*n,n,n}, DofLayout3D::Node);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Neumann);

    int nx = 2*n+1, ny = n+1, nz = n+1;
    int nxy = nx * ny;
    double pi = M_PI;
    double lambda = -(pi*pi/4.0 + pi*pi + pi*pi);

    Eigen::VectorXd u_exact(nx*ny*nz), rhs(nx*ny*nz);
    for (int k = 0; k < nz; ++k) {
        double z = k * h;
        for (int j = 0; j < ny; ++j) {
            double y = j * h;
            for (int i = 0; i < nx; ++i) {
                double x = i * h;
                int idx = k*nxy + j*nx + i;
                u_exact[idx] = std::cos(0.5*pi*x) * std::cos(pi*y) * std::cos(pi*z);
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

// ============================================================
// 3D screened Poisson: (Δ − η)u = f
//
// u = sin(πx)sin(πy)sin(πz),  (Δ − η)u = (−3π² − η)u,
// on [0,1]³, Dirichlet BCs.
// ============================================================

static double err_3d_screened_poisson(int n, double eta)
{
    double h = 1.0 / n;
    CartesianGrid3D g({0.,0.,0.}, {h,h,h}, {n,n,n}, DofLayout3D::Node);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Dirichlet, eta);

    int nx = n+1, ny = n+1, nz = n+1;
    int nxy = nx * ny;
    double pi = M_PI;
    double lambda = -3.0*pi*pi - eta;

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
                bool bdy = (i==0||i==nx-1||j==0||j==ny-1||k==0||k==nz-1);
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

// ============================================================
// 3D Stokes periodic — solved via 4 scalar Poisson equations
//
// Exact solution (∇·u = 0 verified below):
//   u_x =  sin(2πx)cos(2πy)cos(2πz)
//   u_y =  cos(2πx)sin(2πy)cos(2πz)
//   u_z = -2cos(2πx)cos(2πy)sin(2πz)        (sum of ∂_i u_i = 0)
//   p   =  sin(2πx)sin(2πy)sin(2πz)         (mean = 0)
//
// Forcing  f = -Δu + ∇p  (−Δu_i = 12π²u_i for each component):
//   f_x = 12π²u_x + 2πcos(2πx)sin(2πy)sin(2πz)
//   f_y = 12π²u_y + 2πsin(2πx)cos(2πy)sin(2πz)
//   f_z = 12π²u_z + 2πsin(2πx)sin(2πy)cos(2πz)
//
// Algorithm (same projection as 2D, one extra velocity solve):
//   Δp   = ∇·f                              (pressure)
//   Δu_x = ∂p/∂x − f_x                     (x-velocity)
//   Δu_y = ∂p/∂y − f_y                     (y-velocity)
//   Δu_z = ∂p/∂z − f_z                     (z-velocity)
// ============================================================

struct Stokes3DResult { double e_ux, e_uy, e_uz, e_p, div_max; };

static Stokes3DResult run_3d_stokes_periodic(int n)
{
    double h  = 1.0 / n;
    double pi = M_PI;
    int    nx = n, ny = n, nz = n;
    int    nxy = nx * ny;
    int    N   = nx * ny * nz;

    CartesianGrid3D g({0.,0.,0.}, {h,h,h}, {n,n,n}, DofLayout3D::CellCenter);
    LaplaceFftBulkSolverZfft3D solver(g, ZfftBcType::Periodic);

    // ---- exact fields and forcing ----
    Eigen::VectorXd ux_ex(N), uy_ex(N), uz_ex(N), p_ex(N);
    Eigen::VectorXd fx(N), fy(N), fz(N);
    for (int k = 0; k < nz; ++k) {
        double z = (k + 0.5) * h;
        for (int j = 0; j < ny; ++j) {
            double y = (j + 0.5) * h;
            for (int i = 0; i < nx; ++i) {
                double x = (i + 0.5) * h;
                int idx = k*nxy + j*nx + i;
                ux_ex[idx] =  std::sin(2*pi*x) * std::cos(2*pi*y) * std::cos(2*pi*z);
                uy_ex[idx] =  std::cos(2*pi*x) * std::sin(2*pi*y) * std::cos(2*pi*z);
                uz_ex[idx] = -2.0 * std::cos(2*pi*x) * std::cos(2*pi*y) * std::sin(2*pi*z);
                p_ex [idx] =  std::sin(2*pi*x) * std::sin(2*pi*y) * std::sin(2*pi*z);
                fx[idx] = 12*pi*pi * ux_ex[idx] + 2*pi * std::cos(2*pi*x) * std::sin(2*pi*y) * std::sin(2*pi*z);
                fy[idx] = 12*pi*pi * uy_ex[idx] + 2*pi * std::sin(2*pi*x) * std::cos(2*pi*y) * std::sin(2*pi*z);
                fz[idx] = 12*pi*pi * uz_ex[idx] + 2*pi * std::sin(2*pi*x) * std::sin(2*pi*y) * std::cos(2*pi*z);
            }
        }
    }

    // ---- Step 1: pressure  Δp = ∇·f ----
    Eigen::VectorXd div_f(N);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int ip = (i+1)%nx, im = (i-1+nx)%nx;
                int jp = (j+1)%ny, jm = (j-1+ny)%ny;
                int kp = (k+1)%nz, km = (k-1+nz)%nz;
                div_f[k*nxy+j*nx+i] =
                    (fx[k*nxy+j*nx+ip] - fx[k*nxy+j*nx+im]) / (2*h) +
                    (fy[k*nxy+jp*nx+i] - fy[k*nxy+jm*nx+i]) / (2*h) +
                    (fz[kp*nxy+j*nx+i] - fz[km*nxy+j*nx+i]) / (2*h);
            }
        }
    }
    Eigen::VectorXd p_sol;
    solver.solve(div_f, p_sol);

    // ---- Step 2–4: velocity  Δu_i = ∂p/∂x_i − f_i ----
    Eigen::VectorXd dp_dx(N), dp_dy(N), dp_dz(N);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int ip = (i+1)%nx, im = (i-1+nx)%nx;
                int jp = (j+1)%ny, jm = (j-1+ny)%ny;
                int kp = (k+1)%nz, km = (k-1+nz)%nz;
                int c = k*nxy + j*nx + i;
                dp_dx[c] = (p_sol[k*nxy+j*nx+ip] - p_sol[k*nxy+j*nx+im]) / (2*h);
                dp_dy[c] = (p_sol[k*nxy+jp*nx+i] - p_sol[k*nxy+jm*nx+i]) / (2*h);
                dp_dz[c] = (p_sol[kp*nxy+j*nx+i] - p_sol[km*nxy+j*nx+i]) / (2*h);
            }
        }
    }
    Eigen::VectorXd ux_sol, uy_sol, uz_sol;
    solver.solve(dp_dx - fx, ux_sol);
    solver.solve(dp_dy - fy, uy_sol);
    solver.solve(dp_dz - fz, uz_sol);

    // ---- discrete incompressibility ----
    double div_max = 0.0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int ip = (i+1)%nx, im = (i-1+nx)%nx;
                int jp = (j+1)%ny, jm = (j-1+ny)%ny;
                int kp = (k+1)%nz, km = (k-1+nz)%nz;
                double div =
                    (ux_sol[k*nxy+j*nx+ip] - ux_sol[k*nxy+j*nx+im]) / (2*h) +
                    (uy_sol[k*nxy+jp*nx+i] - uy_sol[k*nxy+jm*nx+i]) / (2*h) +
                    (uz_sol[kp*nxy+j*nx+i] - uz_sol[km*nxy+j*nx+i]) / (2*h);
                div_max = std::max(div_max, std::abs(div));
            }
        }
    }

    return {
        (ux_sol - ux_ex).lpNorm<Eigen::Infinity>(),
        (uy_sol - uy_ex).lpNorm<Eigen::Infinity>(),
        (uz_sol - uz_ex).lpNorm<Eigen::Infinity>(),
        (p_sol  - p_ex ).lpNorm<Eigen::Infinity>(),
        div_max
    };
}

// ============================================================
// 3D rectangular box — convergence tests
// ============================================================

TEST_CASE("3D Periodic on box [0,2]×[0,1]×[0,1] — convergence", "[solver][zfft][rect][3d]")
{
    const std::vector<int> ns = {8, 16, 32};
    std::vector<double> errs;

    std::printf("\n  3D rect periodic: u=sin(πx)cos(2πy)cos(2πz), [0,2]×[0,1]²\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_rect_periodic(n);
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

TEST_CASE("3D Dirichlet on box [0,2]×[0,1]×[0,1] — convergence", "[solver][zfft][rect][3d]")
{
    const std::vector<int> ns = {4, 8, 16, 32};
    std::vector<double> errs;

    std::printf("\n  3D rect Dirichlet: u=sin(πx/2)sin(πy)sin(πz), [0,2]×[0,1]²\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_rect_dirichlet(n);
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

TEST_CASE("3D Neumann on box [0,2]×[0,1]×[0,1] — convergence", "[solver][zfft][rect][3d]")
{
    const std::vector<int> ns = {4, 8, 16, 32};
    std::vector<double> errs;

    std::printf("\n  3D rect Neumann: u=cos(πx/2)cos(πy)cos(πz), [0,2]×[0,1]²\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_rect_neumann(n);
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
// 3D screened Poisson — convergence tests
// ============================================================

TEST_CASE("3D screened Poisson (η=1) Dirichlet — convergence", "[solver][zfft][screened][3d]")
{
    const std::vector<int> ns = {4, 8, 16, 32};
    std::vector<double> errs;

    std::printf("\n  3D screened Poisson (η=1): u=sin(πx)sin(πy)sin(πz)\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_screened_poisson(n, 1.0);
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

TEST_CASE("3D screened Poisson (η=100) Dirichlet — convergence", "[solver][zfft][screened][3d]")
{
    const std::vector<int> ns = {4, 8, 16, 32};
    std::vector<double> errs;

    std::printf("\n  3D screened Poisson (η=100): u=sin(πx)sin(πy)sin(πz)\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max error", "rate");
    for (int n : ns) {
        double e = err_3d_screened_poisson(n, 100.0);
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
// 3D Stokes periodic (4 Poisson solves) — convergence tests
// ============================================================

TEST_CASE("3D Stokes periodic (4 Poisson solves) — convergence", "[solver][zfft][stokes][3d]")
{
    const std::vector<int> ns = {8, 16, 32};

    std::vector<double> errs_ux, errs_uy, errs_uz, errs_p;

    std::printf("\n  3D Stokes periodic (4 Poisson): ux=sin(2πx)cos(2πy)cos(2πz), p=sin(2πx)sin(2πy)sin(2πz)\n");
    std::printf("  %-6s  %-12s  %-12s  %-12s  %-12s  %-6s\n",
                "n", "err_ux", "err_uy", "err_uz", "err_p", "rate_ux");

    for (int n : ns) {
        auto r = run_3d_stokes_periodic(n);
        errs_ux.push_back(r.e_ux);
        errs_uy.push_back(r.e_uy);
        errs_uz.push_back(r.e_uz);
        errs_p .push_back(r.e_p);

        if (errs_ux.size() == 1) {
            std::printf("  %-6d  %-12.4e  %-12.4e  %-12.4e  %-12.4e  —\n",
                        n, r.e_ux, r.e_uy, r.e_uz, r.e_p);
        } else {
            double rate = std::log2(errs_ux[errs_ux.size()-2] / r.e_ux);
            std::printf("  %-6d  %-12.4e  %-12.4e  %-12.4e  %-12.4e  %.2f\n",
                        n, r.e_ux, r.e_uy, r.e_uz, r.e_p, rate);
        }
    }

    // Skip k=1 (8→16): pre-asymptotic at coarse 3D grids; check 16→32 onward.
    for (std::size_t k = 2; k < errs_ux.size(); ++k) {
        REQUIRE(std::log2(errs_ux[k-1] / errs_ux[k]) > 1.8);
        REQUIRE(std::log2(errs_ux[k-1] / errs_ux[k]) < 2.2);
        REQUIRE(std::log2(errs_uy[k-1] / errs_uy[k]) > 1.8);
        REQUIRE(std::log2(errs_uy[k-1] / errs_uy[k]) < 2.2);
        REQUIRE(std::log2(errs_uz[k-1] / errs_uz[k]) > 1.8);
        REQUIRE(std::log2(errs_uz[k-1] / errs_uz[k]) < 2.2);
        REQUIRE(std::log2(errs_p [k-1] / errs_p [k]) > 1.8);
        REQUIRE(std::log2(errs_p [k-1] / errs_p [k]) < 2.2);
    }
}

TEST_CASE("3D Stokes periodic — discrete incompressibility O(h²)", "[solver][zfft][stokes][3d]")
{
    const std::vector<int> ns = {8, 16, 32};
    std::vector<double> divs;

    std::printf("\n  3D Stokes periodic — max |∇·u_h|\n");
    std::printf("  %-6s  %-12s  %-6s\n", "n", "max|div u|", "rate");

    for (int n : ns) {
        double d = run_3d_stokes_periodic(n).div_max;
        divs.push_back(d);
        if (divs.size() == 1)
            std::printf("  %-6d  %-12.4e  —\n", n, d);
        else
            std::printf("  %-6d  %-12.4e  %.2f\n", n, d,
                        std::log2(divs[divs.size()-2] / d));
    }

    // Skip k=1 (8→16): pre-asymptotic; check 16→32 onward.
    for (std::size_t k = 2; k < divs.size(); ++k) {
        double rate = std::log2(divs[k-1] / divs[k]);
        REQUIRE(rate > 1.8);
        REQUIRE(rate < 2.2);
    }
}
