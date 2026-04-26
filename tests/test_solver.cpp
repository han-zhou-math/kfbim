#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "core/solver/laplace_fft_bulk_solver_2d.hpp"

#ifdef KFBIM_HAS_FFTW3
#include "core/solver/fftw3_engine.hpp"
#endif

using namespace kfbim;
using Catch::Matchers::WithinAbs;

// ============================================================
// Helpers
// ============================================================

// Apply the periodic 5-point Laplacian to u and write into rhs.
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

    // Arbitrary input
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
    // Grid: nx×ny cells with spacing dx×dy
    int    nx = 32, ny = 24;
    double dx = 1.0 / nx, dy = 1.0 / ny;

    CartesianGrid2D g({0.0, 0.0}, {dx, dy}, {nx, ny}, DofLayout2D::CellCenter);
    auto engine = std::make_unique<FFTW3Engine2D>(nx, ny);
    LaplaceFftBulkSolver2D solver(g, std::move(engine));

    // u_exact is a discrete Fourier mode with zero mean:
    //   u[j*nx+i] = cos(2π*kx*i/nx) * cos(2π*ky*j/ny)  with kx=2, ky=3
    int kx_val = 2, ky_val = 3;
    Eigen::VectorXd u_exact(nx * ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            u_exact[j*nx+i] = std::cos(2.0*M_PI*kx_val*i/nx)
                             * std::cos(2.0*M_PI*ky_val*j/ny);

    // rhs = discrete Laplacian of u_exact
    Eigen::VectorXd rhs;
    apply_laplacian_periodic(u_exact, rhs, nx, ny, dx, dy);

    // Solve
    Eigen::VectorXd u_sol;
    solver.solve(rhs, u_sol);

    REQUIRE(u_sol.size() == nx * ny);
    for (int k = 0; k < nx * ny; ++k)
        REQUIRE_THAT(u_sol[k], WithinAbs(u_exact[k], 1e-10));
}

TEST_CASE("LaplaceFftBulkSolver2D — sine×sine mode, non-square grid", "[solver][fftw3]") {
    // Uses a rectangular grid and a sine×sine mode (also zero-mean)
    int    nx = 20, ny = 16;
    double Lx = 2.0, Ly = 1.5;
    double dx = Lx / nx, dy = Ly / ny;

    CartesianGrid2D g({0.0, 0.0}, {dx, dy}, {nx, ny}, DofLayout2D::CellCenter);
    auto engine = std::make_unique<FFTW3Engine2D>(nx, ny);
    LaplaceFftBulkSolver2D solver(g, std::move(engine));

    // u[i,j] = sin(2π*1*i/nx) * sin(2π*2*j/ny)
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
