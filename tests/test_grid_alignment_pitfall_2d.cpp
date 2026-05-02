#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/grid/cartesian_grid_2d.hpp"
#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/local_cauchy/jump_data.hpp"
#include "core/local_cauchy/local_poly.hpp"
#include "core/problems/laplace_interface_solver_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "core/solver/zfft_bc_type.hpp"
#include "core/transfer/laplace_restrict_2d.hpp"
#include "core/transfer/laplace_spread_2d.hpp"

using namespace kfbim;

namespace {

constexpr double kPi = 3.14159265358979323846;

constexpr double kGL_s[3] = {
    -0.7745966692414834,
     0.0,
     0.7745966692414834
};
constexpr double kGL_w[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
constexpr double kCircleCx = 0.0;
constexpr double kCircleCy = 0.0;

// Harmonic manufactured solution:
//   u_int = exp(x) sin(y),  u_ext = 0.
// Both phases have zero bulk forcing, and the exterior branch satisfies the
// homogeneous Dirichlet condition on the outer box.
double u_ext(double x, double y) {
    (void)x;
    (void)y;
    return 0.0;
}

double u_int(double x, double y) {
    return std::exp(x) * std::sin(y);
}

double f_ext(double x, double y) {
    (void)x;
    (void)y;
    return 0.0;
}

double f_int(double x, double y) {
    (void)x;
    (void)y;
    return 0.0;
}

double ux_ext(double x, double y) {
    (void)x;
    (void)y;
    return 0.0;
}

double uy_ext(double x, double y) {
    (void)x;
    (void)y;
    return 0.0;
}

double ux_int(double x, double y) {
    return std::exp(x) * std::sin(y);
}

double uy_int(double x, double y) {
    return std::exp(x) * std::cos(y);
}

double uxx_ext(double x, double y) {
    (void)x;
    (void)y;
    return 0.0;
}

double uyy_ext(double x, double y) {
    (void)x;
    (void)y;
    return 0.0;
}

double uxy_ext(double x, double y) {
    (void)x;
    (void)y;
    return 0.0;
}

double uxx_int(double x, double y) {
    return std::exp(x) * std::sin(y);
}

double uyy_int(double x, double y) {
    return -std::exp(x) * std::sin(y);
}

double uxy_int(double x, double y) {
    return std::exp(x) * std::cos(y);
}

double C(double x, double y) {
    return u_int(x, y) - u_ext(x, y);
}

double Cx(double x, double y) {
    return ux_int(x, y) - ux_ext(x, y);
}

double Cy(double x, double y) {
    return uy_int(x, y) - uy_ext(x, y);
}

double Cxx(double x, double y) {
    return uxx_int(x, y) - uxx_ext(x, y);
}

double Cxy(double x, double y) {
    return uxy_int(x, y) - uxy_ext(x, y);
}

double Cyy(double x, double y) {
    return uyy_int(x, y) - uyy_ext(x, y);
}

double normal_derivative_int(double x, double y, double nx, double ny) {
    return ux_int(x, y) * nx + uy_int(x, y) * ny;
}

double normal_derivative_ext(double x, double y, double nx, double ny) {
    return ux_ext(x, y) * nx + uy_ext(x, y) * ny;
}

Interface2D make_shifted_unit_circle_panels(int num_panels) {
    const int Nq = 3 * num_panels;
    Eigen::MatrixX2d pts(Nq, 2);
    Eigen::MatrixX2d nml(Nq, 2);
    Eigen::VectorXd wts(Nq);
    Eigen::VectorXi comp = Eigen::VectorXi::Zero(num_panels);

    const double dth = 2.0 * kPi / static_cast<double>(num_panels);
    int q = 0;
    for (int p = 0; p < num_panels; ++p) {
        // Midpoint nodes hit theta = 0, pi/2, pi, 3pi/2 whenever
        // num_panels is divisible by 4.
        const double th_mid = p * dth;
        const double half_dth = 0.5 * dth;
        for (int i = 0; i < 3; ++i) {
            const double th = th_mid + half_dth * kGL_s[i];
            pts(q, 0) = kCircleCx + std::cos(th);
            pts(q, 1) = kCircleCy + std::sin(th);
            nml(q, 0) = std::cos(th);
            nml(q, 1) = std::sin(th);
            wts[q] = kGL_w[i] * half_dth;
            ++q;
        }
    }

    return {pts, nml, wts, 3, comp};
}

int panel_count_for_grid(int N) {
    // Keeps all power-of-two levels divisible by four, while avoiding an
    // unnecessarily dense O(grid * interface) closest-point precompute.
    return std::max(8, N / 4);
}

std::vector<int> requested_levels() {
    return {32, 64, 128, 256, 512, 1024};
}

struct InterfaceSolveData {
    double bulk_err;
    double u_avg_err;
    double un_avg_err;
};

InterfaceSolveData run_interface_solve(int N) {
    const double h = 4.0 / static_cast<double>(N);
    CartesianGrid2D grid({-2.0, -2.0}, {h, h}, {N, N}, DofLayout2D::Node);
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];

    auto iface = make_shifted_unit_circle_panels(panel_count_for_grid(N));
    GridPair2D gp(grid, iface);
    const int Nq = iface.num_points();

    Eigen::VectorXd f_bulk(grid.num_dofs());
    Eigen::VectorXd exact(grid.num_dofs());
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        const bool boundary =
            (n % nx == 0 || n % nx == nx - 1 || n / nx == 0 || n / nx == ny - 1);
        if (gp.domain_label(n) == 1) {
            exact[n] = u_int(x, y);
            f_bulk[n] = boundary ? 0.0 : f_int(x, y);
        } else {
            exact[n] = u_ext(x, y);
            f_bulk[n] = boundary ? 0.0 : f_ext(x, y);
        }
    }

    std::vector<LaplaceJumpData2D> jumps(Nq);
    for (int q = 0; q < Nq; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const double nx_q = iface.normals()(q, 0);
        const double ny_q = iface.normals()(q, 1);
        jumps[q].u_jump = C(x, y);
        jumps[q].un_jump = Cx(x, y) * nx_q + Cy(x, y) * ny_q;
        jumps[q].rhs_derivs = Eigen::VectorXd::Constant(1, f_int(x, y) - f_ext(x, y));
    }

    LaplacePanelSpread2D spread(gp);
    LaplaceFftBulkSolverZfft2D bulk_solver(grid, ZfftBcType::Dirichlet);
    LaplaceQuadraticRestrict2D restrict_op(gp);
    LaplaceInterfaceSolver2D solver(spread, bulk_solver, restrict_op);

    const auto result = solver.solve(jumps, f_bulk);

    double bulk_err = 0.0;
    for (int j = 1; j < ny - 1; ++j) {
        for (int i = 1; i < nx - 1; ++i) {
            const int n = j * nx + i;
            bulk_err = std::max(bulk_err, std::abs(result.u_bulk[n] - exact[n]));
        }
    }

    double u_avg_err = 0.0;
    double un_avg_err = 0.0;
    for (int q = 0; q < Nq; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const double nx_q = iface.normals()(q, 0);
        const double ny_q = iface.normals()(q, 1);
        const double exact_u_avg = 0.5 * (u_int(x, y) + u_ext(x, y));
        const double exact_un_avg =
            0.5 * (normal_derivative_int(x, y, nx_q, ny_q)
                   + normal_derivative_ext(x, y, nx_q, ny_q));

        u_avg_err = std::max(u_avg_err, std::abs(result.u_avg[q] - exact_u_avg));
        un_avg_err = std::max(un_avg_err, std::abs(result.un_avg[q] - exact_un_avg));
    }

    return {bulk_err, u_avg_err, un_avg_err};
}

struct RestrictData {
    double u_avg_err;
    double un_avg_err;
};

RestrictData run_restrict_only(int N) {
    const double h = 4.0 / static_cast<double>(N);
    CartesianGrid2D grid({-2.0, -2.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto iface = make_shifted_unit_circle_panels(panel_count_for_grid(N));
    GridPair2D gp(grid, iface);
    const int Nq = iface.num_points();

    Eigen::VectorXd bulk(grid.num_dofs());
    for (int n = 0; n < grid.num_dofs(); ++n) {
        const auto c = grid.coord(n);
        const double x = c[0];
        const double y = c[1];
        bulk[n] = gp.domain_label(n) == 1 ? u_int(x, y) : u_ext(x, y);
    }

    std::vector<LocalPoly2D> correction(Nq);
    for (int q = 0; q < Nq; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        correction[q].center = {x, y};
        correction[q].coeffs.resize(6);
        correction[q].coeffs << C(x, y), Cx(x, y), Cy(x, y),
                                Cxx(x, y), Cxy(x, y), Cyy(x, y);
    }

    LaplaceQuadraticRestrict2D restrict_op(gp);
    const auto polys = restrict_op.apply(bulk, correction);

    double u_avg_err = 0.0;
    double un_avg_err = 0.0;
    for (int q = 0; q < Nq; ++q) {
        const double x = iface.points()(q, 0);
        const double y = iface.points()(q, 1);
        const double nx_q = iface.normals()(q, 0);
        const double ny_q = iface.normals()(q, 1);
        const double jump_u = C(x, y);
        const double jump_un = Cx(x, y) * nx_q + Cy(x, y) * ny_q;
        const double u_trace = polys[q].coeffs[0];
        const double un_trace =
            polys[q].coeffs[1] * nx_q + polys[q].coeffs[2] * ny_q;

        const double numerical_u_avg = u_trace - 0.5 * jump_u;
        const double numerical_un_avg = un_trace - 0.5 * jump_un;
        const double exact_u_avg = 0.5 * (u_int(x, y) + u_ext(x, y));
        const double exact_un_avg =
            0.5 * (normal_derivative_int(x, y, nx_q, ny_q)
                   + normal_derivative_ext(x, y, nx_q, ny_q));

        u_avg_err = std::max(u_avg_err, std::abs(numerical_u_avg - exact_u_avg));
        un_avg_err = std::max(un_avg_err, std::abs(numerical_un_avg - exact_un_avg));
    }

    return {u_avg_err, un_avg_err};
}

double rate(double coarse, double fine) {
    return std::log2(coarse / fine);
}

} // namespace

TEST_CASE("Grid/interface alignment pitfall: Poisson IIM solve on shifted unit circle",
          "[.][grid_alignment][iim][2d]")
{
    const auto Ns = requested_levels();
    std::vector<InterfaceSolveData> data;
    data.reserve(Ns.size());

    std::printf("\n  Grid alignment pitfall check: Poisson interface solve\n");
    std::printf("  Interface: unit circle centered at (%.2f, %.2f); box: [-2,2]^2\n",
                kCircleCx, kCircleCy);
    std::printf("  Branches: u_int=exp(x)sin(y), u_ext=0\n");
    std::printf("  %6s  %8s  %12s  %8s  %12s  %8s  %12s  %8s\n",
                "N", "Npan", "bulk_err", "rate",
                "u_avg_err", "rate", "un_avg_err", "rate");

    bool all_finite = true;
    for (std::size_t l = 0; l < Ns.size(); ++l) {
        data.push_back(run_interface_solve(Ns[l]));
        all_finite = all_finite
                     && std::isfinite(data[l].bulk_err)
                     && std::isfinite(data[l].u_avg_err)
                     && std::isfinite(data[l].un_avg_err);
        if (l == 0) {
            std::printf("  %6d  %8d  %12.4e  %8s  %12.4e  %8s  %12.4e  %8s\n",
                        Ns[l], panel_count_for_grid(Ns[l]), data[l].bulk_err, "-",
                        data[l].u_avg_err, "-", data[l].un_avg_err, "-");
        } else {
            std::printf("  %6d  %8d  %12.4e  %8.3f  %12.4e  %8.3f  %12.4e  %8.3f\n",
                        Ns[l], panel_count_for_grid(Ns[l]),
                        data[l].bulk_err, rate(data[l - 1].bulk_err, data[l].bulk_err),
                        data[l].u_avg_err, rate(data[l - 1].u_avg_err, data[l].u_avg_err),
                        data[l].un_avg_err, rate(data[l - 1].un_avg_err, data[l].un_avg_err));
        }
    }
    CHECK(all_finite);
    CHECK(data.back().bulk_err < data.front().bulk_err);
    CHECK(data.back().u_avg_err < data.front().u_avg_err);
    CHECK(data.back().un_avg_err < data.front().un_avg_err);
}

TEST_CASE("Grid/interface alignment pitfall: restrict average trace on shifted unit circle",
          "[.][grid_alignment][restrict][2d]")
{
    const auto Ns = requested_levels();
    std::vector<RestrictData> data;
    data.reserve(Ns.size());

    std::printf("\n  Grid alignment pitfall check: restrict-only trace interpolation\n");
    std::printf("  Interface: unit circle centered at (%.2f, %.2f); box: [-2,2]^2\n",
                kCircleCx, kCircleCy);
    std::printf("  %6s  %8s  %12s  %8s  %12s  %8s\n",
                "N", "Npan", "u_avg_err", "rate", "un_avg_err", "rate");

    bool all_finite = true;
    for (std::size_t l = 0; l < Ns.size(); ++l) {
        data.push_back(run_restrict_only(Ns[l]));
        all_finite = all_finite
                     && std::isfinite(data[l].u_avg_err)
                     && std::isfinite(data[l].un_avg_err);
        if (l == 0) {
            std::printf("  %6d  %8d  %12.4e  %8s  %12.4e  %8s\n",
                        Ns[l], panel_count_for_grid(Ns[l]),
                        data[l].u_avg_err, "-", data[l].un_avg_err, "-");
        } else {
            std::printf("  %6d  %8d  %12.4e  %8.3f  %12.4e  %8.3f\n",
                        Ns[l], panel_count_for_grid(Ns[l]),
                        data[l].u_avg_err, rate(data[l - 1].u_avg_err, data[l].u_avg_err),
                        data[l].un_avg_err, rate(data[l - 1].un_avg_err, data[l].un_avg_err));
        }
    }
    CHECK(all_finite);
    CHECK(data.back().u_avg_err < data.front().u_avg_err);
    CHECK(data.back().un_avg_err < data.front().un_avg_err);
}
