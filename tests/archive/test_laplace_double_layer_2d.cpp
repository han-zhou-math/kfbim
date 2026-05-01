// 2nd kind BIE for Interior Dirichlet BVP using double-layer representation.
//
// Formulation:
//   u = u_part + D[phi]
// where:
//   -Δu_part = f,  u_part = 0 on ∂Ω_box
//   D[phi] is double-layer potential with density phi: [u]=phi, [∂u/∂n]=0.
//
// Interior BVP: (1/2 I - K) phi = g - u_part|Γ   (Dirichlet mode)
//
// Manufactured solution: u = sin(πx)sin(πy),  f = 2π²u.
// This u is 0 on the box boundary [0,1]², matching the FFT bulk solver BCs.

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/interface/interface_2d.hpp"
#include "core/geometry/grid_pair_2d.hpp"
#include "core/grid/cartesian_grid_2d.hpp"
#include "core/solver/laplace_zfft_bulk_solver_2d.hpp"
#include "core/solver/zfft_bc_type.hpp"
#include "core/local_cauchy/jump_data.hpp"
#include "core/transfer/laplace_spread_2d.hpp"
#include "core/transfer/laplace_restrict_2d.hpp"
#include "core/operator/laplace_kfbi_operator.hpp"
#include "core/gmres/gmres.hpp"

using namespace kfbim;

static constexpr double kPi = 3.14159265358979323846;

// ── Manufactured solution ─────────────────────────────────────────────────────
// u = sin(πx)sin(πy),  -Δu = 2π²u

static double u_exact(double x, double y) { return std::sin(kPi*x)*std::sin(kPi*y); }
static double f_bulk(double x, double y)  { return 2.0*kPi*kPi*std::sin(kPi*x)*std::sin(kPi*y); }

// ── Star interface ────────────────────────────────────────────────────────────

static const double kGL_s[3] = {-0.7745966692414834, 0.0, +0.7745966692414834};
static const double kGL_w[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};

static Interface2D make_star_panels(int N_panels)
{
    const double cx = 0.5, cy = 0.5, R = 0.28, A = 0.40;
    const int K = 5;
    const int Nq = 3 * N_panels;
    Eigen::MatrixX2d pts(Nq, 2), nml(Nq, 2);
    Eigen::VectorXd  wts(Nq);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(N_panels);
    const double dth = 2.0 * kPi / N_panels;
    int q = 0;
    for (int p = 0; p < N_panels; ++p) {
        const double th_mid   = (p + 0.5) * dth;
        const double half_dth = 0.5 * dth;
        for (int i = 0; i < 3; ++i) {
            const double th   = th_mid + half_dth * kGL_s[i];
            const double r    = R * (1.0 + A * std::cos(K * th));
            const double drdt = -R * A * K * std::sin(K * th);
            pts(q, 0) = cx + r * std::cos(th);
            pts(q, 1) = cy + r * std::sin(th);
            const double tx   = drdt * std::cos(th) - r * std::sin(th);
            const double ty   = drdt * std::sin(th) + r * std::cos(th);
            const double tlen = std::sqrt(tx*tx + ty*ty);
            nml(q, 0) =  ty / tlen;
            nml(q, 1) = -tx / tlen;
            wts[q]    = kGL_w[i] * half_dth * tlen;
            ++q;
        }
    }
    return {pts, nml, wts, 3, comp};
}

// ── Solver helper ─────────────────────────────────────────────────────────────

static double run_double_layer_dirichlet(int N, LaplaceKFBIMode mode)
{
    const double h = 1.0 / N;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {N, N}, DofLayout2D::Node);
    auto d = grid.dof_dims();
    int nx = d[0], ny = d[1];

    auto iface = make_star_panels(N);
    GridPair2D gp(grid, iface);
    const int Nq = iface.num_points();

    std::vector<int> labels(nx * ny);
    for (int n = 0; n < nx * ny; ++n) labels[n] = gp.domain_label(n);

    // 1. Fixed RHS forcing: f = 2π²u
    Eigen::VectorXd base_rhs(nx * ny);
    for (int n = 0; n < nx * ny; ++n) {
        auto c = grid.coord(n);
        bool bdy = (n % nx == 0 || n % nx == nx-1 || n / nx == 0 || n / nx == ny-1);
        base_rhs[n] = bdy ? 0.0 : f_bulk(c[0], c[1]);
    }

    // 2. Fixed f jumps at interface: for this u, f is continuous, so [f]=0
    std::vector<Eigen::VectorXd> rhs_derivs(Nq, Eigen::VectorXd::Zero(1));

    // 3. Components
    LaplacePanelSpread2D        spread(gp);
    LaplaceFftBulkSolverZfft2D  bulk_solver(grid, ZfftBcType::Dirichlet);
    LaplaceQuadraticRestrict2D  restrict_op(gp);

    // 4. KFBI operator: unknown [u] = phi,  [un] = 0
    LaplaceKFBIOperator2D op(spread, bulk_solver, restrict_op,
                              base_rhs, rhs_derivs, mode);

    // 5. GMRES RHS: b = g - u_part|Γ
    // Compute u_part|Γ by applying full operator with phi=0
    Eigen::VectorXd u_part_trace;
    op.apply_full(Eigen::VectorXd::Zero(Nq), u_part_trace);

    Eigen::VectorXd gmres_rhs(Nq);
    for (int q = 0; q < Nq; ++q)
        gmres_rhs[q] = u_exact(iface.points()(q, 0), iface.points()(q, 1)) - u_part_trace[q];

    // 6. Solve (now purely linear!)
    Eigen::VectorXd phi = Eigen::VectorXd::Zero(Nq);
    GMRES gmres(100, 1e-10, 50);
    int iters = gmres.solve(op, gmres_rhs, phi);

    std::printf("    N=%3d  iters=%2d  res=%.2e\n", 
                N, iters, gmres.residuals().empty() ? 0.0 : gmres.residuals().back());

    // 7. Final bulk solve with converged phi
    Eigen::VectorXd u_bulk;
    
    std::vector<LaplaceJumpData2D> final_jumps(Nq);
    for (int q = 0; q < Nq; ++q) {
        final_jumps[q].u_jump     = phi[q];
        final_jumps[q].un_jump    = 0.0;
        final_jumps[q].rhs_derivs = Eigen::VectorXd::Zero(1);
    }
    Eigen::VectorXd rhs = base_rhs;
    spread.apply(final_jumps, rhs);
    bulk_solver.solve(-rhs, u_bulk);

    // 8. Error on target side
    int target_label = 1; // Interior only
    double err = 0.0;
    for (int j = 1; j < ny-1; ++j) {
        for (int i = 1; i < nx-1; ++i) {
            int n = j * nx + i;
            if (gp.domain_label(n) != target_label) continue;
            auto c = grid.coord(n);
            err = std::max(err, std::abs(u_bulk[n] - u_exact(c[0], c[1])));
        }
    }
    return err;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("Laplace 2nd kind BIE (Double Layer): Interior Dirichlet BVP",
          "[kfbi][laplace][double_layer][interior][2d]")
{
    std::printf("\n  Interior Dirichlet BVP (Double Layer, 2nd kind BIE):\n");
    const int Ns[] = {32, 64, 128};
    double errs[3];
    for (int i = 0; i < 3; ++i) {
        errs[i] = run_double_layer_dirichlet(Ns[i], LaplaceKFBIMode::Dirichlet);
        if (i == 0) std::printf("    N=%3d  err=%.4e\n", Ns[i], errs[i]);
        else {
            double rate = std::log2(errs[i-1] / errs[i]);
            std::printf("    N=%3d  err=%.4e  rate=%.2f\n", Ns[i], errs[i], rate);
            REQUIRE(rate > 1.5);
        }
    }
}
