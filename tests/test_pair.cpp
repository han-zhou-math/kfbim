#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <unordered_set>

static constexpr double kPi = 3.14159265358979323846;

#include "src/grid/cartesian_grid_2d.hpp"
#include "src/grid/cartesian_grid_3d.hpp"
#include "src/interface/interface_2d.hpp"
#include "src/interface/interface_3d.hpp"
#include "src/geometry/grid_pair_2d.hpp"
#include "src/geometry/grid_pair_3d.hpp"

using namespace kfbim;

// ============================================================================
// Test factories
// ============================================================================

// Uniformly-sampled circle at (cx,cy) with radius r, N points.
// Each point is its own panel (points_per_panel = 1).
static Interface2D make_circle(double cx, double cy, double r, int N)
{
    Eigen::MatrixX2d pts(N, 2), nml(N, 2);
    Eigen::VectorXd  wts(N);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(N);
    const double dtheta = 2.0 * kPi / N;
    for (int i = 0; i < N; ++i) {
        double th = i * dtheta;
        double ct = std::cos(th), st = std::sin(th);
        pts(i, 0) = cx + r * ct;  pts(i, 1) = cy + r * st;
        nml(i, 0) = ct;           nml(i, 1) = st;
        wts(i)    = r * dtheta;
    }
    return {pts, nml, wts, 1, comp};
}

// Circle represented by 3-point Chebyshev-Lobatto panel geometry.
static Interface2D make_circle_lobatto_panels(double cx, double cy, double r, int N_panels)
{
    constexpr double lobatto_s[3] = {
        -1.0,
         0.0,
         1.0
    };
    constexpr double lobatto_w[3] = {1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0};

    const int Nq = 3 * N_panels;
    Eigen::MatrixX2d pts(Nq, 2), nml(Nq, 2);
    Eigen::VectorXd  wts(Nq);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(N_panels);
    const double dtheta = 2.0 * kPi / N_panels;

    int q = 0;
    for (int p = 0; p < N_panels; ++p) {
        const double theta_mid = (p + 0.5) * dtheta;
        const double half_dtheta = 0.5 * dtheta;
        for (int i = 0; i < 3; ++i) {
            const double th = theta_mid + half_dtheta * lobatto_s[i];
            const double ct = std::cos(th), st = std::sin(th);
            pts(q, 0) = cx + r * ct;
            pts(q, 1) = cy + r * st;
            nml(q, 0) = ct;
            nml(q, 1) = st;
            wts(q) = lobatto_w[i] * half_dtheta * r;
            ++q;
        }
    }

    return {pts, nml, wts, 3, comp, PanelNodeLayout2D::ChebyshevLobatto};
}

// Multi-circle interface: arbitrary number of circles, each its own component.
struct CircleCfg { double cx, cy, r; };

static Interface2D make_multi_circles(const std::vector<CircleCfg>& cfgs, int N_each)
{
    int Nc   = static_cast<int>(cfgs.size());
    int Ntot = Nc * N_each;
    Eigen::MatrixX2d pts(Ntot, 2), nml(Ntot, 2);
    Eigen::VectorXd  wts(Ntot);
    Eigen::VectorXi  comp(Ntot);
    const double dtheta = 2.0 * kPi / N_each;
    for (int c = 0; c < Nc; ++c) {
        double cx = cfgs[c].cx, cy = cfgs[c].cy, r = cfgs[c].r;
        for (int i = 0; i < N_each; ++i) {
            int idx = c * N_each + i;
            double th = i * dtheta;
            double ct = std::cos(th), st = std::sin(th);
            pts(idx, 0) = cx + r * ct;  pts(idx, 1) = cy + r * st;
            nml(idx, 0) = ct;           nml(idx, 1) = st;
            wts(idx)    = r * dtheta;
            comp(idx)   = c;
        }
    }
    return {pts, nml, wts, 1, comp};
}

// Two-circle interface: circles at (0.25, 0.5) and (0.75, 0.5), both r=0.15.
static Interface2D make_two_circles(int N_each)
{
    int Ntot = 2 * N_each;
    Eigen::MatrixX2d pts(Ntot, 2), nml(Ntot, 2);
    Eigen::VectorXd  wts(Ntot);
    Eigen::VectorXi  comp(Ntot);  // panel_components has one entry per panel = one per point here

    const double r = 0.15;
    const double centers[2][2] = {{0.25, 0.5}, {0.75, 0.5}};
    const double dtheta = 2.0 * kPi / N_each;

    for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < N_each; ++i) {
            int idx = c * N_each + i;
            double th = i * dtheta;
            double ct = std::cos(th), st = std::sin(th);
            pts(idx, 0) = centers[c][0] + r * ct;
            pts(idx, 1) = centers[c][1] + r * st;
            nml(idx, 0) = ct;
            nml(idx, 1) = st;
            wts(idx)    = r * dtheta;
            comp(idx)   = c;
        }
    }
    return {pts, nml, wts, 1, comp};
}

// Triangulated cube [lo, hi]^3 — 8 vertices, 12 triangles.
// One quadrature point (centroid) per triangle, outward normals.
static Interface3D make_cube(double lo, double hi)
{
    // 8 corners
    Eigen::MatrixX3d V(8, 3);
    V << lo,lo,lo,  hi,lo,lo,  hi,hi,lo,  lo,hi,lo,  // 0-3 bottom
         lo,lo,hi,  hi,lo,hi,  hi,hi,hi,  lo,hi,hi;  // 4-7 top

    // 12 triangles (2 per face), outward winding viewed from outside
    Eigen::MatrixX3i F(12, 3);
    // bottom (-z): normal (0,0,-1)
    F.row(0)  << 0,2,1;  F.row(1)  << 0,3,2;
    // top (+z): normal (0,0,+1)
    F.row(2)  << 4,5,6;  F.row(3)  << 4,6,7;
    // front (-y): normal (0,-1,0)
    F.row(4)  << 0,1,5;  F.row(5)  << 0,5,4;
    // back (+y): normal (0,+1,0)
    F.row(6)  << 2,3,7;  F.row(7)  << 2,7,6;
    // left (-x): normal (-1,0,0)
    F.row(8)  << 0,4,7;  F.row(9)  << 0,7,3;
    // right (+x): normal (+1,0,0)
    F.row(10) << 1,2,6;  F.row(11) << 1,6,5;

    int Np = 12;
    Eigen::MatrixX3d pts(Np, 3), nml(Np, 3);
    Eigen::VectorXd  wts(Np);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(Np);

    // Face normal directions (per pair of triangles, rows 0-11)
    const double face_nml[12][3] = {
        {0,0,-1}, {0,0,-1},
        {0,0, 1}, {0,0, 1},
        {0,-1,0}, {0,-1,0},
        {0, 1,0}, {0, 1,0},
        {-1,0,0}, {-1,0,0},
        { 1,0,0}, { 1,0,0}
    };

    for (int p = 0; p < Np; ++p) {
        // centroid
        for (int d = 0; d < 3; ++d)
            pts(p, d) = (V(F(p,0), d) + V(F(p,1), d) + V(F(p,2), d)) / 3.0;
        nml.row(p) << face_nml[p][0], face_nml[p][1], face_nml[p][2];
        // area = 0.5 * (hi-lo)^2 for right-triangle faces
        wts(p) = 0.5 * (hi-lo) * (hi-lo);
    }

    return {V, F, pts, nml, wts, 1, comp};
}

// ============================================================================
// Interface2D construction
// ============================================================================

TEST_CASE("Interface2D construction and accessors", "[interface][2d]")
{
    auto iface = make_circle(0.5, 0.5, 0.3, 64);
    REQUIRE(iface.num_points()        == 64);
    REQUIRE(iface.num_panels()        == 64);
    REQUIRE(iface.points_per_panel()  ==  1);
    REQUIRE(iface.num_components()    ==  1);
    REQUIRE(iface.weights().sum()     == Catch::Approx(2.0 * kPi * 0.3).epsilon(1e-3));
}

TEST_CASE("Interface2D rejects bad arguments", "[interface][2d]")
{
    Eigen::MatrixX2d pts(6, 2); pts.setZero();
    Eigen::MatrixX2d nml(6, 2); nml.setZero();
    Eigen::VectorXd  wts(6);    wts.setZero();
    Eigen::VectorXi  comp(6);   comp.setZero();
    REQUIRE_THROWS(Interface2D(pts, nml, wts, 0, comp));   // bad k
    REQUIRE_THROWS(Interface2D(pts, nml, wts, 4, comp));   // 6 not divisible by 4
}

// ============================================================================
// Interface3D construction
// ============================================================================

TEST_CASE("Interface3D construction and accessors", "[interface][3d]")
{
    auto iface = make_cube(0.2, 0.8);
    REQUIRE(iface.num_panels()  == 12);
    REQUIRE(iface.num_points()  == 12);
    REQUIRE(iface.num_vertices()== 8);
}

// ============================================================================
// GridPair2D — circle in [0,1]^2
// ============================================================================

TEST_CASE("GridPair2D construction", "[gridpair][2d]")
{
    // 17×17 Node grid on [0,1]^2 (h = 1/16)
    CartesianGrid2D grid({0.0, 0.0}, {1.0/16, 1.0/16}, {16, 16}, DofLayout2D::Node);
    auto iface = make_circle(0.5, 0.5, 0.3, 128);
    REQUIRE_NOTHROW(GridPair2D(grid, iface));
}

TEST_CASE("GridPair2D domain labeling — circle", "[gridpair][2d]")
{
    const double h  = 1.0 / 32;
    const double r  = 0.3;
    const double cx = 0.5, cy = 0.5;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {32, 32}, DofLayout2D::Node);
    auto iface = make_circle(cx, cy, r, 256);
    GridPair2D gp(grid, iface);

    auto d = grid.dof_dims();
    int nx = d[0], ny = d[1];
    int n_interior_wrong = 0, n_exterior_wrong = 0;

    // Use a safe margin away from the interface to avoid boundary ambiguity.
    const double margin = 2.0 * h;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = i * h, y = j * h;
            double dist_to_circle = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
            int label = gp.domain_label(j * nx + i);
            if (dist_to_circle < r - margin) {
                // Definitely inside
                if (label != 1) n_interior_wrong++;
            } else if (dist_to_circle > r + margin) {
                // Definitely outside
                if (label != 0) n_exterior_wrong++;
            }
        }
    }
    REQUIRE(n_interior_wrong == 0);
    REQUIRE(n_exterior_wrong == 0);
}

TEST_CASE("GridPair2D domain labeling uses oversampled Chebyshev-Lobatto panels",
          "[gridpair][2d][domain]")
{
    auto iface = make_circle_lobatto_panels(0.0, 0.0, 1.0, 4);
    CartesianGrid2D grid({-0.99, -0.99}, {0.99, 0.99}, {2, 2}, DofLayout2D::Node);
    GridPair2D gp(grid, iface);

    // These nodes sit just inside the true circle at quadratic panel boundaries.
    // A polygon through only the stored panel points cuts across those boundaries.
    REQUIRE(gp.domain_label(grid.index(2, 1)) == 1);
    REQUIRE(gp.domain_label(grid.index(1, 2)) == 1);
    REQUIRE(gp.domain_label(grid.index(0, 1)) == 1);
    REQUIRE(gp.domain_label(grid.index(1, 0)) == 1);
    REQUIRE(gp.domain_label(grid.index(1, 1)) == 1);
}

TEST_CASE("GridPair2D two-component labeling", "[gridpair][2d]")
{
    const double h = 1.0 / 32;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {32, 32}, DofLayout2D::Node);
    auto iface = make_two_circles(128);
    GridPair2D gp(grid, iface);

    auto d = grid.dof_dims();
    int nx = d[0];
    const double r = 0.15, margin = 2.0 * h;
    const double centers[2][2] = {{0.25, 0.5}, {0.75, 0.5}};

    for (int c = 0; c < 2; ++c) {
        double ccx = centers[c][0], ccy = centers[c][1];
        // A node clearly inside circle c should have label c+1.
        // Pick (ccx, ccy) itself — snap to nearest grid node.
        int i = static_cast<int>(std::round(ccx / h));
        int j = static_cast<int>(std::round(ccy / h));
        int n = j * nx + i;
        REQUIRE(gp.domain_label(n) == c + 1);
    }

    // A node at the center of [0,1]^2 is outside both circles.
    int ic = static_cast<int>(std::round(0.5 / h));
    REQUIRE(gp.domain_label(ic * nx + ic) == 0);
}

TEST_CASE("GridPair2D closest_bulk_node", "[gridpair][2d]")
{
    const double h = 1.0 / 16;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {16, 16}, DofLayout2D::Node);
    auto iface = make_circle(0.5, 0.5, 0.3, 64);
    GridPair2D gp(grid, iface);

    for (int q = 0; q < iface.num_points(); ++q) {
        int n = gp.closest_bulk_node(q);
        REQUIRE(n >= 0);
        REQUIRE(n < grid.num_dofs());
        // Distance from node to interface point ≤ sqrt(2) * h (at most one diagonal step).
        auto c = grid.coord(n);
        double dx = c[0] - iface.points()(q, 0);
        double dy = c[1] - iface.points()(q, 1);
        REQUIRE(std::sqrt(dx*dx + dy*dy) <= std::sqrt(2.0) * h + 1e-10);
    }
}

TEST_CASE("GridPair2D closest_interface_point", "[gridpair][2d]")
{
    const double h = 1.0 / 16;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {16, 16}, DofLayout2D::Node);
    auto iface = make_circle(0.5, 0.5, 0.3, 64);
    GridPair2D gp(grid, iface);

    // The nearest interface point to any bulk node should be at most
    // the arc-length spacing (≈ 2π*0.3/64 ≈ 0.029) + the grid spacing away.
    double arc = 2.0 * kPi * 0.3 / 64;

    for (int n = 0; n < grid.num_dofs(); ++n) {
        int q = gp.closest_interface_point(n);
        REQUIRE(q >= 0);
        REQUIRE(q < iface.num_points());
    }

    // For a node near the circle, the nearest interface point should be close.
    // Pick (0.8, 0.5) — on the right edge of the circle.
    auto d = grid.dof_dims();
    int i = static_cast<int>(std::round(0.8 / h));
    int j = static_cast<int>(std::round(0.5 / h));
    int n = j * d[0] + i;
    int q = gp.closest_interface_point(n);
    auto c = grid.coord(n);
    double px = iface.points()(q, 0), py = iface.points()(q, 1);
    double dist = std::sqrt((c[0]-px)*(c[0]-px) + (c[1]-py)*(c[1]-py));
    REQUIRE(dist < 3.0 * h);
}

TEST_CASE("GridPair2D near_interface_nodes and is_near_interface", "[gridpair][2d]")
{
    const double h = 1.0 / 16;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {16, 16}, DofLayout2D::Node);
    auto iface = make_circle(0.5, 0.5, 0.3, 64);
    GridPair2D gp(grid, iface);

    const double radius = 2.5 * h;
    auto nodes = gp.near_interface_nodes(radius);
    REQUIRE(!nodes.empty());
    // Every returned node must satisfy is_near_interface.
    for (int n : nodes)
        REQUIRE(gp.is_near_interface(n, radius));
    // Every non-returned node must fail is_near_interface.
    std::vector<bool> in_list(grid.num_dofs(), false);
    for (int n : nodes) in_list[n] = true;
    for (int n = 0; n < grid.num_dofs(); ++n)
        REQUIRE(in_list[n] == gp.is_near_interface(n, radius));
}

TEST_CASE("GridPair2D near_interface_points", "[gridpair][2d]")
{
    const double h = 1.0 / 16;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {16, 16}, DofLayout2D::Node);
    auto iface = make_circle(0.5, 0.5, 0.3, 64);
    GridPair2D gp(grid, iface);

    // Near-interface node should have multiple interface points within 2*h.
    auto nodes = gp.near_interface_nodes(2.0 * h);
    REQUIRE(!nodes.empty());
    int n0 = nodes[0];
    auto pts = gp.near_interface_points(n0, 2.0 * h);
    REQUIRE(!pts.empty());
    // All returned interface points must be within radius.
    auto c = grid.coord(n0);
    for (int q : pts) {
        double dx = c[0] - iface.points()(q, 0);
        double dy = c[1] - iface.points()(q, 1);
        REQUIRE(std::sqrt(dx*dx + dy*dy) < 2.0 * h + 1e-10);
    }
}

// ============================================================================
// GridPair3D — cube interface in [0,1]^3
// ============================================================================

TEST_CASE("GridPair3D construction", "[gridpair][3d]")
{
    CartesianGrid3D grid({0.0, 0.0, 0.0}, {1.0/8, 1.0/8, 1.0/8}, {8, 8, 8}, DofLayout3D::Node);
    auto iface = make_cube(0.2, 0.8);
    REQUIRE_NOTHROW(GridPair3D(grid, iface));
}

TEST_CASE("GridPair3D domain labeling — cube mesh", "[gridpair][3d]")
{
    const double h  = 1.0 / 16;
    const double lo = 0.25, hi = 0.75;
    CartesianGrid3D grid({0.0, 0.0, 0.0}, {h, h, h}, {16, 16, 16}, DofLayout3D::Node);
    auto iface = make_cube(lo, hi);
    GridPair3D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0], ny = d[1];
    int  n_interior_wrong = 0, n_exterior_wrong = 0;
    const double margin = 2.0 * h;

    for (int n = 0; n < grid.num_dofs(); ++n) {
        auto c = grid.coord(n);
        double x = c[0], y = c[1], z = c[2];
        bool should_be_inside =
            (x > lo + margin && x < hi - margin) &&
            (y > lo + margin && y < hi - margin) &&
            (z > lo + margin && z < hi - margin);
        bool should_be_outside =
            (x < lo - margin || x > hi + margin) ||
            (y < lo - margin || y > hi + margin) ||
            (z < lo - margin || z > hi + margin);

        int label = gp.domain_label(n);
        if (should_be_inside  && label != 1) n_interior_wrong++;
        if (should_be_outside && label != 0) n_exterior_wrong++;
    }
    REQUIRE(n_interior_wrong == 0);
    REQUIRE(n_exterior_wrong == 0);
}

TEST_CASE("GridPair3D closest_bulk_node", "[gridpair][3d]")
{
    const double h = 1.0 / 8;
    CartesianGrid3D grid({0.0, 0.0, 0.0}, {h, h, h}, {8, 8, 8}, DofLayout3D::Node);
    auto iface = make_cube(0.2, 0.8);
    GridPair3D gp(grid, iface);

    for (int q = 0; q < iface.num_points(); ++q) {
        int n = gp.closest_bulk_node(q);
        REQUIRE(n >= 0);
        REQUIRE(n < grid.num_dofs());
        auto c = grid.coord(n);
        double dx = c[0] - iface.points()(q, 0);
        double dy = c[1] - iface.points()(q, 1);
        double dz = c[2] - iface.points()(q, 2);
        // Within sqrt(3) * h (one diagonal step in 3D).
        REQUIRE(std::sqrt(dx*dx + dy*dy + dz*dz) <= std::sqrt(3.0) * h + 1e-10);
    }
}

TEST_CASE("GridPair3D near_interface_nodes and near_interface_points", "[gridpair][3d]")
{
    const double h = 1.0 / 8;
    CartesianGrid3D grid({0.0, 0.0, 0.0}, {h, h, h}, {8, 8, 8}, DofLayout3D::Node);
    auto iface = make_cube(0.2, 0.8);
    GridPair3D gp(grid, iface);

    const double radius = 2.5 * h;
    auto nodes = gp.near_interface_nodes(radius);
    REQUIRE(!nodes.empty());
    for (int n : nodes)
        REQUIRE(gp.is_near_interface(n, radius));

    // Consistency: near_interface_points must return only points within radius.
    int n0 = nodes[0];
    auto pts = gp.near_interface_points(n0, radius);
    auto c = grid.coord(n0);
    for (int q : pts) {
        double dx = c[0] - iface.points()(q, 0);
        double dy = c[1] - iface.points()(q, 1);
        double dz = c[2] - iface.points()(q, 2);
        REQUIRE(std::sqrt(dx*dx + dy*dy + dz*dz) < radius + 1e-10);
    }
}

// ============================================================================
// Bounding-box setup: closest_bulk_node and narrow-band closest_interface_point
// verified against brute-force exhaustive search.
// ============================================================================

// Node grid on [-half, +half]^2 with n cells per side.
static CartesianGrid2D make_bbox_grid_2d(double half, int n) {
    double h = 2.0 * half / n;
    return CartesianGrid2D({-half, -half}, {h, h}, {n, n}, DofLayout2D::Node);
}

// Node grid on [-half, +half]^3 with n cells per side.
static CartesianGrid3D make_bbox_grid_3d(double half, int n) {
    double h = 2.0 * half / n;
    return CartesianGrid3D({-half, -half, -half}, {h, h, h}, {n, n, n}, DofLayout3D::Node);
}

// UV sphere: M latitude rings, N longitude segments.  Quadrature = one centroid
// per triangle.  Winding-order is not enforced (ray casting is winding-agnostic).
static Interface3D make_sphere_uv(double cx, double cy, double cz, double r, int M, int N) {
    int Nv = 2 + M * N;
    int Nt = 2 * N * M;

    Eigen::MatrixX3d V(Nv, 3);
    Eigen::MatrixX3i F(Nt, 3);

    V.row(0) << cx, cy, cz + r;       // north pole
    V.row(1) << cx, cy, cz - r;       // south pole

    for (int m = 0; m < M; ++m) {
        double phi   = kPi * (m + 1) / (M + 1);
        double z_ring = cz + r * std::cos(phi);
        double rxy   = r * std::sin(phi);
        for (int nn = 0; nn < N; ++nn) {
            double theta = 2.0 * kPi * nn / N;
            V.row(2 + m * N + nn) << cx + rxy * std::cos(theta),
                                     cy + rxy * std::sin(theta),
                                     z_ring;
        }
    }

    int fi = 0;
    for (int nn = 0; nn < N; ++nn)    // north cap
        F.row(fi++) << 0, 2 + nn, 2 + (nn + 1) % N;
    for (int m = 0; m < M - 1; ++m)  // body quads → 2 triangles
        for (int nn = 0; nn < N; ++nn) {
            int a = 2 + m * N + nn,       b = 2 + m * N + (nn + 1) % N;
            int c = 2 + (m+1) * N + nn,   d = 2 + (m+1) * N + (nn + 1) % N;
            F.row(fi++) << a, c, b;
            F.row(fi++) << b, c, d;
        }
    for (int nn = 0; nn < N; ++nn)    // south cap
        F.row(fi++) << 1, 2 + (M-1)*N + (nn+1)%N, 2 + (M-1)*N + nn;

    Eigen::MatrixX3d pts(Nt, 3), nml(Nt, 3);
    Eigen::VectorXd  wts(Nt);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(Nt);

    for (int t = 0; t < Nt; ++t) {
        for (int d = 0; d < 3; ++d)
            pts(t, d) = (V(F(t,0),d) + V(F(t,1),d) + V(F(t,2),d)) / 3.0;
        // outward normal: radial direction from center
        double nx = pts(t,0)-cx, ny = pts(t,1)-cy, nz = pts(t,2)-cz;
        double len = std::sqrt(nx*nx + ny*ny + nz*nz);
        nml.row(t) << nx/len, ny/len, nz/len;
        // triangle area
        Eigen::Vector3d ab = V.row(F(t,1)) - V.row(F(t,0));
        Eigen::Vector3d ac = V.row(F(t,2)) - V.row(F(t,0));
        wts(t) = 0.5 * ab.cross(ac).norm();
    }
    return {V, F, pts, nml, wts, 1, comp};
}

// Multi-sphere interface: arbitrary list of (center, radius), each its own component.
struct SphereCfg { double cx, cy, cz, r; };

static Interface3D make_multi_spheres(const std::vector<SphereCfg>& cfgs, int M, int N)
{
    int Nc     = static_cast<int>(cfgs.size());
    int Nt_one = 2 * N * M;                // triangles per sphere
    int Nv_one = 2 + M * N;               // vertices per sphere
    int Nt_tot = Nc * Nt_one;
    int Nv_tot = Nc * Nv_one;

    Eigen::MatrixX3d V(Nv_tot, 3);
    Eigen::MatrixX3i F(Nt_tot, 3);
    Eigen::MatrixX3d pts(Nt_tot, 3), nml(Nt_tot, 3);
    Eigen::VectorXd  wts(Nt_tot);
    Eigen::VectorXi  comp(Nt_tot);

    for (int s = 0; s < Nc; ++s) {
        double cx = cfgs[s].cx, cy = cfgs[s].cy, cz = cfgs[s].cz, r = cfgs[s].r;
        int    vOff = s * Nv_one;
        int    tOff = s * Nt_one;

        // vertices
        V.row(vOff + 0) << cx, cy, cz + r;
        V.row(vOff + 1) << cx, cy, cz - r;
        for (int m = 0; m < M; ++m) {
            double phi  = kPi * (m + 1) / (M + 1);
            double zr   = cz + r * std::cos(phi);
            double rxy  = r * std::sin(phi);
            for (int nn = 0; nn < N; ++nn) {
                double theta = 2.0 * kPi * nn / N;
                V.row(vOff + 2 + m * N + nn) << cx + rxy * std::cos(theta),
                                                 cy + rxy * std::sin(theta), zr;
            }
        }

        // triangles (indices local to this sphere, shifted by vOff)
        int fi = tOff;
        for (int nn = 0; nn < N; ++nn)
            F.row(fi++) << vOff + 0,
                           vOff + 2 + nn,
                           vOff + 2 + (nn + 1) % N;
        for (int m = 0; m < M - 1; ++m)
            for (int nn = 0; nn < N; ++nn) {
                int a = vOff + 2 + m*N + nn,       b = vOff + 2 + m*N + (nn+1)%N;
                int c = vOff + 2 + (m+1)*N + nn,   d = vOff + 2 + (m+1)*N + (nn+1)%N;
                F.row(fi++) << a, c, b;
                F.row(fi++) << b, c, d;
            }
        for (int nn = 0; nn < N; ++nn)
            F.row(fi++) << vOff + 1,
                           vOff + 2 + (M-1)*N + (nn+1)%N,
                           vOff + 2 + (M-1)*N + nn;

        // quadrature (centroids + radial normals + triangle areas)
        for (int t = 0; t < Nt_one; ++t) {
            int tt = tOff + t;
            for (int dd = 0; dd < 3; ++dd)
                pts(tt, dd) = (V(F(tt,0),dd) + V(F(tt,1),dd) + V(F(tt,2),dd)) / 3.0;
            double nx = pts(tt,0)-cx, ny = pts(tt,1)-cy, nz = pts(tt,2)-cz;
            double len = std::sqrt(nx*nx + ny*ny + nz*nz);
            nml.row(tt) << nx/len, ny/len, nz/len;
            Eigen::Vector3d ab = V.row(F(tt,1)) - V.row(F(tt,0));
            Eigen::Vector3d ac = V.row(F(tt,2)) - V.row(F(tt,0));
            wts(tt) = 0.5 * ab.cross(ac).norm();
            comp(tt) = s;
        }
    }
    return {V, F, pts, nml, wts, 1, comp};
}

// ─── Non-trivial surface factories ──────────────────────────────────────────

// 2D star polygon: r(θ) = R * (1 + A * cos(k*θ)).
// k tips, amplitude A (keep A < 1 so the shape is star-shaped w.r.t. center).
static Interface2D make_star(double cx, double cy, double R, double A, int k, int N)
{
    Eigen::MatrixX2d pts(N, 2), nml(N, 2);
    Eigen::VectorXd  wts(N);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(N);
    const double dth = 2.0 * kPi / N;
    for (int i = 0; i < N; ++i) {
        double th   = i * dth;
        double r    = R * (1.0 + A * std::cos(k * th));
        double drdt = -R * A * k * std::sin(k * th);
        pts(i, 0) = cx + r * std::cos(th);
        pts(i, 1) = cy + r * std::sin(th);
        // Tangent vector
        double tx = drdt * std::cos(th) - r * std::sin(th);
        double ty = drdt * std::sin(th) + r * std::cos(th);
        double tlen = std::sqrt(tx*tx + ty*ty);
        // Outward normal: 90° CW rotation of unit tangent
        nml(i, 0) =  ty / tlen;
        nml(i, 1) = -tx / tlen;
        wts(i) = tlen * dth;
    }
    return {pts, nml, wts, 1, comp};
}

// Exact inside test for the star (valid because r(θ) > 0 everywhere).
static bool star_contains(double cx, double cy, double R, double A, int k,
                           double x, double y)
{
    double rho = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
    double th  = std::atan2(y - cy, x - cx);
    return rho < R * (1.0 + A * std::cos(k * th));
}

// 3D ellipsoid: semi-axes a, b, c_ax.  UV grid same connectivity as UV sphere.
static Interface3D make_ellipsoid(double cx, double cy, double cz,
                                   double a, double b, double c_ax, int M, int N)
{
    int Nv = 2 + M * N, Nt = 2 * N * M;
    Eigen::MatrixX3d V(Nv, 3);
    Eigen::MatrixX3i F(Nt, 3);

    V.row(0) << cx, cy, cz + c_ax;
    V.row(1) << cx, cy, cz - c_ax;
    for (int m = 0; m < M; ++m) {
        double phi = kPi * (m + 1) / (M + 1);
        for (int nn = 0; nn < N; ++nn) {
            double theta = 2.0 * kPi * nn / N;
            V.row(2 + m*N + nn) << cx + a   * std::sin(phi) * std::cos(theta),
                                    cy + b   * std::sin(phi) * std::sin(theta),
                                    cz + c_ax * std::cos(phi);
        }
    }
    int fi = 0;
    for (int nn = 0; nn < N; ++nn)
        F.row(fi++) << 0, 2 + nn, 2 + (nn + 1) % N;
    for (int m = 0; m < M - 1; ++m)
        for (int nn = 0; nn < N; ++nn) {
            int aa = 2+m*N+nn,       bb = 2+m*N+(nn+1)%N;
            int cc = 2+(m+1)*N+nn,   dd = 2+(m+1)*N+(nn+1)%N;
            F.row(fi++) << aa, cc, bb;
            F.row(fi++) << bb, cc, dd;
        }
    for (int nn = 0; nn < N; ++nn)
        F.row(fi++) << 1, 2+(M-1)*N+(nn+1)%N, 2+(M-1)*N+nn;

    Eigen::MatrixX3d pts(Nt, 3), nml(Nt, 3);
    Eigen::VectorXd  wts(Nt);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(Nt);

    for (int t = 0; t < Nt; ++t) {
        for (int dd = 0; dd < 3; ++dd)
            pts(t, dd) = (V(F(t,0),dd) + V(F(t,1),dd) + V(F(t,2),dd)) / 3.0;
        // Outward normal on ellipsoid: (x/a², y/b², z/c²) normalized
        double nx = (pts(t,0)-cx) / (a*a);
        double ny = (pts(t,1)-cy) / (b*b);
        double nz = (pts(t,2)-cz) / (c_ax*c_ax);
        double nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
        nml.row(t) << nx/nlen, ny/nlen, nz/nlen;
        Eigen::Vector3d ab = V.row(F(t,1)) - V.row(F(t,0));
        Eigen::Vector3d ac = V.row(F(t,2)) - V.row(F(t,0));
        wts(t) = 0.5 * ab.cross(ac).norm();
    }
    return {V, F, pts, nml, wts, 1, comp};
}

// 3D torus: center (cx,cy,cz), major radius R_maj, tube radius R_min.
// N_tor toroidal segments × N_pol poloidal segments → 2*N_tor*N_pol triangles.
static Interface3D make_torus(double cx, double cy, double cz,
                               double R_maj, double R_min,
                               int N_tor, int N_pol)
{
    int Nv     = N_tor * N_pol;
    int Nt_tri = 2 * N_tor * N_pol;

    Eigen::MatrixX3d V(Nv, 3);
    Eigen::MatrixX3i F(Nt_tri, 3);

    auto vidx = [&](int i, int j) -> int {
        return ((i + N_tor) % N_tor) * N_pol + ((j + N_pol) % N_pol);
    };

    for (int i = 0; i < N_tor; ++i) {
        double tor = 2.0 * kPi * i / N_tor;
        double ct  = std::cos(tor), st = std::sin(tor);
        for (int j = 0; j < N_pol; ++j) {
            double pol  = 2.0 * kPi * j / N_pol;
            double r_lat = R_maj + R_min * std::cos(pol);
            V.row(vidx(i, j)) << cx + r_lat * ct,
                                  cy + r_lat * st,
                                  cz + R_min * std::sin(pol);
        }
    }

    int fi = 0;
    for (int i = 0; i < N_tor; ++i)
        for (int j = 0; j < N_pol; ++j) {
            int aa = vidx(i,   j),   bb = vidx(i+1, j);
            int cc = vidx(i,   j+1), dd = vidx(i+1, j+1);
            F.row(fi++) << aa, bb, dd;
            F.row(fi++) << aa, dd, cc;
        }

    Eigen::MatrixX3d pts(Nt_tri, 3), nml(Nt_tri, 3);
    Eigen::VectorXd  wts(Nt_tri);
    Eigen::VectorXi  comp = Eigen::VectorXi::Zero(Nt_tri);

    for (int t = 0; t < Nt_tri; ++t) {
        for (int dd = 0; dd < 3; ++dd)
            pts(t, dd) = (V(F(t,0),dd) + V(F(t,1),dd) + V(F(t,2),dd)) / 3.0;
        // Outward normal: point minus nearest point on major circle
        double px = pts(t,0)-cx, py = pts(t,1)-cy, pz = pts(t,2)-cz;
        double rho  = std::sqrt(px*px + py*py);
        double nx   = px * (1.0 - R_maj / rho);
        double ny   = py * (1.0 - R_maj / rho);
        double nz   = pz;
        double nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
        nml.row(t) << nx/nlen, ny/nlen, nz/nlen;
        Eigen::Vector3d ab = V.row(F(t,1)) - V.row(F(t,0));
        Eigen::Vector3d ac = V.row(F(t,2)) - V.row(F(t,0));
        wts(t) = 0.5 * ab.cross(ac).norm();
    }
    return {V, F, pts, nml, wts, 1, comp};
}

// ─── brute-force helpers ────────────────────────────────────────────────────

static double dist2d(double ax, double ay, double bx, double by) {
    return std::sqrt((ax-bx)*(ax-bx) + (ay-by)*(ay-by));
}
static double dist3d(double ax,double ay,double az,double bx,double by,double bz) {
    return std::sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by)+(az-bz)*(az-bz));
}

// True nearest grid node (exhaustive search).
static double bf_node_dist_2d(const CartesianGrid2D& g, double x, double y) {
    double best = std::numeric_limits<double>::infinity();
    for (int n = 0; n < g.num_dofs(); ++n) {
        auto c = g.coord(n);
        best = std::min(best, dist2d(c[0], c[1], x, y));
    }
    return best;
}
static double bf_node_dist_3d(const CartesianGrid3D& g, double x, double y, double z) {
    double best = std::numeric_limits<double>::infinity();
    for (int n = 0; n < g.num_dofs(); ++n) {
        auto c = g.coord(n);
        best = std::min(best, dist3d(c[0], c[1], c[2], x, y, z));
    }
    return best;
}

// True nearest interface point (exhaustive search).
static double bf_iface_dist_2d(const Interface2D& iface, double cx, double cy) {
    double best = std::numeric_limits<double>::infinity();
    for (int q = 0; q < iface.num_points(); ++q)
        best = std::min(best, dist2d(cx, cy, iface.points()(q,0), iface.points()(q,1)));
    return best;
}
static double bf_iface_dist_3d(const Interface3D& iface, double cx,double cy,double cz) {
    double best = std::numeric_limits<double>::infinity();
    for (int q = 0; q < iface.num_points(); ++q)
        best = std::min(best, dist3d(cx,cy,cz, iface.points()(q,0),iface.points()(q,1),iface.points()(q,2)));
    return best;
}

// ─── 2D tests ───────────────────────────────────────────────────────────────

// Circle of radius 0.25 at the origin; bounding box [-0.5, 0.5]^2, 32 cells.
TEST_CASE("GridPair2D bbox: closest_bulk_node matches brute force for all interface points",
          "[gridpair][2d][bbox]")
{
    auto  iface = make_circle(0.0, 0.0, 0.25, 64);
    auto  grid  = make_bbox_grid_2d(0.5, 32);
    double h    = grid.spacing()[0];
    GridPair2D gp(grid, iface);

    for (int q = 0; q < iface.num_points(); ++q) {
        double xq = iface.points()(q, 0), yq = iface.points()(q, 1);
        int    n  = gp.closest_bulk_node(q);
        auto   cn = grid.coord(n);

        double d_gp = dist2d(cn[0], cn[1], xq, yq);
        double d_bf = bf_node_dist_2d(grid, xq, yq);

        // The returned node achieves the true minimum distance.
        REQUIRE(d_gp == Catch::Approx(d_bf).margin(1e-12));
        // Sanity: within one diagonal grid step.
        REQUIRE(d_gp <= std::sqrt(2.0) * h + 1e-12);
    }
}

TEST_CASE("GridPair2D bbox: narrow-band 1-layer — closest_interface_point matches brute force",
          "[gridpair][2d][bbox][narrow_band]")
{
    auto  iface = make_circle(0.0, 0.0, 0.25, 64);
    auto  grid  = make_bbox_grid_2d(0.5, 32);
    double h    = grid.spacing()[0];
    GridPair2D gp(grid, iface);

    auto band = gp.near_interface_nodes(1.5 * h);
    REQUIRE(!band.empty());

    for (int n : band) {
        auto   cn  = grid.coord(n);
        int    q   = gp.closest_interface_point(n);
        double d_gp = dist2d(cn[0], cn[1], iface.points()(q,0), iface.points()(q,1));
        double d_bf = bf_iface_dist_2d(iface, cn[0], cn[1]);

        REQUIRE(d_gp == Catch::Approx(d_bf).margin(1e-12));
        // Every band node is within the declared radius.
        REQUIRE(d_bf < 1.5 * h + 1e-12);
    }

    // Rough size check: roughly 2 * (circumference / h) nodes straddle the circle.
    double circ = 2.0 * kPi * 0.25;
    REQUIRE(static_cast<int>(band.size()) > static_cast<int>(circ / h));
}

TEST_CASE("GridPair2D bbox: narrow-band 2-layer — closest_interface_point matches brute force",
          "[gridpair][2d][bbox][narrow_band]")
{
    auto  iface = make_circle(0.0, 0.0, 0.25, 64);
    auto  grid  = make_bbox_grid_2d(0.5, 32);
    double h    = grid.spacing()[0];
    GridPair2D gp(grid, iface);

    auto band1 = gp.near_interface_nodes(1.5 * h);
    auto band2 = gp.near_interface_nodes(2.5 * h);
    REQUIRE(band2.size() > band1.size());

    for (int n : band2) {
        auto   cn  = grid.coord(n);
        int    q   = gp.closest_interface_point(n);
        double d_gp = dist2d(cn[0], cn[1], iface.points()(q,0), iface.points()(q,1));
        double d_bf = bf_iface_dist_2d(iface, cn[0], cn[1]);

        REQUIRE(d_gp == Catch::Approx(d_bf).margin(1e-12));
        REQUIRE(d_bf < 2.5 * h + 1e-12);
    }
}

// ─── 3D tests ───────────────────────────────────────────────────────────────

// Sphere of radius 0.3 at origin; bounding box [-0.5, 0.5]^3, 16 cells.
// UV sphere M=4 latitude rings, N=8 longitude segments → 64 triangles.
TEST_CASE("GridPair3D bbox: closest_bulk_node matches brute force for all interface points",
          "[gridpair][3d][bbox]")
{
    auto  iface = make_sphere_uv(0.0, 0.0, 0.0, 0.3, 4, 8);
    auto  grid  = make_bbox_grid_3d(0.5, 16);
    double h    = grid.spacing()[0];
    GridPair3D gp(grid, iface);

    REQUIRE(iface.num_points() == 64);

    for (int q = 0; q < iface.num_points(); ++q) {
        double xq = iface.points()(q,0), yq = iface.points()(q,1), zq = iface.points()(q,2);
        int    n  = gp.closest_bulk_node(q);
        auto   cn = grid.coord(n);

        double d_gp = dist3d(cn[0],cn[1],cn[2], xq,yq,zq);
        double d_bf = bf_node_dist_3d(grid, xq,yq,zq);

        REQUIRE(d_gp == Catch::Approx(d_bf).margin(1e-12));
        REQUIRE(d_gp <= std::sqrt(3.0) * h + 1e-12);
    }
}

TEST_CASE("GridPair3D bbox: narrow-band 1-layer — closest_interface_point matches brute force",
          "[gridpair][3d][bbox][narrow_band]")
{
    auto  iface = make_sphere_uv(0.0, 0.0, 0.0, 0.3, 4, 8);
    auto  grid  = make_bbox_grid_3d(0.5, 16);
    double h    = grid.spacing()[0];
    GridPair3D gp(grid, iface);

    auto band = gp.near_interface_nodes(1.5 * h);
    REQUIRE(!band.empty());

    for (int n : band) {
        auto   cn  = grid.coord(n);
        int    q   = gp.closest_interface_point(n);
        double d_gp = dist3d(cn[0],cn[1],cn[2],
                             iface.points()(q,0),iface.points()(q,1),iface.points()(q,2));
        double d_bf = bf_iface_dist_3d(iface, cn[0],cn[1],cn[2]);

        REQUIRE(d_gp == Catch::Approx(d_bf).margin(1e-12));
        REQUIRE(d_bf < 1.5 * h + 1e-12);
    }
}

TEST_CASE("GridPair3D bbox: narrow-band 2-layer — closest_interface_point matches brute force",
          "[gridpair][3d][bbox][narrow_band]")
{
    auto  iface = make_sphere_uv(0.0, 0.0, 0.0, 0.3, 4, 8);
    auto  grid  = make_bbox_grid_3d(0.5, 16);
    double h    = grid.spacing()[0];
    GridPair3D gp(grid, iface);

    auto band1 = gp.near_interface_nodes(1.5 * h);
    auto band2 = gp.near_interface_nodes(2.5 * h);
    REQUIRE(band2.size() > band1.size());

    for (int n : band2) {
        auto   cn  = grid.coord(n);
        int    q   = gp.closest_interface_point(n);
        double d_gp = dist3d(cn[0],cn[1],cn[2],
                             iface.points()(q,0),iface.points()(q,1),iface.points()(q,2));
        double d_bf = bf_iface_dist_3d(iface, cn[0],cn[1],cn[2]);

        REQUIRE(d_gp == Catch::Approx(d_bf).margin(1e-12));
        REQUIRE(d_bf < 2.5 * h + 1e-12);
    }
}

// ============================================================================
// Domain labeling — multi-component 2D (exhaustive)
// ============================================================================

TEST_CASE("GridPair2D domain labeling — three disjoint circles", "[gridpair][2d][domain]")
{
    // Three circles packed inside [0,1]^2, well-separated.
    std::vector<CircleCfg> cfgs = {
        {0.20, 0.50, 0.12},   // component 0 → label 1
        {0.55, 0.75, 0.10},   // component 1 → label 2
        {0.70, 0.30, 0.10},   // component 2 → label 3
    };
    const double h = 1.0 / 48;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {48, 48}, DofLayout2D::Node);
    auto iface = make_multi_circles(cfgs, 128);
    REQUIRE(iface.num_components() == 3);

    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0];
    const double margin = 2.0 * h;

    // For each circle: nodes clearly inside → label = component + 1.
    // Nodes clearly outside all circles → label = 0.
    int n_wrong = 0;
    for (int jj = 0; jj < d[1]; ++jj) {
        for (int ii = 0; ii < nx; ++ii) {
            double x = ii * h, y = jj * h;
            int label = gp.domain_label(jj * nx + ii);

            int expected = -1;   // -1 = don't check (near boundary)
            bool outside_all = true;
            for (int c = 0; c < 3; ++c) {
                double dist = std::sqrt((x - cfgs[c].cx)*(x - cfgs[c].cx) +
                                        (y - cfgs[c].cy)*(y - cfgs[c].cy));
                if (dist < cfgs[c].r - margin) {
                    expected    = c + 1;
                    outside_all = false;
                    break;
                }
                if (dist < cfgs[c].r + margin)
                    outside_all = false;   // near boundary — skip
            }
            if (outside_all)  expected = 0;

            if (expected >= 0 && label != expected) n_wrong++;
        }
    }
    REQUIRE(n_wrong == 0);
}

TEST_CASE("GridPair2D domain labeling — exhaustive brute-force check on single circle",
          "[gridpair][2d][domain]")
{
    // Fine grid, large circle: every node clearly inside or outside
    const double h  = 1.0 / 64;
    const double r  = 0.30, cx = 0.50, cy = 0.50;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {64, 64}, DofLayout2D::Node);
    auto iface = make_circle(cx, cy, r, 512);
    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0];
    const double margin = 3.0 * h;
    int n_wrong = 0;

    for (int jj = 0; jj < d[1]; ++jj) {
        for (int ii = 0; ii < nx; ++ii) {
            double dist = std::sqrt((ii*h - cx)*(ii*h - cx) + (jj*h - cy)*(jj*h - cy));
            int label = gp.domain_label(jj * nx + ii);
            if (dist < r - margin && label != 1) n_wrong++;
            if (dist > r + margin && label != 0) n_wrong++;
        }
    }
    REQUIRE(n_wrong == 0);
}

// ============================================================================
// Domain labeling — multi-component 3D (UV spheres)
// ============================================================================

TEST_CASE("GridPair3D domain labeling — single UV sphere", "[gridpair][3d][domain]")
{
    const double R  = 0.28;
    auto  iface = make_sphere_uv(0.0, 0.0, 0.0, R, 6, 12);   // fine enough mesh
    auto  grid  = make_bbox_grid_3d(0.5, 20);
    double h    = grid.spacing()[0];
    GridPair3D gp(grid, iface);

    const double margin = 2.0 * h;
    int n_wrong = 0;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        auto   c    = grid.coord(n);
        double dist = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
        int    label = gp.domain_label(n);
        if (dist < R - margin && label != 1) n_wrong++;
        if (dist > R + margin && label != 0) n_wrong++;
    }
    REQUIRE(n_wrong == 0);
}

TEST_CASE("GridPair3D domain labeling — two disjoint UV spheres", "[gridpair][3d][domain]")
{
    // Two spheres along x-axis, well-separated.
    std::vector<SphereCfg> cfgs = {
        {-0.28, 0.0, 0.0, 0.16},   // component 0 → label 1
        { 0.28, 0.0, 0.0, 0.16},   // component 1 → label 2
    };
    auto  iface = make_multi_spheres(cfgs, 5, 10);
    auto  grid  = make_bbox_grid_3d(0.55, 22);
    double h    = grid.spacing()[0];
    GridPair3D gp(grid, iface);

    REQUIRE(iface.num_components() == 2);

    const double margin = 2.0 * h;
    int n_wrong = 0;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        auto   c     = grid.coord(n);
        int    label = gp.domain_label(n);

        bool outside_all = true;
        int  expected    = -1;   // -1 = near boundary, skip

        for (int s = 0; s < 2; ++s) {
            double dx   = c[0] - cfgs[s].cx;
            double dy   = c[1] - cfgs[s].cy;
            double dz   = c[2] - cfgs[s].cz;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < cfgs[s].r - margin) {
                expected    = s + 1;
                outside_all = false;
                break;
            }
            if (dist < cfgs[s].r + margin)
                outside_all = false;
        }
        if (outside_all) expected = 0;

        if (expected >= 0 && label != expected) n_wrong++;
    }
    REQUIRE(n_wrong == 0);
}

// ============================================================================
// Non-trivial surfaces — domain labeling
// ============================================================================

// ── 2D star polygon ──────────────────────────────────────────────────────────

TEST_CASE("GridPair2D domain labeling — star polygon (non-convex, 5 tips)",
          "[gridpair][2d][domain][nontrivial]")
{
    // r(θ) = 0.28 * (1 + 0.40 * cos(5θ))  centered at (0.5, 0.5)
    // r_min = 0.168 (valley), r_max = 0.392 (tip).
    const double cx = 0.5, cy = 0.5;
    const double R = 0.28, A = 0.40;
    const int    k = 5;
    const double h = 1.0 / 64;

    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {64, 64}, DofLayout2D::Node);
    auto iface = make_star(cx, cy, R, A, k, 512);
    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0];

    // Conservative check: nodes definitely inside the inscribed circle,
    // nodes definitely outside the circumscribed circle.
    const double r_inner = R * (1.0 - A);   // 0.168
    const double r_outer = R * (1.0 + A);   // 0.392
    const double margin  = 3.0 * h;         // ≈ 0.047

    int n_conservative_wrong = 0;
    for (int jj = 0; jj < d[1]; ++jj)
        for (int ii = 0; ii < nx; ++ii) {
            double dist  = std::sqrt((ii*h-cx)*(ii*h-cx) + (jj*h-cy)*(jj*h-cy));
            int    label = gp.domain_label(jj * nx + ii);
            if (dist < r_inner - margin && label != 1) n_conservative_wrong++;
            if (dist > r_outer + margin && label != 0) n_conservative_wrong++;
        }
    REQUIRE(n_conservative_wrong == 0);

    // Exact check via star_contains for nodes not within 3h of the boundary.
    // Boundary proximity approximated by |ρ - r(θ_node)|.
    int n_exact_wrong = 0;
    for (int jj = 0; jj < d[1]; ++jj)
        for (int ii = 0; ii < nx; ++ii) {
            double x  = ii * h, y = jj * h;
            double rho   = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
            double th_nd = std::atan2(y-cy, x-cx);
            double r_bnd = R * (1.0 + A * std::cos(k * th_nd));
            if (std::fabs(rho - r_bnd) < margin) continue;   // near boundary — skip

            bool   inside = star_contains(cx, cy, R, A, k, x, y);
            int    label  = gp.domain_label(jj * nx + ii);
            if (inside  && label != 1) n_exact_wrong++;
            if (!inside && label != 0) n_exact_wrong++;
        }
    REQUIRE(n_exact_wrong == 0);
}

// ── 3D ellipsoid ─────────────────────────────────────────────────────────────

TEST_CASE("GridPair3D domain labeling — ellipsoid (a ≠ b ≠ c)",
          "[gridpair][3d][domain][nontrivial]")
{
    // Semi-axes: a=0.35, b=0.25, c=0.20.  Minimum axis = 0.20.
    const double a = 0.35, b = 0.25, c_ax = 0.20;
    auto  iface = make_ellipsoid(0.0, 0.0, 0.0, a, b, c_ax, 10, 16);
    auto  grid  = make_bbox_grid_3d(0.5, 20);
    double h    = grid.spacing()[0];
    GridPair3D gp(grid, iface);

    // Use the ellipsoid metric: val(P) = sqrt((x/a)²+(y/b)²+(z/c)²).
    // A node with val < 1 is inside.  Margin expressed in ellipsoid-metric units:
    //   delta = margin / min_axis  (worst-case stretch factor of the metric).
    const double margin = 1.5 * h;
    const double delta  = margin / c_ax;   // = 0.075/0.20 = 0.375

    int n_wrong = 0;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        auto   co  = grid.coord(n);
        double val = std::sqrt((co[0]/a)*(co[0]/a) +
                               (co[1]/b)*(co[1]/b) +
                               (co[2]/c_ax)*(co[2]/c_ax));
        int label = gp.domain_label(n);
        if (val < 1.0 - delta && label != 1) n_wrong++;
        if (val > 1.0 + delta && label != 0) n_wrong++;
    }
    REQUIRE(n_wrong == 0);

    // Explicit spot checks.
    // Center (0,0,0): val=0 → inside.
    {
        int n0 = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int n = 0; n < grid.num_dofs(); ++n) {
            auto co = grid.coord(n);
            double d = co[0]*co[0] + co[1]*co[1] + co[2]*co[2];
            if (d < best) { best = d; n0 = n; }
        }
        REQUIRE(gp.domain_label(n0) == 1);
    }
    // Corner (±0.5,±0.5,±0.5): val ≫ 1 → outside.
    {
        auto d = grid.dof_dims();
        int n_corner = (d[1]-1) * d[0] + (d[0]-1);  // (nx-1, 0, 0) edge case, use explicit:
        // find node nearest (0.45, 0.45, 0.45)
        int n0 = 0; double best = std::numeric_limits<double>::infinity();
        for (int n = 0; n < grid.num_dofs(); ++n) {
            auto co = grid.coord(n);
            double dd = (co[0]-0.45)*(co[0]-0.45)+(co[1]-0.45)*(co[1]-0.45)+(co[2]-0.45)*(co[2]-0.45);
            if (dd < best) { best = dd; n0 = n; }
        }
        REQUIRE(gp.domain_label(n0) == 0);
    }
}

// ── 3D torus ─────────────────────────────────────────────────────────────────

TEST_CASE("GridPair3D domain labeling — torus (genus-1, hole is exterior)",
          "[gridpair][3d][domain][nontrivial]")
{
    // Torus in the xy-plane: major radius R=0.30, tube radius r=0.14.
    // The "hole" near the origin is exterior; the tube interior is label 1.
    const double R_maj = 0.30, R_min = 0.14;
    auto  iface = make_torus(0.0, 0.0, 0.0, R_maj, R_min, 32, 16);
    auto  grid  = make_bbox_grid_3d(0.5, 20);
    double h    = grid.spacing()[0];
    GridPair3D gp(grid, iface);

    // Distance from a grid node to the torus surface (tube wall):
    //   d_tube(P) = sqrt((sqrt(x²+y²) - R_maj)² + z²)
    // Inside the tube ↔ d_tube < R_min.
    const double margin = 1.5 * h;   // ≈ 0.075

    int n_wrong = 0;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        auto   co     = grid.coord(n);
        double rho    = std::sqrt(co[0]*co[0] + co[1]*co[1]);
        double d_tube = std::sqrt((rho-R_maj)*(rho-R_maj) + co[2]*co[2]);
        int    label  = gp.domain_label(n);

        if (d_tube < R_min - margin && label != 1) n_wrong++;
        if (d_tube > R_min + margin && label != 0) n_wrong++;
    }
    REQUIRE(n_wrong == 0);

    // Key topological check: origin is in the torus hole → exterior (label 0).
    {
        int n0 = 0; double best = std::numeric_limits<double>::infinity();
        for (int n = 0; n < grid.num_dofs(); ++n) {
            auto co = grid.coord(n);
            double d = co[0]*co[0] + co[1]*co[1] + co[2]*co[2];
            if (d < best) { best = d; n0 = n; }
        }
        REQUIRE(gp.domain_label(n0) == 0);
    }

    // Point at (R_maj, 0, 0) is the center of the tube at angle 0 → inside.
    {
        int n0 = 0; double best = std::numeric_limits<double>::infinity();
        for (int n = 0; n < grid.num_dofs(); ++n) {
            auto co = grid.coord(n);
            double d = (co[0]-R_maj)*(co[0]-R_maj) + co[1]*co[1] + co[2]*co[2];
            if (d < best) { best = d; n0 = n; }
        }
        REQUIRE(gp.domain_label(n0) == 1);
    }

    // Count interior nodes: should be comparable to the tube volume / h³.
    // Tube volume = 2π² * R_maj * R_min² ≈ 2*9.87*0.30*0.0196 ≈ 0.116
    // Expected interior nodes ≈ 0.116 / h³ = 0.116 / 1.25e-4 ≈ 928
    int n_interior = 0;
    for (int n = 0; n < grid.num_dofs(); ++n)
        if (gp.domain_label(n) == 1) n_interior++;
    REQUIRE(n_interior > 200);   // loose lower bound
    REQUIRE(n_interior < 4000);  // loose upper bound
}

// ============================================================================
// BFS-specific tests — verify the narrow-band + flood-fill algorithm directly
// ============================================================================

// Helper: build a flat set of "in-band" node indices for quick lookup.
static std::unordered_set<int> band_set(const GridPair2D& gp,
                                        const CartesianGrid2D& grid, double radius) {
    auto nodes = gp.near_interface_nodes(radius);
    return {nodes.begin(), nodes.end()};
}

static std::unordered_set<int> band_set_3d(const GridPair3D& gp,
                                           const CartesianGrid3D& grid, double radius) {
    auto nodes = gp.near_interface_nodes(radius);
    return {nodes.begin(), nodes.end()};
}

// ─── 2D: no label boundary outside the narrow band ───────────────────────────
//
// BFS correctness property: once the band is labeled by the geometry test, BFS
// flood-fills from it — so NO label boundary can exist between two nodes that
// are both outside the band.  Any adjacent non-band pair must share the same label.
TEST_CASE("GridPair2D BFS: no label boundary outside narrow band — circle",
          "[gridpair][2d][bfs]")
{
    const double h  = 1.0 / 32;
    const double r  = 0.30, cx = 0.50, cy = 0.50;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {32, 32}, DofLayout2D::Node);
    auto iface = make_circle(cx, cy, r, 512);
    GridPair2D gp(grid, iface);

    // Use 4h as "definitely outside band": the internal band is ~2h for this mesh.
    const double threshold = 4.0 * h;
    auto in_band = band_set(gp, grid, threshold);

    int violations = 0;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        if (in_band.count(n)) continue;
        for (int nb : grid.neighbors(n)) {
            if (nb < 0 || in_band.count(nb)) continue;
            if (gp.domain_label(n) != gp.domain_label(nb)) ++violations;
        }
    }
    REQUIRE(violations == 0);
}

// ─── 2D: far-field BFS accuracy on large grid with small interface ────────────
//
// The grid is much larger than the interface, so the overwhelming majority of
// labels are assigned by BFS (not the geometry test).  Verify they match the
// analytical ground truth.
TEST_CASE("GridPair2D BFS: far-field accuracy — small circle on large grid",
          "[gridpair][2d][bfs]")
{
    // [-1, 1]^2 with h=0.05; circle of radius 0.15 centred at origin.
    // Band ~ 2h = 0.10 wide; ~97% of the 41×41 nodes are BFS-labeled.
    const int    n_cells = 40;
    const double half    = 1.0;
    const double h       = 2.0 * half / n_cells;
    CartesianGrid2D grid({-half, -half}, {h, h}, {n_cells, n_cells}, DofLayout2D::Node);

    const double r = 0.15;
    auto iface = make_circle(0.0, 0.0, r, 256);
    GridPair2D gp(grid, iface);

    const double margin = 3.0 * h;
    int n_wrong = 0;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        auto c = grid.coord(n);
        double dist = std::sqrt(c[0]*c[0] + c[1]*c[1]);
        int label = gp.domain_label(n);
        if (dist < r - margin && label != 1) ++n_wrong;
        if (dist > r + margin && label != 0) ++n_wrong;
    }
    REQUIRE(n_wrong == 0);
}

// ─── 2D: BFS propagates correct label deep into interior of non-convex shape ─
//
// For the star polygon, the narrow band covers only the non-convex tips and
// the concave valleys.  All nodes far from the boundary rely on BFS.
// Verify labels match the analytical star_contains test everywhere.
TEST_CASE("GridPair2D BFS: far-field accuracy — star polygon (non-convex)",
          "[gridpair][2d][bfs]")
{
    const double h  = 1.0 / 64;
    const double cx = 0.5, cy = 0.5, R = 0.28, A = 0.40;
    const int    k  = 5;
    CartesianGrid2D grid({0.0, 0.0}, {h, h}, {64, 64}, DofLayout2D::Node);
    auto iface = make_star(cx, cy, R, A, k, 512);
    GridPair2D gp(grid, iface);

    auto d  = grid.dof_dims();
    int  nx = d[0];
    const double margin = 3.0 * h;
    int n_wrong = 0;

    for (int jj = 0; jj < d[1]; ++jj) {
        for (int ii = 0; ii < nx; ++ii) {
            double x = ii * h, y = jj * h;
            double rho = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
            double th  = std::atan2(y-cy, x-cx);
            double r_bnd = R * (1.0 + A * std::cos(k * th));
            // Nodes clearly inside or outside: dist to boundary > margin
            if (std::fabs(rho - r_bnd) > margin) {
                bool expected_inside = star_contains(cx, cy, R, A, k, x, y);
                int label = gp.domain_label(jj * nx + ii);
                if (expected_inside && label != 1) ++n_wrong;
                if (!expected_inside && label != 0) ++n_wrong;
            }
        }
    }
    REQUIRE(n_wrong == 0);
}

// ─── 3D: no label boundary outside the narrow band — UV sphere ───────────────
TEST_CASE("GridPair3D BFS: no label boundary outside narrow band — sphere",
          "[gridpair][3d][bfs]")
{
    const double R = 0.25;
    auto  iface = make_sphere_uv(0.0, 0.0, 0.0, R, 10, 20);
    auto  grid  = make_bbox_grid_3d(0.5, 20);
    double h    = grid.spacing()[0];
    GridPair3D gp(grid, iface);

    const double threshold = 4.0 * h;
    auto in_band = band_set_3d(gp, grid, threshold);

    int violations = 0;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        if (in_band.count(n)) continue;
        for (int nb : grid.neighbors(n)) {
            if (nb < 0 || in_band.count(nb)) continue;
            if (gp.domain_label(n) != gp.domain_label(nb)) ++violations;
        }
    }
    REQUIRE(violations == 0);
}

// ─── 3D: exhaustive far-field check — every non-band node matches the sphere ──
//
// Fine UV sphere mesh; every node clearly inside or outside is checked against
// the analytical condition r < R (interior) / r > R (exterior).
TEST_CASE("GridPair3D BFS: exhaustive far-field check — UV sphere",
          "[gridpair][3d][bfs]")
{
    const double R = 0.28;
    auto iface = make_sphere_uv(0.0, 0.0, 0.0, R, 12, 24);
    auto grid  = make_bbox_grid_3d(0.5, 24);
    double h   = grid.spacing()[0];
    GridPair3D gp(grid, iface);

    const double margin = 3.0 * h;
    int n_wrong = 0;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        auto   c    = grid.coord(n);
        double dist = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
        int    label = gp.domain_label(n);
        if (dist < R - margin && label != 1) ++n_wrong;
        if (dist > R + margin && label != 0) ++n_wrong;
    }
    REQUIRE(n_wrong == 0);
}

// ─── 3D: far-field BFS accuracy — small sphere on a large grid ───────────────
//
// Grid much larger than sphere: >90% of nodes labeled by BFS, not ray casting.
TEST_CASE("GridPair3D BFS: far-field accuracy — small sphere on large grid",
          "[gridpair][3d][bfs]")
{
    // [-2, 2]^3, h=0.2, sphere r=0.3 at origin.
    // Band ~ 2h = 0.4; sphere surface-to-corner distance >> band.
    const int    n_cells = 20;
    const double half    = 2.0;
    const double h       = 2.0 * half / n_cells;
    CartesianGrid3D grid({-half,-half,-half}, {h,h,h}, {n_cells,n_cells,n_cells},
                         DofLayout3D::Node);

    const double R = 0.30;
    auto iface = make_sphere_uv(0.0, 0.0, 0.0, R, 8, 16);
    GridPair3D gp(grid, iface);

    const double margin = 3.0 * h;
    int n_wrong = 0;
    for (int n = 0; n < grid.num_dofs(); ++n) {
        auto   c    = grid.coord(n);
        double dist = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
        int    label = gp.domain_label(n);
        if (dist < R - margin && label != 1) ++n_wrong;
        if (dist > R + margin && label != 0) ++n_wrong;
    }
    REQUIRE(n_wrong == 0);
}
