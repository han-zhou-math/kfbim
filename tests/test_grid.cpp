#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cmath>

#include "core/grid/cartesian_grid_2d.hpp"
#include "core/grid/cartesian_grid_3d.hpp"
#include "core/grid/mac_grid_2d.hpp"
#include "core/grid/mac_grid_3d.hpp"

using namespace kfbim;
using Catch::Matchers::WithinAbs;

// ============================================================
// CartesianGrid2D
// ============================================================

TEST_CASE("CartesianGrid2D CellCenter — DOF count and dims", "[grid2d]") {
    // 4 cells in x, 3 cells in y → 12 DOFs
    CartesianGrid2D g({0.0, 0.0}, {1.0, 1.0}, {4, 3}, DofLayout2D::CellCenter);
    CHECK(g.num_dofs() == 12);
    CHECK(g.dof_dims()[0] == 4);
    CHECK(g.dof_dims()[1] == 3);
}

TEST_CASE("CartesianGrid2D CellCenter — structured coord", "[grid2d]") {
    CartesianGrid2D g({0.0, 0.0}, {1.0, 1.0}, {4, 3}, DofLayout2D::CellCenter);
    // Cell center (i=0, j=0) → (0.5, 0.5)
    auto c00 = g.coord(0, 0);
    REQUIRE_THAT(c00[0], WithinAbs(0.5, 1e-14));
    REQUIRE_THAT(c00[1], WithinAbs(0.5, 1e-14));
    // Cell center (i=3, j=2) → (3.5, 2.5)
    auto c32 = g.coord(3, 2);
    REQUIRE_THAT(c32[0], WithinAbs(3.5, 1e-14));
    REQUIRE_THAT(c32[1], WithinAbs(2.5, 1e-14));
}

TEST_CASE("CartesianGrid2D CellCenter — index and flat coord round-trip", "[grid2d]") {
    CartesianGrid2D g({1.0, 2.0}, {0.5, 0.5}, {4, 3}, DofLayout2D::CellCenter);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 4; ++i) {
            int   idx = g.index(i, j);
            auto  c1  = g.coord(i, j);
            auto  c2  = g.coord(idx);
            REQUIRE_THAT(c2[0], WithinAbs(c1[0], 1e-14));
            REQUIRE_THAT(c2[1], WithinAbs(c1[1], 1e-14));
        }
    }
}

TEST_CASE("CartesianGrid2D CellCenter — neighbors", "[grid2d]") {
    CartesianGrid2D g({0.0, 0.0}, {1.0, 1.0}, {4, 3}, DofLayout2D::CellCenter);
    // Interior node idx=5 → (i=1, j=1): all four neighbors exist
    int idx = g.index(1, 1);  // = 1*4 + 1 = 5
    auto nb = g.neighbors(idx);
    CHECK(nb[0] == g.index(0, 1));  // −x
    CHECK(nb[1] == g.index(2, 1));  // +x
    CHECK(nb[2] == g.index(1, 0));  // −y
    CHECK(nb[3] == g.index(1, 2));  // +y

    // Corner (i=0, j=0): two boundary neighbors
    auto nb00 = g.neighbors(g.index(0, 0));
    CHECK(nb00[0] == -1);  // −x boundary
    CHECK(nb00[2] == -1);  // −y boundary
    CHECK(nb00[1] == g.index(1, 0));
    CHECK(nb00[3] == g.index(0, 1));

    // Corner (i=3, j=2): two boundary neighbors
    auto nb32 = g.neighbors(g.index(3, 2));
    CHECK(nb32[1] == -1);  // +x boundary
    CHECK(nb32[3] == -1);  // +y boundary
}

TEST_CASE("CartesianGrid2D FaceX — DOF count and dims", "[grid2d]") {
    // 4x3 cells → FaceX has (4+1)x3 = 15 DOFs
    CartesianGrid2D g({0.0, 0.0}, {1.0, 1.0}, {4, 3}, DofLayout2D::FaceX);
    CHECK(g.num_dofs() == 15);
    CHECK(g.dof_dims()[0] == 5);
    CHECK(g.dof_dims()[1] == 3);
    // FaceX node (i=0, j=0): x-face at left edge of cell (0,0)
    auto c = g.coord(0, 0);
    REQUIRE_THAT(c[0], WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(c[1], WithinAbs(0.5, 1e-14));
}

TEST_CASE("CartesianGrid2D FaceY — DOF count and dims", "[grid2d]") {
    // 4x3 cells → FaceY has 4x(3+1) = 16 DOFs
    CartesianGrid2D g({0.0, 0.0}, {1.0, 1.0}, {4, 3}, DofLayout2D::FaceY);
    CHECK(g.num_dofs() == 16);
    CHECK(g.dof_dims()[0] == 4);
    CHECK(g.dof_dims()[1] == 4);
    // FaceY node (i=0, j=0): y-face at bottom edge of cell (0,0)
    auto c = g.coord(0, 0);
    REQUIRE_THAT(c[0], WithinAbs(0.5, 1e-14));
    REQUIRE_THAT(c[1], WithinAbs(0.0, 1e-14));
}

TEST_CASE("CartesianGrid2D Node — DOF count", "[grid2d]") {
    CartesianGrid2D g({0.0, 0.0}, {1.0, 1.0}, {4, 3}, DofLayout2D::Node);
    CHECK(g.num_dofs() == 20);   // (4+1)*(3+1)
    auto c = g.coord(0, 0);
    REQUIRE_THAT(c[0], WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(c[1], WithinAbs(0.0, 1e-14));
}

// ============================================================
// CartesianGrid3D
// ============================================================

TEST_CASE("CartesianGrid3D CellCenter — DOF count and dims", "[grid3d]") {
    CartesianGrid3D g({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {4, 3, 2},
                      DofLayout3D::CellCenter);
    CHECK(g.num_dofs() == 24);   // 4*3*2
    CHECK(g.dof_dims()[0] == 4);
    CHECK(g.dof_dims()[1] == 3);
    CHECK(g.dof_dims()[2] == 2);
}

TEST_CASE("CartesianGrid3D CellCenter — index / coord round-trip", "[grid3d]") {
    CartesianGrid3D g({0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {4, 3, 2},
                      DofLayout3D::CellCenter);
    for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 3; ++j)
    for (int i = 0; i < 4; ++i) {
        int idx = g.index(i, j, k);
        auto c1 = g.coord(i, j, k);
        auto c2 = g.coord(idx);
        REQUIRE_THAT(c2[0], WithinAbs(c1[0], 1e-14));
        REQUIRE_THAT(c2[1], WithinAbs(c1[1], 1e-14));
        REQUIRE_THAT(c2[2], WithinAbs(c1[2], 1e-14));
    }
}

TEST_CASE("CartesianGrid3D CellCenter — neighbors", "[grid3d]") {
    CartesianGrid3D g({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {4, 3, 2},
                      DofLayout3D::CellCenter);
    // Interior node (1,1,0): only z− is boundary
    int idx = g.index(1, 1, 0);
    auto nb = g.neighbors(idx);
    CHECK(nb[0] == g.index(0, 1, 0));  // −x
    CHECK(nb[1] == g.index(2, 1, 0));  // +x
    CHECK(nb[2] == g.index(1, 0, 0));  // −y
    CHECK(nb[3] == g.index(1, 2, 0));  // +y
    CHECK(nb[4] == -1);                // −z boundary
    CHECK(nb[5] == g.index(1, 1, 1));  // +z

    // Corner (0,0,0): three boundary faces
    auto nb000 = g.neighbors(g.index(0, 0, 0));
    CHECK(nb000[0] == -1);  // −x
    CHECK(nb000[2] == -1);  // −y
    CHECK(nb000[4] == -1);  // −z
}

TEST_CASE("CartesianGrid3D FaceX/Y/Z — DOF counts", "[grid3d]") {
    CartesianGrid3D gx({0,0,0}, {1,1,1}, {4,3,2}, DofLayout3D::FaceX);
    CHECK(gx.num_dofs() == 5*3*2);   // (nx+1)*ny*nz
    CartesianGrid3D gy({0,0,0}, {1,1,1}, {4,3,2}, DofLayout3D::FaceY);
    CHECK(gy.num_dofs() == 4*4*2);   // nx*(ny+1)*nz
    CartesianGrid3D gz({0,0,0}, {1,1,1}, {4,3,2}, DofLayout3D::FaceZ);
    CHECK(gz.num_dofs() == 4*3*3);   // nx*ny*(nz+1)
}

// ============================================================
// MACGrid2D / MACGrid3D — sub-grid consistency
// ============================================================

TEST_CASE("MACGrid2D sub-grid DOF counts", "[mac2d]") {
    MACGrid2D mac({0.0, 0.0}, {0.1, 0.1}, {8, 6});
    CHECK(mac.pressure_grid().num_dofs()   == 8 * 6);      // 48
    CHECK(mac.velocity_grid_x().num_dofs() == 9 * 6);      // 54
    CHECK(mac.velocity_grid_y().num_dofs() == 8 * 7);      // 56
}

TEST_CASE("MACGrid2D sub-grid layouts", "[mac2d]") {
    MACGrid2D mac({0.0, 0.0}, {1.0, 1.0}, {4, 3});
    CHECK(mac.pressure_grid().layout()   == DofLayout2D::CellCenter);
    CHECK(mac.velocity_grid_x().layout() == DofLayout2D::FaceX);
    CHECK(mac.velocity_grid_y().layout() == DofLayout2D::FaceY);
}

TEST_CASE("MACGrid3D sub-grid DOF counts", "[mac3d]") {
    MACGrid3D mac({0,0,0}, {1,1,1}, {4,3,2});
    CHECK(mac.pressure_grid().num_dofs()   == 4*3*2);
    CHECK(mac.velocity_grid_x().num_dofs() == 5*3*2);
    CHECK(mac.velocity_grid_y().num_dofs() == 4*4*2);
    CHECK(mac.velocity_grid_z().num_dofs() == 4*3*3);
}

TEST_CASE("MACGrid2D pressure and velocity grids share spacing/origin", "[mac2d]") {
    std::array<double, 2> origin  = {1.0, 2.0};
    std::array<double, 2> spacing = {0.25, 0.25};
    MACGrid2D mac(origin, spacing, {8, 6});
    auto& pg  = mac.pressure_grid();
    auto& ugx = mac.velocity_grid_x();
    REQUIRE_THAT(pg.origin()[0],   WithinAbs(ugx.origin()[0],   1e-14));
    REQUIRE_THAT(pg.spacing()[0],  WithinAbs(ugx.spacing()[0],  1e-14));
    // FaceX velocity is at x-face centers: x position at i=0 should be origin[0]
    auto c = ugx.coord(0, 0);
    REQUIRE_THAT(c[0], WithinAbs(origin[0], 1e-14));
    REQUIRE_THAT(c[1], WithinAbs(origin[1] + 0.5 * spacing[1], 1e-14));
}
