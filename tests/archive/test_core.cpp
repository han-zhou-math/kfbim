#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>

#include "src/local_cauchy/jump_data.hpp"
#include "src/local_cauchy/local_poly.hpp"
#include "src/grid/dof_layout.hpp"

using namespace kfbim;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Build sanity: Eigen is linked and usable
// ---------------------------------------------------------------------------
TEST_CASE("Eigen link sanity", "[build]") {
    Eigen::Vector2d v(3.0, 4.0);
    REQUIRE_THAT(v.norm(), WithinAbs(5.0, 1e-14));

    Eigen::Vector3d w(1.0, 2.0, 2.0);
    REQUIRE_THAT(w.norm(), WithinAbs(3.0, 1e-14));
}

// ---------------------------------------------------------------------------
// num_monomials: C(max_degree + dim, dim)
// ---------------------------------------------------------------------------
TEST_CASE("num_monomials — 2D", "[jump_data]") {
    // degree 0: {1}
    CHECK(num_monomials(0, 2) == 1);
    // degree 1: {1, x, y}
    CHECK(num_monomials(1, 2) == 3);
    // degree 2: {1, x, y, x², xy, y²}
    CHECK(num_monomials(2, 2) == 6);
    // degree 3: 10 terms
    CHECK(num_monomials(3, 2) == 10);
    // degree 4: 15 terms  (used by 4th-order method)
    CHECK(num_monomials(4, 2) == 15);
}

TEST_CASE("num_monomials — 3D", "[jump_data]") {
    CHECK(num_monomials(0, 3) == 1);
    CHECK(num_monomials(1, 3) == 4);
    // degree 2: {1, x,y,z, x²,xy,xz,y²,yz,z²}
    CHECK(num_monomials(2, 3) == 10);
    CHECK(num_monomials(3, 3) == 20);
    // 4th order 3D: 35 local Cauchy polynomial terms
    CHECK(num_monomials(4, 3) == 35);
}

// ---------------------------------------------------------------------------
// poly_max_degree / rhs_max_degree
// ---------------------------------------------------------------------------
TEST_CASE("poly_max_degree and rhs_max_degree", "[jump_data]") {
    // 2nd-order method
    CHECK(poly_max_degree(2) == 2);
    CHECK(rhs_max_degree(2) == 0);  // only [f] itself
    CHECK(num_monomials(rhs_max_degree(2), 2) == 1);
    CHECK(num_monomials(rhs_max_degree(2), 3) == 1);

    // 4th-order method
    CHECK(poly_max_degree(4) == 4);
    CHECK(rhs_max_degree(4) == 2);
    // [f, f_x, f_y, f_xx, f_xy, f_yy] — 6 entries in 2D
    CHECK(num_monomials(rhs_max_degree(4), 2) == 6);
    // [f, f_x, f_y, f_z, f_xx, f_xy, f_xz, f_yy, f_yz, f_zz] — 10 entries in 3D
    CHECK(num_monomials(rhs_max_degree(4), 3) == 10);
}

// ---------------------------------------------------------------------------
// LocalPoly coefficient counts match num_monomials
// ---------------------------------------------------------------------------
TEST_CASE("LocalPoly2D coefficient sizing", "[local_poly]") {
    int order = 4;
    int n_poly = num_monomials(poly_max_degree(order), 2);   // 15
    int n_rhs  = num_monomials(rhs_max_degree(order), 2);    // 6

    LocalPoly2D poly;
    poly.center = Eigen::Vector2d::Zero();
    poly.coeffs = Eigen::VectorXd::Zero(n_poly);

    CHECK(poly.coeffs.size() == 15);

    LaplaceJumpData2D jump;
    jump.u_jump     = 0.0;
    jump.un_jump    = 1.0;
    jump.rhs_derivs = Eigen::VectorXd::Zero(n_rhs);

    CHECK(jump.rhs_derivs.size() == 6);
}

TEST_CASE("StokesLocalPoly2D coefficient sizing", "[local_poly]") {
    int order   = 4;
    int n_vel   = num_monomials(poly_max_degree(order),     2);  // 15
    int n_press = num_monomials(poly_max_degree(order) - 1, 2);  // 10

    StokesLocalPoly2D poly;
    poly.center     = Eigen::Vector2d::Zero();
    poly.vel_coeffs  = Eigen::MatrixX2d::Zero(n_vel,   2);
    poly.pres_coeffs = Eigen::VectorXd::Zero(n_press);

    CHECK(poly.vel_coeffs.rows()  == 15);
    CHECK(poly.vel_coeffs.cols()  == 2);
    CHECK(poly.pres_coeffs.size() == 10);
}

// ---------------------------------------------------------------------------
// DofLayout enums compile and are distinct
// ---------------------------------------------------------------------------
TEST_CASE("DofLayout2D enum values are distinct", "[grid]") {
    CHECK(DofLayout2D::CellCenter != DofLayout2D::FaceX);
    CHECK(DofLayout2D::FaceX      != DofLayout2D::FaceY);
    CHECK(DofLayout2D::FaceY      != DofLayout2D::Node);
}

TEST_CASE("DofLayout3D enum values are distinct", "[grid]") {
    CHECK(DofLayout3D::CellCenter != DofLayout3D::FaceX);
    CHECK(DofLayout3D::FaceX      != DofLayout3D::FaceY);
    CHECK(DofLayout3D::FaceY      != DofLayout3D::FaceZ);
    CHECK(DofLayout3D::FaceZ      != DofLayout3D::Node);
}
