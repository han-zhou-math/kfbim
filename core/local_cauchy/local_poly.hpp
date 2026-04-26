#pragma once

#include <Eigen/Dense>
#include "jump_data.hpp"  // num_monomials, poly_max_degree

namespace kfbim {

// ---------------------------------------------------------------------------
// Local correction polynomial — value type produced by ILocalCauchySolver::fit()
// and consumed by ILocalCauchySolver::evaluate().
//
// Monomial basis ordering (same convention as rhs_derivs in jump_data.hpp):
// grouped by total degree, then lexicographic within each group.
//
// 2D: 1 | x,y | x²,xy,y² | x³,x²y,xy²,y³ | ...
// 3D: 1 | x,y,z | x²,xy,xz,y²,yz,z² | ...
//
// coeffs length = num_monomials(poly_max_degree(method_order), dim)
// ---------------------------------------------------------------------------

struct LocalPoly2D {
    Eigen::Vector2d center;  // expansion origin = interface quadrature point
    Eigen::VectorXd coeffs;  // length = num_monomials(poly_max_degree(order), 2)
};

struct LocalPoly3D {
    Eigen::Vector3d center;
    Eigen::VectorXd coeffs;  // length = num_monomials(poly_max_degree(order), 3)
};

// ---------------------------------------------------------------------------
// Stokes: velocity and pressure have separate polynomial degrees.
// velocity: poly_max_degree(order)
// pressure: poly_max_degree(order) - 1   (one degree lower for inf-sup stability)
// ---------------------------------------------------------------------------

struct StokesLocalPoly2D {
    Eigen::Vector2d  center;
    Eigen::MatrixX2d vel_coeffs;   // num_monomials(poly_max_degree(order),   2) x 2
    Eigen::VectorXd  pres_coeffs;  // num_monomials(poly_max_degree(order)-1, 2)
};

struct StokesLocalPoly3D {
    Eigen::Vector3d  center;
    Eigen::MatrixX3d vel_coeffs;   // num_monomials(poly_max_degree(order),   3) x 3
    Eigen::VectorXd  pres_coeffs;  // num_monomials(poly_max_degree(order)-1, 3)
};

} // namespace kfbim
