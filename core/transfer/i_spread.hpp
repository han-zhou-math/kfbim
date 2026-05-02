#pragma once

#include <vector>
#include <Eigen/Dense>
#include "../local_cauchy/jump_data.hpp"
#include "../local_cauchy/local_poly.hpp"
#include "../geometry/grid_pair_2d.hpp"
#include "../geometry/grid_pair_3d.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Spread (Layer 1): interface jump data → bulk RHS correction
//
// For each interface quadrature point, apply():
//   1. Calls ILocalCauchySolver::fit() to solve the local Cauchy problem and
//      obtain a correction polynomial.
//   2. Evaluates the polynomial at every nearby bulk node (via GridPair) and
//      accumulates the values into rhs_correction.
//
// The returned vector of LocalPolys is passed to the paired Restrict::apply().
// Most spreads return one polynomial per interface point; center-based spreads
// may return generated expansion-center polynomials instead.
//
// rhs_correction is *accumulated into*, not zeroed — the caller zeros it first
// when starting a fresh iteration, or accumulates across multiple interfaces.
// ---------------------------------------------------------------------------

class ILaplaceSpread2D {
public:
    virtual ~ILaplaceSpread2D() = default;

    // jumps[i]         = jump data at interface quadrature point i
    // rhs_correction   = correction accumulated into, length grid_pair().grid().num_dofs()
    // returns          = fitted local polys for the paired restrict operator
    virtual std::vector<LocalPoly2D> apply(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const = 0;

    virtual const GridPair2D& grid_pair() const = 0;
};

class ILaplaceSpread3D {
public:
    virtual ~ILaplaceSpread3D() = default;

    virtual std::vector<LocalPoly3D> apply(
        const std::vector<LaplaceJumpData3D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const = 0;

    virtual const GridPair3D& grid_pair() const = 0;
};

// ---------------------------------------------------------------------------
// Stokes Spread: correction accumulated separately into each velocity and
// pressure sub-grid RHS (ux, uy, p in 2D; ux, uy, uz, p in 3D).
// Each rhs_* is sized to the DOF count of the corresponding MACGrid sub-grid.
// ---------------------------------------------------------------------------

class IStokesSpread2D {
public:
    virtual ~IStokesSpread2D() = default;

    virtual std::vector<StokesLocalPoly2D> apply(
        const std::vector<StokesJumpData2D>& jumps,
        Eigen::VectorXd&                     rhs_ux,
        Eigen::VectorXd&                     rhs_uy,
        Eigen::VectorXd&                     rhs_p) const = 0;

    virtual const GridPair2D& grid_pair() const = 0;
};

class IStokesSpread3D {
public:
    virtual ~IStokesSpread3D() = default;

    virtual std::vector<StokesLocalPoly3D> apply(
        const std::vector<StokesJumpData3D>& jumps,
        Eigen::VectorXd&                     rhs_ux,
        Eigen::VectorXd&                     rhs_uy,
        Eigen::VectorXd&                     rhs_uz,
        Eigen::VectorXd&                     rhs_p) const = 0;

    virtual const GridPair3D& grid_pair() const = 0;
};

} // namespace kfbim
