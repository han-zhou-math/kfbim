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
// For each interface quadrature point, apply() builds the correction data
// needed to accumulate bulk RHS corrections near the interface.
//
// Spreads return a result object because projection-point correction needs
// surface jump data during restrict.
//
// rhs_correction is *accumulated into*, not zeroed — the caller zeros it first
// when starting a fresh iteration, or accumulates across multiple interfaces.
// ---------------------------------------------------------------------------

enum class LaplaceCorrectionMethod {
    NearestExpansionCenter,
    ProjectionPoint
};

using LaplaceCorrectionMethod2D = LaplaceCorrectionMethod;
using LaplaceCorrectionMethod3D = LaplaceCorrectionMethod;

struct LaplaceCorrectionContext2D {
    LaplaceCorrectionMethod correction_method =
        LaplaceCorrectionMethod::NearestExpansionCenter;

    // Existing center-polynomial correction data. In projection-point mode this
    // may be empty because C(x) is evaluated directly from projected P2 data.
    std::vector<LocalPoly2D> correction_polys;

    // Surface data needed by the projection-point evaluator. Vectors are sized
    // to Interface2D::num_points() when correction_method == ProjectionPoint.
    Eigen::VectorXd u_jump;
    Eigen::VectorXd un_jump;
    Eigen::VectorXd rhs_jump;

    // Screened coefficient in -Delta C + alpha*C = [f].
    double alpha = 0.0;

    // Projection-point cache for grid nodes where C(x) is actually needed.
    NarrowBandProjection2D projection_cache;
};

using LaplaceSpreadResult2D = LaplaceCorrectionContext2D;

class ILaplaceSpread2D {
public:
    virtual ~ILaplaceSpread2D() = default;

    // jumps[i]         = jump data at interface quadrature point i
    // rhs_correction   = correction accumulated into, length grid_pair().grid().num_dofs()
    // returns          = correction context for the paired restrict operator
    virtual LaplaceSpreadResult2D apply(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const = 0;

    virtual const GridPair2D& grid_pair() const = 0;
};

struct LaplaceCorrectionContext3D {
    LaplaceCorrectionMethod correction_method =
        LaplaceCorrectionMethod::NearestExpansionCenter;

    // Existing center-polynomial correction data. In projection-point mode this
    // may be empty because C(x) is evaluated directly from projected P2 data.
    std::vector<LocalPoly3D> correction_polys;

    // Surface data needed by the projection-point evaluator. Vectors are sized
    // to Interface3D::num_points() when correction_method == ProjectionPoint.
    Eigen::VectorXd u_jump;
    Eigen::VectorXd un_jump;
    Eigen::VectorXd rhs_jump;

    // Screened coefficient in -Delta C + alpha*C = [f].
    double alpha = 0.0;

    // Projection-point cache for grid nodes where C(x) is actually needed.
    NarrowBandProjection3D projection_cache;
};

// Temporary compatibility name for the 3D spread -> restrict contract.
using LaplaceSpreadResult3D = LaplaceCorrectionContext3D;

class ILaplaceSpread3D {
public:
    virtual ~ILaplaceSpread3D() = default;

    // The 3D result carries either nearest-center correction polynomials or
    // projection-point surface data for the paired restrict operator.
    virtual LaplaceSpreadResult3D apply(
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
