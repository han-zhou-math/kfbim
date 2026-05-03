#pragma once

#include "i_spread.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Legacy 2D Laplace spread using the Gauss-point panel Cauchy solver.
//
// The Cauchy solve recovers a degree-2 Taylor correction polynomial C = u+ - u-
// at each legacy Gauss interface quadrature point, where u+ is the interior (label 1) and
// u- is the exterior (label 0). The jump is defined as [u] = u_int - u_ext.
// stencil defect at irregular grid nodes:
//
//   rhs[n] += side(nb)-side(n) * C(x_nb) / h_axis^2
//
// for each face neighbor nb across the interface.
// ---------------------------------------------------------------------------

class LaplacePanelSpread2D final : public ILaplaceSpread2D {
public:
    explicit LaplacePanelSpread2D(const GridPair2D& grid_pair,
                                  double            kappa = 0.0);

    std::vector<LocalPoly2D> apply(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const override;

    const GridPair2D& grid_pair() const override { return grid_pair_; }

private:
    const GridPair2D& grid_pair_;
    double            kappa_;
};

// ---------------------------------------------------------------------------
// Preferred 2D Laplace spread for Chebyshev-Lobatto panel geometry.
//
// Interface points are 3 Lobatto geometry/jump DOFs per panel. Correction
// polynomials are solved at four generated panel-center locations. RHS and
// restrict grid-node corrections evaluate the Taylor polynomial from the
// nearest generated expansion center.
// ---------------------------------------------------------------------------

class LaplaceLobattoCenterSpread2D final : public ILaplaceSpread2D {
public:
    explicit LaplaceLobattoCenterSpread2D(const GridPair2D& grid_pair,
                                          double            kappa = 0.0);

    std::vector<LocalPoly2D> apply(
        const std::vector<LaplaceJumpData2D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const override;

    const GridPair2D& grid_pair() const override { return grid_pair_; }

private:
    const GridPair2D& grid_pair_;
    double            kappa_;
};

} // namespace kfbim
