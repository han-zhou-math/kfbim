#pragma once

#include "i_spread.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Concrete 2D Laplace spread using the panel Cauchy solver.
//
// The Cauchy solve recovers a degree-2 Taylor correction polynomial C = u⁺ − u⁻ = [u]
// at each interface quadrature point. Spread then applies the 5-point IIM
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

} // namespace kfbim
