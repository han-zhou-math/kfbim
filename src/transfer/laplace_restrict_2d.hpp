#pragma once

#include "i_restrict.hpp"
#include "laplace_correction_support.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Restrict companion for LaplaceQuadraticPanelCenterSpread2D.
//
// The incoming correction_polys are expansion-center Taylor polynomials, not
// interface-point polynomials.  Grid samples in the fixed six-point square
// restrict stencil are shifted onto the average branch with the nearest
// expansion center:
// interior samples subtract C/2, exterior samples add C/2.
// stencil_radius is retained for API compatibility and nearest-center cache
// construction; it no longer changes the interpolation stencil size.
// ---------------------------------------------------------------------------

class LaplaceQuadraticPanelCenterRestrict2D final : public ILaplaceRestrict2D {
public:
    explicit LaplaceQuadraticPanelCenterRestrict2D(
        const GridPair2D& grid_pair,
        int               stencil_radius = 2);

    std::vector<LocalPoly2D> apply(
        const Eigen::VectorXd&          bulk_solution,
        const LaplaceSpreadResult2D&    spread_result) const override;

    std::vector<LocalPoly2D> apply(
        const Eigen::VectorXd&          bulk_solution,
        const std::vector<LocalPoly2D>& correction_polys) const;

    const GridPair2D& grid_pair() const override { return grid_pair_; }

private:
    LocalPoly2D fit_at_interface_point(
        const Eigen::VectorXd&                    bulk_solution,
        int                                       q,
        const class LaplaceRestrictCorrectionEvaluator2D& correction_evaluator) const;

    const GridPair2D& grid_pair_;
    int               stencil_radius_;
    LaplaceCorrectionSupport2D support_;
};

using LaplaceLobattoCenterRestrict2D = LaplaceQuadraticPanelCenterRestrict2D;

} // namespace kfbim
