#pragma once

#include "i_spread.hpp"

namespace kfbim {

// Preferred 3D Laplace spread for shared P2 triangular patch geometry.
//
// Each parent triangle is subdivided twice; the 16 small-triangle barycenters
// are used as generated correction expansion centers.
class LaplaceQuadraticPatchCenterSpread3D final : public ILaplaceSpread3D {
public:
    explicit LaplaceQuadraticPatchCenterSpread3D(const GridPair3D& grid_pair,
                                                 double            kappa = 0.0);

    std::vector<LocalPoly3D> apply(
        const std::vector<LaplaceJumpData3D>& jumps,
        Eigen::VectorXd&                      rhs_correction) const override;

    const GridPair3D& grid_pair() const override { return grid_pair_; }

private:
    const GridPair3D& grid_pair_;
    double            kappa_;
};

} // namespace kfbim
