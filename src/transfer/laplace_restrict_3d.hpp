#pragma once

#include "i_restrict.hpp"

namespace kfbim {

// Restrict companion for LaplaceQuadraticPatchCenterSpread3D.
class LaplaceQuadraticPatchCenterRestrict3D final : public ILaplaceRestrict3D {
public:
    explicit LaplaceQuadraticPatchCenterRestrict3D(const GridPair3D& grid_pair,
                                                   int               stencil_radius = 2);

    std::vector<LocalPoly3D> apply(
        const Eigen::VectorXd&       bulk_solution,
        const LaplaceSpreadResult3D& spread_result) const override;

    const GridPair3D& grid_pair() const override { return grid_pair_; }

private:
    LocalPoly3D fit_at_interface_point(
        const Eigen::VectorXd&         bulk_solution,
        int                            q,
        const LaplaceSpreadResult3D&   spread_result,
        const std::vector<int>*        nearest_center_for_grid_node,
        const NarrowBandProjection3D*  projection_band) const;

    const GridPair3D& grid_pair_;
    int               stencil_radius_;
};

} // namespace kfbim
