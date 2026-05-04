#pragma once

#include "i_restrict.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Restrict companion for LaplaceLobattoCenterSpread2D.
//
// The incoming correction_polys are expansion-center Taylor polynomials, not
// interface-point polynomials.  Grid samples in each local restrict stencil
// are shifted onto the average branch with the nearest expansion center:
// interior samples subtract C/2, exterior samples add C/2.
// ---------------------------------------------------------------------------

class LaplaceLobattoCenterRestrict2D final : public ILaplaceRestrict2D {
public:
    explicit LaplaceLobattoCenterRestrict2D(const GridPair2D& grid_pair,
                                            int               stencil_radius = 2);

    std::vector<LocalPoly2D> apply(
        const Eigen::VectorXd&          bulk_solution,
        const std::vector<LocalPoly2D>& correction_polys) const override;

    const GridPair2D& grid_pair() const override { return grid_pair_; }

private:
    LocalPoly2D fit_at_interface_point(
        const Eigen::VectorXd&          bulk_solution,
        int                             q,
        const std::vector<LocalPoly2D>& center_polys,
        const std::vector<int>&         nearest_center_for_grid_node) const;

    const GridPair2D& grid_pair_;
    int               stencil_radius_;
};

} // namespace kfbim
