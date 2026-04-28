#pragma once

#include "i_restrict.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Concrete 2D Laplace restrict.
//
// Fits a degree-2 Taylor polynomial to bulk grid values around each interface
// quadrature point, then subtracts the matching correction polynomial produced
// by Spread. The result coefficients are
//   [u, u_x, u_y, u_xx, u_xy, u_yy]
// centered at the interface point.
// ---------------------------------------------------------------------------

class LaplaceQuadraticRestrict2D final : public ILaplaceRestrict2D {
public:
    explicit LaplaceQuadraticRestrict2D(const GridPair2D& grid_pair,
                                        int               stencil_radius = 2);

    std::vector<LocalPoly2D> apply(
        const Eigen::VectorXd&          bulk_solution,
        const std::vector<LocalPoly2D>& correction_polys) const override;

    const GridPair2D& grid_pair() const override { return grid_pair_; }

private:
    LocalPoly2D fit_at_interface_point(const Eigen::VectorXd& bulk_solution,
                                       int                    q) const;

    const GridPair2D& grid_pair_;
    int               stencil_radius_;
};

} // namespace kfbim
