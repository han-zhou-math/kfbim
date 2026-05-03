#pragma once

#include "i_restrict.hpp"

namespace kfbim {

// ---------------------------------------------------------------------------
// Legacy 2D Laplace restrict for Gauss-point correction polynomials.
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
                                       int                    q,
                                       const LocalPoly2D&     corr) const;

    const GridPair2D& grid_pair_;
    int               stencil_radius_;
};

// ---------------------------------------------------------------------------
// Restrict companion for LaplaceLobattoCenterSpread2D.
//
// The incoming correction_polys are expansion-center Taylor polynomials, not
// interface-point polynomials.  Exterior grid samples in each local restrict
// stencil are corrected with the nearest expansion center to that grid node.
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
