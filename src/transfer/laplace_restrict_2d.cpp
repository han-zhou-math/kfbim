#include "laplace_restrict_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "../grid/structured_grid_ops.hpp"
#include "laplace_projection_correction_2d.hpp"

namespace kfbim {

namespace {

constexpr int kQuadraticExpansionCentersPerPanel = 4;

Eigen::Vector2d interface_point(const Interface2D& iface, int q) {
    return iface.points().row(q).transpose();
}

Eigen::Vector2d grid_point(const CartesianGrid2D& grid, int idx) {
    return structured_grid::point(grid, idx);
}

} // namespace

class LaplaceRestrictCorrectionEvaluator2D {
public:
    virtual ~LaplaceRestrictCorrectionEvaluator2D() = default;
    virtual double half_correction(int grid_node, Eigen::Vector2d pt) const = 0;
};

namespace {

class NearestExpansionCenterRestrictCorrectionEvaluator2D final
    : public LaplaceRestrictCorrectionEvaluator2D {
public:
    NearestExpansionCenterRestrictCorrectionEvaluator2D(
        const GridPair2D& grid_pair,
        const LaplaceSpreadResult2D& spread_result)
        : grid_pair_(grid_pair)
        , correction_polys_(spread_result.correction_polys)
    {}

    double half_correction(int grid_node, Eigen::Vector2d pt) const override
    {
        const int center_idx = grid_pair_.nearest_p2_expansion_center(grid_node);
        return 0.5 * evaluate_taylor_poly_2d(correction_polys_[center_idx], pt);
    }

private:
    const GridPair2D& grid_pair_;
    const std::vector<LocalPoly2D>& correction_polys_;
};

class ProjectionPointRestrictCorrectionEvaluator2D final
    : public LaplaceRestrictCorrectionEvaluator2D {
public:
    ProjectionPointRestrictCorrectionEvaluator2D(
        const GridPair2D& grid_pair,
        const LaplaceSpreadResult2D& spread_result)
        : grid_pair_(grid_pair)
        , spread_result_(spread_result)
        , projection_band_(spread_result.projection_cache)
    {}

    double half_correction(int grid_node, Eigen::Vector2d) const override
    {
        return 0.5 * evaluate_projection_point_correction_2d(grid_pair_,
                                                             projection_band_,
                                                             grid_node,
                                                             spread_result_);
    }

private:
    const GridPair2D& grid_pair_;
    const LaplaceSpreadResult2D& spread_result_;
    const NarrowBandProjection2D& projection_band_;
};

std::unique_ptr<LaplaceRestrictCorrectionEvaluator2D> make_restrict_correction_evaluator(
    const GridPair2D& grid_pair,
    const LaplaceSpreadResult2D& spread_result,
    int expected_centers)
{
    const auto& iface = grid_pair.interface();
    if (spread_result.correction_method
        == LaplaceCorrectionMethod2D::NearestExpansionCenter) {
        if (static_cast<int>(spread_result.correction_polys.size())
            != expected_centers) {
            throw std::invalid_argument(
                "LaplaceQuadraticPanelCenterRestrict2D correction_polys size must equal 4*num_panels");
        }
        return std::make_unique<NearestExpansionCenterRestrictCorrectionEvaluator2D>(
            grid_pair, spread_result);
    }

    if (spread_result.correction_method == LaplaceCorrectionMethod2D::ProjectionPoint) {
        require_projection_curve_data(iface, spread_result);
        return std::make_unique<ProjectionPointRestrictCorrectionEvaluator2D>(
            grid_pair, spread_result);
    }

    throw std::invalid_argument(
        "LaplaceQuadraticPanelCenterRestrict2D unsupported correction method");
}

} // namespace

LaplaceQuadraticPanelCenterRestrict2D::LaplaceQuadraticPanelCenterRestrict2D(
    const GridPair2D& grid_pair,
    int               stencil_radius)
    : grid_pair_(grid_pair)
    , stencil_radius_(stencil_radius)
    , support_(build_laplace_correction_support_2d(
          grid_pair,
          "LaplaceQuadraticPanelCenterRestrict2D"))
{
    if (stencil_radius_ < 1)
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D stencil_radius must be positive");
}

std::vector<LocalPoly2D> LaplaceQuadraticPanelCenterRestrict2D::apply(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult2D& spread_result) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();
    const int expected_centers = kQuadraticExpansionCentersPerPanel * iface.num_panels();

    if (bulk_solution.size() != grid.num_dofs())
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D bulk_solution size must equal grid DOF count");
    if (iface.points_per_panel() != 3
        || iface.panel_node_layout() != PanelNodeLayout2D::QuadraticLagrange) {
        throw std::invalid_argument("LaplaceQuadraticPanelCenterRestrict2D requires P2 quadratic 3-point panels");
    }

    const auto correction_evaluator =
        make_restrict_correction_evaluator(grid_pair_,
                                           spread_result,
                                           expected_centers);

    std::vector<LocalPoly2D> result(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        result[q] = fit_at_interface_point(bulk_solution,
                                           q,
                                           *correction_evaluator);
    }

    return result;
}

std::vector<LocalPoly2D> LaplaceQuadraticPanelCenterRestrict2D::apply(
    const Eigen::VectorXd&          bulk_solution,
    const std::vector<LocalPoly2D>& correction_polys) const
{
    LaplaceSpreadResult2D spread_result;
    spread_result.correction_method =
        LaplaceCorrectionMethod2D::NearestExpansionCenter;
    spread_result.correction_polys = correction_polys;
    return apply(bulk_solution, spread_result);
}

LocalPoly2D LaplaceQuadraticPanelCenterRestrict2D::fit_at_interface_point(
    const Eigen::VectorXd&                   bulk_solution,
    int                                      q,
    const LaplaceRestrictCorrectionEvaluator2D& correction_evaluator) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const Eigen::Vector2d center = interface_point(iface, q);
    const std::array<int, 6>& stencil_nodes = support_.restrict_stencils[q];

    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 1> rhs;
    for (int r = 0; r < static_cast<int>(stencil_nodes.size()); ++r) {
        const int idx = stencil_nodes[r];
        const Eigen::Vector2d pt = grid_point(grid, idx);
        const double dx = pt[0] - center[0];
        const double dy = pt[1] - center[1];
        A(r, 0) = 1.0;
        A(r, 1) = dx;
        A(r, 2) = dy;
        A(r, 3) = 0.5 * dx * dx;
        A(r, 4) = dx * dy;
        A(r, 5) = 0.5 * dy * dy;

        double val = bulk_solution[idx];
        const double correction =
            correction_evaluator.half_correction(idx, pt);
        if (grid_pair_.domain_label(idx) == 0)
            val += correction;
        else
            val -= correction;
        rhs[r] = val;
    }

    Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(A);
    if (!lu.isInvertible()) {
        throw std::runtime_error(
            "LaplaceQuadraticPanelCenterRestrict2D singular fixed quadratic interpolation stencil");
    }

    LocalPoly2D poly;
    poly.center = center;
    poly.coeffs = lu.solve(rhs);
    return poly;
}

} // namespace kfbim
