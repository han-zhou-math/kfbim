#include "laplace_restrict_3d.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../grid/structured_grid_ops.hpp"
#include "laplace_projection_correction_3d.hpp"

namespace kfbim {

namespace {

constexpr int kExpansionCentersPerPanel3D = 16;

using K3 = CGAL::Exact_predicates_inexact_constructions_kernel;
using CPoint3 = K3::Point_3;
using CPointWithIndex3 = std::pair<CPoint3, int>;
using CSearchBaseTraits3 = CGAL::Search_traits_3<K3>;
using CSearchTraits3 =
    CGAL::Search_traits_adapter<CPointWithIndex3,
                                CGAL::First_of_pair_property_map<CPointWithIndex3>,
                                CSearchBaseTraits3>;
using CNeighborSearch3 = CGAL::Orthogonal_k_neighbor_search<CSearchTraits3>;
using CSearchTree3 = CNeighborSearch3::Tree;

Eigen::Vector3d interface_point(const Interface3D& iface, int q)
{
    return iface.points().row(q).transpose();
}

Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int idx)
{
    return structured_grid::point(grid, idx);
}

double max_grid_spacing(const CartesianGrid3D& grid)
{
    return structured_grid::max_spacing(grid);
}

class NearestCenterFinder3D {
public:
    explicit NearestCenterFinder3D(const std::vector<LocalPoly3D>& center_polys)
    {
        if (center_polys.empty())
            throw std::invalid_argument("NearestCenterFinder3D requires centers");

        points_.reserve(center_polys.size());
        for (int i = 0; i < static_cast<int>(center_polys.size()); ++i) {
            const Eigen::Vector3d& c = center_polys[i].center;
            points_.emplace_back(CPoint3(c[0], c[1], c[2]), i);
        }
        tree_ = std::make_unique<CSearchTree3>(points_.begin(), points_.end());
    }

    int nearest(Eigen::Vector3d pt) const
    {
        CNeighborSearch3 search(*tree_, CPoint3(pt[0], pt[1], pt[2]), 1);
        return search.begin()->first.second;
    }

private:
    std::vector<CPointWithIndex3> points_;
    std::unique_ptr<CSearchTree3> tree_;
};

int nearest_poly_index(const NearestCenterFinder3D& center_finder,
                       Eigen::Vector3d             pt)
{
    return center_finder.nearest(pt);
}

std::vector<int> build_nearest_center_map(const GridPair3D&               grid_pair,
                                          const NearestCenterFinder3D&    center_finder,
                                          double                          band_radius)
{
    const auto& grid = grid_pair.grid();
    std::vector<int> nearest(grid.num_dofs(), -1);
    for (int idx : grid_pair.near_interface_nodes(band_radius))
        nearest[idx] = nearest_poly_index(center_finder, grid_point(grid, idx));
    return nearest;
}

int nearest_center_for_grid_node(const std::vector<int>& nearest_center_map,
                                 int                     idx)
{
    if (idx >= 0 && idx < static_cast<int>(nearest_center_map.size())
        && nearest_center_map[idx] >= 0) {
        return nearest_center_map[idx];
    }
    throw std::runtime_error(
        "LaplaceQuadraticPatchCenterRestrict3D missing nearest-center map entry");
}

bool profile_projection_transfer_3d()
{
    return std::getenv("KFBIM_PROFILE_INTERFACE_3D") != nullptr
        || std::getenv("KFBIM_PROFILE_TRANSFER_3D") != nullptr;
}

using ProfileClock = std::chrono::steady_clock;

double seconds_since(ProfileClock::time_point start)
{
    return std::chrono::duration<double>(ProfileClock::now() - start).count();
}

} // namespace

class LaplaceRestrictCorrectionEvaluator3D {
public:
    virtual ~LaplaceRestrictCorrectionEvaluator3D() = default;
    virtual double half_correction(int grid_node, Eigen::Vector3d pt) const = 0;
};

namespace {

class NearestExpansionCenterRestrictCorrectionEvaluator3D final
    : public LaplaceRestrictCorrectionEvaluator3D {
public:
    NearestExpansionCenterRestrictCorrectionEvaluator3D(
        const GridPair3D& grid_pair,
        const LaplaceSpreadResult3D& spread_result,
        double band_radius)
        : correction_polys_(spread_result.correction_polys)
        , nearest_center_for_node_(
              build_nearest_center_map(grid_pair,
                                       NearestCenterFinder3D(correction_polys_),
                                       band_radius))
    {}

    double half_correction(int grid_node, Eigen::Vector3d pt) const override
    {
        const int center_idx =
            nearest_center_for_grid_node(nearest_center_for_node_, grid_node);
        return 0.5 * evaluate_taylor_poly_3d(correction_polys_[center_idx], pt);
    }

private:
    const std::vector<LocalPoly3D>& correction_polys_;
    std::vector<int> nearest_center_for_node_;
};

class ProjectionPointRestrictCorrectionEvaluator3D final
    : public LaplaceRestrictCorrectionEvaluator3D {
public:
    ProjectionPointRestrictCorrectionEvaluator3D(
        const GridPair3D& grid_pair,
        const LaplaceSpreadResult3D& spread_result)
        : grid_pair_(grid_pair)
        , spread_result_(spread_result)
        , projection_band_(spread_result.projection_cache)
    {}

    double half_correction(int grid_node, Eigen::Vector3d) const override
    {
        return 0.5 * evaluate_projection_point_correction_3d(grid_pair_,
                                                             projection_band_,
                                                             grid_node,
                                                             spread_result_);
    }

private:
    const GridPair3D& grid_pair_;
    const LaplaceSpreadResult3D& spread_result_;
    const NarrowBandProjection3D& projection_band_;
};

std::unique_ptr<LaplaceRestrictCorrectionEvaluator3D> make_restrict_correction_evaluator(
    const GridPair3D& grid_pair,
    const LaplaceSpreadResult3D& spread_result,
    int expected_centers,
    double band_radius)
{
    const auto& iface = grid_pair.interface();
    if (spread_result.correction_method
        == LaplaceCorrectionMethod3D::NearestExpansionCenter) {
        if (static_cast<int>(spread_result.correction_polys.size())
            != expected_centers) {
            throw std::invalid_argument(
                "LaplaceQuadraticPatchCenterRestrict3D correction_polys size must equal 16*num_panels");
        }
        return std::make_unique<NearestExpansionCenterRestrictCorrectionEvaluator3D>(
            grid_pair, spread_result, band_radius);
    }

    if (spread_result.correction_method
        == LaplaceCorrectionMethod3D::ProjectionPoint) {
        require_projection_surface_data(iface, spread_result);
        if (profile_projection_transfer_3d()) {
            std::printf("      restrict projection_cache nodes=%zu projections=%zu\n",
                        spread_result.projection_cache.nodes().size(),
                        spread_result.projection_cache.projections().size());
        }
        return std::make_unique<ProjectionPointRestrictCorrectionEvaluator3D>(
            grid_pair, spread_result);
    }

    throw std::invalid_argument(
        "LaplaceQuadraticPatchCenterRestrict3D unsupported correction method");
}

} // namespace

LaplaceQuadraticPatchCenterRestrict3D::LaplaceQuadraticPatchCenterRestrict3D(
    const GridPair3D& grid_pair,
    int               stencil_radius)
    : grid_pair_(grid_pair)
    , stencil_radius_(stencil_radius)
{
    if (stencil_radius_ < 1)
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterRestrict3D stencil_radius must be positive");
}

std::vector<LocalPoly3D> LaplaceQuadraticPatchCenterRestrict3D::apply(
    const Eigen::VectorXd&       bulk_solution,
    const LaplaceSpreadResult3D& spread_result) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();
    const int expected_centers =
        kExpansionCentersPerPanel3D * iface.num_panels();

    if (bulk_solution.size() != grid.num_dofs())
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterRestrict3D bulk_solution size must equal grid DOF count");
    if (iface.points_per_panel() != 6
        || iface.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange) {
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterRestrict3D requires P2 QuadraticLagrange panels");
    }

    const double band_radius = (static_cast<double>(stencil_radius_) + 1.0)
                               * std::sqrt(3.0) * max_grid_spacing(grid);

    const auto correction_evaluator =
        make_restrict_correction_evaluator(grid_pair_,
                                           spread_result,
                                           expected_centers,
                                           band_radius);

    const ProfileClock::time_point fit_start = ProfileClock::now();
    std::vector<LocalPoly3D> result(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        result[q] = fit_at_interface_point(bulk_solution,
                                           q,
                                           *correction_evaluator);
    }
    const double t_fit = seconds_since(fit_start);
    if (profile_projection_transfer_3d()) {
        std::printf("      restrict fit iface=%d cache_radius=%d fit %.3fs\n",
                    n_iface,
                    stencil_radius_,
                    t_fit);
    }

    return result;
}

LocalPoly3D LaplaceQuadraticPatchCenterRestrict3D::fit_at_interface_point(
    const Eigen::VectorXd&                      bulk_solution,
    int                                         q,
    const LaplaceRestrictCorrectionEvaluator3D& correction_evaluator) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int closest = grid_pair_.closest_bulk_node(q);
    const Eigen::Vector3d center = interface_point(iface, q);
    const std::array<int, 10> stencil_nodes =
        structured_grid::quadratic_restrict_stencil_nodes_3d(
            "LaplaceQuadraticPatchCenterRestrict3D",
            grid,
            closest,
            center);

    Eigen::Matrix<double, 10, 10> A;
    Eigen::Matrix<double, 10, 1> rhs;
    for (int r = 0; r < static_cast<int>(stencil_nodes.size()); ++r) {
        const int idx = stencil_nodes[r];
        const Eigen::Vector3d pt = grid_point(grid, idx);
        const double dx = pt[0] - center[0];
        const double dy = pt[1] - center[1];
        const double dz = pt[2] - center[2];
        A(r, 0) = 1.0;
        A(r, 1) = dx;
        A(r, 2) = dy;
        A(r, 3) = dz;
        A(r, 4) = 0.5 * dx * dx;
        A(r, 5) = dx * dy;
        A(r, 6) = dx * dz;
        A(r, 7) = 0.5 * dy * dy;
        A(r, 8) = dy * dz;
        A(r, 9) = 0.5 * dz * dz;

        double val = bulk_solution[idx];
        const double correction =
            correction_evaluator.half_correction(idx, pt);
        if (grid_pair_.domain_label(idx) == 0)
            val += correction;
        else
            val -= correction;
        rhs[r] = val;
    }

    Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> lu(A);
    if (!lu.isInvertible()) {
        throw std::runtime_error(
            "LaplaceQuadraticPatchCenterRestrict3D singular fixed quadratic interpolation stencil");
    }

    LocalPoly3D poly;
    poly.center = center;
    poly.coeffs = lu.solve(rhs);
    return poly;
}

} // namespace kfbim
