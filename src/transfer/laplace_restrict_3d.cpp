#include "laplace_restrict_3d.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

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
    const auto c = grid.coord(idx);
    return {c[0], c[1], c[2]};
}

double max_grid_spacing(const CartesianGrid3D& grid)
{
    const auto h = grid.spacing();
    return std::max(h[0], std::max(h[1], h[2]));
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
    const Eigen::VectorXd&          bulk_solution,
    const std::vector<LocalPoly3D>& correction_polys) const
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
    if (static_cast<int>(correction_polys.size()) != expected_centers) {
        throw std::invalid_argument(
            "LaplaceQuadraticPatchCenterRestrict3D correction_polys size must equal 16*num_panels");
    }

    const double band_radius = (static_cast<double>(stencil_radius_) + 1.0)
                               * std::sqrt(3.0) * max_grid_spacing(grid);
    const NearestCenterFinder3D center_finder(correction_polys);
    const std::vector<int> nearest_center_for_node =
        build_nearest_center_map(grid_pair_, center_finder, band_radius);

    std::vector<LocalPoly3D> result(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        result[q] = fit_at_interface_point(bulk_solution,
                                           q,
                                           correction_polys,
                                           nearest_center_for_node);
    }

    return result;
}

LocalPoly3D LaplaceQuadraticPatchCenterRestrict3D::fit_at_interface_point(
    const Eigen::VectorXd&          bulk_solution,
    int                             q,
    const std::vector<LocalPoly3D>& center_polys,
    const std::vector<int>&         nearest_center_for_grid_node_map) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];
    const int nz = dims[2];
    const int nxy = nx * ny;

    const int closest = grid_pair_.closest_bulk_node(q);
    const int kc = closest / nxy;
    const int rem = closest % nxy;
    const int jc = rem / nx;
    const int ic = rem % nx;

    struct Sample {
        int idx;
        double dist2;
    };
    std::vector<Sample> samples;
    const int width = 2 * stencil_radius_ + 1;
    samples.reserve(width * width * width);

    const Eigen::Vector3d center = interface_point(iface, q);
    for (int k = std::max(0, kc - stencil_radius_);
         k <= std::min(nz - 1, kc + stencil_radius_);
         ++k) {
        for (int j = std::max(0, jc - stencil_radius_);
             j <= std::min(ny - 1, jc + stencil_radius_);
             ++j) {
            for (int i = std::max(0, ic - stencil_radius_);
                 i <= std::min(nx - 1, ic + stencil_radius_);
                 ++i) {
                const int idx = grid.index(i, j, k);
                const Eigen::Vector3d pt = grid_point(grid, idx);
                samples.push_back({idx, (pt - center).squaredNorm()});
            }
        }
    }

    if (samples.size() < 10) {
        throw std::runtime_error(
            "LaplaceQuadraticPatchCenterRestrict3D needs at least 10 stencil samples");
    }

    std::sort(samples.begin(), samples.end(),
              [](const Sample& a, const Sample& b) {
                  if (a.dist2 == b.dist2)
                      return a.idx < b.idx;
                  return a.dist2 < b.dist2;
              });

    const int n_rows = static_cast<int>(samples.size());
    Eigen::MatrixXd A(n_rows, 10);
    Eigen::VectorXd rhs(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        const int idx = samples[r].idx;
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
        const int center_idx =
            nearest_center_for_grid_node(nearest_center_for_grid_node_map,
                                         idx);
        const double correction =
            0.5 * evaluate_taylor_poly_3d(center_polys[center_idx], pt);
        if (grid_pair_.domain_label(idx) == 0)
            val += correction;
        else
            val -= correction;
        rhs[r] = val;
    }

    LocalPoly3D poly;
    poly.center = center;
    poly.coeffs = A.colPivHouseholderQr().solve(rhs);
    return poly;
}

} // namespace kfbim
