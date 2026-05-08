#include "grid_pair_2d.hpp"
#include "p2_curve_2d.hpp"
#include "p2_projection_2d.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/property_map.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kfbim {

using K2      = CGAL::Exact_predicates_inexact_constructions_kernel;
using CPoint2 = K2::Point_2;
using CPoly2  = CGAL::Polygon_2<K2>;
using Point2  = std::array<double, 2>;
using CPointWithIndex2 = std::pair<CPoint2, int>;
using CSearchBaseTraits2 = CGAL::Search_traits_2<K2>;
using CSearchTraits2 =
    CGAL::Search_traits_adapter<CPointWithIndex2,
                                CGAL::First_of_pair_property_map<CPointWithIndex2>,
                                CSearchBaseTraits2>;
using CNeighborSearch2 = CGAL::Orthogonal_k_neighbor_search<CSearchTraits2>;
using CSearchTree2 = CNeighborSearch2::Tree;
using ProfileClock2D = std::chrono::steady_clock;

// ============================================================================
// Internal helpers
// ============================================================================

// x-coordinate of the first DOF along each axis for a given layout.
static std::array<double, 2> dof_first(const CartesianGrid2D& g) {
    auto o = g.origin(), h = g.spacing();
    switch (g.layout()) {
    case DofLayout2D::Node:       return {o[0],             o[1]            };
    case DofLayout2D::CellCenter: return {o[0] + 0.5*h[0],  o[1] + 0.5*h[1] };
    case DofLayout2D::FaceX:      return {o[0],             o[1] + 0.5*h[1] };
    case DofLayout2D::FaceY:      return {o[0] + 0.5*h[0],  o[1]            };
    }
    return {o[0], o[1]};
}

// Nearest bulk-node flat index to physical position (x, y).
static int nearest_node(const CartesianGrid2D& g, double x, double y) {
    auto f = dof_first(g);
    auto h = g.spacing();
    auto d = g.dof_dims();
    int nx = d[0], ny = d[1];
    int i = static_cast<int>(std::lround((x - f[0]) / h[0]));
    int j = static_cast<int>(std::lround((y - f[1]) / h[1]));
    i = std::max(0, std::min(nx - 1, i));
    j = std::max(0, std::min(ny - 1, j));
    return j * nx + i;
}

static Point2 midpoint(Point2 a, Point2 b) {
    return {0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])};
}

static bool profile_grid_pair_2d()
{
    return std::getenv("KFBIM_PROFILE_GRID_PAIR_2D") != nullptr
        || std::getenv("KFBIM_PROFILE_INTERFACE_2D") != nullptr;
}

static bool profile_grid_pair_2d_detail()
{
    return std::getenv("KFBIM_PROFILE_GRID_PAIR_2D_DETAIL") != nullptr;
}

static double seconds_since(ProfileClock2D::time_point start)
{
    return std::chrono::duration<double>(ProfileClock2D::now() - start).count();
}

static double squared_distance(Point2 a, Point2 b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    return dx * dx + dy * dy;
}

static const double* quadratic_nodes_for_layout(PanelNodeLayout2D layout)
{
    static constexpr double gl_nodes[3] = {
        -0.77459666924148337704,
         0.0,
         0.77459666924148337704
    };
    static constexpr double lobatto_nodes[3] = {-1.0, 0.0, 1.0};

    switch (layout) {
    case PanelNodeLayout2D::LegacyGaussLegendre:
        return gl_nodes;
    case PanelNodeLayout2D::QuadraticLagrange:
        return lobatto_nodes;
    case PanelNodeLayout2D::Raw:
        return nullptr;
    }
    return nullptr;
}

static Point2 normalize_or(Point2 v, Point2 fallback)
{
    const double len = std::sqrt(v[0] * v[0] + v[1] * v[1]);
    if (len > 1.0e-14)
        return {v[0] / len, v[1] / len};
    return fallback;
}

static void push_cgal_point(CPoly2& poly, Point2 p) {
    poly.push_back(CPoint2(p[0], p[1]));
}

static Point2 quadratic_panel_point(const Interface2D& iface,
                                    int                panel,
                                    const double       nodes[3],
                                    double             s)
{
    const double s0 = nodes[0];
    const double s1 = nodes[1];
    const double s2 = nodes[2];

    const double l0 = ((s - s1) * (s - s2)) / ((s0 - s1) * (s0 - s2));
    const double l1 = ((s - s0) * (s - s2)) / ((s1 - s0) * (s1 - s2));
    const double l2 = ((s - s0) * (s - s1)) / ((s2 - s0) * (s2 - s1));

    const int q0 = iface.point_index(panel, 0);
    const int q1 = iface.point_index(panel, 1);
    const int q2 = iface.point_index(panel, 2);

    return {
        l0 * iface.points()(q0, 0) + l1 * iface.points()(q1, 0) + l2 * iface.points()(q2, 0),
        l0 * iface.points()(q0, 1) + l1 * iface.points()(q1, 1) + l2 * iface.points()(q2, 1)
    };
}

static Point2 quadratic_panel_derivative(const Interface2D& iface,
                                         int                panel,
                                         const double       nodes[3],
                                         double             s)
{
    const double s0 = nodes[0];
    const double s1 = nodes[1];
    const double s2 = nodes[2];

    const double dl0 = (2.0 * s - s1 - s2) / ((s0 - s1) * (s0 - s2));
    const double dl1 = (2.0 * s - s0 - s2) / ((s1 - s0) * (s1 - s2));
    const double dl2 = (2.0 * s - s0 - s1) / ((s2 - s0) * (s2 - s1));

    const int q0 = iface.point_index(panel, 0);
    const int q1 = iface.point_index(panel, 1);
    const int q2 = iface.point_index(panel, 2);

    return {
        dl0 * iface.points()(q0, 0) + dl1 * iface.points()(q1, 0) + dl2 * iface.points()(q2, 0),
        dl0 * iface.points()(q0, 1) + dl1 * iface.points()(q1, 1) + dl2 * iface.points()(q2, 1)
    };
}

static Point2 quadratic_panel_normal(const Interface2D& iface,
                                     int                panel,
                                     const double       nodes[3],
                                     double             s)
{
    const Point2 deriv = quadratic_panel_derivative(iface, panel, nodes, s);
    Point2 normal = normalize_or({deriv[1], -deriv[0]}, {0.0, 0.0});

    const double s0 = nodes[0];
    const double s1 = nodes[1];
    const double s2 = nodes[2];
    const double l0 = ((s - s1) * (s - s2)) / ((s0 - s1) * (s0 - s2));
    const double l1 = ((s - s0) * (s - s2)) / ((s1 - s0) * (s1 - s2));
    const double l2 = ((s - s0) * (s - s1)) / ((s2 - s0) * (s2 - s1));

    const int q0 = iface.point_index(panel, 0);
    const int q1 = iface.point_index(panel, 1);
    const int q2 = iface.point_index(panel, 2);
    const Point2 ref = normalize_or({
        l0 * iface.normals()(q0, 0) + l1 * iface.normals()(q1, 0) + l2 * iface.normals()(q2, 0),
        l0 * iface.normals()(q0, 1) + l1 * iface.normals()(q1, 1) + l2 * iface.normals()(q2, 1)
    }, {iface.normals()(q1, 0), iface.normals()(q1, 1)});

    if (normal[0] * ref[0] + normal[1] * ref[1] < 0.0)
        normal = {-normal[0], -normal[1]};
    return normal;
}

struct DistanceSample2D {
    Point2 point;
    Point2 normal;
    int    component;
};

static std::vector<int> point_components_2d(const Interface2D& iface)
{
    std::vector<int> components(iface.num_points(), 0);
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (int q = 0; q < iface.points_per_panel(); ++q)
            components[iface.point_index(p, q)] = iface.panel_components()(p);
    }
    return components;
}

static bool use_normal_domain_labels(const Interface2D& iface)
{
    return iface.points_per_panel() == 3
        && quadratic_nodes_for_layout(iface.panel_node_layout()) != nullptr;
}

static std::vector<DistanceSample2D> distance_samples_2d(const Interface2D& iface)
{
    const bool add_panel_samples = use_normal_domain_labels(iface);
    std::vector<DistanceSample2D> samples;
    samples.reserve(static_cast<std::size_t>(iface.num_points())
                    + (add_panel_samples ? geometry2d::kP2CenterS.size()
                                           * static_cast<std::size_t>(iface.num_panels())
                                         : 0));

    const std::vector<int> point_components = point_components_2d(iface);
    for (int q = 0; q < iface.num_points(); ++q) {
        samples.push_back({{iface.points()(q, 0), iface.points()(q, 1)},
                           {iface.normals()(q, 0), iface.normals()(q, 1)},
                           point_components[q]});
    }

    if (!add_panel_samples)
        return samples;

    const double* nodes = quadratic_nodes_for_layout(iface.panel_node_layout());
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (double s : geometry2d::kP2CenterS) {
            if (iface.panel_node_layout() == PanelNodeLayout2D::QuadraticLagrange) {
                const Eigen::Vector2d pt = geometry2d::panel_point(iface, p, s);
                const Eigen::Vector2d normal = geometry2d::panel_normal(iface, p, s);
                samples.push_back({{pt[0], pt[1]},
                                   {normal[0], normal[1]},
                                   iface.panel_components()(p)});
            } else {
                samples.push_back({quadratic_panel_point(iface, p, nodes, s),
                                   quadratic_panel_normal(iface, p, nodes, s),
                                   iface.panel_components()(p)});
            }
        }
    }
    return samples;
}

class NearestPointCloud2D {
public:
    explicit NearestPointCloud2D(const std::vector<Point2>& points)
    {
        if (points.empty())
            throw std::invalid_argument("NearestPointCloud2D requires at least one point");

        points_.reserve(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i)
            points_.emplace_back(CPoint2(points[i][0], points[i][1]), i);
        tree_ = std::make_unique<CSearchTree2>(points_.begin(), points_.end());
    }

    int nearest(Point2 query) const
    {
        CNeighborSearch2 search(*tree_, CPoint2(query[0], query[1]), 1);
        return search.begin()->first.second;
    }

private:
    std::vector<CPointWithIndex2> points_;
    std::unique_ptr<CSearchTree2> tree_;
};

static std::vector<CPoly2> build_raw_label_polygons(const Interface2D& iface) {
    const int k = iface.points_per_panel();
    const int Np = iface.num_panels();
    const int Nc = iface.num_components();

    std::vector<CPoly2> polys(Nc);
    for (int p = 0; p < Np; ++p) {
        int c = iface.panel_components()(p);
        for (int qi = 0; qi < k; ++qi) {
            int idx = iface.point_index(p, qi);
            polys[c].push_back(CPoint2(iface.points()(idx, 0), iface.points()(idx, 1)));
        }
    }
    return polys;
}

static std::vector<CPoly2> build_label_polygons(const Interface2D& iface) {
    const int k = iface.points_per_panel();
    if (k != 3)
        return build_raw_label_polygons(iface);

    constexpr double gl_nodes[3] = {
        -0.77459666924148337704,
         0.0,
         0.77459666924148337704
    };
    constexpr double lobatto_nodes[3] = {-1.0, 0.0, 1.0};

    const double* nodes = nullptr;
    switch (iface.panel_node_layout()) {
    case PanelNodeLayout2D::LegacyGaussLegendre:
        nodes = gl_nodes;
        break;
    case PanelNodeLayout2D::QuadraticLagrange:
        nodes = lobatto_nodes;
        break;
    case PanelNodeLayout2D::Raw:
        return build_raw_label_polygons(iface);
    }

    const int Np = iface.num_panels();
    const int Nc = iface.num_components();
    constexpr std::array<double, 4> internal_s = {-0.6, -0.2, 0.2, 0.6};

    std::vector<std::vector<int>> panels_by_component(Nc);
    for (int p = 0; p < Np; ++p)
        panels_by_component[iface.panel_components()(p)].push_back(p);

    std::vector<CPoly2> polys(Nc);
    for (int comp = 0; comp < Nc; ++comp) {
        const auto& panels = panels_by_component[comp];
        if (panels.empty())
            continue;

        std::vector<Point2> shared_starts(panels.size());
        for (int i = 0; i < static_cast<int>(panels.size()); ++i) {
            const int prev_panel = panels[(i + static_cast<int>(panels.size()) - 1)
                                          % static_cast<int>(panels.size())];
            const int curr_panel = panels[i];
            shared_starts[i] = midpoint(quadratic_panel_point(iface, prev_panel, nodes, 1.0),
                                        quadratic_panel_point(iface, curr_panel, nodes, -1.0));
        }

        for (int i = 0; i < static_cast<int>(panels.size()); ++i) {
            const int panel = panels[i];
            push_cgal_point(polys[comp], shared_starts[i]);
            for (double s : internal_s)
                push_cgal_point(polys[comp], quadratic_panel_point(iface, panel, nodes, s));
        }
    }

    return polys;
}

// ============================================================================
// Pimpl
// ============================================================================
struct GridPair2D::Impl {
    std::vector<int>    closest_bulk_node;  // [iface_pt] → grid flat index
    std::vector<int>    closest_iface_pt;   // [bulk_node] → iface pt index
    std::vector<double> min_iface_dist;     // [bulk_node] → Euclidean dist to nearest iface pt
    std::vector<int>    domain_label_vec;   // [bulk_node] → 0=exterior, c+1=interior of component c
};

// ============================================================================
// Constructor
// ============================================================================
GridPair2D::GridPair2D(const CartesianGrid2D& grid, const Interface2D& iface)
    : grid_(grid), interface_(iface), impl_(std::make_unique<Impl>())
{
    const bool profile = profile_grid_pair_2d();
    const bool detail_profile = profile_grid_pair_2d_detail();
    const ProfileClock2D::time_point t_total_start = ProfileClock2D::now();
    const int Nq = iface.num_points();
    const int N  = grid.num_dofs();
    const int Nc = iface.num_components();

    // ------------------------------------------------------------------
    // 1. closest_bulk_node[q]: nearest grid DOF to each interface point.
    //    Computed analytically from grid layout — O(Nq).
    // ------------------------------------------------------------------
    impl_->closest_bulk_node.resize(Nq);
    for (int q = 0; q < Nq; ++q)
        impl_->closest_bulk_node[q] =
            nearest_node(grid, iface.points()(q, 0), iface.points()(q, 1));
    const double t_closest_bulk = seconds_since(t_total_start);

    // ------------------------------------------------------------------
    // 2. closest_iface_pt[n] and min_iface_dist[n].
    //
    //    The closest interface point remains a DOF index for compatibility.
    //    Narrow-band distances and active curved-panel domain labels use the
    //    shared P2 panel-center samples, mirroring the 3D P2 GridPair path.
    // ------------------------------------------------------------------
    const ProfileClock2D::time_point t_sample_start = ProfileClock2D::now();
    const std::vector<DistanceSample2D> distance_samples =
        distance_samples_2d(iface);

    std::vector<Point2> iface_points;
    iface_points.reserve(Nq);
    for (int q = 0; q < Nq; ++q)
        iface_points.push_back({iface.points()(q, 0), iface.points()(q, 1)});

    std::vector<Point2> sample_points;
    sample_points.reserve(distance_samples.size());
    for (const auto& sample : distance_samples)
        sample_points.push_back(sample.point);

    const NearestPointCloud2D iface_cloud(iface_points);
    const NearestPointCloud2D sample_cloud(sample_points);
    const double t_sample_setup = seconds_since(t_sample_start);

    impl_->closest_iface_pt.resize(N, 0);
    impl_->min_iface_dist.resize(N, std::numeric_limits<double>::infinity());
    std::vector<int> closest_sample_idx(N, 0);

    const ProfileClock2D::time_point t_distance_start = ProfileClock2D::now();
    for (int n = 0; n < N; ++n) {
        const auto c = grid.coord(n);
        const Point2 pt{c[0], c[1]};

        const int q = iface_cloud.nearest(pt);
        impl_->closest_iface_pt[n] = q;

        const int sidx = sample_cloud.nearest(pt);
        impl_->min_iface_dist[n] =
            std::sqrt(squared_distance(pt, distance_samples[sidx].point));
        closest_sample_idx[n] = sidx;
    }
    const double t_distance = seconds_since(t_distance_start);

    // ------------------------------------------------------------------
    // 3. domain_label_vec[n]: 0=exterior, comp+1=interior of component comp.
    //
    // Active P2 panel layouts use the same nearest-surface-sample normal-sign
    // classifier as the 3D P2 path. Raw layouts keep the older polygon
    // fallback because their geometry does not define a parametric curve
    // between panel points.
    // ------------------------------------------------------------------

    impl_->domain_label_vec.resize(N, 0);
    const bool normal_labels = use_normal_domain_labels(iface);
    double t_polygon = 0.0;
    std::size_t polygon_vertices = 0;
    const ProfileClock2D::time_point t_label_start = ProfileClock2D::now();
    int label_queries = 0;
    int label_inside = 0;
    int label_boundary = 0;
    int label_outside = 0;
    double t_label_coord = 0.0;
    double t_label_query = 0.0;
    double t_label_assign = 0.0;

    if (normal_labels) {
        constexpr double kBoundaryTol = 1.0e-14;
        for (int n = 0; n < N; ++n) {
            const ProfileClock2D::time_point t_coord_start = ProfileClock2D::now();
            const auto c = grid.coord(n);
            const Point2 pt{c[0], c[1]};
            if (detail_profile)
                t_label_coord += seconds_since(t_coord_start);

            ++label_queries;
            const ProfileClock2D::time_point t_query_start = ProfileClock2D::now();
            const auto& sample = distance_samples[closest_sample_idx[n]];
            const double signed_normal_distance =
                (pt[0] - sample.point[0]) * sample.normal[0]
                + (pt[1] - sample.point[1]) * sample.normal[1];
            if (detail_profile)
                t_label_query += seconds_since(t_query_start);

            const ProfileClock2D::time_point t_assign_start = ProfileClock2D::now();
            if (detail_profile) {
                if (std::abs(signed_normal_distance) <= kBoundaryTol)
                    ++label_boundary;
                else if (signed_normal_distance < 0.0)
                    ++label_inside;
                else
                    ++label_outside;
            }
            if (signed_normal_distance <= kBoundaryTol)
                impl_->domain_label_vec[n] = sample.component + 1;
            if (detail_profile)
                t_label_assign += seconds_since(t_assign_start);
        }
    } else {
        const ProfileClock2D::time_point t_polygon_start = ProfileClock2D::now();
        std::vector<CPoly2> polys = build_label_polygons(iface);
        t_polygon = seconds_since(t_polygon_start);
        for (const auto& poly : polys)
            polygon_vertices += poly.size();

        for (int n = 0; n < N; ++n) {
            const ProfileClock2D::time_point t_coord_start = ProfileClock2D::now();
            auto    c = grid.coord(n);
            CPoint2 p(c[0], c[1]);
            if (detail_profile)
                t_label_coord += seconds_since(t_coord_start);
            for (int comp = 0; comp < Nc; ++comp) {
                if (polys[comp].size() < 3)
                    continue;
                ++label_queries;
                const ProfileClock2D::time_point t_query_start = ProfileClock2D::now();
                auto side = polys[comp].bounded_side(p);
                if (detail_profile)
                    t_label_query += seconds_since(t_query_start);
                const ProfileClock2D::time_point t_assign_start = ProfileClock2D::now();
                if (detail_profile) {
                    if (side == CGAL::ON_BOUNDED_SIDE)
                        ++label_inside;
                    else if (side == CGAL::ON_BOUNDARY)
                        ++label_boundary;
                    else
                        ++label_outside;
                }
                if (side == CGAL::ON_BOUNDED_SIDE || side == CGAL::ON_BOUNDARY) {
                    impl_->domain_label_vec[n] = comp + 1;
                    if (detail_profile)
                        t_label_assign += seconds_since(t_assign_start);
                    break;
                }
                if (detail_profile)
                    t_label_assign += seconds_since(t_assign_start);
            }
        }
    }
    const double t_label = seconds_since(t_label_start);

    if (profile) {
        const double total = seconds_since(t_total_start);
        std::printf("      GridPair2D Ngrid=%d Niface=%d samples=%zu components=%d label_mode=%s polygon_vertices=%zu label_queries=%d\n",
                    N, Nq, distance_samples.size(), Nc,
                    normal_labels ? "nearest-normal" : "polygon",
                    polygon_vertices, label_queries);
        std::printf("        closest_bulk %.6fs sample_setup %.6fs nearest_queries %.6fs build_polygons %.6fs domain_labels %.6fs total %.6fs\n",
                    t_closest_bulk, t_sample_setup, t_distance, t_polygon,
                    t_label, total);
    }
    if (detail_profile) {
        std::printf("        domain_label_detail coord_point %.6fs classify_query %.6fs assign %.6fs inside=%d boundary=%d outside=%d\n",
                    t_label_coord, t_label_query, t_label_assign,
                    label_inside, label_boundary, label_outside);
    }
}

GridPair2D::~GridPair2D() = default;

// ============================================================================
// Query methods
// ============================================================================
int GridPair2D::closest_bulk_node(int q) const {
    return impl_->closest_bulk_node[q];
}

int GridPair2D::closest_interface_point(int n) const {
    return impl_->closest_iface_pt[n];
}

int GridPair2D::domain_label(int n) const {
    return impl_->domain_label_vec[n];
}

bool GridPair2D::is_near_interface(int n, double radius) const {
    return impl_->min_iface_dist[n] < radius;
}

std::vector<int> GridPair2D::near_interface_nodes(double radius) const {
    int N = grid_.num_dofs();
    std::vector<int> result;
    result.reserve(N / 8);
    for (int n = 0; n < N; ++n)
        if (impl_->min_iface_dist[n] < radius)
            result.push_back(n);
    return result;
}

std::vector<int> GridPair2D::near_interface_points(int n, double radius) const {
    auto   c  = grid_.coord(n);
    double cx = c[0], cy = c[1];
    int Nq = interface_.num_points();
    std::vector<int> result;
    for (int q = 0; q < Nq; ++q) {
        double dx = cx - interface_.points()(q, 0);
        double dy = cy - interface_.points()(q, 1);
        if (std::sqrt(dx*dx + dy*dy) < radius)
            result.push_back(q);
    }
    return result;
}

NarrowBandProjection2D GridPair2D::project_near_interface_nodes(double radius) const
{
    return project_p2_near_interface_nodes_2d(*this, radius);
}

NarrowBandProjection2D GridPair2D::project_grid_nodes_to_interface(
    const std::vector<int>& bulk_node_indices) const
{
    return project_p2_grid_nodes_to_interface_2d(*this, bulk_node_indices);
}

} // namespace kfbim
