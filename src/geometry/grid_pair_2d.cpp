#include "grid_pair_2d.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

namespace kfbim {

using K2      = CGAL::Exact_predicates_inexact_constructions_kernel;
using CPoint2 = K2::Point_2;
using CPoly2  = CGAL::Polygon_2<K2>;
using Point2  = std::array<double, 2>;

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
    case PanelNodeLayout2D::ChebyshevLobatto:
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

    // ------------------------------------------------------------------
    // 2. closest_iface_pt[n] and min_iface_dist[n]:
    //    Brute-force O(N * Nq) — precomputed once so all queries are O(1).
    // ------------------------------------------------------------------
    impl_->closest_iface_pt.resize(N, 0);
    impl_->min_iface_dist.resize(N, std::numeric_limits<double>::infinity());

    for (int n = 0; n < N; ++n) {
        auto  c  = grid.coord(n);
        double cx = c[0], cy = c[1];
        for (int q = 0; q < Nq; ++q) {
            double dx = cx - iface.points()(q, 0);
            double dy = cy - iface.points()(q, 1);
            double d  = std::sqrt(dx*dx + dy*dy);
            if (d < impl_->min_iface_dist[n]) {
                impl_->min_iface_dist[n]  = d;
                impl_->closest_iface_pt[n] = q;
            }
        }
    }

    // ------------------------------------------------------------------
    // 3. domain_label_vec[n]: 0=exterior, comp+1=interior of component comp.
    //
    // Build one CGAL::Polygon_2 per component for labels.  Three-point panels
    // are interpreted as quadratic Gauss-Legendre panel curves and oversampled
    // for labeling only; closest-point and narrow-band queries still use the
    // original interface points above.  ON_BOUNDARY -> interior.
    // ------------------------------------------------------------------

    std::vector<CPoly2> polys = build_label_polygons(iface);

    impl_->domain_label_vec.resize(N, 0);
    for (int n = 0; n < N; ++n) {
        auto    c = grid.coord(n);
        CPoint2 p(c[0], c[1]);
        for (int comp = 0; comp < Nc; ++comp) {
            if (polys[comp].size() < 3)
                continue;
            auto side = polys[comp].bounded_side(p);
            if (side == CGAL::ON_BOUNDED_SIDE || side == CGAL::ON_BOUNDARY) {
                impl_->domain_label_vec[n] = comp + 1;
                break;
            }
        }
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

} // namespace kfbim
