#include "grid_pair_2d.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <queue>

namespace kfbim {

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

// 2D winding-number test — true if (px,py) is strictly inside the polygon.
static bool winding_contains(double px, double py,
                              const std::vector<std::array<double, 2>>& poly)
{
    int wn = 0;
    int n  = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        double x0 = poly[i][0],        y0 = poly[i][1];
        double x1 = poly[(i+1)%n][0],  y1 = poly[(i+1)%n][1];
        if (y0 <= py) {
            if (y1 > py) {
                // upward crossing — is point left of the edge?
                if ((x1-x0)*(py-y0) - (px-x0)*(y1-y0) > 0) wn++;
            }
        } else {
            if (y1 <= py) {
                // downward crossing — is point right of the edge?
                if ((x1-x0)*(py-y0) - (px-x0)*(y1-y0) < 0) wn--;
            }
        }
    }
    return wn != 0;
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
    const int k  = iface.points_per_panel();
    const int Np = iface.num_panels();
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
    // Strategy:
    //   a) Narrow band (min_iface_dist < band_radius): run the winding-number
    //      test exactly — these nodes are near the interface and need geometric
    //      classification.
    //   b) Remaining nodes: BFS flood-fill from the labeled band.  No interface
    //      edge can connect two non-band nodes when band_radius >= h, so labels
    //      propagate safely through the grid graph without any geometric test.
    // ------------------------------------------------------------------

    // Build per-component polygons (used only for band nodes).
    std::vector<std::vector<std::array<double, 2>>> polys(Nc);
    for (int p = 0; p < Np; ++p) {
        int c = iface.panel_components()(p);
        for (int qi = 0; qi < k; ++qi) {
            int idx = p * k + qi;
            polys[c].push_back({iface.points()(idx, 0), iface.points()(idx, 1)});
        }
    }

    // Band radius: must be >= max grid spacing so that no interface edge
    // connects two non-band nodes.  Use 2x for safety with discrete quadrature.
    const double h_max      = std::max(grid.spacing()[0], grid.spacing()[1]);
    const double band_radius = 2.0 * h_max;

    constexpr int UNLABELED = -1;
    impl_->domain_label_vec.resize(N, UNLABELED);

    // (a) Label band nodes with winding-number test; seed BFS queue.
    std::queue<int> bfs;
    for (int n = 0; n < N; ++n) {
        if (impl_->min_iface_dist[n] < band_radius) {
            auto  c   = grid.coord(n);
            int   lbl = 0;
            for (int comp = 0; comp < Nc; ++comp) {
                if (winding_contains(c[0], c[1], polys[comp])) {
                    lbl = comp + 1;
                    break;
                }
            }
            impl_->domain_label_vec[n] = lbl;
            bfs.push(n);
        }
    }

    // (b) BFS flood-fill: propagate labels to unlabeled non-band nodes.
    while (!bfs.empty()) {
        int n = bfs.front(); bfs.pop();
        int lbl = impl_->domain_label_vec[n];
        for (int nb : grid.neighbors(n)) {
            if (nb >= 0 && impl_->domain_label_vec[nb] == UNLABELED) {
                impl_->domain_label_vec[nb] = lbl;
                bfs.push(nb);
            }
        }
    }

    // Fallback: any node still unlabeled (disconnected island — degenerate mesh)
    // gets label 0.
    for (int n = 0; n < N; ++n)
        if (impl_->domain_label_vec[n] == UNLABELED)
            impl_->domain_label_vec[n] = 0;
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
