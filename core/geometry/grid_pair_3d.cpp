#include "grid_pair_3d.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <queue>

namespace kfbim {

// ============================================================================
// Internal helpers
// ============================================================================

// First-DOF coordinate per axis for a given 3D layout.
static std::array<double, 3> dof_first(const CartesianGrid3D& g) {
    auto o = g.origin(), h = g.spacing();
    switch (g.layout()) {
    case DofLayout3D::Node:
        return {o[0],            o[1],            o[2]           };
    case DofLayout3D::CellCenter:
        return {o[0]+0.5*h[0],   o[1]+0.5*h[1],   o[2]+0.5*h[2]  };
    case DofLayout3D::FaceX:
        return {o[0],            o[1]+0.5*h[1],   o[2]+0.5*h[2]  };
    case DofLayout3D::FaceY:
        return {o[0]+0.5*h[0],   o[1],            o[2]+0.5*h[2]  };
    case DofLayout3D::FaceZ:
        return {o[0]+0.5*h[0],   o[1]+0.5*h[1],   o[2]           };
    }
    return {o[0], o[1], o[2]};
}

// Nearest bulk-node flat index to physical position (x, y, z).
static int nearest_node(const CartesianGrid3D& g, double x, double y, double z) {
    auto f = dof_first(g);
    auto h = g.spacing();
    auto d = g.dof_dims();
    int nx = d[0], ny = d[1], nz = d[2];
    int i = static_cast<int>(std::lround((x - f[0]) / h[0]));
    int j = static_cast<int>(std::lround((y - f[1]) / h[1]));
    int k = static_cast<int>(std::lround((z - f[2]) / h[2]));
    i = std::max(0, std::min(nx - 1, i));
    j = std::max(0, std::min(ny - 1, j));
    k = std::max(0, std::min(nz - 1, k));
    return k * (nx * ny) + j * nx + i;
}

// Möller–Trumbore ray–triangle intersection.
// Ray origin (ox, oy, oz), direction D = (0, 1e-7, 1) — slightly off pure +z
// to avoid exact edge/vertex alignment with axis-aligned triangle meshes,
// which causes double-counting on shared edges and corrupts parity.
// Returns true if the intersection parameter t > eps (ray hits above origin).
static bool ray_tri_hit(double ox, double oy, double oz,
                         double v0x, double v0y, double v0z,
                         double v1x, double v1y, double v1z,
                         double v2x, double v2y, double v2z)
{
    const double eps = 1e-10;
    // Slightly off-axis: Dy breaks the x=y symmetry that causes shared-edge hits.
    const double Dx = 0.0, Dy = 1e-7, Dz = 1.0;

    double e1x = v1x-v0x, e1y = v1y-v0y, e1z = v1z-v0z;
    double e2x = v2x-v0x, e2y = v2y-v0y, e2z = v2z-v0z;

    // h = D × e2
    double hx = Dy*e2z - Dz*e2y;
    double hy = Dz*e2x - Dx*e2z;
    double hz = Dx*e2y - Dy*e2x;

    double a = e1x*hx + e1y*hy + e1z*hz;
    if (std::fabs(a) < eps) return false;
    double f = 1.0 / a;

    double sx = ox-v0x, sy = oy-v0y, sz = oz-v0z;
    double u = f * (sx*hx + sy*hy + sz*hz);
    if (u < 0.0 || u > 1.0) return false;

    double qx = sy*e1z - sz*e1y;
    double qy = sz*e1x - sx*e1z;
    double qz = sx*e1y - sy*e1x;
    double v = f * (Dx*qx + Dy*qy + Dz*qz);
    if (v < 0.0 || u + v > 1.0) return false;

    double t = f * (e2x*qx + e2y*qy + e2z*qz);
    return t > eps;
}

// ============================================================================
// Pimpl
// ============================================================================
struct GridPair3D::Impl {
    std::vector<int>    closest_bulk_node;
    std::vector<int>    closest_iface_pt;
    std::vector<double> min_iface_dist;
    std::vector<int>    domain_label_vec;
};

// ============================================================================
// Constructor
// ============================================================================
GridPair3D::GridPair3D(const CartesianGrid3D& grid, const Interface3D& iface)
    : grid_(grid), interface_(iface), impl_(std::make_unique<Impl>())
{
    const int Nq = iface.num_points();
    const int N  = grid.num_dofs();
    const int k  = iface.points_per_panel();
    const int Np = iface.num_panels();
    const int Nc = iface.num_components();

    // ------------------------------------------------------------------
    // 1. closest_bulk_node[q]: nearest grid DOF to each interface point.
    // ------------------------------------------------------------------
    impl_->closest_bulk_node.resize(Nq);
    for (int q = 0; q < Nq; ++q)
        impl_->closest_bulk_node[q] = nearest_node(
            grid,
            iface.points()(q, 0), iface.points()(q, 1), iface.points()(q, 2));

    // ------------------------------------------------------------------
    // 2. closest_iface_pt[n] and min_iface_dist[n]:
    //    Brute-force O(N * Nq).
    // ------------------------------------------------------------------
    impl_->closest_iface_pt.resize(N, 0);
    impl_->min_iface_dist.resize(N, std::numeric_limits<double>::infinity());

    for (int n = 0; n < N; ++n) {
        auto   c  = grid.coord(n);
        double cx = c[0], cy = c[1], cz = c[2];
        for (int q = 0; q < Nq; ++q) {
            double dx = cx - iface.points()(q, 0);
            double dy = cy - iface.points()(q, 1);
            double dz = cz - iface.points()(q, 2);
            double d  = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d < impl_->min_iface_dist[n]) {
                impl_->min_iface_dist[n]   = d;
                impl_->closest_iface_pt[n] = q;
            }
        }
    }

    // ------------------------------------------------------------------
    // 3. domain_label_vec[n]: narrow-band ray-cast + BFS flood-fill.
    //
    // (a) Band nodes (min_iface_dist < band_radius): ray-cast exactly.
    // (b) Remaining nodes: BFS from labeled band — safe because no
    //     interface edge can connect two non-band nodes when
    //     band_radius >= h_max.
    // ------------------------------------------------------------------
    const double h_max = std::max({grid.spacing()[0], grid.spacing()[1], grid.spacing()[2]});

    // Max distance from any quadrature point to a vertex of its panel.
    // Accounts for coarse meshes where centroids can be far from corners.
    double max_panel_radius = 0.0;
    for (int p = 0; p < Np; ++p) {
        for (int vi = 0; vi < 3; ++vi) {
            int vidx = iface.panels()(p, vi);
            double dx = iface.points()(p,0) - iface.vertices()(vidx,0);
            double dy = iface.points()(p,1) - iface.vertices()(vidx,1);
            double dz = iface.points()(p,2) - iface.vertices()(vidx,2);
            max_panel_radius = std::max(max_panel_radius, std::sqrt(dx*dx+dy*dy+dz*dz));
        }
    }
    const double band_radius = 2.0 * h_max + max_panel_radius;

    constexpr int UNLABELED = -1;
    impl_->domain_label_vec.resize(N, UNLABELED);

    std::queue<int> bfs;
    for (int n = 0; n < N; ++n) {
        if (impl_->min_iface_dist[n] < band_radius) {
            auto   c  = grid.coord(n);
            double px = c[0], py = c[1], pz = c[2];

            std::vector<int> hits(Nc, 0);
            for (int p = 0; p < Np; ++p) {
                int va = iface.panels()(p, 0);
                int vb = iface.panels()(p, 1);
                int vc = iface.panels()(p, 2);
                if (ray_tri_hit(px, py, pz,
                                iface.vertices()(va, 0), iface.vertices()(va, 1), iface.vertices()(va, 2),
                                iface.vertices()(vb, 0), iface.vertices()(vb, 1), iface.vertices()(vb, 2),
                                iface.vertices()(vc, 0), iface.vertices()(vc, 1), iface.vertices()(vc, 2)))
                    hits[iface.panel_components()(p)]++;
            }

            int lbl = 0;
            for (int comp = 0; comp < Nc; ++comp) {
                if (hits[comp] % 2 == 1) { lbl = comp + 1; break; }
            }
            impl_->domain_label_vec[n] = lbl;
            bfs.push(n);
        }
    }

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

    for (int n = 0; n < N; ++n)
        if (impl_->domain_label_vec[n] == UNLABELED)
            impl_->domain_label_vec[n] = 0;
}

GridPair3D::~GridPair3D() = default;

// ============================================================================
// Query methods
// ============================================================================
int GridPair3D::closest_bulk_node(int q) const {
    return impl_->closest_bulk_node[q];
}

int GridPair3D::closest_interface_point(int n) const {
    return impl_->closest_iface_pt[n];
}

int GridPair3D::domain_label(int n) const {
    return impl_->domain_label_vec[n];
}

bool GridPair3D::is_near_interface(int n, double radius) const {
    return impl_->min_iface_dist[n] < radius;
}

std::vector<int> GridPair3D::near_interface_nodes(double radius) const {
    int N = grid_.num_dofs();
    std::vector<int> result;
    result.reserve(N / 8);
    for (int n = 0; n < N; ++n)
        if (impl_->min_iface_dist[n] < radius)
            result.push_back(n);
    return result;
}

std::vector<int> GridPair3D::near_interface_points(int n, double radius) const {
    auto   c  = grid_.coord(n);
    double cx = c[0], cy = c[1], cz = c[2];
    int Nq = interface_.num_points();
    std::vector<int> result;
    for (int q = 0; q < Nq; ++q) {
        double dx = cx - interface_.points()(q, 0);
        double dy = cy - interface_.points()(q, 1);
        double dz = cz - interface_.points()(q, 2);
        if (std::sqrt(dx*dx + dy*dy + dz*dz) < radius)
            result.push_back(q);
    }
    return result;
}

} // namespace kfbim
