#include "grid_pair_3d.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <unordered_map>

namespace kfbim {

using K3      = CGAL::Exact_predicates_inexact_constructions_kernel;
using CPoint3 = K3::Point_3;
using CMesh3  = CGAL::Surface_mesh<CPoint3>;
using CSideOf = CGAL::Side_of_triangle_mesh<CMesh3, K3>;

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
    // 3. domain_label_vec[n]: 0=exterior, comp+1=interior of component comp.
    //
    // Build one CGAL::Surface_mesh per component and query every grid node
    // via CGAL::Side_of_triangle_mesh (AABB-accelerated).
    // ON_BOUNDARY → interior (nodes exactly on the surface are interior).
    // ------------------------------------------------------------------

    impl_->domain_label_vec.resize(N, 0);
    for (int comp = 0; comp < Nc; ++comp) {
        CMesh3 mesh;
        std::unordered_map<int, CMesh3::Vertex_index> vmap;
        auto get_or_add = [&](int gv) {
            auto [it, inserted] = vmap.emplace(gv, CMesh3::Vertex_index{});
            if (inserted)
                it->second = mesh.add_vertex(CPoint3(
                    iface.vertices()(gv, 0),
                    iface.vertices()(gv, 1),
                    iface.vertices()(gv, 2)));
            return it->second;
        };
        for (int p = 0; p < Np; ++p) {
            if (iface.panel_components()(p) != comp) continue;
            mesh.add_face(get_or_add(iface.panels()(p, 0)),
                          get_or_add(iface.panels()(p, 1)),
                          get_or_add(iface.panels()(p, 2)));
        }

        CSideOf side_of(mesh);
        for (int n = 0; n < N; ++n) {
            if (impl_->domain_label_vec[n] != 0) continue;
            auto   c    = grid.coord(n);
            auto   side = side_of(CPoint3(c[0], c[1], c[2]));
            if (side == CGAL::ON_BOUNDED_SIDE || side == CGAL::ON_BOUNDARY)
                impl_->domain_label_vec[n] = comp + 1;
        }
    }
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
