#include "grid_pair_3d.hpp"
#include "p2_surface_3d.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kfbim {

using K3      = CGAL::Exact_predicates_inexact_constructions_kernel;
using CPoint3 = K3::Point_3;
using CMesh3  = CGAL::Surface_mesh<CPoint3>;
using CSideOf = CGAL::Side_of_triangle_mesh<CMesh3, K3>;
using CPointWithIndex3 = std::pair<CPoint3, int>;
using CSearchBaseTraits3 = CGAL::Search_traits_3<K3>;
using CSearchTraits3 =
    CGAL::Search_traits_adapter<CPointWithIndex3,
                                CGAL::First_of_pair_property_map<CPointWithIndex3>,
                                CSearchBaseTraits3>;
using CNeighborSearch3 = CGAL::Orthogonal_k_neighbor_search<CSearchTraits3>;
using CSearchTree3 = CNeighborSearch3::Tree;

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

struct DistanceSample3D {
    Eigen::Vector3d point;
    Eigen::Vector3d normal;
    int             component;
};

static std::vector<int> point_components_3d(const Interface3D& iface)
{
    std::vector<int> components(iface.num_points(), 0);
    for (int p = 0; p < iface.num_panels(); ++p) {
        for (int q = 0; q < iface.points_per_panel(); ++q)
            components[iface.point_index(p, q)] = iface.panel_components()[p];
    }
    return components;
}

static std::vector<DistanceSample3D> distance_samples_3d(const Interface3D& iface)
{
    std::vector<DistanceSample3D> samples;
    samples.reserve(static_cast<std::size_t>(iface.num_points())
                    + 16 * static_cast<std::size_t>(iface.num_panels()));
    const std::vector<int> point_components = point_components_3d(iface);
    for (int q = 0; q < iface.num_points(); ++q) {
        samples.push_back({iface.points().row(q).transpose(),
                           iface.normals().row(q).transpose(),
                           point_components[q]});
    }

    if (iface.panel_node_layout() == PanelNodeLayout3D::QuadraticLagrange) {
        for (int p = 0; p < iface.num_panels(); ++p) {
            for (const auto& ref_tri : geometry3d::subdivided_reference_triangles()) {
                const std::array<Eigen::Vector3d, 3> tri = {
                    geometry3d::panel_point(iface, p, ref_tri.bary[0]),
                    geometry3d::panel_point(iface, p, ref_tri.bary[1]),
                    geometry3d::panel_point(iface, p, ref_tri.bary[2])};
                const Eigen::Vector3d center = (tri[0] + tri[1] + tri[2]) / 3.0;
                Eigen::Vector3d normal = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
                const double len = normal.norm();
                if (len > 1.0e-14)
                    normal /= len;
                else
                    normal = iface.normals().row(iface.point_index(p, 0)).transpose();
                const Eigen::Vector3d ref =
                    iface.normals().row(iface.point_index(p, 0)).transpose();
                if (normal.dot(ref) < 0.0)
                    normal = -normal;
                samples.push_back({center, normal, iface.panel_components()[p]});
            }
        }
    }

    return samples;
}

class NearestPointCloud3D {
public:
    explicit NearestPointCloud3D(const std::vector<Eigen::Vector3d>& points)
    {
        if (points.empty())
            throw std::invalid_argument("NearestPointCloud3D requires at least one point");

        points_.reserve(points.size());
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            const Eigen::Vector3d& p = points[i];
            points_.emplace_back(CPoint3(p[0], p[1], p[2]), i);
        }
        tree_ = std::make_unique<CSearchTree3>(points_.begin(), points_.end());
    }

    int nearest(Eigen::Vector3d query) const
    {
        CNeighborSearch3 search(*tree_, CPoint3(query[0], query[1], query[2]), 1);
        return search.begin()->first.second;
    }

    std::vector<int> nearest_k(Eigen::Vector3d query, int k) const
    {
        if (k <= 0)
            throw std::invalid_argument("NearestPointCloud3D nearest_k requires k > 0");
        k = std::min(k, static_cast<int>(points_.size()));

        CNeighborSearch3 search(*tree_, CPoint3(query[0], query[1], query[2]), k);
        std::vector<int> result;
        result.reserve(k);
        for (auto it = search.begin(); it != search.end(); ++it)
            result.push_back(it->first.second);
        return result;
    }

private:
    std::vector<CPointWithIndex3> points_;
    std::unique_ptr<CSearchTree3> tree_;
};

static double squared_distance(Eigen::Vector3d a, Eigen::Vector3d b)
{
    return (a - b).squaredNorm();
}

struct ProjectionSeed3D {
    Eigen::Vector3d point;
    Eigen::Vector3d barycentric;
    int             panel;
    int             component;
};

struct ProjectionEval3D {
    Eigen::Vector3d point;
    Eigen::Vector3d normal;
    Eigen::Vector2d tangential_equations;
    double          distance;
    double          signed_distance;
    double          tangential_residual;
    double          merit;
};

static Eigen::Vector3d grid_point(const CartesianGrid3D& grid, int idx)
{
    const auto c = grid.coord(idx);
    return {c[0], c[1], c[2]};
}

static bool finite_vector(Eigen::Vector2d v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]);
}

static bool reference_triangle_contains(Eigen::Vector3d bary, double tol)
{
    return bary[0] >= -tol
        && bary[1] >= -tol
        && bary[2] >= -tol
        && std::abs(bary.sum() - 1.0) <= tol;
}

static Eigen::Vector2d clamp_uv_to_reference_triangle(Eigen::Vector2d uv)
{
    if (!finite_vector(uv))
        return Eigen::Vector2d::Zero();

    const Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
    if (reference_triangle_contains(bary, 0.0))
        return uv;

    auto best = Eigen::Vector2d(0.0, std::max(0.0, std::min(1.0, uv[1])));
    double best_dist2 = (best - uv).squaredNorm();

    auto try_candidate = [&](Eigen::Vector2d candidate) {
        const double dist2 = (candidate - uv).squaredNorm();
        if (dist2 < best_dist2) {
            best = candidate;
            best_dist2 = dist2;
        }
    };

    try_candidate({std::max(0.0, std::min(1.0, uv[0])), 0.0});

    double edge_u = 0.5 * (uv[0] - uv[1] + 1.0);
    edge_u = std::max(0.0, std::min(1.0, edge_u));
    try_candidate({edge_u, 1.0 - edge_u});

    return best;
}

static ProjectionEval3D projection_eval(const Interface3D&  iface,
                                        int                 panel,
                                        Eigen::Vector3d     target,
                                        Eigen::Vector2d     uv)
{
    const Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
    const Eigen::Vector3d point = geometry3d::panel_point(iface, panel, bary);
    const Eigen::Vector3d normal =
        geometry3d::panel_oriented_normal(iface, panel, bary);
    const Eigen::Vector3d r = point - target;
    const Eigen::Vector3d Xu = geometry3d::panel_tangent_u(iface, panel, bary);
    const Eigen::Vector3d Xv = geometry3d::panel_tangent_v(iface, panel, bary);

    ProjectionEval3D eval;
    eval.point = point;
    eval.normal = normal;
    eval.tangential_equations = {r.dot(Xu), r.dot(Xv)};
    eval.distance = r.norm();
    eval.signed_distance = (target - point).dot(normal);
    eval.merit = eval.tangential_equations.squaredNorm();

    Eigen::Matrix2d metric;
    metric << Xu.dot(Xu), Xu.dot(Xv),
              Xv.dot(Xu), Xv.dot(Xv);
    const double det = metric.determinant();
    if (std::abs(det) > 1.0e-28) {
        const Eigen::Vector2d coeff =
            metric.fullPivLu().solve(eval.tangential_equations);
        const Eigen::Vector3d tangential = coeff[0] * Xu + coeff[1] * Xv;
        eval.tangential_residual = tangential.norm();
    } else {
        eval.tangential_residual = std::sqrt(eval.merit);
    }

    return eval;
}

static SurfaceProjection3D make_surface_projection(const Interface3D&  iface,
                                                   int                 grid_node,
                                                   const ProjectionSeed3D& seed,
                                                   Eigen::Vector3d     target,
                                                   Eigen::Vector2d     uv,
                                                   int                 iterations,
                                                   bool                converged)
{
    Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
    if (!reference_triangle_contains(bary, 1.0e-8)) {
        uv = clamp_uv_to_reference_triangle(uv);
        bary = geometry3d::barycentric_from_uv(uv);
        converged = false;
    }

    const ProjectionEval3D eval =
        projection_eval(iface, seed.panel, target, uv);

    SurfaceProjection3D projection;
    projection.grid_node = grid_node;
    projection.panel = seed.panel;
    projection.component = seed.component;
    projection.barycentric = bary;
    projection.point = eval.point;
    projection.normal = eval.normal;
    projection.signed_distance = eval.signed_distance;
    projection.distance = eval.distance;
    projection.tangential_residual = eval.tangential_residual;
    projection.iterations = iterations;
    projection.converged = converged;
    return projection;
}

static SurfaceProjection3D project_from_seed(const Interface3D&     iface,
                                             int                    grid_node,
                                             Eigen::Vector3d        target,
                                             const ProjectionSeed3D& seed)
{
    constexpr int kMaxIterations = 30;
    constexpr double kResidualTol = 1.0e-11;
    constexpr double kStepTol = 1.0e-13;

    Eigen::Vector2d uv = geometry3d::uv_from_barycentric(seed.barycentric);
    int iterations = 0;
    bool converged = false;

    const Eigen::Vector3d Xuu = geometry3d::panel_second_uu(iface, seed.panel);
    const Eigen::Vector3d Xuv = geometry3d::panel_second_uv(iface, seed.panel);
    const Eigen::Vector3d Xvv = geometry3d::panel_second_vv(iface, seed.panel);

    for (int iter = 0; iter < kMaxIterations; ++iter) {
        const Eigen::Vector3d bary = geometry3d::barycentric_from_uv(uv);
        const ProjectionEval3D eval =
            projection_eval(iface, seed.panel, target, uv);
        if (eval.tangential_residual <= kResidualTol
            && reference_triangle_contains(bary, 1.0e-8)) {
            converged = true;
            iterations = iter;
            break;
        }

        const Eigen::Vector3d point =
            geometry3d::panel_point(iface, seed.panel, bary);
        const Eigen::Vector3d r = point - target;
        const Eigen::Vector3d Xu =
            geometry3d::panel_tangent_u(iface, seed.panel, bary);
        const Eigen::Vector3d Xv =
            geometry3d::panel_tangent_v(iface, seed.panel, bary);

        Eigen::Matrix2d J;
        J(0, 0) = Xu.dot(Xu) + r.dot(Xuu);
        J(0, 1) = Xv.dot(Xu) + r.dot(Xuv);
        J(1, 0) = Xu.dot(Xv) + r.dot(Xuv);
        J(1, 1) = Xv.dot(Xv) + r.dot(Xvv);

        if (std::abs(J.determinant()) <= 1.0e-28)
            break;

        const Eigen::Vector2d delta = J.fullPivLu().solve(eval.tangential_equations);
        if (!finite_vector(delta))
            break;

        bool accepted = false;
        Eigen::Vector2d accepted_uv = uv;
        Eigen::Vector2d accepted_step = Eigen::Vector2d::Zero();
        for (double damping = 1.0; damping >= 1.0 / 64.0; damping *= 0.5) {
            Eigen::Vector2d candidate = uv - damping * delta;
            if (candidate[0] < -0.5 || candidate[1] < -0.5
                || candidate[0] + candidate[1] > 1.5) {
                candidate = clamp_uv_to_reference_triangle(candidate);
            }
            if (!finite_vector(candidate))
                continue;

            const ProjectionEval3D candidate_eval =
                projection_eval(iface, seed.panel, target, candidate);
            if (candidate_eval.merit <= eval.merit || damping <= 1.0 / 64.0) {
                accepted = true;
                accepted_uv = candidate;
                accepted_step = damping * delta;
                break;
            }
        }

        if (!accepted)
            break;

        uv = accepted_uv;
        iterations = iter + 1;
        if (accepted_step.norm() <= kStepTol)
            break;
    }

    const Eigen::Vector3d final_bary = geometry3d::barycentric_from_uv(uv);
    const ProjectionEval3D final_eval =
        projection_eval(iface, seed.panel, target, uv);
    converged = converged
        || (final_eval.tangential_residual <= kResidualTol
            && reference_triangle_contains(final_bary, 1.0e-8));

    return make_surface_projection(iface,
                                   grid_node,
                                   seed,
                                   target,
                                   uv,
                                   iterations,
                                   converged);
}

static std::vector<ProjectionSeed3D> projection_seeds_3d(const Interface3D& iface)
{
    const std::vector<Eigen::Vector3d> center_bary =
        geometry3d::expansion_center_barycentrics();
    std::vector<ProjectionSeed3D> seeds;
    seeds.reserve(static_cast<std::size_t>(iface.num_panels())
                  * center_bary.size());

    for (int p = 0; p < iface.num_panels(); ++p) {
        for (const Eigen::Vector3d& bary : center_bary) {
            seeds.push_back({geometry3d::panel_point(iface, p, bary),
                             bary,
                             p,
                             iface.panel_components()[p]});
        }
    }
    return seeds;
}

static bool better_projection(const SurfaceProjection3D& candidate,
                              const SurfaceProjection3D& current)
{
    if (current.panel < 0)
        return true;
    if (candidate.converged != current.converged)
        return candidate.converged;
    if (candidate.converged)
        return candidate.distance < current.distance;
    if (candidate.tangential_residual != current.tangential_residual)
        return candidate.tangential_residual < current.tangential_residual;
    return candidate.distance < current.distance;
}

// ============================================================================
// Constructor
// ============================================================================
GridPair3D::GridPair3D(const CartesianGrid3D& grid, const Interface3D& iface)
    : grid_(grid), interface_(iface), impl_(std::make_unique<Impl>())
{
    const int Nq = iface.num_points();
    const int N  = grid.num_dofs();
    const int Np = iface.num_panels();
    const int Nc = iface.num_components();
    const std::vector<DistanceSample3D> distance_samples =
        distance_samples_3d(iface);
    std::vector<int> closest_sample_idx(N, 0);

    std::vector<Eigen::Vector3d> iface_points;
    iface_points.reserve(Nq);
    for (int q = 0; q < Nq; ++q)
        iface_points.push_back(iface.points().row(q).transpose());

    std::vector<Eigen::Vector3d> sample_points;
    sample_points.reserve(distance_samples.size());
    for (const auto& sample : distance_samples)
        sample_points.push_back(sample.point);

    const NearestPointCloud3D iface_cloud(iface_points);
    const NearestPointCloud3D sample_cloud(sample_points);

    // ------------------------------------------------------------------
    // 1. closest_bulk_node[q]: nearest grid DOF to each interface point.
    // ------------------------------------------------------------------
    impl_->closest_bulk_node.resize(Nq);
    for (int q = 0; q < Nq; ++q)
        impl_->closest_bulk_node[q] = nearest_node(
            grid,
            iface.points()(q, 0), iface.points()(q, 1), iface.points()(q, 2));

    // ------------------------------------------------------------------
    // 2. closest_iface_pt[n] and min_iface_dist[n].
    //    The closest interface point is a DOF index. Narrow-band distances
    //    use extra samples for curved P2 patches so face interiors are not
    //    missed by coarse six-node surface data.
    // ------------------------------------------------------------------
    impl_->closest_iface_pt.resize(N, 0);
    impl_->min_iface_dist.resize(N, std::numeric_limits<double>::infinity());

    for (int n = 0; n < N; ++n) {
        const auto c = grid.coord(n);
        const Eigen::Vector3d pt(c[0], c[1], c[2]);

        const int q = iface_cloud.nearest(pt);
        impl_->closest_iface_pt[n] = q;

        const int sidx = sample_cloud.nearest(pt);
        impl_->min_iface_dist[n] =
            std::sqrt(squared_distance(pt, distance_samples[sidx].point));
        closest_sample_idx[n] = sidx;
    }

    // ------------------------------------------------------------------
    // 3. domain_label_vec[n]: 0=exterior, comp+1=interior of component comp.
    //
    // Build one CGAL::Surface_mesh per component and query every grid node
    // via CGAL::Side_of_triangle_mesh (AABB-accelerated).
    // ON_BOUNDARY → interior (nodes exactly on the surface are interior).
    // ------------------------------------------------------------------

    impl_->domain_label_vec.resize(N, 0);
    if (iface.panel_node_layout() == PanelNodeLayout3D::QuadraticLagrange) {
        for (int n = 0; n < N; ++n) {
            const auto c = grid.coord(n);
            const Eigen::Vector3d pt(c[0], c[1], c[2]);
            const auto& sample = distance_samples[closest_sample_idx[n]];
            if ((pt - sample.point).dot(sample.normal) < 0.0)
                impl_->domain_label_vec[n] = sample.component + 1;
        }
        return;
    }

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
bool NarrowBandProjection3D::has_projection(int bulk_node_idx) const
{
    return bulk_node_idx >= 0
        && bulk_node_idx < static_cast<int>(projection_index_by_grid_node_.size())
        && projection_index_by_grid_node_[bulk_node_idx] >= 0;
}

const SurfaceProjection3D& NarrowBandProjection3D::projection(int bulk_node_idx) const
{
    if (!has_projection(bulk_node_idx))
        throw std::out_of_range("NarrowBandProjection3D missing grid-node projection");
    return projections_[projection_index_by_grid_node_[bulk_node_idx]];
}

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

NarrowBandProjection3D GridPair3D::project_near_interface_nodes(double radius) const
{
    if (radius < 0.0)
        throw std::invalid_argument("GridPair3D projection radius must be nonnegative");
    if (interface_.points_per_panel() != 6
        || interface_.panel_node_layout() != PanelNodeLayout3D::QuadraticLagrange) {
        throw std::invalid_argument(
            "GridPair3D::project_near_interface_nodes requires P2 QuadraticLagrange panels");
    }

    NarrowBandProjection3D band;
    band.radius_ = radius;
    band.nodes_ = near_interface_nodes(radius);
    band.projection_index_by_grid_node_.assign(grid_.num_dofs(), -1);
    band.projections_.reserve(band.nodes_.size());

    if (band.nodes_.empty())
        return band;

    const std::vector<ProjectionSeed3D> seeds = projection_seeds_3d(interface_);
    std::vector<Eigen::Vector3d> seed_points;
    seed_points.reserve(seeds.size());
    for (const auto& seed : seeds)
        seed_points.push_back(seed.point);
    const NearestPointCloud3D seed_cloud(seed_points);

    constexpr int kRetrySeeds = 64;
    const int k = std::min(kRetrySeeds, static_cast<int>(seeds.size()));

    for (int node : band.nodes_) {
        const Eigen::Vector3d target = grid_point(grid_, node);
        const std::vector<int> seed_indices = seed_cloud.nearest_k(target, k);

        SurfaceProjection3D best;
        for (std::size_t i = 0; i < seed_indices.size(); ++i) {
            const SurfaceProjection3D candidate =
                project_from_seed(interface_, node, target, seeds[seed_indices[i]]);
            if (better_projection(candidate, best))
                best = candidate;
            if (i == 0 && candidate.converged)
                break;
        }

        band.projection_index_by_grid_node_[node] =
            static_cast<int>(band.projections_.size());
        band.projections_.push_back(best);
    }

    return band;
}

} // namespace kfbim
