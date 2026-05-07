#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "../grid/cartesian_grid_3d.hpp"
#include "../interface/interface_3d.hpp"

namespace kfbim {

struct SurfaceProjection3D {
    int             grid_node = -1;
    int             panel = -1;
    int             component = -1;
    Eigen::Vector3d barycentric = Eigen::Vector3d::Zero();
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    double          signed_distance = 0.0;
    double          distance = 0.0;
    double          tangential_residual = 0.0;
    int             iterations = 0;
    bool            converged = false;
};

class NarrowBandProjection3D {
public:
    double radius() const { return radius_; }
    const std::vector<int>& nodes() const { return nodes_; }
    const std::vector<SurfaceProjection3D>& projections() const { return projections_; }

    bool has_projection(int bulk_node_idx) const;
    const SurfaceProjection3D& projection(int bulk_node_idx) const;

private:
    friend class GridPair3D;

    double radius_ = 0.0;
    std::vector<int> nodes_;
    std::vector<SurfaceProjection3D> projections_;
    std::vector<int> projection_index_by_grid_node_;
};

// Owns the CGAL spatial structures relating a CartesianGrid3D and an Interface3D.
// Built once at setup; queries are read-only after construction.
class GridPair3D {
public:
    GridPair3D(const CartesianGrid3D& grid, const Interface3D& interface);
    ~GridPair3D();
    GridPair3D(const GridPair3D&)            = delete;
    GridPair3D& operator=(const GridPair3D&) = delete;

    // interface point index → nearest bulk node
    int closest_bulk_node(int interface_pt_idx) const;

    // bulk node index → nearest interface point (single nearest, any component)
    int closest_interface_point(int bulk_node_idx) const;

    // domain label: 0 = Ω⁻ (exterior), 1,2,... = Ω⁺ (interior) of each component
    int domain_label(int bulk_node_idx) const;

    bool is_near_interface(int bulk_node_idx, double radius) const;

    // all bulk node indices within radius of any interface point
    std::vector<int> near_interface_nodes(double radius) const;

    // all interface point indices within radius of a given bulk node
    // (may span multiple components; used by Corrector to accumulate all contributions)
    std::vector<int> near_interface_points(int bulk_node_idx, double radius) const;

    // P2 curved-surface projections for all grid nodes in a narrow band.
    // Projection results include the parent panel and barycentric coordinates
    // for later surface interpolation/differentiation.
    NarrowBandProjection3D project_near_interface_nodes(double radius) const;

    const CartesianGrid3D& grid()      const { return grid_; }
    const Interface3D&     interface() const { return interface_; }

private:
    const CartesianGrid3D& grid_;
    const Interface3D&     interface_;

    // CGAL structures built in constructor (pimpl to avoid leaking CGAL headers)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfbim
