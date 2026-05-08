#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "../grid/cartesian_grid_3d.hpp"
#include "../interface/interface_3d.hpp"
#include "p2_projection_3d.hpp"

namespace kfbim {

// Owns the spatial structures relating a CartesianGrid3D and an Interface3D.
// Built once at setup; queries are read-mostly after construction.
class GridPair3D {
public:
    GridPair3D(const CartesianGrid3D& grid, const Interface3D& interface);
    ~GridPair3D();
    GridPair3D(const GridPair3D&)            = delete;
    GridPair3D& operator=(const GridPair3D&) = delete;

    // interface point index → nearest bulk node
    int closest_bulk_node(int interface_pt_idx) const;

    // bulk node index → nearest interface point (lazy compatibility query)
    int closest_interface_point(int bulk_node_idx) const;

    // bulk node index → nearest active P2 expansion center. Available only for
    // QuadraticLagrange six-node triangular patches.
    int nearest_p2_expansion_center(int bulk_node_idx) const;

    // domain label: 0 = Ω⁻ (exterior), 1,2,... = Ω⁺ (interior) of each component
    int domain_label(int bulk_node_idx) const;

    bool is_near_interface(int bulk_node_idx, double radius) const;

    // all bulk node indices within radius of active geometry samples
    // (P2 expansion centers for QuadraticLagrange patches)
    std::vector<int> near_interface_nodes(double radius) const;

    // all interface point indices within radius of a given bulk node
    // (may span multiple components; used by Corrector to accumulate all contributions)
    std::vector<int> near_interface_points(int bulk_node_idx, double radius) const;

    // P2 curved-surface projections for all grid nodes in a narrow band.
    // Projection results include the parent panel and barycentric coordinates
    // for later surface interpolation/differentiation.
    NarrowBandProjection3D project_near_interface_nodes(double radius) const;

    // P2 curved-surface projections for an explicit grid-node support set.
    // Uses the nearest generated P2 expansion center to choose a parent panel,
    // initializes with the closest point on the flat vertex triangle, then
    // applies curved-patch Newton. If Newton does not converge, the flat
    // triangle parameter is returned with converged=false.
    NarrowBandProjection3D project_grid_nodes_to_interface(
        const std::vector<int>& bulk_node_indices) const;

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
