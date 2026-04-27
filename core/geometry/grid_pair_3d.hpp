#pragma once

#include <vector>
#include "../grid/cartesian_grid_3d.hpp"
#include "../interface/interface_3d.hpp"

namespace kfbim {

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

    // domain label: 0 = exterior, 1,2,... = interior of each component
    int domain_label(int bulk_node_idx) const;

    bool is_near_interface(int bulk_node_idx, double radius) const;

    // all bulk node indices within radius of any interface point
    std::vector<int> near_interface_nodes(double radius) const;

    // all interface point indices within radius of a given bulk node
    // (may span multiple components; used by Corrector to accumulate all contributions)
    std::vector<int> near_interface_points(int bulk_node_idx, double radius) const;

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
