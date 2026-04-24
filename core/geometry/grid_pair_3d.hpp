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

    // index of the bulk grid node closest to interface quadrature point i
    int closest_bulk_node(int interface_pt_idx) const;

    // index of the interface quadrature point closest to bulk node idx
    int closest_interface_point(int bulk_node_idx) const;

    // domain label of bulk node idx (0 = exterior, 1,2,... = interior domains)
    int domain_label(int bulk_node_idx) const;

    bool is_near_interface(int bulk_node_idx, double radius) const;

    // all bulk node indices whose closest interface point is within radius
    std::vector<int> near_interface_nodes(double radius) const;

    const CartesianGrid3D& grid()      const { return grid_; }
    const Interface3D&     interface_() const { return interface_; }

private:
    const CartesianGrid3D& grid_;
    const Interface3D&     interface_;

    // CGAL structures built in constructor (pimpl to avoid leaking CGAL headers)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfbim
