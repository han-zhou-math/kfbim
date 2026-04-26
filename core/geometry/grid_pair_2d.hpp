#pragma once

#include <vector>
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"

namespace kfbim {

class GridPair2D {
public:
    GridPair2D(const CartesianGrid2D& grid, const Interface2D& interface);

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

    const CartesianGrid2D& grid()       const { return grid_; }
    const Interface2D&     interface_() const { return interface_; }

private:
    const CartesianGrid2D& grid_;
    const Interface2D&     interface_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfbim
