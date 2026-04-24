#pragma once

#include <vector>
#include "../grid/cartesian_grid_2d.hpp"
#include "../interface/interface_2d.hpp"

namespace kfbim {

class GridPair2D {
public:
    GridPair2D(const CartesianGrid2D& grid, const Interface2D& interface);

    int  closest_bulk_node(int interface_pt_idx)          const;
    int  closest_interface_point(int bulk_node_idx)        const;
    int  domain_label(int bulk_node_idx)                   const;
    bool is_near_interface(int bulk_node_idx, double radius) const;

    std::vector<int> near_interface_nodes(double radius) const;

    const CartesianGrid2D& grid()       const { return grid_; }
    const Interface2D&     interface_() const { return interface_; }

private:
    const CartesianGrid2D& grid_;
    const Interface2D&     interface_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kfbim
