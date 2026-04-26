#pragma once

#include <Eigen/Dense>

namespace kfbim {

// Triangulated surface interface in 3D.
//
// In 3D, Gauss quadrature points on a triangle are not natural isoparametric
// nodes (they don't sit at corners or edge midpoints), so geometry and DOFs
// are kept separate:
//   - vertices + panels define the geometry (P1 or P2 triangulation)
//   - points, normals, weights are the quadrature/DOF locations
//
// Each panel has k = points_per_panel quadrature points (uniform across all
// panels). Points are stored panel-major: panel p occupies global indices
// [p*k, (p+1)*k). point_index(p, q) = p * points_per_panel + q.
//
// Supports multiple disconnected components via panel_components.
class Interface3D {
public:
    // vertices:         Nv x 3, geometric vertex coordinates
    // panels:           Np x 3, triangle connectivity (indices into vertices)
    // points:           Nq x 3, quadrature/DOF coordinates, panel-major order
    // normals:          Nq x 3, outward unit normals at each quadrature point
    // weights:          Nq,     quadrature weights (area-weighted)
    // points_per_panel: k,      uniform; Np = Nq / k
    // panel_components: Np,     0-indexed component id; all-zeros for single interface
    Interface3D(Eigen::MatrixX3d vertices,
                Eigen::MatrixX3i panels,
                Eigen::MatrixX3d points,
                Eigen::MatrixX3d normals,
                Eigen::VectorXd  weights,
                int              points_per_panel,
                Eigen::VectorXi  panel_components);

    int num_vertices()     const { return static_cast<int>(vertices_.rows()); }
    int num_panels()       const { return static_cast<int>(panels_.rows()); }
    int num_points()       const { return static_cast<int>(points_.rows()); }
    int points_per_panel() const { return points_per_panel_; }
    int num_components()   const { return panel_components_.maxCoeff() + 1; }

    int point_index(int panel, int local_q) const {
        return panel * points_per_panel_ + local_q;
    }

    const Eigen::MatrixX3d& vertices()         const { return vertices_; }
    const Eigen::MatrixX3i& panels()           const { return panels_; }
    const Eigen::MatrixX3d& points()           const { return points_; }
    const Eigen::MatrixX3d& normals()          const { return normals_; }
    const Eigen::VectorXd&  weights()          const { return weights_; }
    const Eigen::VectorXi&  panel_components() const { return panel_components_; }

private:
    Eigen::MatrixX3d vertices_;
    Eigen::MatrixX3i panels_;
    Eigen::MatrixX3d points_;
    Eigen::MatrixX3d normals_;
    Eigen::VectorXd  weights_;
    int              points_per_panel_;
    Eigen::VectorXi  panel_components_;
};

} // namespace kfbim
