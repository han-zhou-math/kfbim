#pragma once

#include <Eigen/Dense>

namespace kfbim {

// Triangulated surface interface in 3D.
// Stores quadrature data explicitly; geometry description (how points/normals
// were generated from a parametric surface) lives above this class.
class Interface3D {
public:
    // points:  Nq x 3, quadrature point coordinates
    // normals: Nq x 3, outward unit normals at each quadrature point
    // weights: Nq,     quadrature weights (area-weighted)
    // panels:  Np x 3, triangle connectivity (indices into points)
    Interface3D(Eigen::MatrixX3d points,
                Eigen::MatrixX3d normals,
                Eigen::VectorXd  weights,
                Eigen::MatrixX3i panels);

    int num_points() const { return static_cast<int>(points_.rows()); }
    int num_panels() const { return static_cast<int>(panels_.rows()); }

    const Eigen::MatrixX3d& points()  const { return points_; }
    const Eigen::MatrixX3d& normals() const { return normals_; }
    const Eigen::VectorXd&  weights() const { return weights_; }
    const Eigen::MatrixX3i& panels()  const { return panels_; }

private:
    Eigen::MatrixX3d points_;
    Eigen::MatrixX3d normals_;
    Eigen::VectorXd  weights_;
    Eigen::MatrixX3i panels_;
};

} // namespace kfbim
