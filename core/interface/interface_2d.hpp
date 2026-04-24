#pragma once

#include <Eigen/Dense>

namespace kfbim {

// Piecewise-linear curve interface in 2D.
// panels: Np x 2, line segment connectivity (indices into points)
class Interface2D {
public:
    // points:  Nq x 2, quadrature point coordinates
    // normals: Nq x 2, outward unit normals
    // weights: Nq,     quadrature weights (arc-length-weighted)
    // panels:  Np x 2, segment connectivity
    Interface2D(Eigen::MatrixX2d points,
                Eigen::MatrixX2d normals,
                Eigen::VectorXd  weights,
                Eigen::MatrixX2i panels);

    int num_points() const { return static_cast<int>(points_.rows()); }
    int num_panels() const { return static_cast<int>(panels_.rows()); }

    const Eigen::MatrixX2d& points()  const { return points_; }
    const Eigen::MatrixX2d& normals() const { return normals_; }
    const Eigen::VectorXd&  weights() const { return weights_; }
    const Eigen::MatrixX2i& panels()  const { return panels_; }

private:
    Eigen::MatrixX2d points_;
    Eigen::MatrixX2d normals_;
    Eigen::VectorXd  weights_;
    Eigen::MatrixX2i panels_;
};

} // namespace kfbim
