#include "interface_3d.hpp"
#include <stdexcept>

namespace kfbim {

Interface3D::Interface3D(Eigen::MatrixX3d vertices,
                         Eigen::MatrixX3i panels,
                         Eigen::MatrixX3d points,
                         Eigen::MatrixX3d normals,
                         Eigen::VectorXd  weights,
                         int              points_per_panel,
                         Eigen::VectorXi  panel_components)
    : vertices_(std::move(vertices))
    , panels_(std::move(panels))
    , points_(std::move(points))
    , normals_(std::move(normals))
    , weights_(std::move(weights))
    , points_per_panel_(points_per_panel)
    , panel_components_(std::move(panel_components))
{
    if (points_per_panel_ <= 0)
        throw std::invalid_argument("points_per_panel must be positive");
    if (points_.rows() % points_per_panel_ != 0)
        throw std::invalid_argument("num_points must be divisible by points_per_panel");
    if (normals_.rows() != points_.rows())
        throw std::invalid_argument("normals row count must equal points row count");
    if (weights_.size() != points_.rows())
        throw std::invalid_argument("weights size must equal num_points");
    if (panel_components_.size() != num_panels())
        throw std::invalid_argument("panel_components size must equal num_panels");
}

} // namespace kfbim
