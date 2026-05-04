#include "interface_2d.hpp"
#include <stdexcept>

namespace kfbim {

namespace {

Eigen::MatrixXi make_panel_major_connectivity(int num_points, int points_per_panel)
{
    if (points_per_panel <= 0)
        throw std::invalid_argument("points_per_panel must be positive");
    if (num_points % points_per_panel != 0)
        throw std::invalid_argument("num_points must be divisible by points_per_panel");

    const int num_panels = num_points / points_per_panel;
    Eigen::MatrixXi panel_point_indices(num_panels, points_per_panel);
    for (int p = 0; p < num_panels; ++p)
        for (int q = 0; q < points_per_panel; ++q)
            panel_point_indices(p, q) = p * points_per_panel + q;
    return panel_point_indices;
}

} // namespace

Interface2D::Interface2D(Eigen::MatrixX2d points,
                         Eigen::MatrixX2d normals,
                         Eigen::VectorXd  weights,
                         int              points_per_panel,
                         Eigen::VectorXi  panel_components,
                         PanelNodeLayout2D panel_node_layout)
    : points_(std::move(points))
    , normals_(std::move(normals))
    , weights_(std::move(weights))
    , points_per_panel_(points_per_panel)
    , panel_point_indices_(make_panel_major_connectivity(static_cast<int>(points_.rows()),
                                                         points_per_panel))
    , panel_components_(std::move(panel_components))
    , panel_node_layout_(panel_node_layout)
{
    if (normals_.rows() != points_.rows())
        throw std::invalid_argument("normals row count must equal points row count");
    if (weights_.size() != points_.rows())
        throw std::invalid_argument("weights size must equal num_points");
    if (panel_point_indices_.rows() <= 0)
        throw std::invalid_argument("panel_point_indices must contain at least one panel");
    if (panel_components_.size() != num_panels())
        throw std::invalid_argument("panel_components size must equal num_panels");
}

Interface2D::Interface2D(Eigen::MatrixX2d points,
                         Eigen::MatrixX2d normals,
                         Eigen::VectorXd  weights,
                         int              points_per_panel,
                         Eigen::MatrixXi  panel_point_indices,
                         Eigen::VectorXi  panel_components,
                         PanelNodeLayout2D panel_node_layout)
    : points_(std::move(points))
    , normals_(std::move(normals))
    , weights_(std::move(weights))
    , points_per_panel_(points_per_panel)
    , panel_point_indices_(std::move(panel_point_indices))
    , panel_components_(std::move(panel_components))
    , panel_node_layout_(panel_node_layout)
{
    if (points_per_panel_ <= 0)
        throw std::invalid_argument("points_per_panel must be positive");
    if (normals_.rows() != points_.rows())
        throw std::invalid_argument("normals row count must equal points row count");
    if (weights_.size() != points_.rows())
        throw std::invalid_argument("weights size must equal num_points");
    if (panel_point_indices_.cols() != points_per_panel_)
        throw std::invalid_argument("panel_point_indices column count must equal points_per_panel");
    if (panel_point_indices_.rows() <= 0)
        throw std::invalid_argument("panel_point_indices must contain at least one panel");
    for (int p = 0; p < panel_point_indices_.rows(); ++p) {
        for (int q = 0; q < panel_point_indices_.cols(); ++q) {
            const int idx = panel_point_indices_(p, q);
            if (idx < 0 || idx >= points_.rows())
                throw std::invalid_argument("panel_point_indices contains an out-of-range point index");
        }
    }
    if (panel_components_.size() != num_panels())
        throw std::invalid_argument("panel_components size must equal num_panels");
}

} // namespace kfbim
