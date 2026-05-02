#pragma once

#include <Eigen/Dense>

namespace kfbim {

enum class PanelNodeLayout2D {
    LegacyGaussLegendre,
    ChebyshevLobatto,
    Raw,

    // Backward-compatible alias for older call sites.
    GaussLegendre = LegacyGaussLegendre
};

// Isoparametric piecewise-polynomial curve interface in 2D.
//
// The interface is divided into Np panels. Each panel has k = points_per_panel
// Chebyshev-Lobatto, legacy Gauss-Legendre, or raw points that serve as both
// the geometric nodes (defining a degree k-1 polynomial curve segment) and the
// DOF/quadrature locations.
//
// Points are stored panel-major: panel p occupies global indices [p*k, (p+1)*k).
// point_index(p, q) = p * points_per_panel + q.
//
// Supports multiple disconnected components (e.g. several circles) via
// panel_components: one integer component id per panel.
class Interface2D {
public:
    // points:           Nq x 2, all quadrature/DOF coordinates, panel-major order
    // normals:          Nq x 2, outward unit normals at each point
    // weights:          Nq,     quadrature weights (arc-length-weighted)
    // points_per_panel: k,      uniform across all panels; Np = Nq / k
    // panel_components: Np,     0-indexed component id; all-zeros for single interface
    Interface2D(Eigen::MatrixX2d points,
                Eigen::MatrixX2d normals,
                Eigen::VectorXd  weights,
                int              points_per_panel,
                Eigen::VectorXi  panel_components,
                PanelNodeLayout2D panel_node_layout = PanelNodeLayout2D::LegacyGaussLegendre);

    int num_points()       const { return static_cast<int>(points_.rows()); }
    int points_per_panel() const { return points_per_panel_; }
    int num_panels()       const { return num_points() / points_per_panel_; }
    int num_components()   const { return panel_components_.maxCoeff() + 1; }

    // global index of the q-th local point on panel p
    int point_index(int panel, int local_q) const {
        return panel * points_per_panel_ + local_q;
    }

    const Eigen::MatrixX2d& points()           const { return points_; }
    const Eigen::MatrixX2d& normals()          const { return normals_; }
    const Eigen::VectorXd&  weights()          const { return weights_; }
    const Eigen::VectorXi&  panel_components() const { return panel_components_; }
    PanelNodeLayout2D       panel_node_layout() const { return panel_node_layout_; }

private:
    Eigen::MatrixX2d points_;
    Eigen::MatrixX2d normals_;
    Eigen::VectorXd  weights_;
    int              points_per_panel_;
    Eigen::VectorXi  panel_components_;
    PanelNodeLayout2D panel_node_layout_;
};

} // namespace kfbim
