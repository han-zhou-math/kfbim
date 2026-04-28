#include "laplace_restrict_2d.hpp"

#include <algorithm>
#include <stdexcept>

namespace kfbim {

namespace {

Eigen::Vector2d interface_point(const Interface2D& iface, int q) {
    return iface.points().row(q).transpose();
}

Eigen::Vector2d grid_point(const CartesianGrid2D& grid, int idx) {
    const auto c = grid.coord(idx);
    return {c[0], c[1]};
}

} // namespace

LaplaceQuadraticRestrict2D::LaplaceQuadraticRestrict2D(const GridPair2D& grid_pair,
                                                       int               stencil_radius)
    : grid_pair_(grid_pair)
    , stencil_radius_(stencil_radius)
{
    if (stencil_radius_ < 1)
        throw std::invalid_argument("LaplaceQuadraticRestrict2D stencil_radius must be positive");
}

std::vector<LocalPoly2D> LaplaceQuadraticRestrict2D::apply(
    const Eigen::VectorXd&          bulk_solution,
    const std::vector<LocalPoly2D>& correction_polys) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const int n_iface = iface.num_points();

    if (bulk_solution.size() != grid.num_dofs())
        throw std::invalid_argument("LaplaceQuadraticRestrict2D bulk_solution size must equal grid DOF count");
    if (static_cast<int>(correction_polys.size()) != n_iface)
        throw std::invalid_argument("LaplaceQuadraticRestrict2D correction_polys size must equal interface point count");

    std::vector<LocalPoly2D> result(n_iface);
    for (int q = 0; q < n_iface; ++q) {
        result[q] = fit_at_interface_point(bulk_solution, q);

        const auto& corr = correction_polys[q];
        if (corr.coeffs.size() == 0)
            continue;
        if (corr.coeffs.size() != result[q].coeffs.size())
            throw std::invalid_argument("LaplaceQuadraticRestrict2D correction polynomial degree mismatch");
        if ((corr.center - result[q].center).norm() > 1e-12)
            throw std::invalid_argument("LaplaceQuadraticRestrict2D correction polynomial center mismatch");

        result[q].coeffs -= corr.coeffs;
    }

    return result;
}

LocalPoly2D LaplaceQuadraticRestrict2D::fit_at_interface_point(
    const Eigen::VectorXd& bulk_solution,
    int                    q) const
{
    const auto& grid = grid_pair_.grid();
    const auto& iface = grid_pair_.interface();
    const auto dims = grid.dof_dims();
    const int nx = dims[0];
    const int ny = dims[1];

    const int closest = grid_pair_.closest_bulk_node(q);
    const int ic = closest % nx;
    const int jc = closest / nx;

    struct Sample {
        int idx;
        double dist2;
    };
    std::vector<Sample> samples;
    samples.reserve((2 * stencil_radius_ + 1) * (2 * stencil_radius_ + 1));

    const Eigen::Vector2d center = interface_point(iface, q);
    for (int j = std::max(0, jc - stencil_radius_);
         j <= std::min(ny - 1, jc + stencil_radius_);
         ++j) {
        for (int i = std::max(0, ic - stencil_radius_);
             i <= std::min(nx - 1, ic + stencil_radius_);
             ++i) {
            const int idx = grid.index(i, j);
            const Eigen::Vector2d pt = grid_point(grid, idx);
            samples.push_back({idx, (pt - center).squaredNorm()});
        }
    }

    if (samples.size() < 6)
        throw std::runtime_error("LaplaceQuadraticRestrict2D needs at least 6 stencil samples");

    std::sort(samples.begin(), samples.end(),
              [](const Sample& a, const Sample& b) {
                  if (a.dist2 == b.dist2)
                      return a.idx < b.idx;
                  return a.dist2 < b.dist2;
              });

    const int n_rows = static_cast<int>(samples.size());
    Eigen::MatrixXd A(n_rows, 6);
    Eigen::VectorXd rhs(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        const Eigen::Vector2d pt = grid_point(grid, samples[r].idx);
        const double dx = pt[0] - center[0];
        const double dy = pt[1] - center[1];
        A(r, 0) = 1.0;
        A(r, 1) = dx;
        A(r, 2) = dy;
        A(r, 3) = 0.5 * dx * dx;
        A(r, 4) = dx * dy;
        A(r, 5) = 0.5 * dy * dy;
        rhs[r] = bulk_solution[samples[r].idx];
    }

    LocalPoly2D poly;
    poly.center = center;
    poly.coeffs = A.colPivHouseholderQr().solve(rhs);
    return poly;
}

} // namespace kfbim
