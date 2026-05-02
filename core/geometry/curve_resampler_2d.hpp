#pragma once

#include "curve_2d.hpp"
#include "../interface/interface_2d.hpp"
#include <vector>

namespace kfbim {

// ---------------------------------------------------------------------------
// CurveResampler2D
//
// Resamples a parametric curve into a set of quasi-uniform panels based on
// arc length. This guarantees that the distance between Gauss points is roughly
// proportional to the panel length, preventing ill-conditioning in local solves.
// ---------------------------------------------------------------------------
class CurveResampler2D {
public:
    // Discretize the curve into Interface2D panels.
    // target_L_h_ratio determines the target panel arc length L relative to h.
    // It defaults to 4.0 to securely guarantee that arc_h_ratio > 0.8.
    static Interface2D discretize(const ICurve2D& curve, double h, double target_L_h_ratio = 4.0);

private:
    struct ArcLengthMap {
        std::vector<double> t_vals;
        std::vector<double> s_vals;
        double total_length;
        
        double get_t(double s) const;
    };

    static ArcLengthMap build_arc_length_map(const ICurve2D& curve, int num_samples = 10000);
};

} // namespace kfbim
