// plic.cpp - analytic PLIC geometry in the unit square (see plic.hpp).
#include "plic.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace plic {

double area(double mx, double my, double a) {
    // reflect to mx, my >= 0 (X -> 1 - X shifts alpha by -mx)
    if (mx < 0.0) { a -= mx; mx = -mx; }
    if (my < 0.0) { a -= my; my = -my; }
    if (a <= 0.0) return 0.0;
    const double s = mx + my;
    if (a >= s) return 1.0;
    if (mx > my) std::swap(mx, my);          // now 0 <= mx <= my, my > 0
    double v;
    if (a <= mx) v = a * a / (2.0 * mx * my);                       // triangle
    else if (a <= my) v = (2.0 * a - mx) / (2.0 * my);              // trapezoid
    else v = 1.0 - (s - a) * (s - a) / (2.0 * mx * my);             // square minus triangle
    return std::clamp(v, 0.0, 1.0);
}

double alpha(double mx, double my, double f) {
    const double n1 = std::min(std::fabs(mx), std::fabs(my));
    const double n2 = std::max(std::fabs(mx), std::fabs(my));
    f = std::clamp(f, 0.0, 1.0);
    double a;
    if (n2 <= 0.0) a = 0.0;                                         // no normal: undefined
    else if (f <= 0.5 * n1 / n2) a = std::sqrt(2.0 * f * n1 * n2);
    else if (f <= 1.0 - 0.5 * n1 / n2) a = f * n2 + 0.5 * n1;
    else a = n1 + n2 - std::sqrt(2.0 * n1 * n2 * (1.0 - f));
    if (mx < 0.0) a += mx;
    if (my < 0.0) a += my;
    return a;
}

double rectArea(double mx, double my, double a, double X0, double X1, double Y0, double Y1) {
    const double dX = X1 - X0, dY = Y1 - Y0;
    if (dX <= 0.0 || dY <= 0.0) return 0.0;
    // X = X0 + dX*xi, Y = Y0 + dY*eta maps the rectangle onto the unit square
    return dX * dY * area(mx * dX, my * dY, a - mx * X0 - my * Y0);
}

bool segment(double mx, double my, double a, double P[4]) {
    static const double VX[4] = {0, 1, 1, 0}, VY[4] = {0, 0, 1, 1};
    double phi[4];
    for (int k = 0; k < 4; ++k) phi[k] = mx * VX[k] + my * VY[k] - a;
    int n = 0;
    for (int k = 0; k < 4 && n < 2; ++k) {
        const int l = (k + 1) & 3;
        if ((phi[k] <= 0.0) == (phi[l] <= 0.0)) continue;
        const double t = phi[k] / (phi[k] - phi[l]);
        P[2 * n] = VX[k] + t * (VX[l] - VX[k]);
        P[2 * n + 1] = VY[k] + t * (VY[l] - VY[k]);
        ++n;
    }
    return n == 2;
}

void mycNormal(const double s[3][3], double& mx, double& my) {
    // column sums (heights along y) and row sums (widths along x)
    const double hl = s[0][0] + s[0][1] + s[0][2], hr = s[2][0] + s[2][1] + s[2][2];
    const double wb = s[0][0] + s[1][0] + s[2][0], wt = s[0][2] + s[1][2] + s[2][2];
    // centered columns, y = y(x): slope from heights, dominant component +-1
    double cx0 = 0.5 * (hl - hr), cy0 = wb >= wt ? 1.0 : -1.0;
    double n = std::fabs(cx0) + 1.0;
    cx0 /= n; cy0 /= n;
    // centered rows, x = x(y)
    double cy1 = 0.5 * (wb - wt), cx1 = hl >= hr ? 1.0 : -1.0;
    n = std::fabs(cy1) + 1.0;
    cx1 /= n; cy1 /= n;
    // best centered candidate: the one whose dominant component is larger
    double ccx, ccy, ccDom;
    if (std::fabs(cy0) >= std::fabs(cx1)) { ccx = cx0; ccy = cy0; ccDom = std::fabs(cy0); }
    else { ccx = cx1; ccy = cy1; ccDom = std::fabs(cx1); }
    // Youngs (weighted 3x3 gradient)
    double yx = (s[0][0] + 2.0 * s[0][1] + s[0][2]) - (s[2][0] + 2.0 * s[2][1] + s[2][2]);
    double yy = (s[0][0] + 2.0 * s[1][0] + s[2][0]) - (s[0][2] + 2.0 * s[1][2] + s[2][2]);
    n = std::fabs(yx) + std::fabs(yy);
    if (n > 1e-14) {
        yx /= n; yy /= n;
        // the centered estimate saturates (too large dominant component) when the
        // interface leaves the 3-cell column: then Youngs is more reliable
        if (ccDom > std::max(std::fabs(yx), std::fabs(yy))) { mx = yx; my = yy; return; }
    }
    mx = ccx; my = ccy;
}

}  // namespace plic
