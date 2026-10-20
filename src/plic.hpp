// plic.hpp - analytic PLIC geometry in the unit square (internal helpers of the VOF module).
//
// Convention (same as vof::Line): phase 1 = { (X,Y) in [0,1]^2 : mx*X + my*Y <= alpha }.
// Normals of any sign are accepted; area/rectArea also accept non-normalized normals,
// alpha() and the normal routines use |mx| + |my| = 1.
// Formulas: Scardovelli & Zaleski, JCP 164 (2000), written without cancellation so that
// they stay accurate for nearly degenerate normals (mx or my -> 0) and f -> 0 / 1.
#pragma once

namespace plic {

// Fraction of the unit square occupied by phase 1.
double area(double mx, double my, double alpha);

// Inverse of area(): line constant such that area(mx, my, alpha) == f (|mx|+|my| = 1).
double alpha(double mx, double my, double f);

// Area of phase 1 inside the sub-rectangle [X0,X1]x[Y0,Y1] of the unit cell (in unit-cell
// area units, i.e. at most (X1-X0)(Y1-Y0)).
double rectArea(double mx, double my, double alpha, double X0, double X1, double Y0, double Y1);

// End points (P[0],P[1]) - (P[2],P[3]) of the interface segment inside the unit square.
// Returns false if the line does not cross the square.
bool segment(double mx, double my, double alpha, double P[4]);

// Mixed-Youngs-Centered normal (Aulisa et al. 2007) from the 3x3 block s[i][j] of
// fractions, i = x offset + 1, j = y offset + 1. Output normalized |mx| + |my| = 1,
// pointing out of phase 1 (along -grad f).
void mycNormal(const double s[3][3], double& mx, double& my);

}  // namespace plic
