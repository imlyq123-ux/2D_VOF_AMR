// common.hpp - basic types shared by every module.
#pragma once
#include <cmath>
#include <string>

struct Vec2 {
    double x = 0.0, y = 0.0;
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    double norm() const { return std::sqrt(x * x + y * y); }
};

// Domain sides. Index order is used everywhere (Params::bc, Mesh::Face::bside).
enum Side { LEFT = 0, RIGHT = 1, BOTTOM = 2, TOP = 3 };

inline double sq(double a) { return a * a; }
