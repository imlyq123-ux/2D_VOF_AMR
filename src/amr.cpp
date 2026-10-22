// amr.cpp - interface band + velocity wavelet refinement criterion.
#include "amr.hpp"

#include <algorithm>
#include "vof.hpp"
#ifndef _OPENMP
#pragma GCC diagnostic ignored "-Wunknown-pragmas"  // serial build: omp pragmas are inert
#endif

namespace amr {

namespace {
constexpr int kParMin = 16384;  // smaller loops run serially (OpenMP start-up cost)
}

std::vector<char> interfacialCells(const Mesh& m, const std::vector<double>& f) {
    std::vector<char> mark(m.nCells(), 0);
    const auto& leaves = m.leaves();
    const auto& faces = m.faces();
#pragma omp parallel if (m.nLeaves() > kParMin)
    {
#pragma omp for schedule(static)
        for (int k = 0; k < m.nLeaves(); ++k) {
            const int c = leaves[k];
            if (f[c] > vof::EPS && f[c] < 1.0 - vof::EPS) mark[c] = 1;
        }
        // sharp jumps between pure cells also count as interface (marks only set 0 -> 1)
#pragma omp for schedule(static)
        for (int q = 0; q < m.nFaces(); ++q) {
            const Mesh::Face& F = faces[q];
            if (F.L < 0 || F.R < 0 || std::fabs(f[F.L] - f[F.R]) <= 0.5) continue;
#pragma omp atomic write
            mark[F.L] = 1;
#pragma omp atomic write
            mark[F.R] = 1;
        }
    }
    return mark;
}

std::vector<int> computeFlags(const State& s) {
    const Mesh& m = s.m();
    const Params& prm = s.prm;
    const int Lmax = m.maxLevel();
    const std::vector<double>& f = s.F(s.id.f);

    // ---- interface band, measured in max-level cells
    std::vector<char> band(m.nCells(), 0);
    const std::vector<char> itf = interfacialCells(m, f);
    const int b = prm.interfaceBuffer;
    const auto& leaves = m.leaves();
    const int nl = m.nLeaves();
#pragma omp parallel for schedule(dynamic, 256) if (nl > kParMin)
    for (int k = 0; k < nl; ++k) {
        const int c = leaves[k];
        if (!itf[c]) continue;
        const int sh = Lmax - m.level(c);
        const int i0 = (m.ix(c) << sh) - b, i1 = ((m.ix(c) + 1) << sh) - 1 + b;
        const int j0 = (m.iy(c) << sh) - b, j1 = ((m.iy(c) + 1) << sh) - 1 + b;
        for (int j = j0; j <= j1; ++j)
            for (int i = i0; i <= i1; ++i) {
                const int n = m.locate(Lmax, i, j);
#pragma omp atomic write
                band[n] = 1;
            }
    }

    std::vector<int> flags(m.nLeaves(), -1);
    // ---- velocity wavelet estimate
    std::vector<char> refineVel(m.nCells(), 0), keepVel(m.nCells(), 0);
    if (prm.adaptVelocity && !prm.vofOnly && s.id.u >= 0) {
        for (int comp : {s.id.u, s.id.v}) {
            std::vector<double> q = s.F(comp);
            m.restrict(q);
#pragma omp parallel for schedule(static) if (nl > kParMin)
            for (int k = 0; k < nl; ++k) {
                const int c = leaves[k];
                const int p = m.parent(c);
                if (p < 0) continue;
                const int L = m.level(p), i = m.ix(p), j = m.iy(p);
                // unlimited bilinear prediction from the parent level
                const double gx = 0.5 * (q[m.locate(L, i + 1, j)] - q[m.locate(L, i - 1, j)]);
                const double gy = 0.5 * (q[m.locate(L, i, j + 1)] - q[m.locate(L, i, j - 1)]);
                const double sx = (m.ix(c) & 1) ? 0.25 : -0.25, sy = (m.iy(c) & 1) ? 0.25 : -0.25;
                const double e = std::fabs(q[c] - (q[p] + sx * gx + sy * gy));
                if (e > prm.uTol) refineVel[c] = 1;
                if (e > prm.uTol / 1.5) keepVel[c] = 1;
            }
        }
    }
#pragma omp parallel for schedule(static) if (nl > kParMin)
    for (int k = 0; k < nl; ++k) {
        const int c = leaves[k];
        if (band[c]) flags[k] = m.level(c) < Lmax ? 1 : 0;
        else if (refineVel[c]) flags[k] = 1;
        else if (keepVel[c]) flags[k] = 0;
    }
    return flags;
}

}  // namespace amr
