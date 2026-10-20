// mesh.cpp - quadtree AMR mesh implementation (see mesh.hpp for conventions).
#include "mesh.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace {
inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return std::fabs(a) < std::fabs(b) ? a : b;
}
}  // namespace

Mesh::Mesh(const Config& cfg) : cfg_(cfg) {
    if (cfg_.minLevel < 0 || cfg_.maxLevel < cfg_.minLevel || cfg_.maxLevel > 14)
        throw std::runtime_error("Mesh: invalid minLevel/maxLevel");
    h_.resize(cfg_.maxLevel + 1);
    for (int L = 0; L <= cfg_.maxLevel; ++L) h_[L] = cfg_.L0 / double(1 << L);
    std::vector<std::vector<uint8_t>> exists(cfg_.maxLevel + 1);
    for (int L = 0; L <= cfg_.maxLevel; ++L)
        exists[L].assign((size_t)nx(L) * ny(L), L <= cfg_.minLevel ? 1 : 0);
    rebuild(exists);
}

int Mesh::find(int L, int i, int j) const {
    if (L < 0 || L > cfg_.maxLevel || i < 0 || j < 0 || i >= nx(L) || j >= ny(L)) return -1;
    return mapAt(L, i, j);
}

int Mesh::locate(int L, int i, int j, int* mirror) const {
    int m = 0;
    while (L > cfg_.maxLevel) { i >>= 1; j >>= 1; --L; }   // arithmetic shift = floor (g++)
    const int n = nx(L), mm = ny(L);
    if (i < 0 || i >= n) {
        if (cfg_.periodicX) {
            i = ((i % n) + n) % n;
        } else {
            i = (i < 0) ? -1 - i : 2 * n - 1 - i;
            i = std::clamp(i, 0, n - 1);
            m |= 1;
        }
    }
    if (j < 0 || j >= mm) {
        if (cfg_.periodicY) {
            j = ((j % mm) + mm) % mm;
        } else {
            j = (j < 0) ? -1 - j : 2 * mm - 1 - j;
            j = std::clamp(j, 0, mm - 1);
            m |= 2;
        }
    }
    if (mirror) *mirror = m;
    for (int l = L; l >= 0; --l) {
        int c = mapAt(l, i, j);
        if (c >= 0) return c;
        i >>= 1;
        j >>= 1;
    }
    return -1;  // unreachable: level 0 always exists
}

int Mesh::leafAt(Vec2 p) const {
    const int L = cfg_.maxLevel;
    int i = (int)std::floor((p.x - cfg_.x0) / h_[L]);
    int j = (int)std::floor((p.y - cfg_.y0) / h_[L]);
    i = std::clamp(i, 0, nx(L) - 1);
    j = std::clamp(j, 0, ny(L) - 1);
    return locate(L, i, j);
}

// ------------------------------------------------------------------ fields

int Mesh::addField(const std::string& name, Prolongator p) {
    if (fieldId(name) >= 0) throw std::runtime_error("Mesh::addField: duplicate field " + name);
    names_.push_back(name);
    data_.emplace_back(nCells(), 0.0);
    prolong_.push_back(p ? std::move(p) : Prolongator(&Mesh::prolongLinear));
    return (int)data_.size() - 1;
}

int Mesh::fieldId(const std::string& name) const {
    for (size_t k = 0; k < names_.size(); ++k)
        if (names_[k] == name) return (int)k;
    return -1;
}

void Mesh::restrict(int id) { restrict(data_[id]); }

void Mesh::restrict(std::vector<double>& d) const {
    for (int c = nCells() - 1; c >= 0; --c) {
        if (isLeaf(c)) continue;
        const int* ch = &child_[4 * c];
        d[c] = 0.25 * (d[ch[0]] + d[ch[1]] + d[ch[2]] + d[ch[3]]);
    }
}

void Mesh::prolongInject(const Mesh&, const std::vector<double>& d, int p, double out[4]) {
    out[0] = out[1] = out[2] = out[3] = d[p];
}

void Mesh::prolongLinear(const Mesh& m, const std::vector<double>& d, int p, double out[4]) {
    const int L = m.level(p), i = m.ix(p), j = m.iy(p);
    const double v = d[p];
    const double vl = d[m.locate(L, i - 1, j)], vr = d[m.locate(L, i + 1, j)];
    const double vb = d[m.locate(L, i, j - 1)], vt = d[m.locate(L, i, j + 1)];
    const double gx = minmod(v - vl, vr - v), gy = minmod(v - vb, vt - v);
    for (int k = 0; k < 4; ++k) {
        const double sx = (k & 1) ? 0.25 : -0.25, sy = (k >> 1) ? 0.25 : -0.25;
        out[k] = v + sx * gx + sy * gy;
    }
}

// ------------------------------------------------------------------ rebuild

void Mesh::rebuild(const std::vector<std::vector<uint8_t>>& exists) {
    const int Lmax = cfg_.maxLevel;
    // keep old state for field transfer
    std::vector<std::vector<int>> oldMap;
    oldMap.swap(map_);
    std::vector<std::vector<double>> oldData;
    oldData.swap(data_);

    map_.resize(Lmax + 1);
    lev_.clear(); ci_.clear(); cj_.clear(); parent_.clear();
    for (int L = 0; L <= Lmax; ++L) {
        map_[L].assign((size_t)nx(L) * ny(L), -1);
        for (int j = 0; j < ny(L); ++j)
            for (int i = 0; i < nx(L); ++i) {
                if (!exists[L][(size_t)j * nx(L) + i]) continue;
                const int c = (int)lev_.size();
                mapAt(L, i, j) = c;
                lev_.push_back(L);
                ci_.push_back(i);
                cj_.push_back(j);
                parent_.push_back(L > 0 ? mapAt(L - 1, i >> 1, j >> 1) : -1);
                if (L > 0 && parent_.back() < 0) throw std::runtime_error("Mesh::rebuild: orphan cell");
            }
    }
    const int nc = (int)lev_.size();
    child_.assign(4 * (size_t)nc, -1);
    for (int c = 0; c < nc; ++c)
        if (parent_[c] >= 0) child_[4 * parent_[c] + (ci_[c] & 1) + 2 * (cj_[c] & 1)] = c;
    for (int c = 0; c < nc; ++c)
        if (child_[4 * c] >= 0)
            for (int k = 1; k < 4; ++k)
                if (child_[4 * c + k] < 0) throw std::runtime_error("Mesh::rebuild: incomplete children");

    leaves_.clear();
    leafIdx_.assign(nc, -1);
    for (int c = 0; c < nc; ++c)
        if (isLeaf(c)) {
            leafIdx_[c] = (int)leaves_.size();
            leaves_.push_back(c);
        }

    // ---- field transfer: copy cells that already existed, prolong new ones.
    data_.assign(oldData.size(), std::vector<double>(nc, 0.0));
    if (!oldData.empty()) {
        std::vector<uint8_t> isNew(nc, 0);
        for (int c = 0; c < nc; ++c) {
            const int L = lev_[c];
            const int o = oldMap.empty() ? -1 : oldMap[L][(size_t)cj_[c] * nx(L) + ci_[c]];
            if (o >= 0) {
                for (size_t f = 0; f < data_.size(); ++f) data_[f][c] = oldData[f][o];
            } else {
                isNew[c] = 1;
            }
        }
        // cells are sorted by level: parents of level-L cells are final once we reach L
        double out[4];
        for (int c = 0; c < nc; ++c) {
            if (child_[4 * c] < 0 || !isNew[child_[4 * c]]) continue;
            for (size_t f = 0; f < data_.size(); ++f) {
                prolong_[f](*this, data_[f], c, out);
                for (int k = 0; k < 4; ++k) data_[f][child_[4 * c + k]] = out[k];
            }
        }
    }
    buildFaces();
}

void Mesh::buildFaces() {
    faces_.clear();
    const bool perX = cfg_.periodicX, perY = cfg_.periodicY;
    for (int c : leaves_) {
        const int L = lev_[c], i = ci_[c], j = cj_[c];
        const double hh = h_[L];
        const Vec2 xc = center(c);
        for (int s = 0; s < 4; ++s) {
            int a = i + (s == RIGHT) - (s == LEFT);
            int b = j + (s == TOP) - (s == BOTTOM);
            const int dir = (s <= RIGHT) ? 0 : 1;
            const bool plus = (s == RIGHT || s == TOP);
            Face F;
            F.dir = dir;
            F.area = hh;
            F.level = L;
            F.x = dir == 0 ? Vec2(xc.x + (plus ? 0.5 : -0.5) * hh, xc.y)
                           : Vec2(xc.x, xc.y + (plus ? 0.5 : -0.5) * hh);
            bool outside = false;
            if (a < 0 || a >= nx(L)) {
                if (perX) a = (a + nx(L)) % nx(L); else outside = true;
            }
            if (b < 0 || b >= ny(L)) {
                if (perY) b = (b + ny(L)) % ny(L); else outside = true;
            }
            if (outside) {
                F.bside = s;
                F.dist = 0.5 * hh;
                if (plus) { F.L = c; F.R = -1; } else { F.L = -1; F.R = c; }
                faces_.push_back(F);
                continue;
            }
            int n = mapAt(L, a, b);
            if (n >= 0) {
                if (!isLeaf(n) || !plus) continue;  // finer side / other owner adds it
                F.dist = hh;
                F.L = c; F.R = n;
                faces_.push_back(F);
            } else {
                n = locate(L, a, b);  // coarser leaf (level L-1 by balance)
                F.dist = 0.5 * hh + 0.5 * h_[lev_[n]];
                if (plus) { F.L = c; F.R = n; } else { F.L = n; F.R = c; }
                faces_.push_back(F);
            }
        }
    }
    // CSR cell -> faces
    const int nl = nLeaves();
    cellFaceOff_.assign(nl + 1, 0);
    for (const Face& F : faces_) {
        if (F.L >= 0) cellFaceOff_[leafIdx_[F.L] + 1]++;
        if (F.R >= 0) cellFaceOff_[leafIdx_[F.R] + 1]++;
    }
    for (int k = 0; k < nl; ++k) cellFaceOff_[k + 1] += cellFaceOff_[k];
    cellFaces_.assign(cellFaceOff_[nl], -1);
    std::vector<int> pos(cellFaceOff_.begin(), cellFaceOff_.end() - 1);
    for (int f = 0; f < (int)faces_.size(); ++f) {
        if (faces_[f].L >= 0) cellFaces_[pos[leafIdx_[faces_[f].L]]++] = f;
        if (faces_[f].R >= 0) cellFaces_[pos[leafIdx_[faces_[f].R]]++] = f;
    }
}

// ------------------------------------------------------------------ adaptation

bool Mesh::adapt(const std::vector<int>& flags) {
    if ((int)flags.size() != nLeaves()) throw std::runtime_error("Mesh::adapt: flags size mismatch");
    const int Lmax = cfg_.maxLevel, Lmin = cfg_.minLevel;
    for (auto& d : data_) restrict(d);

    std::vector<std::vector<uint8_t>> ex(Lmax + 1);
    for (int L = 0; L <= Lmax; ++L) {
        ex[L].assign((size_t)nx(L) * ny(L), 0);
    }
    for (int c = 0; c < nCells(); ++c) ex[lev_[c]][(size_t)cj_[c] * nx(lev_[c]) + ci_[c]] = 1;
    auto E = [&](int L, int i, int j) -> uint8_t& { return ex[L][(size_t)j * nx(L) + i]; };
    bool changed = false;

    // create the 4 children of (L,i,j) (and, recursively, any missing ancestors)
    std::function<void(int, int, int)> ensure = [&](int L, int i, int j) {
        if (E(L, i, j)) return;
        ensure(L - 1, i >> 1, j >> 1);
        const int pi = i & ~1, pj = j & ~1;
        E(L, pi, pj) = E(L, pi + 1, pj) = E(L, pi, pj + 1) = E(L, pi + 1, pj + 1) = 1;
        changed = true;
    };

    // 1. requested refinement
    for (int k = 0; k < nLeaves(); ++k) {
        const int c = leaves_[k];
        if (flags[k] > 0 && lev_[c] < Lmax) ensure(lev_[c] + 1, 2 * ci_[c], 2 * cj_[c]);
    }
    // 2. balance: if a cell exists at level L, all neighbours of its parent exist
    auto wrapOk = [&](int L, int& a, int& b) {
        if (a < 0 || a >= nx(L)) {
            if (!cfg_.periodicX) return false;
            a = (a + nx(L)) % nx(L);
        }
        if (b < 0 || b >= ny(L)) {
            if (!cfg_.periodicY) return false;
            b = (b + ny(L)) % ny(L);
        }
        return true;
    };
    for (int L = Lmax; L >= 2; --L) {
        for (int j = 0; j < ny(L); j += 2)
            for (int i = 0; i < nx(L); i += 2) {
                if (!E(L, i, j)) continue;
                const int pi = i >> 1, pj = j >> 1;
                for (int dj = -1; dj <= 1; ++dj)
                    for (int di = -1; di <= 1; ++di) {
                        int a = pi + di, b = pj + dj;
                        if ((di || dj) && wrapOk(L - 1, a, b)) ensure(L - 1, a, b);
                    }
            }
    }
    // 3. coarsening: parents whose 4 children are flagged leaves
    for (int p = 0; p < nCells(); ++p) {
        if (isLeaf(p) || lev_[p] < Lmin) continue;
        const int Lp = lev_[p], i = ci_[p], j = cj_[p];
        bool ok = true;
        for (int k = 0; k < 4 && ok; ++k) {
            const int ch = child_[4 * p + k];
            ok = isLeaf(ch) && flags[leafIdx_[ch]] < 0;
        }
        if (!ok) continue;
        // children must still be leaves after refinement/balance
        if (Lp + 2 <= Lmax && E(Lp + 2, 4 * i, 4 * j) | E(Lp + 2, 4 * i + 2, 4 * j) |
                                  E(Lp + 2, 4 * i, 4 * j + 2) | E(Lp + 2, 4 * i + 2, 4 * j + 2))
            continue;
        // no neighbour of p may have grandchildren
        if (Lp + 2 <= Lmax) {
            for (int dj = -1; dj <= 1 && ok; ++dj)
                for (int di = -1; di <= 1 && ok; ++di) {
                    int a = i + di, b = j + dj;
                    if (!(di || dj) || !wrapOk(Lp, a, b)) continue;
                    for (int k = 0; k < 4 && ok; ++k) {
                        const int ca = 2 * a + (k & 1), cb = 2 * b + (k >> 1);
                        if (E(Lp + 2, 2 * ca, 2 * cb)) ok = false;
                    }
                }
        }
        if (!ok) continue;
        for (int k = 0; k < 4; ++k) E(Lp + 1, 2 * i + (k & 1), 2 * j + (k >> 1)) = 0;
        changed = true;
    }
    if (!changed) return false;
    rebuild(ex);
    return true;
}

int Mesh::refineWhere(const std::function<bool(const Mesh&, int)>& pred) {
    int passes = 0;
    for (;;) {
        std::vector<int> flags(nLeaves(), 0);
        bool any = false;
        for (int k = 0; k < nLeaves(); ++k) {
            const int c = leaves_[k];
            if (lev_[c] < cfg_.maxLevel && pred(*this, c)) { flags[k] = 1; any = true; }
        }
        if (!any || !adapt(flags)) break;
        ++passes;
    }
    return passes;
}

std::string Mesh::checkConsistency() const {
    std::ostringstream err;
    for (int c : leaves_) {
        const int L = lev_[c];
        if (L < cfg_.minLevel || L > cfg_.maxLevel) err << "leaf " << c << " level out of range\n";
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di) {
                int a = ci_[c] + di, b = cj_[c] + dj;
                if ((a < 0 || a >= nx(L)) && !cfg_.periodicX) continue;
                if ((b < 0 || b >= ny(L)) && !cfg_.periodicY) continue;
                const int n = locate(L, a, b);
                if (lev_[n] < L - 1) err << "balance violated at leaf " << c << "\n";
            }
        // each side covered exactly by faces of total length h
        double side[4] = {0, 0, 0, 0};
        for (const int* f = cellFacesBegin(c); f != cellFacesEnd(c); ++f) {
            const Face& F = faces_[*f];
            const bool plus = (F.L == c);
            side[F.dir * 2 + (plus ? 1 : 0)] += F.area;
        }
        for (int s = 0; s < 4; ++s)
            if (std::fabs(side[s] - size(c)) > 1e-12 * size(c))
                err << "leaf " << c << " side " << s << " face length " << side[s] << " != " << size(c) << "\n";
    }
    return err.str();
}
