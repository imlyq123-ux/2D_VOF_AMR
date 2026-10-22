// mesh.cpp - quadtree AMR mesh implementation (see mesh.hpp for conventions).
#include "mesh.hpp"

#include <algorithm>
#include <atomic>
#include <sstream>
#include <stdexcept>
#ifndef _OPENMP
#pragma GCC diagnostic ignored "-Wunknown-pragmas"  // serial build: omp pragmas are inert
#endif

namespace {
constexpr int kPar = 16384;  // smaller loops run serially (OpenMP start-up cost)

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
    std::vector<TileMap<uint8_t>> exists;
    for (int L = 0; L <= cfg_.maxLevel; ++L) {
        exists.emplace_back(nx(L), ny(L), 0);
        if (L <= cfg_.minLevel)
            for (int j = 0; j < ny(L); ++j)
                for (int i = 0; i < nx(L); ++i) exists[L].set(i, j, 1);
    }
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
    // level by level, finest parents first; the parents of one level are independent
#pragma omp parallel if (nCells() > kPar)
    for (int L = cfg_.maxLevel - 1; L >= 0; --L) {
        const int b = levelStart_[L], e = levelStart_[L + 1];
#pragma omp for schedule(static)
        for (int c = b; c < e; ++c) {
            if (isLeaf(c)) continue;
            const int* ch = &child_[4 * c];
            d[c] = 0.25 * (d[ch[0]] + d[ch[1]] + d[ch[2]] + d[ch[3]]);
        }
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

void Mesh::rebuild(const std::vector<TileMap<uint8_t>>& exists) {
    static std::atomic<uint64_t> lastRevision{0};
    revision_ = ++lastRevision;
    const int Lmax = cfg_.maxLevel;
    // keep old state for field transfer
    std::vector<TileMap<int>> oldMap;
    oldMap.swap(map_);
    std::vector<std::vector<double>> oldData;
    oldData.swap(data_);

    // cells level by level, row-major within a level: the roots, then for each row of
    // the previous level the lower children of its refined cells (left to right), then
    // their upper children
    lev_.clear(); ci_.clear(); cj_.clear(); parent_.clear(); child_.clear();
    levelStart_.assign(Lmax + 2, 0);
    auto add = [&](int L, int i, int j, int parent) {
        lev_.push_back(L);
        ci_.push_back(i);
        cj_.push_back(j);
        parent_.push_back(parent);
        for (int k = 0; k < 4; ++k) child_.push_back(-1);
    };
    for (int j = 0; j < ny(0); ++j)
        for (int i = 0; i < nx(0); ++i) add(0, i, j, -1);
    for (int L = 0; L <= Lmax; ++L) {
        const int b = levelStart_[L], e = levelStart_[L + 1] = (int)lev_.size();
        if (L == Lmax) break;
        for (int r0 = b, r1; r0 < e; r0 = r1) {
            for (r1 = r0; r1 < e && cj_[r1] == cj_[r0]; ++r1) {}  // one row: [r0, r1)
            for (int half = 0; half < 2; ++half)
                for (int c = r0; c < r1; ++c) {
                    const int i = 2 * ci_[c], j = 2 * cj_[c];
                    if (!exists[L + 1].get(i, j)) continue;
                    for (int k = 2 * half; k < 2 * half + 2; ++k) {
                        if (!exists[L + 1].get(i + (k & 1), j + (k >> 1)))
                            throw std::runtime_error("Mesh::rebuild: incomplete children");
                        child_[4 * c + k] = (int)lev_.size();
                        add(L + 1, i + (k & 1), j + (k >> 1), c);
                    }
                }
        }
    }
    const int nc = (int)lev_.size();
    size_t nExist = 0;
    for (const auto& e : exists) nExist += e.count();
    if (nExist != (size_t)nc) throw std::runtime_error("Mesh::rebuild: orphan cell");

    map_.clear();
    for (int L = 0; L <= Lmax; ++L) {
        map_.emplace_back(nx(L), ny(L), -1);
        for (int c = levelStart_[L]; c < levelStart_[L + 1]; ++c) map_[L].set(ci_[c], cj_[c], c);
    }

    leaves_.clear();
    leafIdx_.assign(nc, -1);
    for (int c = 0; c < nc; ++c)
        if (isLeaf(c)) {
            leafIdx_[c] = (int)leaves_.size();
            leaves_.push_back(c);
        }

    // ---- field transfer: copy cells that already existed, prolong new ones.
    const int nf = (int)oldData.size();
    data_.assign(nf, std::vector<double>(nc, 0.0));
    if (nf > 0) {
        std::vector<uint8_t> isNew(nc, 0);
#pragma omp parallel for schedule(static) if (nc > kPar)
        for (int c = 0; c < nc; ++c) {
            const int o = oldMap.empty() ? -1 : oldMap[lev_[c]].get(ci_[c], cj_[c]);
            if (o >= 0) {
                for (int f = 0; f < nf; ++f) data_[f][c] = oldData[f][o];
            } else {
                isNew[c] = 1;
            }
        }
        // parents level by level: the prolongation of a level-L parent reads cells of
        // level <= L only, which are final once the coarser levels are done
        for (int L = 0; L < Lmax; ++L) {
            const int b = levelStart_[L], e = levelStart_[L + 1];
#pragma omp parallel for schedule(dynamic, 256) if (e - b > kPar / 4)
            for (int c = b; c < e; ++c) {
                if (child_[4 * c] < 0 || !isNew[child_[4 * c]]) continue;
                double out[4];
                for (int f = 0; f < nf; ++f) {
                    prolong_[f](*this, data_[f], c, out);
                    for (int k = 0; k < 4; ++k) data_[f][child_[4 * c + k]] = out[k];
                }
            }
        }
    }
    buildFaces();
}

void Mesh::buildFaces() {
    const bool perX = cfg_.periodicX, perY = cfg_.periodicY;
    // Faces owned by leaf c, in side order: boundary faces, faces towards a coarser leaf,
    // and +x/+y faces towards a same-level leaf (the finer side owns 2:1 faces).
    auto ownFaces = [&](int c, auto emit) {
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
                emit(F);
                continue;
            }
            int n = mapAt(L, a, b);
            if (n >= 0) {
                if (!isLeaf(n) || !plus) continue;  // finer side / other owner adds it
                F.dist = hh;
                F.L = c; F.R = n;
                emit(F);
            } else {
                n = locate(L, a, b);  // coarser leaf (level L-1 by balance)
                F.dist = 0.5 * hh + 0.5 * h_[lev_[n]];
                if (plus) { F.L = c; F.R = n; } else { F.L = n; F.R = c; }
                emit(F);
            }
        }
    };
    // the faces are listed leaf by leaf: one pass on small meshes, else count, prefix
    // sum and fill in parallel
    const int nl = nLeaves();
    if (nl <= kPar) {
        faces_.clear();
        for (int c : leaves_) ownFaces(c, [&](const Face& F) { faces_.push_back(F); });
    } else {
        std::vector<int> off(nl + 1, 0);
#pragma omp parallel for schedule(static)
        for (int k = 0; k < nl; ++k) {
            int cnt = 0;
            ownFaces(leaves_[k], [&](const Face&) { ++cnt; });
            off[k + 1] = cnt;
        }
        for (int k = 0; k < nl; ++k) off[k + 1] += off[k];
        faces_.resize(off[nl]);
#pragma omp parallel for schedule(static)
        for (int k = 0; k < nl; ++k) {
            int q = off[k];
            ownFaces(leaves_[k], [&](const Face& F) { faces_[q++] = F; });
        }
    }
    // CSR cell -> faces
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

    std::vector<TileMap<uint8_t>> ex;
    for (int L = 0; L <= Lmax; ++L) ex.emplace_back(nx(L), ny(L), 0);
    for (int c = 0; c < nCells(); ++c) ex[lev_[c]].set(ci_[c], cj_[c], 1);
    auto E = [&](int L, int i, int j) { return ex[L].get(i, j); };
    // set the 4 siblings of (L,i,j) to v
    auto setSiblings = [&](int L, int i, int j, uint8_t v) {
        const int pi = i & ~1, pj = j & ~1;
        for (int k = 0; k < 4; ++k) ex[L].set(pi + (k & 1), pj + (k >> 1), v);
    };
    bool changed = false;

    // create the 4 children of (L,i,j) (and, recursively, any missing ancestors)
    std::function<void(int, int, int)> ensure = [&](int L, int i, int j) {
        if (E(L, i, j)) return;
        ensure(L - 1, i >> 1, j >> 1);
        setSiblings(L, i, j, 1);
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
    // (ensure() at level L-1 only adds cells at levels < L, so ex[L] is stable while
    // it is traversed; the result is the closure, independent of the traversal order)
    for (int L = Lmax; L >= 2; --L) {
        ex[L].forEach([&](int i, int j, uint8_t) {  // one of each 4 siblings
            const int pi = i >> 1, pj = j >> 1;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    int a = pi + di, b = pj + dj;
                    if ((di || dj) && wrapOk(L - 1, a, b)) ensure(L - 1, a, b);
                }
        }, 2);
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
        if (Lp + 2 <= Lmax && (E(Lp + 2, 4 * i, 4 * j) | E(Lp + 2, 4 * i + 2, 4 * j) |
                               E(Lp + 2, 4 * i, 4 * j + 2) | E(Lp + 2, 4 * i + 2, 4 * j + 2)))
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
        setSiblings(Lp + 1, 2 * i, 2 * j, 0);
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
