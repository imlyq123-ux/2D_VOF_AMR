// mesh.hpp - 2D cell-based quadtree AMR mesh.
//
// Structure
//   * nx0 x ny0 root cells (level 0) of size L0, lower-left corner (x0, y0).
//   * A cell (L, i, j) has size h(L) = L0 / 2^L and covers
//       [x0 + i h, x0 + (i+1) h] x [y0 + j h, y0 + (j+1) h],  0 <= i < nx(L), 0 <= j < ny(L).
//   * Every non-leaf cell has exactly 4 children. Child k of (L,i,j) is
//       (L+1, 2i + (k&1), 2j + (k>>1)).
//   * The tree is kept 2:1 balanced including diagonal neighbours: two leaves that
//     touch (even at a corner) differ by at most one level.
//   * Leaves always satisfy minLevel <= level <= maxLevel.
//
// Storage
//   * Cells (leaves AND their ancestors) have a compact index c in [0, nCells()).
//     Cells are ordered by level (coarse first), so a reverse loop visits children
//     before parents.
//   * Fields are std::vector<double> of size nCells() owned by the mesh; they are
//     transferred automatically by adapt() (copy / prolongation / restriction).
//   * Only leaf values are "the solution". Parent values are meaningful only after
//     restrict(id) (average of children).
//
// Faces
//   * faces() lists every face between leaves plus every boundary face. A face
//     between a coarse and a fine leaf has the fine leaf's length (a coarse cell side
//     touching two fine cells therefore carries two faces).
//   * Orientation: dir = 0 for faces normal to x, 1 for faces normal to y. The
//     normal points from cell L to cell R (i.e. +x or +y).
//   * Boundary faces: on the LEFT/BOTTOM sides L = -1 and R = interior cell; on the
//     RIGHT/TOP sides L = interior cell and R = -1. bside gives the Side, -1 inside.
//   * Index of face lists change on every adapt(); face data must be rebuilt.
//
// Thread safety: all const methods may be called concurrently (OpenMP loops over
// leaves()/faces() are fine).
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "common.hpp"

class Mesh {
public:
    struct Config {
        int nx0 = 1, ny0 = 1;
        double x0 = 0.0, y0 = 0.0, L0 = 1.0;
        int minLevel = 3, maxLevel = 7;
        bool periodicX = false, periodicY = false;
    };

    struct Face {
        int L = -1, R = -1;   // cell indices (not leaf ordinals); -1 = boundary
        int dir = 0;          // 0: normal along x, 1: normal along y
        int bside = -1;       // Side for boundary faces, -1 for interior faces
        double area = 0.0;    // face length
        double dist = 0.0;    // distance between L and R centres along the normal
                              // (h/2 for boundary faces, 1.5 h_fine for coarse/fine)
        Vec2 x;               // face midpoint
        int level = 0;        // level of the finer adjacent cell
    };

    // Prolongation: fill out[k], k = 0..3 (child numbering as above), from the parent
    // cell `parent`. Called during adapt() on the NEW mesh: every cell at the parent
    // level or coarser already holds its final value, so locate() at the parent's
    // level gives valid neighbour data.
    using Prolongator =
        std::function<void(const Mesh&, const std::vector<double>& data, int parent, double out[4])>;

    explicit Mesh(const Config& cfg);

    // ---------------------------------------------------------------- geometry
    const Config& config() const { return cfg_; }
    int minLevel() const { return cfg_.minLevel; }
    int maxLevel() const { return cfg_.maxLevel; }
    int nx(int L) const { return cfg_.nx0 << L; }
    int ny(int L) const { return cfg_.ny0 << L; }
    double h(int L) const { return h_[L]; }
    double xmin() const { return cfg_.x0; }
    double ymin() const { return cfg_.y0; }
    double xmax() const { return cfg_.x0 + cfg_.nx0 * cfg_.L0; }
    double ymax() const { return cfg_.y0 + cfg_.ny0 * cfg_.L0; }
    bool periodicX() const { return cfg_.periodicX; }
    bool periodicY() const { return cfg_.periodicY; }

    // ---------------------------------------------------------------- cells
    int nCells() const { return (int)lev_.size(); }
    int level(int c) const { return lev_[c]; }
    int ix(int c) const { return ci_[c]; }
    int iy(int c) const { return cj_[c]; }
    bool isLeaf(int c) const { return child_[4 * c] < 0; }
    int parent(int c) const { return parent_[c]; }
    int child(int c, int k) const { return child_[4 * c + k]; }
    double size(int c) const { return h_[lev_[c]]; }
    double volume(int c) const { return h_[lev_[c]] * h_[lev_[c]]; }
    Vec2 center(int c) const {
        double hh = h_[lev_[c]];
        return {cfg_.x0 + (ci_[c] + 0.5) * hh, cfg_.y0 + (cj_[c] + 0.5) * hh};
    }

    // Exact lookup of cell (L,i,j); -1 if it does not exist or is out of range.
    int find(int L, int i, int j) const;

    // Stencil lookup. Returns the cell that "represents" (L,i,j):
    //   - the cell itself if it exists (leaf, or parent whose value must be restricted),
    //   - otherwise the coarser leaf that contains it.
    // Out-of-domain indices are wrapped for periodic directions and MIRRORED otherwise
    // (i = -1 -> 0, i = -2 -> 1, i = nx -> nx-1, ...). If mirror != nullptr it receives
    // a bit mask: bit 0 set if the x index was mirrored, bit 1 if the y index was
    // mirrored (caller applies the BC, e.g. flips the normal velocity).
    int locate(int L, int i, int j, int* mirror = nullptr) const;

    // Leaf containing physical point p (clamped into the domain).
    int leafAt(Vec2 p) const;

    // ---------------------------------------------------------------- leaves
    const std::vector<int>& leaves() const { return leaves_; }
    int nLeaves() const { return (int)leaves_.size(); }
    int leafIndex(int c) const { return leafIdx_[c]; }   // -1 for non-leaves

    // ---------------------------------------------------------------- faces
    const std::vector<Face>& faces() const { return faces_; }
    int nFaces() const { return (int)faces_.size(); }
    // Faces touching leaf c (CSR by leaf ordinal). Face f is on the +dir side of c if
    // faces()[f].L == c, else on the -dir side.
    const int* cellFacesBegin(int c) const { return &cellFaces_[cellFaceOff_[leafIdx_[c]]]; }
    const int* cellFacesEnd(int c) const { return &cellFaces_[cellFaceOff_[leafIdx_[c] + 1]]; }

    // ---------------------------------------------------------------- fields
    // Default prolongation is limited (minmod) linear reconstruction, which is
    // conservative (mean of the 4 children equals the parent value).
    int addField(const std::string& name, Prolongator p = Prolongator());
    void setProlongator(int id, Prolongator p) { prolong_[id] = std::move(p); }
    int fieldId(const std::string& name) const;   // -1 if unknown
    const std::string& fieldName(int id) const { return names_[id]; }
    int nFields() const { return (int)data_.size(); }
    std::vector<double>& field(int id) { return data_[id]; }
    const std::vector<double>& field(int id) const { return data_[id]; }

    // Parents <- average of children, bottom-up (all levels).
    void restrict(int id);
    void restrict(std::vector<double>& data) const;

    // Built-in prolongators (usable from other modules).
    static void prolongLinear(const Mesh&, const std::vector<double>&, int parent, double out[4]);
    static void prolongInject(const Mesh&, const std::vector<double>&, int parent, double out[4]);

    // ---------------------------------------------------------------- adaptation
    // flags[leafOrdinal]: +1 refine, -1 coarsen (honoured only if all 4 siblings agree
    // and balance/minLevel allow), 0 keep. Refinement wins over coarsening; extra
    // refinement is added automatically to keep the 2:1 balance.
    // Returns true if the mesh changed (then faces/leaves/indices are rebuilt and all
    // fields transferred).
    bool adapt(const std::vector<int>& flags);

    // Refine every leaf for which pred(c) is true (up to maxLevel), repeating until
    // nothing changes. Used for initial meshes. Returns number of adapt() passes.
    int refineWhere(const std::function<bool(const Mesh&, int)>& pred);

    // Diagnostic: check internal consistency (tree, balance, faces). Returns "" if OK.
    std::string checkConsistency() const;

private:
    Config cfg_;
    std::vector<double> h_;
    // per-level dense maps (L, i, j) -> cell index or -1
    std::vector<std::vector<int>> map_;
    // per-cell arrays
    std::vector<int> lev_, ci_, cj_, parent_, child_, leafIdx_;
    std::vector<int> leaves_;
    std::vector<Face> faces_;
    std::vector<int> cellFaceOff_, cellFaces_;
    // fields
    std::vector<std::string> names_;
    std::vector<std::vector<double>> data_;
    std::vector<Prolongator> prolong_;

    int& mapAt(int L, int i, int j) { return map_[L][(size_t)j * nx(L) + i]; }
    int mapAt(int L, int i, int j) const { return map_[L][(size_t)j * nx(L) + i]; }
    // Rebuild compact arrays from an existence map (exists[L][j*nx+i]).
    void rebuild(const std::vector<std::vector<uint8_t>>& exists);
    void buildFaces();
};
