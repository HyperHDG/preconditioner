// Q1 decompositions on arrays: the coarse hat functions of a Cartesian grid over a bounding
// box, evaluated at points. PETSc-free; net2as and the Python binding both build on it.
#pragma once
#include <cstdint>
#include <vector>

namespace netdecomp {

// Sparse triplets, rows = points (offset by row0), cols = coarse functions.
struct Coo {
  std::vector<std::int64_t> rows, cols;
  std::vector<double> vals;
  std::int64_t nnz() const { return (std::int64_t)vals.size(); }
};

// A Cartesian grid of cells[d] cells per axis over [lo, hi]; hats are numbered with the first
// axis fastest: flat = i + n0 * (j + n1 * k), n_d = cells[d] + 1.
struct Q1Grid {
  int dim;
  std::vector<int> cells, nodes_per_axis;
  std::vector<double> lo, hi, h;
  Q1Grid(int dim, const int *cells, const double *lo, const double *hi);
  std::int64_t n_hats() const;
};

// All hats of the grid at the points (n x sdim, row-major; the first dim coordinates are
// used): a point lies in cell floor((x - lo + eps) / h), clamped to the grid, so points on the
// upper box faces belong to the last cell (closed cover; the reference's lower-side tolerance
// eps is kept). Entries with |value| <= support_eps are dropped: open hat supports.
Coo q1_hats(const Q1Grid &g, const double *points, std::int64_t n, int sdim, std::int64_t row0,
            double eps = 1e-10, double support_eps = 1e-14);

// The interior hats, those that vanish on the box boundary: new index per hat, -1 for a
// boundary hat; n_interior receives their count.
std::vector<std::int64_t> q1_interior(const Q1Grid &g, std::int64_t *n_interior);

// The reference's boundary rule: a point is on the boundary when no interior hat is positive
// at it. One flag per point of the hats' rows (row0 .. row0 + n).
std::vector<char> q1_boundary_points(const Coo &hats, const std::vector<std::int64_t> &interior,
                                     std::int64_t n, std::int64_t row0, double eps = 1e-10);

}  // namespace netdecomp
