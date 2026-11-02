#include "q1.hxx"
#include <cmath>

namespace netdecomp {

Q1Grid::Q1Grid(int dim_, const int *cells_, const double *lo_, const double *hi_)
    : dim(dim_), cells(cells_, cells_ + dim_), nodes_per_axis(dim_), lo(lo_, lo_ + dim_), hi(hi_, hi_ + dim_), h(dim_) {
  for (int d = 0; d < dim; d++) {
    nodes_per_axis[d] = cells[d] + 1;
    h[d] = (hi[d] - lo[d]) / cells[d];
  }
}

std::int64_t Q1Grid::n_hats() const {
  std::int64_t n = 1;
  for (int d = 0; d < dim; d++) n *= nodes_per_axis[d];
  return n;
}

Coo q1_hats(const Q1Grid &g, const double *points, std::int64_t n, int sdim, std::int64_t row0, double eps, double support_eps) {
  Coo out;
  const int corners = 1 << g.dim;
  out.rows.reserve(n * corners);
  out.cols.reserve(n * corners);
  out.vals.reserve(n * corners);
  std::vector<std::int64_t> stride(g.dim, 1);
  for (int d = 1; d < g.dim; d++) stride[d] = stride[d - 1] * g.nodes_per_axis[d - 1];
  std::vector<int> cell(g.dim);
  std::vector<double> local(g.dim);
  for (std::int64_t k = 0; k < n; k++) {
    for (int d = 0; d < g.dim; d++) {
      const double x = points[k * sdim + d];
      int c = (int)std::floor((x - g.lo[d] + eps) / g.h[d]);
      if (c < 0) c = 0;
      if (c > g.cells[d] - 1) c = g.cells[d] - 1;
      cell[d] = c;
      local[d] = (x - g.lo[d]) / g.h[d] - c;
    }
    for (int corner = 0; corner < corners; corner++) {
      double w = 1.0;
      std::int64_t col = 0;
      for (int d = 0; d < g.dim; d++) {
        const int bit = (corner >> d) & 1;
        w *= bit ? local[d] : 1.0 - local[d];
        col += (cell[d] + bit) * stride[d];
      }
      if (std::fabs(w) <= support_eps) continue;
      out.rows.push_back(row0 + k);
      out.cols.push_back(col);
      out.vals.push_back(w);
    }
  }
  return out;
}

std::vector<std::int64_t> q1_interior(const Q1Grid &g, std::int64_t *n_interior) {
  std::vector<std::int64_t> map(g.n_hats(), -1);
  std::int64_t next = 0;
  for (std::int64_t flat = 0; flat < g.n_hats(); flat++) {
    std::int64_t rest = flat;
    bool interior = true;
    for (int d = 0; d < g.dim; d++) {
      const int i = (int)(rest % g.nodes_per_axis[d]);
      rest /= g.nodes_per_axis[d];
      if (i == 0 || i == g.cells[d]) interior = false;
    }
    if (interior) map[flat] = next++;
  }
  *n_interior = next;
  return map;
}

std::vector<char> q1_boundary_points(const Coo &hats, const std::vector<std::int64_t> &interior, std::int64_t n, std::int64_t row0, double eps) {
  std::vector<double> sum(n, 0.0);
  for (std::int64_t t = 0; t < hats.nnz(); t++)
    if (interior[hats.cols[t]] >= 0) sum[hats.rows[t] - row0] += hats.vals[t];
  std::vector<char> boundary(n);
  for (std::int64_t k = 0; k < n; k++) boundary[k] = sum[k] < eps;
  return boundary;
}

}  // namespace netdecomp
