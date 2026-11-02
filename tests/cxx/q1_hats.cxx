// netdecomp::q1_hats on a grid of points: partition of unity, interior count, boundary rule.
#include "q1.hxx"
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  const int n = 33, cells[2] = {4, 4};
  const double lo[2] = {0, 0}, hi[2] = {1, 1};
  std::vector<double> pts;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) { pts.push_back((double)j / (n - 1)); pts.push_back((double)i / (n - 1)); pts.push_back(0.0); }
  netdecomp::Q1Grid g(2, cells, lo, hi);
  auto hats = netdecomp::q1_hats(g, pts.data(), n * n, 3, 100);
  std::vector<double> sum(n * n, 0.0);
  for (std::int64_t t = 0; t < hats.nnz(); t++) {
    if (hats.rows[t] < 100 || hats.rows[t] >= 100 + n * n || hats.cols[t] < 0 || hats.cols[t] >= g.n_hats()) { std::puts("index out of range"); return 1; }
    sum[hats.rows[t] - 100] += hats.vals[t];
  }
  for (double s : sum) if (std::fabs(s - 1.0) > 1e-12) { std::printf("partition of unity violated: %g\n", s); return 1; }
  std::int64_t n_int;
  auto interior = netdecomp::q1_interior(g, &n_int);
  if (n_int != 9 || g.n_hats() != 25) { std::puts("interior count"); return 1; }
  auto bnd = netdecomp::q1_boundary_points(hats, interior, n * n, 100);
  int nb = 0;
  for (char b : bnd) nb += b;
  if (nb != 4 * (n - 1)) { std::printf("boundary points: %d\n", nb); return 1; }
  std::puts("ok");
  return 0;
}
