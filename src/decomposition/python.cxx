// netprecond._decomposition: the netdecomp array functions for Python. Arrays in, arrays out;
// the sparse matrices and the Decomposition are assembled on the Python side.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include "q1.hxx"

namespace nb = nanobind;
using namespace nb::literals;

using Points = nb::ndarray<const double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;
using Int64 = nb::ndarray<nb::numpy, std::int64_t, nb::shape<-1>>;
using Float64 = nb::ndarray<nb::numpy, double, nb::shape<-1>>;
using Bool = nb::ndarray<nb::numpy, bool, nb::shape<-1>>;

template <typename T>
static nb::ndarray<nb::numpy, T, nb::shape<-1>> to_numpy(std::vector<T> &&v) {
  auto *heap = new std::vector<T>(std::move(v));
  nb::capsule owner(heap, [](void *p) noexcept { delete static_cast<std::vector<T> *>(p); });
  return nb::ndarray<nb::numpy, T, nb::shape<-1>>(heap->data(), {heap->size()}, owner);
}

static netdecomp::Q1Grid grid_of(const std::vector<int> &cells, const std::vector<double> &lo, const std::vector<double> &hi) {
  if (lo.size() != cells.size() || hi.size() != cells.size()) throw nb::value_error("cells, lo and hi must have the same length");
  return netdecomp::Q1Grid((int)cells.size(), cells.data(), lo.data(), hi.data());
}

NB_MODULE(_decomposition, m) {
  m.doc() = "netdecomp: Q1 hats on a Cartesian grid, evaluated at points (see src/decomposition/q1.hxx)";
  m.def("q1_hats", [](Points points, std::vector<int> cells, std::vector<double> lo, std::vector<double> hi, double eps, double support_eps) {
      auto g = grid_of(cells, lo, hi);
      if ((int)points.shape(1) < g.dim) throw nb::value_error("points need at least len(cells) coordinates");
      auto coo = netdecomp::q1_hats(g, points.data(), (std::int64_t)points.shape(0), (int)points.shape(1), 0, eps, support_eps);
      return nb::make_tuple(to_numpy(std::move(coo.rows)), to_numpy(std::move(coo.cols)), to_numpy(std::move(coo.vals)), g.n_hats());
    }, "points"_a, "cells"_a, "lo"_a, "hi"_a, "eps"_a = 1e-10, "support_eps"_a = 1e-14,
    "All hats at the points as COO (rows, cols, vals, n_hats); hats numbered with the first axis fastest");
  m.def("q1_interior", [](std::vector<int> cells, std::vector<double> lo, std::vector<double> hi) {
      std::int64_t n;
      auto map = netdecomp::q1_interior(grid_of(cells, lo, hi), &n);
      return nb::make_tuple(to_numpy(std::move(map)), n);
    }, "cells"_a, "lo"_a, "hi"_a, "Interior index per hat (-1 for a boundary hat) and their count");
  m.def("q1_boundary_points", [](Int64 rows, Int64 cols, Float64 vals, Int64 interior, std::int64_t n, double eps) {
      netdecomp::Coo coo;
      coo.rows.assign(rows.data(), rows.data() + rows.shape(0));
      coo.cols.assign(cols.data(), cols.data() + cols.shape(0));
      coo.vals.assign(vals.data(), vals.data() + vals.shape(0));
      std::vector<std::int64_t> inter(interior.data(), interior.data() + interior.shape(0));
      auto flags = netdecomp::q1_boundary_points(coo, inter, n, 0, eps);
      auto *heap = new std::vector<char>(std::move(flags));
      nb::capsule owner(heap, [](void *p) noexcept { delete static_cast<std::vector<char> *>(p); });
      return Bool(reinterpret_cast<bool *>(heap->data()), {heap->size()}, owner);
    }, "rows"_a, "cols"_a, "vals"_a, "interior"_a, "n"_a, "eps"_a = 1e-10,
    "Per point: no interior hat is positive there (the reference's boundary rule)");
}
