// laplacian.cxx -- the simplest use of PCNet2AS, PETSc and HDF5 only, no HyperHDG.
//
// Graph Laplacian of a .geo.h5 network (fiber_network's make_geo format: /domain/points,
// /domain/edges, /domain/types_points), unit edge weights or 1/length, Dirichlet rows and
// columns of the typed nodes replaced by the identity, right-hand side sin(1 + i) on the free
// nodes. Solved with CG; the preconditioner is net2as unless -pc_type says otherwise. Every
// component of the network must contain a Dirichlet node (a grid or a clamped box does).
//
//   ./laplacian -domain grid.geo.h5 -net2as_p 3 -net2as_cb_type q1 -net2as_cb_trim
//
// prints one JSON line {"domain": .., "nodes": .., "iterations": .., "residuals": [..]} with the
// true residual norm divided by |b| after every iteration, which tests/ compares with the
// Python implementations on the identical file.
#include <petsc.h>
#include <hdf5.h>
#include "net2as.hxx"
#include <cmath>
#include <string>
#include <vector>

// read a whole 1D or 2D dataset, converted to the memory type (HDF5 converts integer widths)
template <typename T>
static std::vector<T> read_dataset(hid_t file, const char *name, hid_t memtype, hsize_t *ncols) {
  hid_t ds = H5Dopen2(file, name, H5P_DEFAULT);
  PetscCheckAbort(ds >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "dataset %s missing", name);
  hid_t sp = H5Dget_space(ds);
  hsize_t dims[2] = {0, 1};
  int rank = H5Sget_simple_extent_ndims(sp);
  H5Sget_simple_extent_dims(sp, dims, NULL);
  if (rank == 1) dims[1] = 1;
  std::vector<T> data(dims[0] * dims[1]);
  PetscCheckAbort(H5Dread(ds, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()) >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "reading %s failed", name);
  H5Sclose(sp);
  H5Dclose(ds);
  *ncols = dims[1];
  return data;
}

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  char domain[PETSC_MAX_PATH_LEN] = "", weights[32] = "unit";
  PetscBool set;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-domain", domain, sizeof(domain), &set));
  PetscCheck(set, PETSC_COMM_WORLD, PETSC_ERR_USER, "-domain <file.geo.h5> is required");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-weights", weights, sizeof(weights), NULL));
  const bool inverse_length = std::string(weights) == "inverse_length";
  PetscCheck(inverse_length || std::string(weights) == "unit", PETSC_COMM_WORLD, PETSC_ERR_USER, "-weights unit|inverse_length");

  hid_t file = H5Fopen(domain, H5F_ACC_RDONLY, H5P_DEFAULT);
  PetscCheck(file >= 0, PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "cannot open %s", domain);
  hsize_t sdim, two, one;
  const auto points = read_dataset<double>(file, "/domain/points", H5T_NATIVE_DOUBLE, &sdim);
  const auto edges = read_dataset<PetscInt>(file, "/domain/edges", sizeof(PetscInt) == 8 ? H5T_NATIVE_INT64 : H5T_NATIVE_INT, &two);
  const auto types = read_dataset<int>(file, "/domain/types_points", H5T_NATIVE_INT, &one);
  H5Fclose(file);
  const PetscInt N = points.size() / sdim, M = edges.size() / 2;
  PetscCheck(sdim == 3 && two == 2, PETSC_COMM_WORLD, PETSC_ERR_FILE_READ, "expected points (n,3) and edges (m,2)");

  // weights and degrees (every rank reads the whole file; these are test-size networks)
  std::vector<PetscReal> w(M);
  std::vector<PetscInt> deg(N, 0);
  for (PetscInt e = 0; e < M; e++) {
    const PetscInt a = edges[2 * e], b = edges[2 * e + 1];
    PetscReal len = 0;
    for (hsize_t d = 0; d < sdim; d++) len += std::pow(points[a * sdim + d] - points[b * sdim + d], 2);
    w[e] = inverse_length ? 1.0 / std::sqrt(len) : 1.0;
    deg[a]++; deg[b]++;
  }

  Mat A;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscInt rs, re;
  PetscCall(MatSetUp(A));   // sizes known; preallocation below through MatXAIJSetPreallocation
  PetscCall(MatGetOwnershipRange(A, &rs, &re));
  std::vector<PetscInt> dnnz(re - rs, 1), onnz(re - rs, 0);
  for (PetscInt e = 0; e < M; e++)
    for (int s = 0; s < 2; s++) {
      const PetscInt r = edges[2 * e + s], c = edges[2 * e + 1 - s];
      if (r >= rs && r < re) ((c >= rs && c < re) ? dnnz : onnz)[r - rs]++;
    }
  PetscCall(MatXAIJSetPreallocation(A, 1, dnnz.data(), onnz.data(), NULL, NULL));
  for (PetscInt e = 0; e < M; e++)
    for (int s = 0; s < 2; s++) {
      const PetscInt r = edges[2 * e + s], c = edges[2 * e + 1 - s];
      if (r >= rs && r < re) {
        PetscCall(MatSetValue(A, r, c, -w[e], ADD_VALUES));
        PetscCall(MatSetValue(A, r, r, w[e], ADD_VALUES));
      }
    }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  std::vector<PetscInt> dirichlet;
  for (PetscInt r = rs; r < re; r++) if (types[r] != 0) dirichlet.push_back(r);
  PetscCall(MatZeroRowsColumns(A, dirichlet.size(), dirichlet.data(), 1.0, NULL, NULL));

  Vec b, x;
  PetscCall(MatCreateVecs(A, &x, &b));
  for (PetscInt r = rs; r < re; r++)
    PetscCall(VecSetValue(b, r, types[r] != 0 ? 0.0 : std::sin(1.0 + r), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  // what net2as needs besides the matrix: owned coordinates and owned edges (first node owned)
  std::vector<PetscReal> coords(points.begin() + rs * sdim, points.begin() + re * sdim);
  std::vector<PetscInt> owned_edges;
  for (PetscInt e = 0; e < M; e++)
    if (edges[2 * e] >= rs && edges[2 * e] < re) { owned_edges.push_back(edges[2 * e]); owned_edges.push_back(edges[2 * e + 1]); }

  KSP ksp;
  PC pc;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT, 500));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCRegister("net2as", PCCreate_Net2AS));
  PetscCall(PCSetType(pc, "net2as"));
  PetscCall(KSPSetFromOptions(ksp));
  PCType pc_type;
  PetscCall(PCGetType(pc, &pc_type));
  if (strcmp(pc_type, "net2as") == 0)
    PetscCall(PCNet2ASSetDomain(pc, re - rs, N, sdim, coords.data(), owned_edges.size() / 2, owned_edges.data()));
  PetscCall(KSPSetResidualHistory(ksp, NULL, PETSC_DECIDE, PETSC_TRUE));
  PetscCall(KSPSolve(ksp, b, x));

  const PetscReal *hist;
  PetscInt nh, its;
  PetscReal nb;
  PetscCall(KSPGetResidualHistory(ksp, &hist, &nh));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(VecNorm(b, NORM_2, &nb));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "{\"domain\": \"%s\", \"nodes\": %" PetscInt_FMT ", \"dirichlet\": %" PetscInt_FMT ", \"iterations\": %" PetscInt_FMT ", \"residuals\": [",
                        domain, N, (PetscInt)dirichlet.size(), its));
  for (PetscInt k = 1; k < nh; k++)   // hist[0] is the initial residual
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%.17g", k > 1 ? ", " : "", (double)(hist[k] / nb)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "]}\n"));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
