// laplacian.cxx -- the simplest use of PCNet2AS, PETSc only, no network file.
//
// Graph Laplacian of an n x n grid on the unit square with unit edge weights, Dirichlet rows
// and columns on the box boundary replaced by the identity, right-hand side sin(1 + i) on the
// interior nodes. Solved with CG; the preconditioner is net2as unless -pc_type says otherwise.
//
//   ./laplacian -n 33 -net2as_p 3 -net2as_cb_type q1 -ksp_rtol 1e-10
//
// prints one JSON line {"n": .., "iterations": .., "residuals": [..]} with the true residual
// norm divided by |b| after every iteration, which tests/ compares with the Python
// implementations on the identical problem (tests/problems.py builds the same matrix).
#include <petsc.h>
#include "net2as.hxx"
#include <cmath>
#include <vector>

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscInt n = 33;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  const PetscInt N = n * n;
  auto idx = [n](PetscInt i, PetscInt j) { return i * n + j; };   // i along y, j along x

  Mat A;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 5, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL));
  PetscInt rs, re;
  PetscCall(MatGetOwnershipRange(A, &rs, &re));

  std::vector<PetscInt> dirichlet, edges;
  std::vector<PetscReal> coords;
  for (PetscInt r = rs; r < re; r++) {
    const PetscInt i = r / n, j = r % n;
    PetscInt nb[4], k = 0;
    if (j > 0) nb[k++] = idx(i, j - 1);
    if (j < n - 1) nb[k++] = idx(i, j + 1);
    if (i > 0) nb[k++] = idx(i - 1, j);
    if (i < n - 1) nb[k++] = idx(i + 1, j);
    for (PetscInt q = 0; q < k; q++) PetscCall(MatSetValue(A, r, nb[q], -1.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, r, r, (PetscScalar)k, INSERT_VALUES));
    if (i == 0 || i == n - 1 || j == 0 || j == n - 1) dirichlet.push_back(r);
    coords.push_back((PetscReal)j / (n - 1));   // net2as wants 3 coordinates per point
    coords.push_back((PetscReal)i / (n - 1));
    coords.push_back(0.0);
    if (j < n - 1) { edges.push_back(r); edges.push_back(idx(i, j + 1)); }   // owned: first node is owned
    if (i < n - 1) { edges.push_back(r); edges.push_back(idx(i + 1, j)); }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatZeroRowsColumns(A, dirichlet.size(), dirichlet.data(), 1.0, NULL, NULL));

  Vec b, x;
  PetscCall(MatCreateVecs(A, &x, &b));
  for (PetscInt r = rs; r < re; r++) {
    const PetscInt i = r / n, j = r % n;
    const bool bnd = (i == 0 || i == n - 1 || j == 0 || j == n - 1);
    PetscCall(VecSetValue(b, r, bnd ? 0.0 : std::sin(1.0 + r), INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

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
    PetscCall(PCNet2ASSetDomain(pc, re - rs, N, 3, coords.data(), edges.size() / 2, edges.data()));
  PetscCall(KSPSetResidualHistory(ksp, NULL, PETSC_DECIDE, PETSC_TRUE));
  PetscCall(KSPSolve(ksp, b, x));

  const PetscReal *hist;
  PetscInt nh;
  PetscInt its;
  PetscReal nb;
  PetscCall(KSPGetResidualHistory(ksp, &hist, &nh));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(VecNorm(b, NORM_2, &nb));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "{\"n\": %" PetscInt_FMT ", \"iterations\": %" PetscInt_FMT ", \"residuals\": [", n, its));
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
