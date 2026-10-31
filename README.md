# netprecond

Two-level additive Schwarz preconditioners for spatial network models (fiber networks,
graph Laplacians, Timoshenko beam networks on graphs), in three generations:

1. `python/netprecond/gortz_hellman_malqvist_22.py`: the reference implementation of
   Görtz, Hellman, Målqvist, *Iterative solution of spatial network models by subspace
   decomposition*, Math. Comp. 93 (2024), by Andreas Rupp. Q1 coarse space on a Cartesian
   grid, the grid cells as subdomains, scipy only.
2. `python/netprecond/jprecond.py`: the same method rewritten and optimized, then extended by
   partition-of-unity coarse spaces on arbitrary subdomain partitions.
3. `src/net2as.cxx`: the PETSc preconditioner `PCNet2AS` (`-pc_type net2as`, options
   `-net2as_*`): parallel, `cb_type q1` for the original numerics, `cb_type pu` for the new
   coarse spaces with graph-distance overlap and ParHIP/METIS partitions.

This is the first import: the files as they were in HyperHDG's `jprecond/ne9` branch and in
`fiber_network`, with their history. Slimming comes in later pull requests, each guarded by
the tests in `tests/`.

## Layout

- `src/`: `net2as`, `petsc_util`, the optional `petsc_parhip` backend. CMake targets.
- `python/netprecond/`: the Python preconditioners, `gortz_constants.py` (the network
  constants of the Görtz, Hellman, Målqvist assumptions), `sdsize.py` (subdomain size plots).
- `tools/`: `frontal_distribution` (cholmod frontal sizes), `partition_net.py` (METIS
  partition of a `.geo.h5`), `petsc_mat_to_vtkhdf.py` and `netcoarse.py` (PETSc matrix to
  VTKHDF, two versions to be unified).
- `examples/`: `fiber_network_diffusion.py`, `fiber_network_elastic.py`, `ne18-08-dd-ref.py`
  (HyperHDG's Python module assembles the matrix, the Python preconditioners solve);
  `hyperhdg/network.cxx` (HyperHDG assembles, net2as preconditions; built with
  `-DNETPRECOND_HYPERHDG=ON -DHYPERHDG_DIR=...`).

## Simplest example

`examples/laplacian.cxx` needs nothing but PETSc: it builds the graph Laplacian of an n x n
grid with Dirichlet data on the box boundary and solves it with CG and net2as:

```
./build/laplacian -n 33 -net2as_cb_type q1 -net2as_p 3 -net2as_cb_trim
```

prints the iteration count and the residual history as JSON. `-net2as_p 3` is a 4 x 4 coarse
grid; `-net2as_cb_trim` removes the coarse DoFs on the boundary, which makes q1 mode the
reference method exactly (`tests/test_q1_net2as.py` checks that against the Python
reference on the identical problem). Any PETSc option applies, `-pc_type none` for comparison.

## Build

```
export PKG_CONFIG_PATH=<petsc prefix>/lib/pkgconfig      # or PETSC_DIR / PETSC_ARCH
cmake -B build && cmake --build build
pip install -r requirements.txt && pip install -e .
python -m pytest
```

The tests pin the numerics: generation 1 against 2 (same coarse space, same CG history on
a grid Laplacian) and 1 against 3 through `build/laplacian`, with the recorded histories in
`tests/golden/`. Later pull requests slim the code against these.

`spack.yaml` describes an environment with PETSc (hdf5, metis, mumps).
