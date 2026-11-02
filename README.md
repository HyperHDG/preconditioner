# netprecond

Two-level additive Schwarz preconditioners for spatial network models (fiber networks,
graph Laplacians, Timoshenko beam networks on graphs), in three generations:

1. `python/netprecond/reference/gortz_hellman_malqvist_22.py`: the reference implementation
   of Görtz, Hellman, Målqvist, *Iterative solution of spatial network models by subspace
   decomposition*, Math. Comp. 93 (2024), by Andreas Rupp. Q1 coarse space on a Cartesian
   grid, the hat supports as subdomains, scipy only. Frozen: it is the oracle of the tests.
2. `python/netprecond/`: the Python tier. A `Decomposition` is data, the coarse basis and the
   subdomains; `q1()` builds the reference's from the C++ library below; `AdditiveSchwarz(A,
   dec)` is the two-level preconditioner on any decomposition, factorized once, callable, scipy
   only. New decompositions are prototyped here as functions returning that data. (This grew
   out of jprecond.py, the 2025 rewrite of the reference that net2as was then written from;
   see the git history.)
   `src/decomposition/`: the decompositions as PETSc-free C++ on arrays, one implementation
   used by net2as and, through a nanobind binding, by the Python tier.
3. `src/net2as.cxx`: the PETSc preconditioner `PCNet2AS` (`-pc_type net2as`, options
   `-net2as_*`): parallel, `cb_type q1` for the original numerics, `cb_type pu` for the new
   coarse spaces with graph-distance overlap and ParHIP/METIS partitions.

Production and prototypes meet in the same `AdditiveSchwarz`. net2as itself is still the
import from HyperHDG's `jprecond/ne9` branch; its slimming comes in pull requests guarded by
the tests in `tests/`.

## Layout

- `src/`: `decomposition/` (the netdecomp library and its Python binding), `net2as`,
  `petsc_util`, the optional `petsc_parhip` backend. CMake targets.
- `python/netprecond/`: `decomposition.py` (the contract, `expand` for block dofs, `q1`),
  `schwarz.py` (`AdditiveSchwarz`), `pcg.py` (PCG with the Lanczos condition estimate),
  `problems.py` (graph Laplacians of `.geo.h5` networks), `reference/` (frozen),
  `gortz_constants.py` (the network constants of the paper's assumptions), `sdsize.py`.
- `tools/`: `frontal_distribution` (cholmod frontal sizes), `partition_net.py` (METIS
  partition of a `.geo.h5`), `petsc_mat_to_vtkhdf.py` and `netcoarse.py` (PETSc matrix to
  VTKHDF, two versions to be unified).
- `examples/`: `laplacian_net2as.cxx`, `laplacian_q1.py`, `hyperhdg_net_q1.py`, the sweeps,
  `zenodo_fiber_network.sh`; `hyperhdg/network.cxx` (HyperHDG assembles, net2as
  preconditions; built with `-DNETPRECOND_HYPERHDG=ON -DHYPERHDG_DIR=...`, untested here).

## Simplest example

`examples/laplacian_net2as.cxx` needs nothing but PETSc and HDF5: it reads a `.geo.h5` network
from fiber_network's `make_geo`, builds its graph Laplacian with Dirichlet data on the typed
nodes and solves it with CG and net2as. A grid is a network like any other:

```
python submodules/fiber_network/fiber_network/make_geo.py --grid 33 -t 1e-6 --dirichlet xmin xmax ymin ymax -o grid.geo.h5
./build/laplacian_net2as -domain grid.geo.h5 -net2as_cb_type q1 -net2as_p 3 -net2as_cb_trim
python examples/laplacian_q1.py --domain grid.geo.h5 -H 4         # the same in Python, generation 2
```

The driver prints the iteration count and the residual history as JSON, the Python script
YAML as it goes. `-net2as_p 3` is a 4 x 4 coarse grid; `-net2as_cb_trim` removes the coarse
DoFs on the boundary, which makes q1 mode the reference method exactly
(`tests/test_q1_net2as.py` checks that against the Python reference on the identical file).
Any PETSc option applies, `-pc_type none` for comparison. `-t 1e-6` types one node layer as
boundary; make_geo's default band of 2 % of the extent is several layers on a fine grid.

## HDG systems from HyperHDG

`examples/hyperhdg_net_q1.py` does the same for the HDG trace systems HyperHDG assembles on the
network: diffusion, and Timoshenko beams with six dofs per node. One module instantiates both
classes, `Diffusion` and `Timoshenko`, on the 3D file topology for a degree; HyperHDG's Python
interface compiles it on first use into `.cache/hyperhdg/`. Networks are three-dimensional;
`--flatten` solves on a copy with z = 0, the network projected to the plane. The q1
decomposition is expanded to the dofs per node and restricted to the dofs the system carries:
HyperHDG leaves the rows of Dirichlet dofs empty, and make_geo's Dirichlet band reaches inside
the box, so those dofs must leave every subdomain or the blocks are singular.

```
python examples/hyperhdg_net_q1.py --domain mikado.geo.h5 --problem diffusion -H 4
python examples/hyperhdg_net_q1.py --domain mikado.geo.h5 --problem timoshenko --degree 2 -H 4
```

`examples/hyperhdg_cube_q1.py` is the same preconditioner on plain HDG diffusion on the unit
square (HyperHDG's `Cubic<2,2>` topology, its own constant-coefficient test problem): the
hypernodes are the cell faces at their barycenters, each with degree+1 dofs. A sanity check
that the method is not tied to networks; the condition number stays near 4 (degree 1) and 7
(degree 2) from 16 to 256 cells per axis and 4 to 32 coarse cells per axis.

```
python examples/hyperhdg_cube_q1.py -n 64 --degree 2 -H 8
```

HyperHDG is the pinned submodule at its `h5-io` branch, which reads `.geo.h5`. The build needs
HDF5 (`HDF5_ROOT` if it is not on the default search path); `hyperhdg.py` is imported from the
submodule's `python/` directory, the one path manipulation left, until HyperHDG's Python
interface is a package. Known limit: the Timoshenko solver's local frame is degenerate on
axis-aligned edges, so beams run on mikado and real networks, not on grids.

`examples/zenodo_fiber_network.sh` fetches the paper networks of
[zenodo.org/records/20410514](https://zenodo.org/records/20410514) (CC BY 4.0), converts net2
(8 x 8 mm, 1.6 million nodes) whole and as a 2 x 2 mm piece, and the examples run on both.

`tests/verify_reproducibles.py` runs HyperHDG's cython-era reproducibles, which this example
replaces, on the same networks and compares the CG histories; it compiles both interfaces, so
it is run by hand.

## Sweeps: the condition number is bounded

`examples/sweep_grid_q1.sh`, `examples/sweep_mikado_q1.sh` and `examples/sweep_hyperhdg_net_q1.sh` run the reference method (as
jprecond q1, which the tests show to be the same) with PCG on graph Laplacians and record the
Lanczos estimate of the condition number of the preconditioned operator (`python/netprecond/pcg.py`,
the estimate PETSc's `-ksp_compute_eigenvalues` gives for CG). The theory (Görtz, Hellman,
Målqvist 2024, Thm. 4.6) bounds it independently of the number of nodes and of the coarse
grid size, by the network constants of their Assumption 3.5 only.

On grids from `make_geo --grid`, 289 to 66049 nodes and 4 to 32 coarse cells per axis, the condition number stays
between 3.0 and 3.9 and PCG needs 13 to 16 iterations. On mikado networks generated by
fiber_network's `make_geo --mikado` with a fixed seed, it is 9 to 11 for 20k to 236k nodes with 4
coarse cells per axis and grows to 17 with 16 cells, which is the network constants at work:
finer coarse cells see fewer fibers each. Networks too sparse for a coarse grid make the coarse
problem singular (empty hats) and such runs are dropped from the plots.

Through HyperHDG (`sweep_hyperhdg_net_q1.sh`, degree 1): diffusion on grids gives 3.2 to 3.7 and 14
to 17 iterations, on mikado networks 7 to 10 with 4 cells per axis. Timoshenko beams, six dofs
per node, give 20 at 124k dofs and 16 at 404k dofs with 4 cells, but 50 to 85 with 8 cells and
hundreds on the smallest networks: plain hats replicated per component are a weak coarse space
for bending, which is what net2as's per-component coarse degrees (`-net2as_cb_degs`) address.
One run of either sweep is
`examples/laplacian_q1.py --domain x.geo.h5 -H H`; it prints YAML as it goes, one residual per PCG
iteration. The sweeps need GNU parallel, yq and jq; sweepplot and
fiber_network's `make_geo` come from the pinned submodules, so the generated networks and the
numbers above are reproducible from a fresh clone.

## Build

```
git submodule update --init --recursive                    # fiber_network, sweepplot, nanobind, HyperHDG (h5-io), pinned
pip install -r requirements.txt && pip install -e .        # in a venv on an externally managed Python
export PKG_CONFIG_PATH=<petsc prefix>/lib/pkgconfig      # or PETSC_DIR / PETSC_ARCH
cmake -B build -DPython_EXECUTABLE=$(which python) && cmake --build build
python -m pytest && (cd build && ctest)
```

The build puts the Python binding of the decomposition library, `netprecond._decomposition`,
into the package directory, so the editable install sees it. Without PETSc,
`-DNETPRECOND_PETSC=OFF` builds the library, the binding and its C++ test only, which is all
the Python tier needs. The examples and tests import the installed package; nothing is
path-hacked.

The tests pin the numerics to the frozen reference: the q1 decomposition has its coarse
basis and its subdomains, `AdditiveSchwarz` on it has its CG history on a grid Laplacian, and
net2as in q1 mode has it through `build/laplacian_net2as`; the recorded histories are in
`tests/golden/`. A number may only change together with its golden file.

`spack.yaml` describes an environment with PETSc (hdf5, metis, mumps).
