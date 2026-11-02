#!/bin/bash
# The reference method on HDG systems assembled by HyperHDG: diffusion on grids and mikado
# networks, Timoshenko beams on mikado networks (its local frame is degenerate on axis-aligned
# edges, so no grids), against the network size and the coarse grid size. Networks from the
# pinned fiber_network's make_geo; HyperHDG compiles its module on the first run (minutes).
# Needs the package installed, the submodules, GNU parallel, yq, jq, and HDF5_ROOT if HDF5 is
# not on the default search path.
set -eo pipefail
cd "$(dirname "$0")/.."
: ${SWEEPPLOT:=python submodules/sweepplot/sweepplot.py}
: ${MAKE_GEO:=python submodules/fiber_network/fiber_network/make_geo.py}
: ${OUTDIR:=output}
: ${NS:=17 33 65}                 # grid points per axis (diffusion)
: ${MASSES:=50 100 200 400}       # mikado total fiber length (both problems)
: ${HS:=4 8}                      # coarse cells per axis
: ${DEGREE:=1}
: ${SEED:=1}
: ${JOBS:=2}                      # Timoshenko factorizations are memory hungry
NAME=$(basename -s .sh $0)
OUT=$OUTDIR/$NAME
LOG=$OUT/data.jsonl
mkdir -p $OUT
git rev-parse HEAD > $OUT/rev; git diff-index --quiet HEAD || echo dirty >> $OUT/rev

for n in $NS; do
  [ -e $OUT/grid-$n.geo.h5 ] || $MAKE_GEO --grid $n -t 1e-6 --dirichlet xmin xmax ymin ymax -o $OUT/grid-$n.geo.h5 > $OUT/grid-$n.log 2>&1
done
for m in $MASSES; do
  [ -e $OUT/mikado-$m.geo.h5 ] || $MAKE_GEO --mikado $m --seed $SEED --dirichlet xmin xmax ymin ymax -o $OUT/mikado-$m.geo.h5 > $OUT/mikado-$m.log 2>&1
done
# compile the HyperHDG module once, before parallel asks for it from several jobs at once
python examples/hyperhdg_net_q1.py --domain $OUT/grid-17.geo.h5 --problem diffusion --degree $DEGREE -H 4 > /dev/null

RUN="python examples/hyperhdg_net_q1.py --degree $DEGREE"
{ for n in $NS; do for H in $HS; do echo "$RUN --domain $OUT/grid-$n.geo.h5 --problem diffusion -H $H --label grid-$n"; done; done
  for m in $MASSES; do for H in $HS; do for p in diffusion timoshenko; do echo "$RUN --domain $OUT/mikado-$m.geo.h5 --problem $p -H $H --label mikado-$m"; done; done; done
} | parallel -j $JOBS --bar --results $OUT/runs.json || true     # parallel exits with the number of failed jobs
yq -i '.Stdout |= (select(. != "") | from_yaml) // .' $OUT/runs.json
jq -c 'select(.Exitval == 0) | .Stdout | .network = (.label | split("-")[0])' $OUT/runs.json > $LOG

for p in diffusion timoshenko; do
  jq -c "select(.problem == \"$p\")" $LOG \
    | $SWEEPPLOT -x dofs -y cond -g H --group0 network --log x --xbase 2 --xlabel 'dofs' --ylabel 'cond' \
                 --title "$p" --save $OUT/cond-dofs-$p.png --nshow
  echo "== $p: cond"; jq -c "select(.problem == \"$p\")" $LOG | $SWEEPPLOT -x dofs -y cond -g H --group0 network --table -
  echo "== $p: iterations"; jq -c "select(.problem == \"$p\")" $LOG | $SWEEPPLOT -x dofs -y iterations -g H --group0 network --table -
done
