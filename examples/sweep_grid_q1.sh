#!/bin/bash
# Condition number of the q1-preconditioned grid Laplacian against the number of nodes and
# against the coarse grid size: both stay bounded (Goertz, Hellman, Maalqvist 2024, Thm. 4.6; the
# bound depends on the network constants only). The grids come from the pinned fiber_network
# submodule's make_geo. Needs the package installed (pip install -e .), the submodules
# (git submodule update --init), GNU parallel, yq (github.com/mikefarah/yq), jq.
set -eo pipefail
cd "$(dirname "$0")/.."
: ${SWEEPPLOT:=python sweepplot/sweepplot.py}                       # pinned submodules; or installed tools
: ${MAKE_GEO:=python fiber_network/fiber_network/make_geo.py}
: ${OUTDIR:=output}
: ${NS:=17 33 65 129 257}        # points per axis
: ${HS:=4 8 16 32}               # coarse cells per axis (a cell needs nodes inside: H < N - 1; other pairs fail and are skipped)
: ${JOBS:=4}
NAME=$(basename -s .sh $0)
OUT=$OUTDIR/$NAME
LOG=$OUT/data.jsonl
mkdir -p $OUT
git rev-parse HEAD > $OUT/rev; git diff-index --quiet HEAD || echo dirty >> $OUT/rev

# the grids, Dirichlet on the whole boundary, which the reference method assumes
for n in $NS; do
  [ -e $OUT/grid-$n.geo.h5 ] || $MAKE_GEO --grid $n --dirichlet xmin xmax ymin ymax -o $OUT/grid-$n.geo.h5 > $OUT/grid-$n.log 2>&1
done

parallel -j $JOBS --bar --results $OUT/runs.json \
  "python examples/q1.py --domain $OUT/grid-{1}.geo.h5 -H {2} --label {1}" ::: $NS ::: $HS \
  || true        # parallel exits with the number of failed jobs; the invalid pairs
yq -i '.Stdout |= (select(. != "") | from_yaml) // .' $OUT/runs.json   # the YAML each run printed; a failed run may be empty
jq -c 'select(.Exitval == 0) | .Stdout | .grid = .label' $OUT/runs.json > $LOG

$SWEEPPLOT -x nodes -y cond -g H --log x --xbase 2 --ylim 0,20 --save $OUT/cond-nodes.png --tikz $OUT/cond-nodes --nshow < $LOG
$SWEEPPLOT -x H -y cond -g grid --log x --xbase 2 --xlabel 'coarse cells per axis' --ylim 0,20 --save $OUT/cond-H.png --tikz $OUT/cond-H --nshow < $LOG
$SWEEPPLOT -x nodes -y iterations -g H --log x --xbase 2 --save $OUT/iterations-nodes.png --nshow < $LOG
$SWEEPPLOT -x nodes -y cond -g H --table $OUT/cond < $LOG
$SWEEPPLOT -x nodes -y cond -g H --table - < $LOG
$SWEEPPLOT -x nodes -y iterations -g H --table - < $LOG
