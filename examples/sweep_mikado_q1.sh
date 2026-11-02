#!/bin/bash
# As sweep_grid_q1.sh, on mikado fiber networks against the network size (total fiber length
# MASS, fixed seed) and the coarse grid size. The networks come from the pinned fiber_network
# submodule's make_geo and are generated, not stored: the numbers depend on the generator
# version, which the submodule fixes. Needs the package installed, GNU parallel, yq, jq.
set -eo pipefail
cd "$(dirname "$0")/.."
: ${SWEEPPLOT:=python sweepplot/sweepplot.py}                       # pinned submodules; or installed tools
: ${MAKE_GEO:=python fiber_network/fiber_network/make_geo.py}
: ${OUTDIR:=output}
: ${MASSES:=50 100 200 400 800}     # total fiber length on the unit square, fiber length 0.05
: ${HS:=4 8 16}                     # coarse cells per axis; too fine for a sparse net -> singular coarse problem, skipped
: ${SEED:=1}
: ${JOBS:=4}
NAME=$(basename -s .sh $0)
OUT=$OUTDIR/$NAME
LOG=$OUT/data.jsonl
mkdir -p $OUT
git rev-parse HEAD > $OUT/rev; git diff-index --quiet HEAD || echo dirty >> $OUT/rev

# Dirichlet on the whole bounding box, which the reference method assumes
for m in $MASSES; do
  [ -e $OUT/mikado-$m.geo.h5 ] || $MAKE_GEO --mikado $m --seed $SEED --dirichlet xmin xmax ymin ymax \
                                           -o $OUT/mikado-$m.geo.h5 > $OUT/mikado-$m.log 2>&1
done

parallel -j $JOBS --bar --results $OUT/runs.json \
  "python examples/q1.py --domain $OUT/mikado-{1}.geo.h5 -H {2} --label {1}" ::: $MASSES ::: $HS \
  || true
yq -i '.Stdout |= (select(. != "") | from_yaml) // .' $OUT/runs.json   # the YAML each run printed; a failed run may be empty
jq -c 'select(.Exitval == 0) | .Stdout | .mass = .label' $OUT/runs.json > $LOG

$SWEEPPLOT -x nodes -y cond -g H --log x --xbase 2 --save $OUT/cond-nodes.png --tikz $OUT/cond-nodes --nshow < $LOG
$SWEEPPLOT -x H -y cond -g mass --log x --xbase 2 --xlabel 'coarse cells per axis' --save $OUT/cond-H.png --tikz $OUT/cond-H --nshow < $LOG
$SWEEPPLOT -x nodes -y cond -g H --table - < $LOG
$SWEEPPLOT -x nodes -y iterations -g H --table - < $LOG
