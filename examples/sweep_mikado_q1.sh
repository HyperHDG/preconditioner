#!/bin/bash
# As sweep_grid_q1.sh, on mikado fiber networks against the network size (total fiber length
# MASS, fixed seed) and the coarse grid size. The networks come from fiber_network's make_geo
# (github.com/HyperHDG/fiber_network; its pip install -e . puts make_geo on the PATH) and are
# generated, not stored. Needs the package installed, GNU parallel, yq, jq, sweepplot.
set -eo pipefail
cd "$(dirname "$0")/.."
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
  [ -e $OUT/mikado-$m.geo.h5 ] || make_geo --mikado $m --seed $SEED --dirichlet xmin xmax ymin ymax \
                                           -o $OUT/mikado-$m.geo.h5 > $OUT/mikado-$m.log 2>&1
done

parallel -j $JOBS --bar --results $OUT/runs.json \
  "python examples/q1.py --domain $OUT/mikado-{1}.geo.h5 -H {2} --label {1}" ::: $MASSES ::: $HS \
  || true
yq -i '.Stdout |= (select(. != "") | from_yaml) // .' $OUT/runs.json   # the YAML each run printed; a failed run may be empty
jq -c 'select(.Exitval == 0) | .Stdout | .mass = .label' $OUT/runs.json > $LOG

sweepplot -x nodes -y cond -g H --log x --xbase 2 --save $OUT/cond-nodes.png --tikz $OUT/cond-nodes --nshow < $LOG
sweepplot -x H -y cond -g mass --log x --xbase 2 --xlabel 'coarse cells per axis' --save $OUT/cond-H.png --tikz $OUT/cond-H --nshow < $LOG
sweepplot -x nodes -y cond -g H --table - < $LOG
sweepplot -x nodes -y iterations -g H --table - < $LOG
