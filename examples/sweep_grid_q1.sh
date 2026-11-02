#!/bin/bash
# Condition number of the q1-preconditioned grid Laplacian against the number of nodes and
# against the coarse grid size: both stay bounded (Goertz, Hellman, Maalqvist 2024, Thm. 4.6; the
# bound depends on the network constants only). Needs the package installed (pip install -e .),
# the submodules (git submodule update --init), GNU parallel, yq (github.com/mikefarah/yq), jq.
set -eo pipefail
cd "$(dirname "$0")/.."
: ${SWEEPPLOT:=python sweepplot/sweepplot.py}                       # pinned submodule; or an installed sweepplot
: ${OUTDIR:=output}
: ${NS:=17 33 65 129 257}        # points per axis
: ${HS:=4 8 16 32}               # coarse cells per axis (needs H < N - 1; other pairs are skipped)
: ${JOBS:=4}
NAME=$(basename -s .sh $0)
OUT=$OUTDIR/$NAME
LOG=$OUT/data.jsonl
mkdir -p $OUT
git rev-parse HEAD > $OUT/rev; git diff-index --quiet HEAD || echo dirty >> $OUT/rev

parallel -j $JOBS --bar --results $OUT/runs.json \
  "python examples/q1.py --grid {1} -H {2}" ::: $NS ::: $HS \
  || true        # parallel exits with the number of failed jobs; the invalid pairs
yq -i '.Stdout |= (select(. != "") | from_yaml) // .' $OUT/runs.json   # the YAML each run printed; a failed run may be empty
jq -c 'select(.Exitval == 0) | .Stdout' $OUT/runs.json > $LOG

$SWEEPPLOT -x nodes -y cond -g H --log x --xbase 2 --ylim 0,20 --save $OUT/cond-nodes.png --tikz $OUT/cond-nodes --nshow < $LOG
$SWEEPPLOT -x H -y cond -g grid --log x --xbase 2 --xlabel 'coarse cells per axis' --ylim 0,20 --save $OUT/cond-H.png --tikz $OUT/cond-H --nshow < $LOG
$SWEEPPLOT -x nodes -y iterations -g H --log x --xbase 2 --save $OUT/iterations-nodes.png --nshow < $LOG
$SWEEPPLOT -x nodes -y cond -g H --table $OUT/cond < $LOG
$SWEEPPLOT -x nodes -y cond -g H --table - < $LOG
$SWEEPPLOT -x nodes -y iterations -g H --table - < $LOG
