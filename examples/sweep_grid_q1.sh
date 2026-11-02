#!/bin/bash
# Condition number of the q1-preconditioned grid Laplacian against the number of nodes and
# against the coarse grid size: both should stay bounded (Goertz, Hellman, Maalqvist 2024,
# Thm. 4.6; the bound depends on the network constants only). The reference method via jprecond
# (tests/ show they agree). Needs GNU parallel, jq, sweepplot (github.com/HyperHDG/sweepplot).
set -eo pipefail
cd "$(dirname "$0")/.."
: ${OUTDIR:=output}
: ${NS:=17 33 65 129 257}        # points per axis
: ${HS:=4 8 16 32}               # coarse cells per axis (must be < n - 1)
: ${JOBS:=4}
NAME=$(basename -s .sh $0)
OUT=$OUTDIR/$NAME
LOG=$OUT/data.jsonl
mkdir -p $OUT
git rev-parse HEAD > $OUT/rev; git diff-index --quiet HEAD || echo dirty >> $OUT/rev

parallel -j $JOBS --bar --results $OUT/runs.json \
  "python examples/grid_q1.py -n {1} -H {2}" ::: $NS ::: $HS \
  || true        # parallel exits with the number of failed jobs; failures are filtered below
# one record per successful run; parallel's Stdout holds the JSON line (invalid n/H pairs exit 2)
jq -c 'select(.Exitval == 0) | .Stdout | fromjson' $OUT/runs.json > $LOG

sweepplot -x nodes -y cond -g H --log x --xbase 2 --xlabel 'nodes' --ylabel 'cond' \
          --ylim 0,20 --save $OUT/cond-nodes.png --tikz $OUT/cond-nodes --nshow < $LOG
sweepplot -x H -y cond -g n --log x --xbase 2 --xlabel 'coarse cells per axis' --ylabel 'cond' \
          --ylim 0,20 --save $OUT/cond-H.png --tikz $OUT/cond-H --nshow < $LOG
sweepplot -x nodes -y iterations -g H --log x --xbase 2 --save $OUT/iterations-nodes.png --nshow < $LOG
sweepplot -x nodes -y cond -g H --table $OUT/cond < $LOG
sweepplot -x nodes -y cond -g H --table - < $LOG
sweepplot -x nodes -y iterations -g H --table - < $LOG
