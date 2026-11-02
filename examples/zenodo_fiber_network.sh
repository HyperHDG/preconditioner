#!/bin/bash
# Fetch the paper fiber networks of Holten, Rupp et al., "Paper Fiber Network Models for Elastic
# Wave Simulation", Zenodo, https://doi.org/10.5281/zenodo.20410514 (CC BY 4.0), and convert them
# with fiber_network's make_geo into .geo.h5 domains the examples take. net2 (8 x 8 mm, 1.6M
# nodes, 238 MB) is fetched and converted whole and as a clamped 2 x 2 mm piece for the
# scipy-based examples; net3 (40 x 40 mm, 41M nodes, 6 GB) only with NET3=1. Units kg, um, s.
#
#   examples/zenodo_fiber_network.sh
#   python examples/hyperhdg_q1.py --domain output/zenodo/net2-crop.geo.h5 --problem timoshenko -H 8
#   ./build/laplacian_net2as -domain output/zenodo/net2.geo.h5 -net2as_p 15 -net2as_cb_type q1 -net2as_cb_trim
set -eo pipefail
cd "$(dirname "$0")/.."
: ${MAKE_GEO:=python submodules/fiber_network/fiber_network/make_geo.py}
: ${DATA:=data/zenodo-20410514}          # the downloads, kept; ignored by git
: ${OUT:=output/zenodo}
: ${CROP:=0.25}                          # fraction of the sheet kept per axis for the crop
RECORD=https://zenodo.org/api/records/20410514/files
mkdir -p $DATA $OUT
nets="net2"; [ "${NET3:-0}" = 1 ] && nets="net2 net3"
for name in $nets; do
  [ -e $DATA/$name.zip ] || curl -sSL -o $DATA/$name.zip $RECORD/$name.zip/content
  [ -d $DATA/$name ] || unzip -q $DATA/$name.zip -d $DATA
  # Dirichlet on the whole bounding box, which the reference method assumes
  [ -e $OUT/$name.geo.h5 ] || $MAKE_GEO -i $DATA/$name --dirichlet xmin xmax ymin ymax -o $OUT/$name.geo.h5 > $OUT/$name.log 2>&1
  [ -e $OUT/$name-crop.geo.h5 ] || $MAKE_GEO -i $DATA/$name --clamp-xy $CROP --dirichlet xmin xmax ymin ymax -o $OUT/$name-crop.geo.h5 > $OUT/$name-crop.log 2>&1
done
ls -la $OUT/*.geo.h5
