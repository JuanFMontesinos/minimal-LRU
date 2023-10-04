DST_DIR=/mnt/pro980/resurrecting_rnns
mkdir -p $DST_DIR

rm -rf $DST_DIR
mkdir -p $DST_DIR

./bin/download_lra.sh
./bin/download_aan.sh

