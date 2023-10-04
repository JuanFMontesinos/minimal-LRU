DST_DIR=/mnt/pro980/resurrecting_rnns
mkdir -p $DST_DIR

# Clone and unpack the LRA object.
# This can take a long time, so get comfortable.
rm -rf $DST_DIR/lra_release.gz $DST_DIR/lra_release  # Clean out any old datasets.
wget -v https://storage.googleapis.com/long-range-arena/lra_release.gz -P $DST_DIR

# Add a progress bar because this can be slow.
pv $DST_DIR/lra_release.gz | tar -zx -C $DST_DIR
