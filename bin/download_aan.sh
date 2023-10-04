DST_DIR=/mnt/pro980/resurrecting_rnns
mkdir -p $DST_DIR

# Download the raw AAN data from the TutorialBank Corpus.
wget -v https://github.com/Yale-LILY/TutorialBank/blob/master/resources-v2022-clean.tsv -P $DST_DIR
