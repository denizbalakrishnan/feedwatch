#!/bin/sh
# make_zip.sh: builds the source zip for the hackathon's Round 1 form.
#
# It packs exactly the files committed to git, so nothing stray gets in:
# no raw data, no dev tools, no videos. The .gitattributes file also keeps
# out data/, because that file is derived from a dataset we did not make and
# the form asks for our own code only.
#
# Usage, from the project folder:   sh scripts/make_zip.sh
set -e
mkdir -p submission
git archive --format=zip --output=submission/source.zip HEAD
ls -l submission/source.zip
