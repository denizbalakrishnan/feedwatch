#!/bin/sh
# make_zip.sh: builds the source zip for the hackathon's Round 1 form.
#
# It packs only files committed to git, so raw data, dev tools and videos
# cannot get in. The .gitattributes file also keeps out two committed folders:
# data/, because that file is derived from a dataset we did not make and the
# form asks for our own code only, and demo/, which holds the video's
# voiceover script.
#
# Usage, from the project folder:   sh scripts/make_zip.sh
set -e
mkdir -p submission
git archive --format=zip --output=submission/source.zip HEAD
ls -l submission/source.zip
