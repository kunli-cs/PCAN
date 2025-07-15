#!/usr/bin/env bash

# set up environment
# conda env create -f environment.yml
source activate gym
# pip install mmengine
# pip install --upgrade youtube-dl

DATA_DIR="/home/peco/data/gym"
ANNO_DIR="/home/peco/data/gym/annotations"
python download.py ${ANNO_DIR}/annotation.json ${DATA_DIR}/videos

source deactivate gym
conda remove -n gym --all
