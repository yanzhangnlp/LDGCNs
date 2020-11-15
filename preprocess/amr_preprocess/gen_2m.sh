#!/bin/bash

# CHANGE THIS
REPO_DIR=..

# CONSTANTS
DATA_DIR=../data
PREPROC_DIR=../data/tmp_amr
ORIG_AMR_DIR=../data/abstract_meaning_representation_amr_2.0/data/alignments/split
FINAL_AMR_DIR=../data/amr

#####
# CREATE FOLDER STRUCTURE

mkdir -p ../data/tmp_amr/train

mkdir -p ../data/amr/

#####
cat ${ORIG_AMR_DIR}/train/* > ${PREPROC_DIR}/train/raw_amrs.txt

#####
# CONVERT ORIGINAL AMR SEMBANK TO ONELINE FORMAT
# for SPLIT in train; do
python split_amr.py ../data/tmp_amr/train/raw_amrs.txt ../data/tmp_amr/train/surface.txt ../data/tmp_amr/train/graphs.txt
python preproc_amr.py  ../data/tmp_amr/train/graphs.txt  ../data/tmp_amr/train/surface.txt   ../data/amr/train.amr   ../data/amr/train_surface.pp.txt --mode LINE_GRAPH --triples-output ../data/amr/train.grh --anon --map-output ../data/amr/train_map.pp.txt --anon-surface ../data/amr/train.snt --nodes-scope ../data/amr/train_nodes.scope.pp.txt --scope
paste ${FINAL_AMR_DIR}/${SPLIT}.amr ${FINAL_AMR_DIR}/${SPLIT}.grh > ${FINAL_AMR_DIR}/${SPLIT}.amrgrh
# done




# python3 global_node.py --input_dir ${FINAL_AMR_DIR}/

# for SPLIT in train dev test; do
# 	cp ${FINAL_AMR_DIR}/${SPLIT}.amr_g  ${FINAL_AMR_DIR}/${SPLIT}.amr
# 	cp ${FINAL_AMR_DIR}/${SPLIT}.grh_g  ${FINAL_AMR_DIR}/${SPLIT}.grh
# 	cp ${FINAL_AMR_DIR}/${SPLIT}.amrgrh_g  ${FINAL_AMR_DIR}/${SPLIT}.amrgrh
# done

# rm ${FINAL_AMR_DIR}/*_g

# echo '{"d": 1, "r": 2, "s": 3, "g": 4}' > ${FINAL_AMR_DIR}/edge_vocab.json




