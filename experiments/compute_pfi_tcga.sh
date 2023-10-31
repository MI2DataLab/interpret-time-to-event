#!/bin/zsh
PREFXS=("blca" "lgg" "brca" "hnsc" "kirc" "luad" "lusc" "ov" "paad" "skcm" "stad")
for PREFX in $PREFXS
do
    python src/main_pfi.py \
    --path-train-data datasets_filtered/${PREFX}_train.csv \
    --path-val-data datasets_filtered/${PREFX}_valid.csv \
    --seed 0 \
    --path-save outputs/pfi_${PREFX}.csv
done