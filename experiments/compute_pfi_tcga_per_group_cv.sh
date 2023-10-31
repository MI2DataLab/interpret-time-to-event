#!/bin/zsh
PREFXS=("blca" "lgg" "brca" "hnsc" "kirc" "luad" "lusc" "ov" "paad" "skcm" "stad")
for PREFX in $PREFXS
do
    for SPLIT_ID in {1..10}
    do
        python src/main_pfi.py \
        --path-train-data datasets_filtered/${PREFX}_train.csv \
        --path-val-data datasets_filtered/${PREFX}_valid.csv \
        --path-json datasets_filtered/${PREFX}_cv10split.json \
        --split-id $SPLIT_ID \
        --seed 0 \
        --path-save outputs/gpfi_${PREFX}_split=${SPLIT_ID}.csv \
        --permute-groups \
        --n-permutations 50
    done
done