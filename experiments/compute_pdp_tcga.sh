#!/bin/zsh
PREFXS=("brca" "hnsc" "kirc" "luad" "lusc" "ov" "paad" "skcm" "stad")
CROSS_VAL=true
COMPUTE_DIFF_VAR=false
FULL_TRAIN=false
for PREFX in $PREFXS
do
    if $CROSS_VAL
    then
        for SPLIT_ID in {1..10}
        do
            python src/main_pdp.py \
            --path-train-data datasets_filtered/${PREFX}_train.csv \
            --path-val-data datasets_filtered/${PREFX}_valid.csv \
            --path-save outputs/pdp_diff-var=${COMPUTE_DIFF_VAR}_full-train=${FULL_TRAIN}_split=${SPLIT_ID}_${PREFX}.csv \
            --path-json datasets_filtered/${PREFX}_cv10split.json \
            --split-id $SPLIT_ID \
            --path-perf-csv outputs/performance.csv \
            --data-id $PREFX \
            --model-id rsf \
            --reduction $COMPUTE_DIFF_VAR \
            --merge $FULL_TRAIN \
            --seed 0 \
            --ibs-only
        done
    else
            python src/main_pdp.py \
            --path-train-data datasets_filtered/${PREFX}_train.csv \
            --path-val-data datasets_filtered/${PREFX}_valid.csv \
            --path-save outputs/pdp_diff-var=${COMPUTE_DIFF_VAR}_full-train=${FULL_TRAIN}_${PREFX}.csv \
            --path-perf-csv outputs/performance.csv \
            --data-id $PREFX \
            --model-id rsf \
            --reduction $COMPUTE_DIFF_VAR \
            --merge $FULL_TRAIN \
            --seed 0
    fi
done