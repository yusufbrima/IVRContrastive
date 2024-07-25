#!/bin/bash

PYTHON_SCRIPT="ft_full.py"

# Define the parameter arrays
modelstrs=("resnet18" "dense121")
contrastive_methods=("triplet" "supcon")

# Fix target_class to "chimpanzee_ir"
TARGET_CLASS="chimpanzee_ir"

# Outer loop to iterate over experiments 0 to 50
for EXPERIMENT in {0..50}; do
    # Loop through all combinations
    for modelstr in "${modelstrs[@]}"; do
        for contrastive_method in "${contrastive_methods[@]}"; do
            echo "Running experiment $EXPERIMENT with modelstr=$modelstr, target_class=$TARGET_CLASS, contrastive_method=$contrastive_method"

            python3 "$PYTHON_SCRIPT" \
                --experiment "$EXPERIMENT" \
                --modelstr "$modelstr" \
                --target_class "$TARGET_CLASS" \
                --contrastive_method "$contrastive_method"

            echo "Completed experiment $EXPERIMENT with modelstr=$modelstr, target_class=$TARGET_CLASS, contrastive_method=$contrastive_method"
            echo "--------------------"
        done
    done
    echo "Completed all combinations for experiment $EXPERIMENT."
done

echo "All experiments completed."
