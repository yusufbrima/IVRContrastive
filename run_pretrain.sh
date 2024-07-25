#!/bin/bash

PYTHON_SCRIPT="contrastive.py"
EXPERIMENT_COUNTER=1

for target_class in "chimpanzee_ir"; do
    for modelstr in "dense121" "resnet18"; do
        for contrastive_method in "supcon" "triplet"; do
            echo "Running experiment $EXPERIMENT_COUNTER with target_class=$target_class, modelstr=$modelstr, contrastive_method=$contrastive_method"
            python3 "$PYTHON_SCRIPT" --experiment "$EXPERIMENT_COUNTER" --target_class "$target_class" --modelstr "$modelstr" --contrastive_method "$contrastive_method"
            # ((EXPERIMENT_COUNTER++))
        done
    done
done

echo "All experiments completed."