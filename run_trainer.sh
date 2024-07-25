#!/bin/bash

# Define the target class
TARGET_CLASS="chimpanzee_ir"

# Iterate over experiment numbers from 0 to 50
for EXP in {0..50}
do
    # Iterate over two model strings
    for MODEL in "dense121" "resnet18"
    do
        echo "Running experiment number $EXP with model $MODEL"
        python trainer.py --modelstr $MODEL --experiment $EXP --target_class $TARGET_CLASS
    done
done
