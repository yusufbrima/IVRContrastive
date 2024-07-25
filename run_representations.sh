#!/bin/bash

# Define arrays of options for each parameter
models=("dense121" "resnet18")
target_classes=("chimpanzee_ir")
model_types=("classifier" "contrastive")
methods=("supcon" "triplet")

# define experiment ids for the best dense121 and resnet18 classifiers 

# Nested for loops to cover all combinations
for model in "${models[@]}"; do
    for target_class in "${target_classes[@]}"; do
        for model_type in "${model_types[@]}"; do
            for method in "${methods[@]}"; do
                # Set the experiment number based on both the model and method
                if [ "$model" == "dense121" ] && [ "$model_type" == "classifier" ]; then
                    experiment=7
                elif [ "$model" == "resnet18" ] && [ "$model_type" == "classifier" ]; then
                    experiment=22
                else
                    experiment=1  # Default experiment number for all other cases
                fi

                # Execute the Python script with the current combination of arguments and experiment flag
                echo "Running: $model $target_class $model_type $method with experiment $experiment"
                python extract_rep.py --modelstr $model --target_class $target_class --model_type $model_type --method $method --experiment $experiment
                echo "Done: $model $target_class $model_type $method with experiment $experiment"
            done
        done
    done
done
