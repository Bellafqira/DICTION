#!/bin/bash

source .venv/bin/activate

methods=("UCHIDA") # values: ("DICTION", "DEEPSIGNS", "UCHIDA", "RES_ENCRYPT")
models=("MLP_RIGA") # values: ("MLP" "CNN" "RESNET18" "MLP_RIGA")
operations=("WATERMARKING" "PRUNING" "FINE_TUNING" "OVERWRITING" "SHOW") # values: ("TRAIN" "WATERMARKING" "PRUNING" "OVERWRITING" "FINE_TUNING" "SHOW" "PIA")

for method in "${methods[@]}"; do
    for model in "${models[@]}"; do
        for operation in "${operations[@]}"; do
            echo -e "\nRunning $method with $model and $operation"
            # Create the output directory if it does not exist
            mkdir -p outs/"$operation"/"$method"
            # Execute the python script and output the results to a file
            python test_case.py --method "$method" --model "$model" --operation "$operation" | tee -a outs/"$operation"/"$method"/"$model".txt
        done
    done
done

echo "Operations completed."