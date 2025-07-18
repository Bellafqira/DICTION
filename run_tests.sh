#!/bin/bash

methods=("DICTION") # values: ("DICTION", "DEEPSIGNS", "UCHIDA", "RES_ENCRYPT", "HUFUNET")
models=("RESNET18") # values: ("MLP" "CNN" "RESNET18" "MLP_RIGA")
operations=("DUMMY_NEURONS" "DISTILLATION") # values: ("TRAIN" "WATERMARKING" "PRUNING" "OVERWRITING" "FINE_TUNING" "SHOW" "PIA" "DUMMY_NEURONS" "DISTILLATION")

for method in "${methods[@]}"; do
    for model in "${models[@]}"; do
        for operation in "${operations[@]}"; do
            echo -e "\nRunning $method with $model and $operation"

            # Determine the output directory
            if [ "$operation" == "TRAIN" ]; then
                output_dir="outs/$operation"
            else
                output_dir="outs/$operation/$method"
            fi

            # Createt the output directory if it does not exist
            mkdir -p "$output_dir"

            # Execute the python script and output the results to a file
            python test_case.py --method "$method" --model "$model" --operation "$operation" | tee -a "$output_dir/$model.txt"
        done
    done
done

echo "Operations completed."
