#!/bin/bash

source venv/bin/activate

methods=("DICTION") # values: ("DICTION", "DEEPSIGNS", "UCHIDA", "RES_ENCRYPT")
models=("RESNET18") # values: ("MLP", "CNN", "RESNET18", "MLP_RIGA)
operations=("SHOW") #  values: ("WATERMARKING", "TRAIN", "FINE_TUNING", "OVERWRITING", "PRUNING", "SHOW", "PIA")

for method in "${methods[@]}"; do
    for model in "${models[@]}"; do
        for operation in "${operations[@]}"; do
            echo -e "\nRunning $method with $model and $operation"
            python test_case.py --method $method --model $model --operation $operation | tee out.txt
        done
    done
done