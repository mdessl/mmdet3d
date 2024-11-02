#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
WORK_DIR=${4:-'work_dirs/modality_robustness'}
PORT=${5:-29500}

# Create work directory
mkdir -p $WORK_DIR

# Define modalities and removal rates
MODALITIES=("img")
REMOVAL_RATES=(0.0 1.0) #0.1 0.3 0.5 0.7 0.9 

# Run tests for each combination
for MODALITY in "${MODALITIES[@]}"; do
    for RATE in "${REMOVAL_RATES[@]}"; do
        TEST_DIR="${WORK_DIR}/${MODALITY}_removal_${RATE//./_}"
        mkdir -p $TEST_DIR
        
        echo "Testing ${MODALITY} removal rate: ${RATE}"
        
        ./tools/dist_test.sh \
            $CONFIG \
            $CHECKPOINT \
            $GPUS \
            --work-dir $TEST_DIR \
            --cfg-options \
            test_dataloader.dataset.metainfo.version=v1.0-mini \
            train_dataloader.dataset.dataset.metainfo.version=v1.0-mini \
            custom_hooks="[dict()]" \
            "custom_hooks.0.type=ModalityDropHook" \
            "custom_hooks.0.modality=$MODALITY" \
            "custom_hooks.0.drop_rate=$RATE" \          
            2>&1 | tee $TEST_DIR/test_log.txt
            
        # Increment PORT to avoid conflicts
        PORT=$((PORT + 1))
    done
done 