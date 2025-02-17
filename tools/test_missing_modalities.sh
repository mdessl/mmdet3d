#!/usr/bin/env bash

# Check for minimum required arguments
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 CONFIG CHECKPOINT GPUS [PORT] [GPU_IDS] [additional options...]"
    exit 1
fi

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${4:-29500}  # Default port 29500 if not specified
GPU_IDS=${5:-"all"}  # Default to all GPUs if not specified
CFG_OPTIONS="${@:6}"  # Capture remaining arguments starting from the 6th argument

# Array of ratios to test
RATIOS=(1.0) #0.0 0.1 0.3 0.5 0.7 0.9 
MODALITIES=("camera") # "camera" "lidar" 

# Create a timestamp for unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_WORK_DIR="work_dirs/missing_modality_tests_${TIMESTAMP}"

# Function to wait for a process and check its exit status
wait_and_check() {
    local pid=$1
    wait $pid
    local status=$?
    if [ $status -ne 0 ]; then
        echo "Error: Test failed with exit code $status"
        exit $status
    fi
}

for MODALITY in "${MODALITIES[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        echo "Testing with missing ${MODALITY} at ratio ${RATIO}"
        
        # Create specific work directory for this test
        WORK_DIR="${BASE_WORK_DIR}/${MODALITY}_${RATIO}"
        mkdir -p ${WORK_DIR}
        
        # Prepare the command based on whether GPU_IDS is specified
        if [ "$GPU_IDS" = "all" ]; then
            # Run without CUDA_VISIBLE_DEVICES
            PORT=${PORT} ./tools/dist_test.sh \
                ${CONFIG} \
                ${CHECKPOINT} \
                ${GPUS} \
                --missing_modality ${MODALITY} \
                --missing_ratio ${RATIO} \
                --work-dir ${WORK_DIR} \
                --cfg-options test_dataloader.dataset.metainfo.version=v1.0-mini train_dataloader.dataset.dataset.metainfo.version=v1.0-mini \
                2>&1 | tee "${WORK_DIR}/test.log"
        else
            # Run with specified GPUs
            CUDA_VISIBLE_DEVICES=${GPU_IDS} PORT=${PORT} ./tools/dist_test.sh \
                ${CONFIG} \
                ${CHECKPOINT} \
                ${GPUS} \
                --missing_modality ${MODALITY} \
                --missing_ratio ${RATIO} \
                --work-dir ${WORK_DIR} \
                --cfg-options test_dataloader.dataset.metainfo.version=v1.0-mini train_dataloader.dataset.dataset.metainfo.version=v1.0-mini \
                2>&1 | tee "${WORK_DIR}/test.log"
        fi
            
        # Wait for the test to complete before starting the next one
        wait_and_check $!
        
        echo "Completed test for ${MODALITY} at ratio ${RATIO}"
        echo "----------------------------------------"
        
        # Add a small delay between tests to ensure resources are freed
        sleep 2
    done
done


