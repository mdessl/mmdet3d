#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
CFG_OPTIONS="$@"  # Capture remaining arguments

# Array of ratios to test
RATIOS=(1.0) #0.0 0.1 0.3 0.5 0.7 0.9 
MODALITIES=("lidar") # "camera" "lidar" 

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
        
        # Run the test and wait for it to complete
        ./tools/dist_test.sh \
            ${CONFIG} \
            ${CHECKPOINT} \
            ${GPUS} \
            --missing_modality ${MODALITY} \
            --missing_ratio ${RATIO} \
            --work-dir ${WORK_DIR} \
            2>&1 | tee "${WORK_DIR}/test.log"
        
        #--cfg-options test_dataloader.dataset.metainfo.version=v1.0-mini train_dataloader.dataset.dataset.metainfo.version=v1.0-mini

        # Wait for the test to complete before starting the next one
        wait_and_check $!
        
        echo "Completed test for ${MODALITY} at ratio ${RATIO}"
        echo "----------------------------------------"
        
        # Add a small delay between tests to ensure resources are freed
        sleep 2
    done
done

