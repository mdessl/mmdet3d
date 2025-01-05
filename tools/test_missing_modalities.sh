#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3  # Remove first 3 arguments

# Default modalities
DEFAULT_MODALITIES=("camera")

# Parse named arguments
MODALITIES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --modalities)
            # Split the comma-separated modalities into array
            IFS=',' read -ra MODALITIES <<< "$2"
            shift 2
            ;;
        *)
            # Store other arguments
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Use default modalities if none provided
if [ ${#MODALITIES[@]} -eq 0 ]; then
    MODALITIES=("${DEFAULT_MODALITIES[@]}")
fi

# Array of ratios to test
RATIOS=(0.0 1.0)

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

