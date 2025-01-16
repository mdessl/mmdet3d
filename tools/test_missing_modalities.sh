#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3  # Remove first 3 arguments

# Default modalities
DEFAULT_MODALITIES=("lidar")

# Parse named arguments
MODALITIES=()
BASE_WORK_DIR=""
RATIOS=(1.0)  # Default ratios

while [[ $# -gt 0 ]]; do
    case $1 in
        --modalities)
            IFS=',' read -ra MODALITIES <<< "$2"
            shift 2
            ;;
        --work-dir)
            BASE_WORK_DIR="$2"
            shift 2
            ;;
        --ratios)
            IFS=',' read -ra RATIOS <<< "$2"
            shift 2
            ;;
        *)
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Use default modalities if none provided
if [ ${#MODALITIES[@]} -eq 0 ]; then
    MODALITIES=("${DEFAULT_MODALITIES[@]}")
fi

# Use default work dir if none provided
if [ -z "$BASE_WORK_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BASE_WORK_DIR="work_dirs/missing_modality_tests_${TIMESTAMP}"
fi

# Create base work directory
mkdir -p "${BASE_WORK_DIR}"

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

echo "${MODALITIES[@]}"

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
        
        wait_and_check $!
        
        echo "Completed test for ${MODALITY} at ratio ${RATIO}"
        echo "----------------------------------------"
        
        sleep 2
    done
done

