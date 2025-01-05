#!/bin/bash

# Define base paths
TARGET_DIR="/mmdet3d/data/nuscenes"
MULTICORRUPT_DIR="/MultiCorrupt/multicorrupt"

# Define config+checkpoint pairs with test modalities
declare -A CONFIGS_AND_CHECKPOINTS=(
    # sbnet
    ["/mmdet3d/projects/BEVFusion/configs/sbnet_256.py:both"]="/mmdet3d/data/nuscenes/sbnet.pth"
    
    # BEVFusion (test both camera and lidar corruptions)
    ["/mmdet3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py:both"]="/mmdet3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
    
)

# Exit on error
set -e

# Function to safely create symbolic links
create_safe_link() {
    local source=$1
    local target=$2
    local backup="${target}.original"
    
    if [ -e "$source" ]; then
        # Backup original data if it exists and isn't already backed up
        if [ -e "$target" ] && [ ! -e "$backup" ]; then
            mv "$target" "$backup"
        fi
        ln -sf "$source" "$target"
    else
        echo "Warning: Source $source does not exist, skipping..."
    fi
}

# Add this function to restore original data
restore_original_data() {
    local target_dir="$TARGET_DIR"
    
    # Restore LIDAR data
    if [ -e "${target_dir}/samples/LIDAR_TOP.original" ]; then
        rm -f "${target_dir}/samples/LIDAR_TOP"
        mv "${target_dir}/samples/LIDAR_TOP.original" "${target_dir}/samples/LIDAR_TOP"
    fi
    if [ -e "${target_dir}/sweeps/LIDAR_TOP.original" ]; then
        rm -f "${target_dir}/sweeps/LIDAR_TOP"
        mv "${target_dir}/sweeps/LIDAR_TOP.original" "${target_dir}/sweeps/LIDAR_TOP"
    fi
    
    # Restore camera data
    for cam in CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT; do
        if [ -e "${target_dir}/samples/${cam}.original" ]; then
            rm -f "${target_dir}/samples/${cam}"
            mv "${target_dir}/samples/${cam}.original" "${target_dir}/samples/${cam}"
        fi
    done
}

# Add trap to ensure cleanup on script exit
trap restore_original_data EXIT

# Function to detect modality type for a given version
detect_modality() {
    local version_dir="$1"
    local has_lidar=false
    local has_cameras=false

    # Check for LIDAR_TOP in both samples and sweeps
    if [ -d "${version_dir}/samples/LIDAR_TOP" ] && [ -d "${version_dir}/sweeps/LIDAR_TOP" ]; then
        has_lidar=true
    fi

    # Check for any camera folder in samples
    if [ -d "${version_dir}/samples" ]; then
        for cam in CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT; do
            if [ -d "${version_dir}/samples/${cam}" ]; then
                has_cameras=true
                break
            fi
        done
    fi

    # Determine modality type
    if $has_lidar && $has_cameras; then
        echo "both"
    elif $has_lidar; then
        echo "lidar"
    elif $has_cameras; then
        echo "camera"
    else
        echo "unknown"
    fi
}

# Function to link modality
link_modality() {
    local version_dir=$1
    local modality=$2

    case $modality in
        "both")
            create_safe_link "${version_dir}/samples/LIDAR_TOP" "${TARGET_DIR}/samples/LIDAR_TOP"
            create_safe_link "${version_dir}/sweeps/LIDAR_TOP" "${TARGET_DIR}/sweeps/LIDAR_TOP"
            for cam in CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT; do
                create_safe_link "${version_dir}/samples/${cam}" "${TARGET_DIR}/samples/${cam}"
            done
            ;;
            
        "lidar")
            create_safe_link "${version_dir}/samples/LIDAR_TOP" "${TARGET_DIR}/samples/LIDAR_TOP"
            create_safe_link "${version_dir}/sweeps/LIDAR_TOP" "${TARGET_DIR}/sweeps/LIDAR_TOP"
            ;;
            
        "camera")
            for cam in CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT; do
                create_safe_link "${version_dir}/samples/${cam}" "${TARGET_DIR}/samples/${cam}"
            done
            ;;
    esac
}

# Get all corruption directories
corrupt_dirs=($(ls -d ${MULTICORRUPT_DIR}/*))

if [ ${#corrupt_dirs[@]} -eq 0 ]; then
    echo "No corruption directories found in ${MULTICORRUPT_DIR}"
    exit 1
fi

# Process each corruption type
for corrupt_dir in "${corrupt_dirs[@]}"; do
    corruption_type=$(basename "$corrupt_dir")
    echo "Processing corruption type: ${corruption_type}"

    # Get all version directories for this corruption type
    versions=($(ls -d ${corrupt_dir}/* | grep -E '/[0-9]+$' | sort -n))

    if [ ${#versions[@]} -eq 0 ]; then
        echo "No version directories found in ${corrupt_dir}"
        continue
    fi

    # Process each version
    for version_dir in "${versions[@]}"; do
        version=$(basename "$version_dir")
        echo "Processing version ${version} for corruption type ${corruption_type}..."
        
        # Detect modality type
        modality=$(detect_modality "$version_dir")
        echo "Detected modality: ${modality}"
        
        if [ "$modality" = "unknown" ]; then
            echo "Warning: Could not determine modality type for version ${version}, skipping..."
            continue
        fi

        # Link the detected modality
        link_modality "$version_dir" "$modality"

        # Run tests for each config+checkpoint pair
        for config_with_modality in "${!CONFIGS_AND_CHECKPOINTS[@]}"; do
            # Split config path and test modality
            config="${config_with_modality%:*}"
            test_modality="${config_with_modality#*:}"
            checkpoint="${CONFIGS_AND_CHECKPOINTS[$config_with_modality]}"
            
            # Skip if current modality doesn't match what we want to test
            case "$test_modality" in
                "both")
                    # Test all modalities
                    ;;
                "camera")
                    # Only test when lidar is corrupted
                    if [ "$modality" != "lidar" ]; then
                        continue
                    fi
                    ;;
                "lidar")
                    # Only test when camera is corrupted
                    if [ "$modality" != "camera" ]; then
                        continue
                    fi
                    ;;
                *)
                    echo "Unknown test modality: ${test_modality}"
                    continue
                    ;;
            esac
            
            echo "Running test for corruption type ${corruption_type}, version ${version}"
            echo "Using config: ${config}"
            echo "Using checkpoint: ${checkpoint}"
            echo "Testing modality: ${test_modality}"
            
            if ! bash tools/test_missing_modalities.sh "${config}" "${checkpoint}" 1; then
                echo "Test failed for corruption type ${corruption_type}, version ${version}"
                echo "With config: ${config}"
                continue
            fi
        done
    done
done

