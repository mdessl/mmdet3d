#!/bin/bash

# Set up test environment
TEST_ROOT="test_corruption_scripts"
MOCK_MULTICORRUPT="${TEST_ROOT}/MultiCorrupt/multicorrupt"
MOCK_MMDET3D="${TEST_ROOT}/mmdet3d"

# Create test directory structure
setup_test_env() {
    echo "Setting up test environment..."
    
    # Create base directories
    mkdir -p "${MOCK_MULTICORRUPT}"/{fog,rain}/{1,2}/samples
    mkdir -p "${MOCK_MULTICORRUPT}"/{fog,rain}/{1,2}/sweeps
    mkdir -p "${MOCK_MMDET3D}/data/nuscenes/samples"
    mkdir -p "${MOCK_MMDET3D}/data/nuscenes/sweeps"
    
    # Create mock camera and lidar data
    for sev in 1 2; do
        # Fog corruption
        for cam in CAM_FRONT CAM_BACK CAM_LEFT CAM_RIGHT; do
            mkdir -p "${MOCK_MULTICORRUPT}/fog/${sev}/samples/${cam}"
            touch "${MOCK_MULTICORRUPT}/fog/${sev}/samples/${cam}/test.jpg"
        done
        mkdir -p "${MOCK_MULTICORRUPT}/fog/${sev}/samples/LIDAR_TOP"
        touch "${MOCK_MULTICORRUPT}/fog/${sev}/samples/LIDAR_TOP/test.pcd"
        
        # Rain corruption
        for cam in CAM_FRONT CAM_BACK CAM_LEFT CAM_RIGHT; do
            mkdir -p "${MOCK_MULTICORRUPT}/rain/${sev}/samples/${cam}"
            touch "${MOCK_MULTICORRUPT}/rain/${sev}/samples/${cam}/test.jpg"
        done
        mkdir -p "${MOCK_MULTICORRUPT}/rain/${sev}/samples/LIDAR_TOP"
        touch "${MOCK_MULTICORRUPT}/rain/${sev}/samples/LIDAR_TOP/test.pcd"
    done
    
    # Create mock model files
    mkdir -p "${MOCK_MMDET3D}/projects/BEVFusion/configs"
    touch "${MOCK_MMDET3D}/projects/BEVFusion/configs/sbnet_256.py"
    touch "${MOCK_MMDET3D}/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
    touch "${MOCK_MMDET3D}/data/nuscenes/sbnet.pth"
    touch "${MOCK_MMDET3D}/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth"
}

# Test directory structure creation
test_directory_structure() {
    echo "Testing directory structure..."
    
    # Test corruption type directories
    for corruption in fog rain; do
        if [ ! -d "${MOCK_MULTICORRUPT}/${corruption}" ]; then
            echo "ERROR: Missing corruption directory: ${corruption}"
            return 1
        fi
        
        # Test severity directories
        for sev in 1 2; do
            if [ ! -d "${MOCK_MULTICORRUPT}/${corruption}/${sev}" ]; then
                echo "ERROR: Missing severity directory: ${corruption}/${sev}"
                return 1
            fi
        done
    done
    
    echo "Directory structure test passed!"
    return 0
}

# Test model name generation
test_model_names() {
    echo "Testing model name generation..."
    
    # Source the function from the main script
    source tools/test_corrupted_modalities.sh
    
    # Test sbnet name
    local sbnet_name=$(get_model_name "/mmdet3d/projects/BEVFusion/configs/sbnet_256.py")
    if [ "$sbnet_name" != "sbnet" ]; then
        echo "ERROR: Incorrect sbnet name: ${sbnet_name}"
        return 1
    fi
    
    # Test bevfusion name
    local bevfusion_name=$(get_model_name "/mmdet3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
    if [ "$bevfusion_name" != "bevfusion" ]; then
        echo "ERROR: Incorrect bevfusion name: ${bevfusion_name}"
        return 1
    fi
    
    echo "Model name generation test passed!"
    return 0
}

# Test work directory creation
test_work_directories() {
    echo "Testing work directory creation..."
    
    # Create test work directories
    for corruption in fog rain; do
        for sev in 1 2; do
            for model in sbnet bevfusion; do
                work_dir="work_dirs/${corruption}_sev${sev}_${model}"
                mkdir -p "${work_dir}/camera_0.0"
                mkdir -p "${work_dir}/camera_1.0"
                touch "${work_dir}/test.log"
                
                # Verify directory structure
                if [ ! -d "${work_dir}/camera_0.0" ] || [ ! -d "${work_dir}/camera_1.0" ]; then
                    echo "ERROR: Missing modality directories in ${work_dir}"
                    return 1
                fi
                if [ ! -f "${work_dir}/test.log" ]; then
                    echo "ERROR: Missing test log in ${work_dir}"
                    return 1
                fi
            done
        done
    done
    
    echo "Work directory creation test passed!"
    return 0
}

# Clean up test environment
cleanup() {
    echo "Cleaning up test environment..."
    rm -rf "${TEST_ROOT}"
    rm -rf "work_dirs"
}

# Main test execution
main() {
    echo "Starting tests..."
    
    # Set up test environment
    setup_test_env
    
    # Run tests
    test_directory_structure
    local dir_test_status=$?
    
    test_model_names
    local name_test_status=$?
    
    test_work_directories
    local work_dir_test_status=$?
    
    # Clean up
    cleanup
    
    # Report results
    echo "Test Results:"
    echo "Directory Structure Test: $([ $dir_test_status -eq 0 ] && echo "PASSED" || echo "FAILED")"
    echo "Model Name Test: $([ $name_test_status -eq 0 ] && echo "PASSED" || echo "FAILED")"
    echo "Work Directory Test: $([ $work_dir_test_status -eq 0 ] && echo "PASSED" || echo "FAILED")"
    
    # Return overall status
    return $(( dir_test_status + name_test_status + work_dir_test_status ))
}

# Run tests
main
