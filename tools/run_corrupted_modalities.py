from contextlib import contextmanager
from pathlib import Path
import shutil
import os

MULTICORRUPT_DIR = "/MultiCorrupt/multicorrupt"
TARGET_DIR = "/mmdet3d/data/nuscenes"

CONFIGS_AND_CHECKPOINTS = {
    "/mmdet3d/projects/BEVFusion/configs/sbnet_256.py": (
        "/mmdet3d/data/nuscenes/sbnet.pth",
        "sbnet"
    ),
    "/mmdet3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py": (
        "/mmdet3d//bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth",
        "bevfusion"
    ),
}
@contextmanager
def directory_link_manager(src_dir: str, target_dir: str):
    """Safely manage directory symlinks with backups."""
    src_path = Path(src_dir).resolve()
    target_path = Path(target_dir)
    original_dirs = {}

    try:
        # Process sweeps and samples directories
        for data_type in ['sweeps', 'samples']:
            src_data_dir = src_path / data_type
            if not src_data_dir.exists():
                continue

            # Process all subdirectories in sweeps/samples
            for dir_path in src_data_dir.iterdir():
                if not dir_path.is_dir():
                    continue

                # Create target path
                target = target_path / data_type / dir_path.name

                if target.exists() and not target.is_symlink():
                    # Create backup with _original suffix at same level
                    backup_dir = target.parent / f"{target.name}_original"
                    backup_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(target, backup_dir)
                    original_dirs[str(target)] = backup_dir
                elif target.exists():
                    target.unlink()

                # Create symlink
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(dir_path)

        yield

    finally:
        # Restore original directories
        for target_path_str, backup_dir in original_dirs.items():
            target = Path(target_path_str)
            if target.exists():
                if target.is_symlink():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            # Restore from backup
            if backup_dir.exists():
                shutil.move(backup_dir, target)

def run_test(config: str, checkpoint: str, gpus: int = 1, work_dir: str = None):
    """Run corruption test using the test_corrupted_modalities.sh script."""
    import subprocess
    
    # Ensure work_dir exists if provided
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
    
    cmd = [
        "bash",
        "tools/test_missing_modalities.sh",
        config,
        checkpoint,
        str(gpus)
    ]
    
    if work_dir:
        cmd.extend(["--work-dir", work_dir])
    
    try:
        # Always show output in terminal
        process = subprocess.run(
            cmd,
            check=True,
            text=True
        )
        
 
        return True
            
    except subprocess.CalledProcessError:
        return False

def main():
    # Process each corruption directory
    for corrupt_dir in Path(MULTICORRUPT_DIR).iterdir():
        if not corrupt_dir.is_dir():
            continue
            
        # Process each severity version
        for version_dir in sorted([d for d in corrupt_dir.iterdir() if d.name.isdigit()]):
            print(f"Testing {corrupt_dir.name} version {version_dir.name}")
            
            # Run tests with managed data state
            with directory_link_manager(str(version_dir), TARGET_DIR):
                for config, (checkpoint, model_name) in CONFIGS_AND_CHECKPOINTS.items():
                    work_dir = f"work_dirs/{model_name}/{corrupt_dir.name}_sev{version_dir.name}"
                    print("done storing with sys linking")
                    run_test(config, checkpoint, work_dir=work_dir)

if __name__ == "__main__":
    main()