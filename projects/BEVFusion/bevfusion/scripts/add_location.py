import argparse
import mmcv
from nuscenes.nuscenes import NuScenes
from os import path as osp
import pickle
import mmengine
import numpy as np
from pyquaternion import Quaternion


def add_location_to_nuscenes_infos(
    pkl_path, nusc_dataroot, out_path=None, version="v1.0-trainval"
):
    """Add location information to an existing NuScenes info pkl file.

    Args:
        pkl_path (str): Path to the existing .pkl file
        nusc_dataroot (str): Path to the NuScenes dataset root
        out_path (str, optional): Path to save the updated .pkl file. If None,
            will overwrite the input file
        version (str): NuScenes dataset version. Default: 'v1.0-trainval'
    """
    print(f"Adding location information to {pkl_path}")

    # Load existing data
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # Initialize NuScenes
    nusc = NuScenes(version=version, dataroot=nusc_dataroot, verbose=True)

    # Add location to each info
    for info in data["data_list"]:
        # Get scene info through sample -> scene
        sample = nusc.get("sample", info["token"])
        scene = nusc.get("scene", sample["scene_token"])

        # Get location from scene's log
        location = nusc.get("log", scene["log_token"])["location"]

        # Get ego pose information
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', lidar_token)
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

        # Add scene_token, location, and ego pose info
        info["scene_token"] = scene["token"]
        info["location"] = location
        info["ego2global_translation"] = pose_record['translation']
        info["ego2global_rotation"] = pose_record['rotation']

        # Create ego2global transformation matrix
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(pose_record['rotation']).rotation_matrix
        ego2global[:3, 3] = pose_record['translation']
        info["ego2global"] = ego2global

    # Save updated data
    save_path = out_path if out_path is not None else pkl_path
    mmengine.dump(data, save_path)
    print(f"Successfully saved updated info file to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Add location to NuScenes info file')
    parser.add_argument(
        'version',
        help='v1.0-trainval or v1.0-mini'
    ) 
    parser.add_argument('--pkl_path', default="/root/mmdet3d/data/nuscenes/nuscenes_infos_train.pkl", help='Path to the input .pkl file')
    parser.add_argument('--nusc_root', default="/root/mmdet3d/data/nuscenes", help='NuScenes dataset root path')
    parser.add_argument(
        '--out-path', 
        default=None,
        help='Path to save the updated .pkl file. If not specified, will overwrite input file'
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    add_location_to_nuscenes_infos(
        args.pkl_path,
        args.nusc_root,
        args.out_path,
        args.version
    ) 