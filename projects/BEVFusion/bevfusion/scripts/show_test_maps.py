import os
import json
import glob

def parse_results():
    # Find all json files in the work_dirs/missing_modality_tests_* directories
    base_dir = "work_dirs"
    test_dirs = glob.glob(os.path.join(base_dir, "missing_modality_tests_*"))
    
    # Sort by timestamp to get the latest first
    test_dirs.sort(reverse=True)
    
    if not test_dirs:
        print("No test directories found!")
        return
    
    # Use the most recent test directory
    latest_dir = test_dirs[1]
    results = []
    
    # Loop through all subdirectories
    for subdir in os.listdir(latest_dir):
        if not os.path.isdir(os.path.join(latest_dir, subdir)):
            continue
            
        # Parse modality and ratio from directory name
        modality, ratio = subdir.split('_')
        ratio = float(ratio)
        
        # Find the json file in the timestamp subdirectory
        json_files = glob.glob(os.path.join(latest_dir, subdir, "*/*.json"))
        if not json_files:
            continue
            
        # Read the json file
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            
        # Extract mAP value
        map_value = data.get("NuScenes metric/pred_instances_3d_NuScenes/mAP", None)
        
        if map_value is not None:
            results.append({
                'modality': modality,
                'ratio': ratio,
                'mAP': map_value
            })
    
    # Sort results by modality and ratio
    results.sort(key=lambda x: (x['modality'], x['ratio']))
    
    # Print results in a formatted table
    print("\nResults from directory:", os.path.basename(latest_dir))
    print("\nModality | Removal Ratio | mAP")
    print("-" * 35)
    for result in results:
        print(f"{result['modality']:<8} | {result['ratio']:^12.1%} | {result['mAP']:^.4f}")

if __name__ == "__main__":
    parse_results()