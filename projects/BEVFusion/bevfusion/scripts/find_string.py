import pickle

def find_path_in_dict(d, search_str, path=""):
    """
    Recursively search for values containing search_str in nested dictionary
    """
    found_paths = []
    
    if isinstance(d, dict):
        for key, value in d.items():
            new_path = f"{path}.{key}" if path else key
            
            # Check if the value is string and contains our search
            if isinstance(value, str) and search_str in value:
                found_paths.append((new_path, value))
            
            # Recurse if value is dict or list
            if isinstance(value, (dict, list)):
                found_paths.extend(find_path_in_dict(value, search_str, new_path))
                         
    elif isinstance(d, list):
        for i, item in enumerate(d):
            new_path = f"{path}[{i}]"
            if isinstance(item, str) and search_str in item:
                found_paths.append((new_path, item))
            if isinstance(item, (dict, list)):
                found_paths.extend(find_path_in_dict(item, search_str, new_path))
                
    return found_paths

# Load all pkl files and search in them
pkl_files = [
    "/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl"
]

search_string = "data/nuscenes/samples"

for pkl_file in pkl_files:
    try:
        with open(pkl_file, 'rb') as f:
            print(f"\nSearching in {pkl_file}...")
            data = pickle.load(f)
            
            # Keep track of unique paths
            unique_paths = set()
            
            # Check first few items
            for idx, item in enumerate(data["data_list"]):
                results = find_path_in_dict(item, search_string)
                for path, _ in results:
                    unique_paths.add(path)
            print(results)
            print("\nUnique paths found:")
            for path in sorted(unique_paths):
                print(f"Path: {path}")
                
    except FileNotFoundError:
        print(f"File not found: {pkl_file}")
    except Exception as e:
        print(f"Error processing {pkl_file}: {e}")