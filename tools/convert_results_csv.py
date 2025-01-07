from pathlib import Path
import json
import pandas as pd
from typing import Dict, Set, Tuple
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, base_path: str = "/mmdet3d/work_dirs"):
        self.base_path = Path(base_path)
        self.MAP_KEY = "NuScenes metric/pred_instances_3d_NuScenes/mAP"
        self.NDS_KEY = "NuScenes metric/pred_instances_3d_NuScenes/NDS"
        
        # Create output directories
        self.output_dir = Path("evaluation_results")
        self.model_comparison_dir = self.output_dir / "model_comparison"
        self.sensor_comparison_dir = self.output_dir / "sensor_comparison"
        
        for dir_path in [self.model_comparison_dir, self.sensor_comparison_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_sensor_types(self) -> Set[str]:
        """Dynamically discover all unique sensor types across all directories."""
        sensor_types = set()
        for model_dir in self.base_path.iterdir():
            if not model_dir.is_dir():
                continue
            for corruption_dir in model_dir.iterdir():
                if not corruption_dir.is_dir():
                    continue
                for sensor_dir in corruption_dir.iterdir():
                    if sensor_dir.is_dir():
                        sensor_types.add(sensor_dir.name)
        return sensor_types

    def get_newest_timestamp_dir(self, path: Path) -> Path:
        """Find the newest timestamp directory in the given path."""
        try:
            timestamp_dirs = [d for d in path.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[8] == '_']
            if not timestamp_dirs:
                return None
            return max(timestamp_dirs, key=lambda x: int(x.name.replace('_', '')))
        except Exception as e:
            logger.error(f"Error finding newest timestamp in {path}: {e}")
            return None

    def read_metrics(self, json_path: Path) -> Tuple[float, float]:
        """Read mAP and NDS metrics from a json file."""
        try:
            with open(json_path) as f:
                data = json.load(f)
                return data.get(self.MAP_KEY), data.get(self.NDS_KEY)
        except Exception as e:
            logger.error(f"Error reading metrics from {json_path}: {e}")
            return None, None

    def collect_metrics(self) -> Dict[str, Dict[str, float]]:
        """Collect all raw metrics in a structured format."""
        raw_data = {}
        
        logger.info(f"Searching in base path: {self.base_path}")
        
        for model_dir in self.base_path.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            logger.info(f"Processing model: {model_name}")

            for corruption_dir in model_dir.iterdir():
                if not corruption_dir.is_dir():
                    continue
                corruption_name = corruption_dir.name
                logger.info(f"Processing corruption: {corruption_name}")

                for sensor_dir in corruption_dir.iterdir():
                    if not sensor_dir.is_dir():
                        continue
                    sensor_type = sensor_dir.name
                    logger.info(f"Processing sensor: {sensor_type}")

                    newest_dir = self.get_newest_timestamp_dir(sensor_dir)
                    if not newest_dir:
                        logger.warning(f"No timestamp directory found in {sensor_dir}")
                        continue

                    json_path = newest_dir / f"{newest_dir.name}.json"
                    if not json_path.exists():
                        logger.warning(f"No json file found at {json_path}")
                        continue

                    map_value, nds_value = self.read_metrics(json_path)
                    if map_value is None or nds_value is None:
                        logger.warning(f"Could not read metrics from {json_path}")
                        continue
                        
                    key = f"{model_name}_{sensor_type}_{corruption_name}"
                    logger.info(f"Found metrics for {key}: mAP={map_value:.3f}, NDS={nds_value:.3f}")
                    raw_data[key] = {"mAP": map_value, "NDS": nds_value}

        logger.info(f"Total entries collected: {len(raw_data)}")
        return raw_data

    def create_model_comparison(self, raw_data: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create DataFrames comparing models (rows) vs corruptions (columns) for camera_0.0"""
        models = ["bevfusion", "sbnet"]
        corruptions = set()
        metrics = {"mAP": {}, "NDS": {}}

        # Extract all corruption types
        for key in raw_data.keys():
            if "camera_0.0" in key:
                model, sensor, corruption = key.split("_", 2)
                corruptions.add(corruption)
        
        corruptions = sorted(list(corruptions))  # Sort for consistent ordering

        # Organize data
        for model in models:
            metrics["mAP"][model] = {}
            metrics["NDS"][model] = {}
            for corruption in corruptions:
                key = f"{model}_camera_0.0_{corruption}"
                if key in raw_data:
                    metrics["mAP"][model][corruption] = raw_data[key]["mAP"]
                    metrics["NDS"][model][corruption] = raw_data[key]["NDS"]

        return pd.DataFrame.from_dict(metrics["mAP"]).T, pd.DataFrame.from_dict(metrics["NDS"]).T

    def create_sensor_comparison(self, raw_data: Dict[str, Dict[str, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create DataFrames comparing sensor+model combinations (rows) vs corruptions (columns)"""
        models = ["bevfusion", "sbnet"]
        sensor_mapping = {
            "camera_0.0": "both modalities",
            "camera_1.0": "camera 100% removed",
            "lidar_1.0": "lidar 100% removed"
        }
        
        # Initialize metrics dictionaries
        metrics = {"mAP": {}, "NDS": {}}
        
        # Extract corruptions correctly
        corruptions = set()
        for key in raw_data.keys():
            # Example key: bevfusion_camera_0.0_beamsreducing_sev1
            parts = key.split('_')
            # Join the corruption parts (everything after sensor)
            corruption = '_'.join(parts[3:])
            corruptions.add(corruption)
        
        corruptions = sorted(list(corruptions))
        print(f"Found corruptions: {corruptions}")
        
        # For each sensor-model combination
        for internal_sensor, display_name in sensor_mapping.items():
            for model in models:
                row_name = f"{display_name} ({model})"
                metrics["mAP"][row_name] = {}
                metrics["NDS"][row_name] = {}
                
                # For each corruption
                for corruption in corruptions:
                    key = f"{model}_{internal_sensor}_{corruption}"
                    if key in raw_data:
                        metrics["mAP"][row_name][corruption] = raw_data[key]["mAP"]
                        metrics["NDS"][row_name][corruption] = raw_data[key]["NDS"]
                    else:
                        # Fill with NaN for missing data
                        metrics["mAP"][row_name][corruption] = float('nan')
                        metrics["NDS"][row_name][corruption] = float('nan')

        # Convert to DataFrames
        map_df = pd.DataFrame.from_dict(metrics["mAP"], orient='index')
        nds_df = pd.DataFrame.from_dict(metrics["NDS"], orient='index')
        
        # Clean up column names
        map_df.columns = [col.replace('_sev', ' S') for col in map_df.columns]
        nds_df.columns = [col.replace('_sev', ' S') for col in nds_df.columns]
        
        print("\nCreated DataFrames with shapes:")
        print(f"mAP: {map_df.shape}")
        print(f"NDS: {nds_df.shape}")
        print("\nColumns:", map_df.columns.tolist())
        print("\nIndex:", map_df.index.tolist())
        
        return map_df, nds_df

def format_for_output(df: pd.DataFrame, metric_name: str, output_dir: Path, prefix: str = "") -> None:
    """Format DataFrame for output"""
    if df.empty:
        print(f"Warning: Empty DataFrame for {metric_name}")
        return
        
    # Clean up column names
    df.columns = [col.replace('_sev', ' S').replace('_', ' ') for col in df.columns]
    df = df.round(3)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save different formats
    base_name = f"{prefix}_{metric_name.lower()}" if prefix else f"metrics_{metric_name.lower()}"
    
    # Save CSV
    df.to_csv(output_dir / f"{base_name}.csv")
    
    # Save LaTeX with thick lines after pairs
    latex_lines = df.to_latex(
        float_format="%.3f",
        column_format='l' + 'r' * len(df.columns),
        escape=False
    ).split('\n')
    
    modified_lines = []
    for i, line in enumerate(latex_lines):
        modified_lines.append(line)
        if line.strip().startswith('both modalities (sbnet)') or \
           line.strip().startswith('camera 100\\% removed (sbnet)'):
            modified_lines.append('\\midrule[1pt]')
    
    with open(output_dir / f"{base_name}.tex", 'w') as f:
        f.write('\n'.join(modified_lines))
    
    # Save Markdown
    with open(output_dir / f"{base_name}.md", 'w') as f:
        f.write(df.to_markdown(floatfmt=".3f"))
    
    # Print to console
    print(f"\n{prefix} {metric_name} Metrics:")
    print(df.to_string(float_format=lambda x: '{:.3f}'.format(x)))

def main():
    collector = MetricsCollector()
    raw_data = collector.collect_metrics()

    # Version 1: Model comparison (rows: models, columns: corruptions)
    df_map_v1, df_nds_v1 = collector.create_model_comparison(raw_data)
    format_for_output(df_map_v1, "mAP", collector.model_comparison_dir, "models")
    format_for_output(df_nds_v1, "NDS", collector.model_comparison_dir, "models")

    # Version 2: Sensor comparison (rows: sensor+model combinations, columns: corruptions)
    df_map_v2, df_nds_v2 = collector.create_sensor_comparison(raw_data)
    format_for_output(df_map_v2, "mAP", collector.sensor_comparison_dir, "sensors")
    format_for_output(df_nds_v2, "NDS", collector.sensor_comparison_dir, "sensors")

if __name__ == "__main__":
    main()