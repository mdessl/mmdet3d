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
        sensor_models = set()
        corruptions = set()
        metrics = {"mAP": {}, "NDS": {}}

        # Extract all combinations and corruptions
        for key in raw_data.keys():
            model, sensor, corruption = key.split("_", 2)
            sensor_models.add(f"{sensor}_{model}")
            corruptions.add(corruption)
        
        sensor_models = sorted(list(sensor_models))  # Sort for consistent ordering
        corruptions = sorted(list(corruptions))      # Sort for consistent ordering

        # Organize data
        for sensor_model in sensor_models:
            metrics["mAP"][sensor_model] = {}
            metrics["NDS"][sensor_model] = {}
            model = sensor_model.split("_")[1]
            sensor = sensor_model.split("_")[0]
            
            for corruption in corruptions:
                key = f"{model}_{sensor}_{corruption}"
                if key in raw_data:
                    metrics["mAP"][sensor_model][corruption] = raw_data[key]["mAP"]
                    metrics["NDS"][sensor_model][corruption] = raw_data[key]["NDS"]

        return pd.DataFrame.from_dict(metrics["mAP"]).T, pd.DataFrame.from_dict(metrics["NDS"]).T

def format_for_output(df: pd.DataFrame, metric_name: str, output_dir: Path, prefix: str = "") -> None:
    """Format DataFrame for LaTeX and presentation-friendly output"""
    # Clean up column names
    df.columns = [col.replace('_sev', ' S').replace('_', ' ') for col in df.columns]
    df = df.round(3)
    
    # Save different formats
    base_name = f"{prefix}_{metric_name.lower()}" if prefix else f"metrics_{metric_name.lower()}"
    
    df.to_csv(output_dir / f"{base_name}.csv")
    
    latex_str = df.to_latex(float_format="%.3f")
    with open(output_dir / f"{base_name}.tex", 'w') as f:
        f.write(latex_str)
    
    markdown_str = df.to_markdown(floatfmt=".3f")
    with open(output_dir / f"{base_name}.md", 'w') as f:
        f.write(markdown_str)
    
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