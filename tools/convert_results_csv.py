from pathlib import Path
import json
import pandas as pd
from typing import Dict, Set, Tuple
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D  # For custom legend handles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, base_path: str = "/work_dirs2"):
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
        """Find the newest timestamp directory that contains a valid JSON file."""
        try:
            # Get all timestamp directories sorted by timestamp (newest first)
            timestamp_dirs = [
                d for d in path.iterdir() 
                if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'
            ]
            if not timestamp_dirs:
                return None
            
            # Sort by timestamp, newest first
            timestamp_dirs.sort(key=lambda x: int(x.name.replace('_', '')), reverse=True)
            
            # Look through directories until we find one with a JSON file
            for dir_path in timestamp_dirs:
                json_path = dir_path / f"{dir_path.name}.json"
                if json_path.exists():
                    return dir_path
                
            return None
        except Exception as e:
            logger.error(f"Error finding timestamp dir with JSON in {path}: {e}")
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

    def plot_sensor_metrics(self, raw_data: Dict[str, Dict[str, float]]) -> None:
        logger.info("Starting plot_sensor_metrics")
        logger.info(f"Number of raw data entries: {len(raw_data)}")
        
        # Set style and figure size
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = [18, 12]
        
        # Define sensor configurations to plot with their specific corruption types
        sensor_configs = [
            {
                'sensors': ('camera_0.0', 'lidar_0.0'),
                'title': 'Results under corrupted modalities',
                'corruptions': {
                    'beamsreducing': ('red', 'Beam Reducing'),
                    'brightness': ('green', 'Brightness'),
                    'dark': ('blue', 'Dark'),
                    'fog': ('purple', 'Fog'),
                    'motionblur': ('orange', 'Motion Blur'),
                    'pointsreducing': ('brown', 'Point Reducing')
                }
            },
            {
                'sensors': 'camera_1.0',
                'title': 'Results under corrupted modalities and 100% Camera Removed',
                'corruptions': {
                    'beamsreducing': ('red', 'Beam Reducing'),
                    'fog': ('purple', 'Fog'),
                    'motionblur': ('orange', 'Motion Blur'),
                    'pointsreducing': ('brown', 'Point Reducing')
                }
            },
            {
                'sensors': 'lidar_1.0',
                'title': 'Results under corrupted modalities and 100% LIDAR Removed',
                'corruptions': {
                    'brightness': ('green', 'Brightness'),
                    'dark': ('blue', 'Dark'),
                    'fog': ('purple', 'Fog'),
                    'motionblur': ('orange', 'Motion Blur')
                }
            }
        ]
        
        for config in sensor_configs:
            logger.info(f"\nProcessing config for: {config['title']}")
            
            fig = plt.figure(figsize=(18, 12))
            n_corruptions = len(config['corruptions'])
            n_rows = (n_corruptions + 2) // 3
            n_cols = 3
            
            for idx, (corruption_base, (color, display_name)) in enumerate(config['corruptions'].items()):
                ax = plt.subplot(n_rows, n_cols, idx + 1)
                
                # First, find all possible severity levels for this corruption
                all_severities = set()
                for key in raw_data.keys():
                    if corruption_base in key:
                        severity_str = key.split('_')[-1]
                        severity = int(severity_str[3:]) if severity_str.startswith('sev') else int(severity_str)
                        all_severities.add(severity)
                all_severities = sorted(list(all_severities))
                
                for model in ['bevfusion', 'sbnet']:
                    severities = []
                    values = []
                    
                    # Create data points for all severity levels
                    for severity in all_severities:
                        for key, metrics in raw_data.items():
                            if (model in key and 
                                corruption_base in key and 
                                (severity_str := key.split('_')[-1]) and
                                (int(severity_str[3:]) if severity_str.startswith('sev') else int(severity_str)) == severity):
                                
                                sensor_match = False
                                if isinstance(config['sensors'], tuple):
                                    # Accept either camera_0.0 or lidar_0.0 for base case
                                    sensor_match = 'camera_0.0' in key or 'lidar_0.0' in key
                                else:
                                    sensor_match = config['sensors'] in key
                                    
                                if sensor_match:
                                    severities.append(severity)
                                    values.append(metrics['mAP'])
                                    break
                    
                    if severities and values:
                        # Plot line with larger markers
                        linestyle = '--' if model == 'bevfusion' else '-'
                        line = ax.plot(severities, values, color=color, linestyle=linestyle, marker='o',
                                     label=model.capitalize(), linewidth=2.0, markersize=4)[0]
                        
                        # Create a separate line object for the legend with smaller markers
                        legend_line = Line2D([0], [0], color=color, linestyle=linestyle, marker='o',
                                           label=model.capitalize(), linewidth=2.0, markersize=1)
                        
                        # Replace the line in the legend
                        handles, labels = ax.get_legend_handles_labels()
                        handles[-1] = legend_line
                        ax.legend(handles=handles)
                
                # Set x-axis ticks to show all possible severity levels
                ax.set_xticks(all_severities)
                ax.set_xticklabels([str(s) for s in all_severities])
                
                ax.set_xlabel('Severity Level')
                ax.set_ylabel('mAP')
                ax.set_title(display_name)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 0.7)
                ax.legend()
            
            # Adjust layout and title
            plt.tight_layout()
            fig.suptitle(f'{config["title"]}', y=1.02, fontsize=16)
            
            # Save plot
            filename = f'map_plot_all_corruptions_{config["title"].lower().replace(" ", "_").replace("-", "_")}.png'
            save_path = self.sensor_comparison_dir / filename
            logger.info(f"Attempting to save plot to: {save_path}")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Successfully saved plot: {filename}")
            plt.close(fig)

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
    
    # Add plotting
    collector.plot_sensor_metrics(raw_data)

if __name__ == "__main__":
    main()