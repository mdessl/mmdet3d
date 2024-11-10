import os
import yaml
from collections import OrderedDict


def format_value(value):
    """Format a value into a Python-style string."""
    if isinstance(value, (list, tuple)):
        # Special handling for pipeline configurations
        if value and isinstance(value[0], dict) and "type" in value[0]:
            lines = ["["]
            for item in value:
                lines.append("    dict(")
                # Sort keys to put 'type' first
                sorted_keys = sorted(item.keys(), key=lambda x: (x != "type", x))
                for k in sorted_keys:
                    v = item[k]
                    formatted_v = format_value(v)
                    if k == "type":
                        lines[-1] = (
                            f"    dict(type='{v}',"  # Combine type with dict opening
                        )
                    elif isinstance(v, (list, dict)) and len(str(v)) > 40:
                        # Multi-line for complex/long values
                        lines.append(f"         {k}=")
                        v_lines = format_value(v).split("\n")
                        lines.extend("         " + line for line in v_lines)
                        lines[-1] += ","
                    else:
                        # Single line for simple values
                        lines.append(f"         {k}={formatted_v},")
                lines.append("    ),")
            lines.append("]")
            return "\n".join(lines)

        # Regular list handling
        if not value:
            return "[]"
        if any(isinstance(x, (list, tuple)) for x in value):
            items = [format_value(item) for item in value]
            return "[\n" + ",\n".join(f"    {item}" for item in items) + "\n]"
        return f'[{", ".join(format_value(x) for x in value)}]'

    elif isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        for k, v in value.items():
            formatted_v = format_value(v)
            if isinstance(v, (dict, list)) and len(str(v)) > 40:
                lines.append(f"    {k}:")
                v_lines = formatted_v.split("\n")
                lines.extend("    " + line for line in v_lines)
            else:
                lines.append(f"    {k}: {formatted_v}")
        lines.append("}")
        return "\n".join(lines)

    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        if "${" in value:
            return value
        return f"'{value}'"
    elif value is None:
        return "None"
    return str(value)


def format_dict(d, indent=0):
    """Format a dictionary into a Python-style config string."""
    if not d:
        return "dict()"

    lines = ["dict("]
    indent_str = "    " * (indent + 1)

    for key, value in d.items():
        if isinstance(value, dict):
            formatted_value = format_dict(value, indent + 1)
            lines.append(f"{indent_str}{key}={formatted_value},")
        else:
            formatted_value = format_value(value)
            if "\n" in str(formatted_value):
                lines.append(f"{indent_str}{key}={formatted_value},")
            else:
                lines.append(f"{indent_str}{key}={formatted_value},")

    lines.append("    " * indent + ")")
    return "\n".join(lines)


def merge_config(base, update):
    """Recursively merge two configs"""
    for k, v in update.items():
        if k not in base:
            base[k] = v
        elif isinstance(v, dict) and isinstance(base[k], dict):
            merge_config(base[k], v)
        else:
            base[k] = v
    return base


def convert_to_python_config(config_dict):
    """Convert config dictionary to Python format string."""
    output = []

    # Handle top-level special variables first
    special_vars = ["_base_", "point_cloud_range", "voxel_size", "image_size"]
    for var in special_vars:
        if var in config_dict:
            value = config_dict[var]
            if var == "_base_":
                if isinstance(value, list):
                    paths = [f"    '{path}'" for path in value]
                    output.append(f"_base_ = [\n" + ",\n".join(paths) + "\n]")
                else:
                    output.append(f"_base_ = ['{value}']")
            else:
                output.append(f"{var} = {format_value(value)}")

    # Handle remaining configuration
    for key, value in config_dict.items():
        if key not in special_vars:
            if isinstance(value, dict):
                output.append(f"{key} = {format_dict(value)}")
            else:
                output.append(f"{key} = {format_value(value)}")

    return "\n\n".join(output)


def get_full_config(config_path, output_format="dict"):
    """
    Get full configuration from config path.
    Args:
        config_path: Path to config file
        output_format: 'dict' or 'python' for different output formats
    """
    parts = config_path.split("/")
    config_files = []
    current_path = []

    for part in parts:
        current_path.append(part)
        if os.path.exists("/".join(current_path)):
            if part.endswith(".yaml"):
                config_files.append("/".join(current_path))
            elif os.path.exists("/".join(current_path + ["default.yaml"])):
                config_files.append("/".join(current_path + ["default.yaml"]))

    final_config = OrderedDict()
    for cfg_file in config_files:
        with open(cfg_file, "r") as f:
            config = yaml.safe_load(f)
            if config:
                final_config = merge_config(final_config, config)

    if output_format == "python":
        return convert_to_python_config(final_config)
    return final_config


# Example usage
if __name__ == "__main__":
    config_path = "configs/nuscenes/seg/fusion-bev256d2-lss.yaml"

    # Get dictionary format
    dict_config = get_full_config(config_path, output_format="dict")
    print("Dictionary format:")
    print(dict_config)
    print("\n" + "=" * 80 + "\n")

    # Get Python format
    python_config = get_full_config(config_path, output_format="python")
    print("Python format:")
    print(python_config)
