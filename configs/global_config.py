import yaml
import os
import datetime
from pathlib import Path

cfg = None  # Global variable


def load_config(config_file):
    global cfg  # Explicitly declaring that we are modifying the global variable

    # Check if config_file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)  # Modify global cfg
        except yaml.YAMLError as exc:
            print(f"Error loading YAML: {exc}")


def save_config(save_path, save_tag):
    save_path = Path(save_path)  # Ensure Path object
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
    config_filename = save_path / f"{save_tag}_{timestamp}.yaml"  # Define filename

    # Save config as YAML
    with open(config_filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Config saved to {config_filename}")
    return config_filename
