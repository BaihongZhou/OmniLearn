import yaml
import os

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

