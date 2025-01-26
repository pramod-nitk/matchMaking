# src/utils.py
import logging
import yaml
def load_config(config_path: str) -> dict:
    """
    Loads a YAML config file and returns it as a dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logger(level=logging.INFO):
    """
    Loads a YAML config file and returns it as a dictionary.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
