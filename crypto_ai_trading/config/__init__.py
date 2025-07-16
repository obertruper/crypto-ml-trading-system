"""
>4C;L :>=D83C@0F88
"""

from pathlib import Path
import yaml

def load_config(config_path: str = None) -> dict:
    """03@C7:0 :>=D83C@0F88 87 YAML D09;0"""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

__all__ = ['load_config']