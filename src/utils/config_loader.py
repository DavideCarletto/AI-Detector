import yaml
import os 


def load_configs():

    with open('configs/config.yaml', 'r') as f:
        global_cfg = yaml.safe_load(f)

    return global_cfg


cfg = load_configs()