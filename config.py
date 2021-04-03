import os
import json
import glob
import copy
import importlib
import torch
import numpy as np
from auto_LiRPA.utils import logger

def update_dict(d, u, show_warning = False):
    for k, v in u.items():
        if k not in d and show_warning:
            print("\033[91m Warning: key {} not found in config. Make sure to double check spelling and config option name. \033[0m".format(k))
        if isinstance(v, dict):
            if k not in d or not isinstance(d[k], dict):
                d[k] = v
            else:
                d[k] = update_dict(d.get(k, {}), v, show_warning)
        else:
            d[k] = v
    return d

def load_config(path):
    with open("config/defaults.json") as f:
        config = json.load(f)
    if path is not None:
        logger.info("Loading config file: {}".format(path))
        with open(path) as f:
            update_dict(config, json.load(f))
    return config