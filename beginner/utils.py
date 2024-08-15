#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:53:58 2024

@author: aboubakr
"""

import logging
import json

def load_config(config_path):
    """
    Parameters
    ----------
    config_path : string
        Path to the configuration file.

    Raises
    ------
    e
        FileNotFoundError if if the wrong_path is inputed
        Logging Error if the json file cannot be decoded / parsed

    Returns
    -------
    config : dict
        all the config attributes with their corresponding value.

    """
    try:
        logging.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {config_path}")
        raise e
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON configuration: {config_path}")
        raise e