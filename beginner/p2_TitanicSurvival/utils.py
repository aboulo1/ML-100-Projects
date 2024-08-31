#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:45:39 2024

@author: aboubakr
"""

import logging
import json
import os

def load_json(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data

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
        
def load_config_decorator(config_file_path):
    """
    A decorator factory
    Parameters
    ----------
    config_file_path : str
        Decorator to load the configs.

    Returns
    -------
    the decorator

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Load configurations
            config = load_config(remove_leading_slash(config_file_path))
            # Pass the configurations to the function
            return func(config, *args, **kwargs)
        return wrapper
    return decorator

def remove_leading_slash(path):
    """
    Removes the leading slash from a file path if it starts with one.

    Parameters:
    path (str): The file path.

    Returns:
    str: The file path without a leading slash.
    """
    if os.path.exists('/.dockerenv') and path.startswith('/'): #running in docker
        return path[1:]
    return path