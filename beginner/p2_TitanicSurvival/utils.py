#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:45:39 2024

@author: aboubakr
"""

import logging
import json
import os
import pandas as pd
from sklearn.metrics import confusion_matrix


def load_json(filepath):
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
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data

def dump_json(data, filepath):
    with open(filepath, 'w') as fp:
        json.dump(data,fp)
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
            config = load_config(config_file_path)
            # Pass the configurations to the function
            return func(config, *args, **kwargs)
        return wrapper
    return decorator

def load_csv_data(file_path: str):
    """
    Helper function to load a CSV file and handle errors.
    """
    try:
        print(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        print(f"File not found: {file_path}")
        raise e



def weighted_f1_score(y_true, y_pred):
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Precision and Recall
    precision_pos = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall_pos = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    precision_neg = tn / (tn + fn) if (tn + fn) != 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    # F1-Scores for both classes
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) != 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) != 0 else 0
    
    # Support (number of instances for each class)
    support_pos = tp + fn
    support_neg = tn + fp
    
    # Weighted F1-Score
    total_support = support_pos + support_neg
    weighted_f1 = (f1_pos * support_pos + f1_neg * support_neg) / total_support
    
    return weighted_f1