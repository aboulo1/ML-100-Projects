#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:50:37 2024

@author: aboubakr
"""

#%% Import needed libraries
import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# current_dir = os.getcwd()
# project_root = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.append(project_root)
# sys.path.append(os.path.abspath(os.path.join(project_root, '..')))
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.abspath(project_root))
from utils import load_config_decorator, remove_leading_slash
from scripts.Processing import preprocessing
#%% Fetch config paths

config_path = os.path.join(project_root, 'config.json')

class DataPipeline:
    """
    A data pipeline class for loading, preprocessing, and splitting data for the iris classification model.

    This class handles:
        - the loading of data, either from the sklearn's iris dataset or from a CSV file,
        - preprocessing of the data,
        - splitting into training and test sets.

    Attributes:
    ------
        data_path (str): Path to the data file if not using the iris dataset.
        is_iris_loaded (bool): Flag to indicate whether to load the iris dataset from sklearn.
        target_name (str): The name of the target column in the dataset.
        data (DataFrame or Bunch): The loaded dataset.
        feature_names (list): List of feature names in the dataset.
        retrain (bool): Flag to indicate whether the preprocessing should be done for retraining.
        X_train (DataFrame or ndarray): Training features.
        X_test (DataFrame or ndarray): Test features.
        y_train (Series or ndarray): Training labels.
        y_test (Series or ndarray): Test labels.
    """
    @load_config_decorator(config_path)
    def __init__(config, self,  is_iris_loaded : bool = True):
        
        """
        Initializes the pipeline with the given parameters
        Parameters
        ------
        config : dict
            contains all the configuration infos passed down through the decorator
        is_iris_loaded : bool
            If True, loads the iris dataset from sklearn. Otherwise, reads data from a CSV file.
        """
        #%% access config file
        self.data_path = os.path.join(project_root, config['data_path'])
        self.is_iris_loaded = is_iris_loaded
        self.target_name = config['target_column']
        self.data = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """
        Loads the dataset based on the is_iris_loaded flag.

        If is_iris_loaded is True, the iris dataset is loaded from sklearn.
        Otherwise, the dataset is loaded from a CSV file specified by data_path.

        Raises:
        ------
            FileNotFoundError: If the CSV file specified by data_path is not found.
        """
        if self.is_iris_loaded:
            self.data = load_iris()
            self.feature_names = self.data.feature_names
        else:
            try:
                print(f"Loading data from {self.data_path}")
                self.data = pd.read_csv(remove_leading_slash(self.data_path))
            except FileNotFoundError as e:
                print(f"File not found: {self.data_path}")
                raise e
                
                
    def preprocess (self):
        """
        Preprocesses the dataset by splitting it into training and test sets, and applying necessary transformations.

        The data is split into features and labels, and then split into training and test sets.
        If retrain is True, the new models will be saved (cf preprocessing function documentation)

        Raises:
        ------
            KeyError: If the target column is not found in the dataset.
        """
        try:
            print(f"Preprocessing data: target column is {self.target_name}")
            if self.is_iris_loaded: #the format is different from a dataframe read
                X = self.data.data
                y = self.data[self.target_name]
            else : # the case of a dataFrame
                X = self.data.drop(columns=[self.target_name])
                y = self.data[self.target_name]
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            self.X_train = preprocessing(X_train, train=True, feature_names = self.feature_names)
            self.X_test = preprocessing(X_test, train=False, feature_names = self.feature_names)
            self.y_train = y_train
            self.y_test = y_test
            print("Preprocessing completed successfully")
        except KeyError as e:
            print(f"Column not found in data: {self.target_column}")
            raise e
            
    def get_data(self):
        """
        Loads and preprocesses the data, returning the training and test sets.

        This method combines the load_data and preprocess steps, ensuring that the data is ready for modeling.

        Returns:
        ------
            tuple: A tuple containing X_train, X_test, y_train, y_test.
        """
        self.load_data()
        self.preprocess()
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_raw_data(self, input_data : np.ndarray | pd.DataFrame):
        """

        Parameters
        ----------
        input_data : np.ndarray | pd.DataFrame
            the data to preprocess in order to evaluate it with our model

        Returns
        -------
        TYPE : np.ndarray | pd.DataFrame
            the processed data ready to be fed to the model.

        """
        self.load_data()
        return preprocessing(input_data, train=False, feature_names = self.feature_names)
