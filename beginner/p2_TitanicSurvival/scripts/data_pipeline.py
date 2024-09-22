import sys
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#% Set project directory
current_dir = os.getcwd()
sys.path.append(os.path.abspath(current_dir))
from utils import load_config_decorator, load_csv_data
from scripts.Processing import preprocessing

config_path = os.path.join(current_dir, 'config.json')

class DataPipeline:
    """
    A data pipeline class for loading, preprocessing training and test data for the titanic survival model.

    This class handles:
        - the loading of data from a CSV file,
        - preprocessing of the data,
        - splitting into training and test sets.
        - preprocessing raw inputs to accomodate them to our models usage

    Attributes:
    ------
        train_path (str): Path to the train data file.
        test_path (str): Path to the test data file.
        target_name (str): The name of the target column in the dataset.
        X_train (DataFrame or ndarray): Training features.
        X_test (DataFrame or ndarray): Test features.
        y_train (Series or ndarray): Training labels.
        y_test (Series or ndarray): Test labels.
    """

    @load_config_decorator(config_path)
    def __init__(config, self):
        """
        Initializes the pipeline with the given parameters
        Parameters
        ------
        config : dict
            contains all the configuration infos passed down through the decorator
        """
        #%% Access the config file
        self.train_path = os.path.join(current_dir, config['train_path'])
        self.test_path = os.path.join(current_dir, config['test_path'])
        self.target_name = 'Survived'
        self.X_train = None
        self.X_test = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None

    def load_data(self, 
                  train_test : bool = True, 
                  train : bool = False, 
                  test : bool = False):
        
        """
        Loads train, test, or both datasets from specified file paths.

        Args:
            train_test (bool): If True, loads both train and test datasets.
            train (bool): If True, loads the train dataset.
            test (bool): If True, loads the test dataset.

        Returns:
            DataFrame or Tuple[DataFrame, DataFrame]: The loaded train and/or test datasets.
        
        Raises:
            ValueError: If both 'train' and 'test' are set to True without setting 'train_test' to True.
            FileNotFoundError: If the specified file path for train or test is not found.
            ValueErrorr : If no input parameter is set to True
        """
        if train and test :
            raise ValueError("If you need both train and test data, set train_test = True")
        
        if train or train_test : train_data = load_csv_data(self.train_path)
        if test or train_test : test_data = load_csv_data(self.test_path)
        
        if train_test : return train_data, test_data
        if train : return train_data
        if test : return test_data 

        raise ValueError("At least one of 'train', 'test', or 'train_test' must be True.")
    
    def preprocess(self):
        """
        Preprocesses the dataset by splitting it into training and test sets, and applying necessary transformations.

        The data is split into features and labels depending on its origin in training and test sets.

        Raises:
        ------
            KeyError: If the target column is not found in the dataset.
        """
        train_data, test_data = self.load_data()
        try : 
            print(f"Preprocessing data: target column is {self.target_name} 🔥")
            X,y = preprocessing(train_data, train=True)
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X,y, train_size=0.8, random_state=42, stratify=y)
            self.X_test, _ = preprocessing(test_data, train=False, target_data=False)
            print(f"Preprocessing completed successfully 💯")
        except :
            raise ValueError("Something went wrong in preprocessing 😔")
        
    def get_data(self):
        """
        Preprocesses the data, returning the training and test sets.

        This method starts with preprocess steps, ensuring that the data is ready for modeling.

        Returns:
        ------
            tuple: A tuple containing X_train, X_valid, X_test, y_train, y_test.
        """
        self.preprocess()
        return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid
    
    def preprocess_raw_data(self, input_data : pd.DataFrame):
        """

        Parameters
        ----------
        input_data : pd.DataFrame
            the data to preprocess in order to evaluate it with our model

        Returns
        -------
        TYPE :  pd.DataFrame
            the processed data ready to be fed to the model.

        """
        return preprocessing(input_data, train=False, target_data=False)
    
    def preprocess_api(self, input_data : pd.DataFrame):
        """

        Parameters
        ----------
        input_data : pd.DataFrame
            the data to preprocess in order to evaluate it with our model

        Returns
        -------
        TYPE :  pd.DataFrame
            the processed data ready to be fed to the model.

        """
        return preprocessing(input_data, train=False, target_data=False)[0]
        