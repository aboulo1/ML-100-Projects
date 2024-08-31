#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:29:45 2024

@author: aboubakr
"""
#%% load needed libraries
import sys
import os
from joblib import load, dump
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.abspath(project_root))
from utils import load_config_decorator
#%% Fetch config paths

config_path = os.path.join(current_dir, 'config.json')

class ModelPipeline:
    """
   A pipeline class for training, evaluating, saving, and loading iris classifier models.

   This class is designed to manage the steps of a machine learning model, including training,
   evaluation, and saving the model to disk. It defaults to using a linear SVM if no model is provided.
   
   Attributes:
   ------
       model (object): The machine learning model to be used in the pipeline. Defaults to a linear SVM.
       model_save_path (str): The path where the trained model will be saved or loaded from.
   """
    @load_config_decorator(config_path)
    def __init__(config, self, model = None):
        """
        Initializes the ModelPipeline with the specified model or a default linear SVM.

        Parameters:
        ------
            config : dict
                contains all the configuration infos passed down through the decorator
            model : object, optional
                A machine learning model instance. If not provided, defaults to SVC with a linear kernel.
        """
        if model is None:
            self.model = SVC(kernel='linear', probability=True)
        else:
            self.model = model
        self.model_save_path = os.path.join(current_dir,config['model_save_path'])
    
    def train (self, X_train, y_train):
        """
        Trains the model on the provided training data.

        Parameters:
        ------
            X_train (array-like): Training data features.
            y_train (array-like): Training data labels.

        Raises:
        ------
            Exception: If the training process fails, an exception is raised and logged.
        """
        try:
            print("Training model")
            self.model.fit(X_train, y_train)
            print("Model training completed")
        except Exception as e:
            print("Model training failed")
            raise e
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the provided test data and logs the performance metrics.

        Parameters:
        ------
            X_test (array-like): Test data features.
            y_test (array-like): Test data labels.

        Returns:
        ------
            float: The accuracy of the model on the test data.

        Raises:
        ------
            Exception: If the evaluation process fails, an exception is raised and logged.
        """
        try:
            print("Evaluating model")
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            model_name = type(self.model).__name__
            print(f"{model_name} Accuracy: {accuracy:.2f}")
            print(f"Model Classification Report: {report}")
            return accuracy
        except Exception as e:
            print("Model Evaluation Failed")
            raise e
    
    def save_model(self):
        """
        Saves the trained model to the specified path.

        Raises:
        ------
            Exception: If the model cannot be saved, an exception is raised and logged.
        """
        try:
            print(f"Saving model to {self.model_save_path}")
            dump(self.model, self.model_save_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Saving model to {self.model_save_path} failed")
            raise e
        return
    
    def load_model(self):
        """
        Loads the model from the specified path.

        Raises:
        ------
            FileNotFoundError: If the model file does not exist at the specified path.
        """
        try:
            print(f"Loading model from {self.model_save_path}")
            self.model = load(self.model_save_path)
            print("Model loaded successfully")
        except FileNotFoundError as e:
            print(f"Model file not found in: {self.model_save_path}")
            raise e