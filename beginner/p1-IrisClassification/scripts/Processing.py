#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:35:59 2024

@author: aboubakr
"""
import os
import json
import pandas as pd
import numpy as np 
from typing import Union
from sklearn.preprocessing import StandardScaler
from joblib import load, dump

project_root = '/Users/aboubakr/ML-100-Projects/beginner/p1-IrisClassification'
models_path = os.path.join(project_root, 'models')

# paths to save or load our model processors
scaler_filepath = os.path.join(models_path,'feature_standard_scaler.pkl')
IQR_filepath = os.path.join(models_path,'feature_IQR.json')

# Create the processing function
def preprocessing(iris : Union [np.ndarray, pd.DataFrame],
                  train : bool = False,
                  save_scaler : bool = True,
                  save_IQR : bool = True) :
    """
        Preprocess the input data to train or fit data for our iris classification
        The input argument data is a mandatory 4 columns np ndarray or pd.DataFrame
        Parameters
        ----------
        data : Union [np.ndarray, pd.DataFrame], mandatory
            The input data to process
        train : a boolean that indicates if we're processing in the training phase or not
            if train = True : then we will be saving Scalers, Outlier parameters etc
            if train = False : then we'll be reading model preprocessing files
        
        save_scaler : a boolean to indicate if we want to save a new scaler during training
        save_IQR : a boolean to indicate if we want to save new IQR param during training

        Raises
        ------
        ValueError
            If the data is not on the right type or doesn't contain the amount
            of features (4) that's required.
        FileNotFoundError
            If you are in test mode but didn't train and save your scaler or
            outlier detection parameters
        Example of call :
        ------
            from sklearn.datasets import load_iris
            iris = load_iris()
            preprocessing(iris, train=True, save_scaler=False, save_IQR=False)
            
            OR
            
            df = pd.DataFrame(data= np.c_[iris['data']], columns= iris['feature_names'])
            preprocessing(df, train=True, save_scaler=False, save_IQR=False)
          
    """
    
    if isinstance(iris, pd.DataFrame):
        if iris.shape[1] !=4 :
            raise ValueError("The input for training or fitting must have 4 columns")
    elif isinstance(iris.data, np.ndarray):
        if iris.data.shape[1] !=4 :
            raise ValueError("The input for training or fitting must have 4 columns")
        iris = pd.DataFrame(data=np.c_[iris['data']], columns=iris['feature_names'])
    else:
        raise ValueError("The input data must be a numpy ndarray or a pandas DataFrame")
        
    iris = iris.rename(columns = {'sepal length (cm)' : 'SepalLength',
                                  'sepal width (cm)' : 'SepalWidth',
                                  'petal length (cm)' : 'PetalLength',
                                  'petal width (cm)' : 'PetalWidth'})
        
    # During the EDA we've noticed a strong correlation between petal sizes and sepal length
    # Thus we'll be dropping the later. 
    # We're for now keeping sepal width eventhough it's decorrelated with the other
    # features and target, it didn't affect the clustering of target when plotted
    # in respect to petal sizes.
    
    iris = iris.drop(columns = ['SepalLength'])
    # just to be sure that the columns order doesn't create errors in scaling
    # Let's split the data between X : the features 
    
    X = iris[['PetalLength','PetalWidth','SepalWidth']]
    
    if train : # Standardize, cap outliers and save to files
    
        # We've also noticed that all these variables follow a normal distribution 
        # Hence we are going to use a standardScaler and convert back to a DF
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        scaled_X = pd.DataFrame(scaled_X, columns=X.columns)
        # Save scaler in the models directory for further usage
        if save_scaler : dump(scaler, scaler_filepath)
        
        # Now let's handle outliers via a capping 
        # For that we must get the first and third quantile to set an upper and lower bound
        IQR = {col : {'Q1' : scaled_X[col].quantile(0.25),
                      'Q3' : scaled_X[col].quantile(0.75)} for col in scaled_X.columns}
        capped_X = scaled_X.apply(lambda col: capping(col, IQR[col.name]))
        
        # Save the IQR dictionnary to a json file
        if save_IQR :
            with open(IQR_filepath, 'w') as file:
                json.dump(IQR, file, indent=4)
    
    else : #Read the trained parameters and apply them to data
        # Load the scaler and transform the data
        try:
            scaler = load(scaler_filepath)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise FileNotFoundError("To transform this dataset you have to first train your standardScaler")
        scaled_X = scaler.transform(X)
        scaled_X = pd.DataFrame(scaled_X, columns=X.columns)
        
        # Load the quantiles and cap outliers
        try:
            with open(IQR_filepath, 'r') as file:
                IQR = json.dump(file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise FileNotFoundError("To cap this dataset you have to first set your outliers quantiles")
            scaled_X = scaler.transform(X)
        capped_X = scaled_X.apply(lambda col: capping(col, IQR[col.name]))
    
    
    return capped_X

#Our capping function
def capping(column : pd.Series, column_IQR : dict):
    """

    Parameters
    ----------
    column : pd.Series
        The series that we will perform the capping on
    column_IQR : dict
        The dictionnary containing the First and third quantile of coluln

    Returns
    -------
    pd.Series
        a series where value that are considered outliers are replaced by 
        lower or upper bound.

    """
    Q1 = column_IQR['Q1']
    Q3 = column_IQR['Q3']
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column.clip(lower=lower_bound, upper=upper_bound)
        
        