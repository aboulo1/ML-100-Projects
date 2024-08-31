#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:41:09 2024

@author: aboubakr
"""

"""
- Drop Unnecessary cols
-Sex is a key feature
-Class is a key feature
-Extracting title from name '([A-Za-z]+)\.' , mapper at config['title_map']
-SibSp => to lonely, couple(=1) other
-Parch => idem as Sibsp
-Numeric Means based on train models/numeric_means.json'
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked', 'Title']
target = 'Survived'
"""
import os
import sys
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from joblib import load, dump
#%% Set project directory
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.abspath(project_root))
from utils import load_config_decorator, load_json

#%% Fetch config paths
config_path = os.path.join(project_root, 'config.json')

#%% Preprocessing function
@load_config_decorator(config_path)
def preprocessing(config,
               titanic : pd.DataFrame,
               train : bool = False,
               save_scaler : bool = True):
    """

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    titanic : np.ndarray | pd.DataFrame
        DESCRIPTION.
    feature_names : list, optional
        DESCRIPTION. The default is [].
    train : bool, optional
        DESCRIPTION. The default is False.
    save_scaler : bool, optional
        DESCRIPTION. The default is True.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    y = titanic.Survived
    # After the EDA we've decided to keep the following columns :
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']
    # Get Title from name
    titanic['Title'] = titanic.Name.str.extract('([A-Za-z]+)\.')
    title_map_fp = os.path.join(project_root, config['title_map'])
    title_mapper = load_json(title_map_fp)
    titanic.Title = titanic.Title.replace(title_mapper)
    try:
        data = titanic[features]
    except KeyError as e:
        print(f"Error: {e}")
        raise KeyError("The set provided is missing mandatory columns")
    # Fill na
    dtypes = data.dtypes
    numerical_col = dtypes[dtypes != 'object'].index
    means_fp = os.path.join(project_root, config['means'])
    means = load_json(means_fp)
    for col in numerical_col:
        data[col] = data[col].fillna(means[col])
    
    # Polynomial Features
    for_poly_feat = ['Age','Fare']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(data[for_poly_feat])
    poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(for_poly_feat))
    data = pd.concat([data, poly_features_df.drop(columns = for_poly_feat)], axis = 1)
    
    #Interaction Term between Pclass and Fare
    data['Pclass_Fare'] = data['Pclass'].multiply(data['Fare'])
    #Domain Specific Transformation
    # Log and Fare as it follows a normal distribution
    data['Log_fare'] = np.log1p(data['Fare'])
    
    # Family Size
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    #3- single, couple, other on Sibsp
    data.SibSp = data.SibSp.apply(lambda x : 'single' if x == 0 else ( 'couple' if x == 1 else 'other'))
    #4- Same on Parch
    data.Parch = data.Parch.apply(lambda x : 'alone' if x == 0 else ( 'duo' if x == 1 else 'other'))
    dtypes = data.dtypes
    numerical_col = dtypes[dtypes != 'object'].index.tolist()
    #Scaling numerical columns
    scaler_filepath = os.path.join(project_root, config['scaler_save_path'])
    if train:
        scaler = StandardScaler()
        data[numerical_col] = scaler.fit_transform(data[numerical_col])
        if save_scaler : dump(scaler, scaler_filepath)
    
    else:
        try:
            scaler = load(scaler_filepath)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise FileNotFoundError("To transform this dataset you have to first train your standardScaler")
        data[numerical_col] = scaler.transform(data[numerical_col])
        
    #Handle Sex:
    data.Sex = data.Sex == 'male'
    #Object columns
    dummy_cols = ['SibSp','Parch','Embarked'] #Enums in APP
    X = pd.get_dummies(data, columns=dummy_cols)   
    return X,y


























