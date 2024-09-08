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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from joblib import load, dump
#% Set project directory
current_dir = os.getcwd()
sys.path.append(os.path.abspath(current_dir))
from utils import load_config_decorator, load_json

#% Fetch config paths
ref_dir = current_dir
#ref_dir = os.path.abspath(os.path.join(current_dir, '..')) #Use this for model prototyping #TODO do it better
config_path = os.path.join(ref_dir, 'config.json')
print(ref_dir)


#% Preprocessing function
@load_config_decorator(config_path)
def preprocessing(config,
               titanic : pd.DataFrame,
               target_data : bool = True,
               train : bool = False,
               save_scaler : bool = True,
               save_encoder : bool = True):
    """

    Parameters
    ----------
    config : TYPE
        DESCRIPTION.
    titanic : pd.DataFrame
        DESCRIPTION.
    train : bool, optional
        DESCRIPTION. The default is False.
    save_scaler : bool, optional
        DESCRIPTION. The default is True.
    save_encoder : bool, optional
        DESCRIPTION. The default is True.

    Raises
    ------
    ValueError
        DESCRIPTION.
    FileNotFoundError
        DESCRIPTION

    Returns
    -------
    X,y

    """
    if target_data : y = titanic.Survived
    else : y = None
    # After the EDA we've decided to keep the following columns :
    #features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']
    # Get Title from name
    titanic['Title'] = titanic.Name.str.extract('([A-Za-z]+)\.')
    title_map_fp = os.path.join(ref_dir, config['title_map'])
    title_mapper = load_json(title_map_fp)
    titanic.Title = titanic.Title.replace(title_mapper)

    #Drop unnecessary columns
    drop_cols = ['Name','Cabin','Ticket', 'PassengerId','Survived']
    data = titanic.drop(drop_cols, axis=1, errors='ignore')

    # Fill na
    dtypes = data.dtypes
    numerical_col = dtypes[dtypes != 'object'].index
    means_fp = os.path.join(ref_dir, config['means'])
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
    #Handle Sex:
    data.Sex = data.Sex == 'male'

    scaler_filepath = os.path.join(ref_dir, config['scaler_save_path'])
    encoder_filepath = os.path.join(ref_dir, config['encoder_save_path'])
    dummy_cols = ['SibSp','Parch','Embarked', 'Title'] #Enums in APP
    if train:
        #Scaling numerical columns
        scaler = StandardScaler()
        data[numerical_col] = scaler.fit_transform(data[numerical_col])
        if save_scaler : dump(scaler, scaler_filepath)
        #LabelEncoding
        le = LabelEncoder()
        data[dummy_cols] = data[dummy_cols].apply(le.fit_transform)
        if save_encoder : dump(le, encoder_filepath)

    else:
        #Scaling numerical data
        try:
            scaler = load(scaler_filepath)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise FileNotFoundError("To transform this dataset you have to first train your Standard Scaler")
        data[numerical_col] = scaler.transform(data[numerical_col])
        
        #LabelEncoder
        try:
            le = load(encoder_filepath)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise FileNotFoundError("To transform this dataset you have to first train your Label Encoder")
        data[dummy_cols] = data[dummy_cols].apply(le.fit_transform)

    X = pd.get_dummies(data, 'Title')
    return data,y



























# %%
