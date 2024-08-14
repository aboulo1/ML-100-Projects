#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:35:19 2024

@author: aboubakr
"""

#%% Imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Processing import preprocessing
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
#%% Load the dataset and split the data set

iris = load_iris()

# before doing anything let's split the data. We don't want to do preprocessing on
# test data that's theoriticlly unseedn to the model

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                                                    test_size=0.3, random_state=42)
#%% Preprocess the data
X_train_processed = preprocessing(X_train, train=True, feature_names = iris.feature_names)

#%% Preprocess the testing set
X_test_processed = preprocessing(X_test, feature_names = iris.feature_names)
#%% Logistic Regression
LR = LogisticRegression()
LR.fit(X_train_processed, y_train)
y_pred_lr = LR.predict(X_test_processed)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f" The accuracy of the Logistic Regression model is {accuracy_lr}")
print("It has the following classification report")
print(classification_report(y_test, y_pred_lr))
# It's amazing we have a precision of 100% jaja
#%% RandomForestClassifier
RF = RandomForestClassifier(criterion = 'entropy', n_estimators = 25)
RF.fit(X_train_processed, y_train)
y_pred_rf = RF.predict(X_test_processed)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f" The accuracy of the Random Forest model is {accuracy_rf}")
print("It has the following classification report")
print(classification_report(y_test, y_pred_rf))
# Oh wow 100% too of F1-score
#%% SupportVectorMachine
SVM = svm.SVC(kernel="linear")
SVM.fit(X_train_processed, y_train)
y_pred_SVM = SVM.predict(X_test_processed)

accuracy_svm = accuracy_score(y_test, y_pred_SVM)
print(f" The accuracy of the Logistic Regression model is {accuracy_svm}")
print("It has the following classification report")
print(classification_report(y_test, y_pred_SVM))
# 95% accuracy let's use a plot to see how it separated the classes and what he had wrong
#%% Plot the decision boundary
# source : https://medium.com/geekculture/svm-classification-with-sklearn-svm-svc-how-to-plot-a-decision-boundary-with-margins-in-2d-space-7232cb3962c0
X = X_train_processed.values[:,:2]
y = y_train
sns.set(style="whitegrid")
# Create a scatter plot of the data points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", edgecolor="k", s=100)
coef = SVM.coef_
intercept = SVM.intercept_
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
margin_colors = ['red', 'green', 'blue']
for i in range(coef.shape[0]):
    # Calculate the decision boundary line
    decision_boundary = -(coef[i, 0] * x_vals + intercept[i]) / coef[i, 1]

    # Plot the decision boundary
    plt.plot(x_vals, decision_boundary, label=f'Boundary {iris.feature_names[i]}')
    
    # Calculate and plot the margins
    margin = 1 / np.sqrt(np.sum(coef[i] ** 2))
    margin_boundary_upper = np.c_[x_vals, decision_boundary + margin][:, 1]
    margin_boundary_lower = np.c_[x_vals, decision_boundary - margin][:, 1]


    # Fill the area between the margins with a semi-transparent color
    plt.fill_between(x_vals, margin_boundary_lower, margin_boundary_upper, color=margin_colors[i], alpha=0.1)

x_min = X_test_processed[X_test_processed.columns[0]].min()
x_max = X_test_processed[X_test_processed.columns[0]].max()
y_min = X_test_processed[X_test_processed.columns[1]].min()
y_max = X_test_processed[X_test_processed.columns[1]].max()
plt.xlim(x_min-0.25, x_max+0.25)
plt.ylim(y_min-0.25, y_max+0.25)    
plt.xlabel(f"{iris.feature_names[0]}")
plt.ylabel(f"{iris.feature_names[1]}")
plt.title('SVM Decision Boundary Training')
plt.legend(prop = { "size": 5 }, loc ="upper left")
plt.show()

#%% plot the decision boundary seen with X_test

correct = y_test == y_pred_SVM
X_test_processed['correct'] = correct

#Set dictionnaries to affiliate different colors and markers depending on accuracy of precision
colors = {0: 'blue', 1: 'black', 2: 'green'}
colors_wrong = {0: 'yellow', 1: 'red', 2: 'orange'}
markers = {True: 'o', False: 'x'}

for i in range(coef.shape[0]):
    # Calculate the decision boundary line
    decision_boundary = -(coef[i, 0] * x_vals + intercept[i]) / coef[i, 1]
    # Plot the decision boundary
    plt.plot(x_vals, decision_boundary, label=f'Boundary {iris.feature_names[i]}')
    # Fill the area between the margins with a semi-transparent color
    margin = 1 / np.sqrt(np.sum(coef[i] ** 2))
    margin_boundary_upper = np.c_[x_vals, decision_boundary + margin][:, 1]
    margin_boundary_lower = np.c_[x_vals, decision_boundary - margin][:, 1]
    plt.fill_between(x_vals, margin_boundary_lower, 
                     margin_boundary_upper, color=margin_colors[i], alpha=0.1)
    #Plot the right and wrong predictions
    for is_correct in [True, False]:
        subset = X_test_processed[(y_test==i) & (X_test_processed.correct == is_correct)]
        plt.scatter(x=subset[subset.columns[0]], y=subset[subset.columns[1]],
                        color = colors[i] if is_correct else colors_wrong[i], 
                        marker = markers[is_correct],
                        label=f'{iris.feature_names[i]} - {"Correct" if is_correct else "Wrong"}')
    
plt.xlim(x_min-0.25, x_max+0.25)
plt.ylim(y_min-0.25, y_max+0.25)
plt.xlabel(f"{iris.feature_names[0]}")
plt.ylabel(f"{iris.feature_names[1]}")
plt.title('SVM Decision Boundary Testing ')
plt.legend(prop = { "size": 7 }, loc ="upper left")
plt.show()
