#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:58:10 2024

@author: aboubakr
"""

from data_pipeline import DataPipeline
from model_pipeline import ModelPipeline

def main():
    """
    A quick function to train, evaluate and save the model in a few lines
    All the preprocessing is handled in the pipelines

    Returns
    -------
    None.

    """
    # Data Pipeline
    pipeline = DataPipeline() #could add is_iris_loaded=False
    X_train, X_test, y_train, y_test = pipeline.get_data()
    
    # Model Pipeline
    model_pipeline = ModelPipeline()
    model_pipeline.train(X_train, y_train)
    accuracy = model_pipeline.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    model_pipeline.save_model()
    
if __name__ == "__main__":
    main()
    