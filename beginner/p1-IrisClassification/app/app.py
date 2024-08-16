#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 08:15:43 2024

@author: aboubakr
"""
import sys
import os
import numpy as np
from typing import Annotated
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from enum import Enum

current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(os.path.abspath(os.path.join(project_root, 'scripts')))
from model_pipeline import ModelPipeline
from data_pipeline import DataPipeline
#%% Initialize the FastAPI app

app = FastAPI(title="Predict Flower Specy",
              description= "Fetch the specy of a flower base on his physical size")
#%% Load model
# Global variables to hold the cached model and data pipeline
model = None
data_pipeline = None

# Load model and data pipeline at startup
@app.on_event("startup")
def load_model_and_data_pipeline():
    global model, data_pipeline
    data_pipeline = DataPipeline()
    model_pipeline = ModelPipeline()
    model_pipeline.load_model()
    model = model_pipeline.model
#%% Define the input data model

class InputData(BaseModel):
    SepalLength : float | None = Field(default = None, description="The flower's sepal length in cm", example=3)
    SepalWidth : float = Field(..., description="The flower's sepal width in cm", example=2.7)
    PetalLength : float = Field(..., description="The flower's petal length in cm", example=1.8)
    PetalWidth : float = Field(..., description="The flower's sepal length in cm", example=3)
    
class PredictionResult(str, Enum):
    setosa = 'setosa'
    versicolor = 'versicolor'
    virginica = 'virginica'


class PredictProbaResult(BaseModel):
    __root__: dict[PredictionResult, float]
    
prediction_list = list(PredictionResult)
    
#%% Prediction endpoint
@app.post("/predict", response_model=list[PredictionResult])
def predict(data: Annotated[list[InputData] , 
                            Body(embed=True, 
                                 description = "The specifications of the flower to classify")]):
    input_data = np.array([[d.SepalLength, d.SepalWidth, d.PetalLength, d.PetalWidth] for d in data])
    processed_input = data_pipeline.preprocess_raw_data(input_data)
    prediction = model.predict(processed_input)
    predicted_class = [prediction_list[int(pred)] for pred in prediction]
    return predicted_class
#%% Let's do a predict proba endpoint to play with dicts
#Make sure that the loaded model was trained with probability = True

@app.post("/predict_proba", response_model=list[PredictProbaResult])
def predict_proba(data: Annotated[list[InputData], Body(description="The specifications of the flower to classify")]):
    input_data = np.array([[d.SepalLength, d.SepalWidth, d.PetalLength, d.PetalWidth] for d in data])
    processed_input = data_pipeline.preprocess_raw_data(input_data)
    try:
        probabilities = model.predict_proba(processed_input)
    except AttributeError as e:
        print("Model was not trained with probabiliy enabled")
        raise e
    return [{prediction_list[i]: float(prob[i]) for i in range(len(prob))} for prob in probabilities ]
 