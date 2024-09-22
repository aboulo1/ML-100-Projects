import sys
import os
import pandas as pd
from typing import Annotated
from fastapi import FastAPI, Body, Query
from pydantic import BaseModel, Field
from enum import Enum

current_dir = os.getcwd()
sys.path.append(current_dir)
from scripts.model_pipeline import ModelPipeline
from scripts.data_pipeline import DataPipeline
#%% Initialize the FastAPI app

fastapp = FastAPI(title="Predict Titanic Survival Chances",
              description= "Compute the chance of an indiviudual surviving the titanic based on several data")
#%% Load model
# Global variables to hold the cached model and data pipeline
model = None
data_pipeline = None
cols = None

# Load model and data pipeline at startup
@fastapp.on_event("startup")
def load_model_and_data_pipeline():
    """
    To load all the needed models and pipelines so that we don't reload them at each call'

    Returns
    -------
    None, but renders in glabal values our data and model pipelines

    """
    global model, data_pipeline, cols
    data_pipeline = DataPipeline()
    model_pipeline = ModelPipeline()
    model_pipeline.load_model()
    model = model_pipeline.model
    cols = ["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

class BSex (str, Enum):
    male = "male"
    female = "female"

class Pclass (int, Enum):
    first = 1
    second = 2
    third = 3
class EmbarkationPort (str, Enum):
    cherbourg = 'C'
    queenstown = 'Q'
    southampton = 'S'
# Define the input data model
class InputData(BaseModel):
    """
    Body of the prediction endpoints. It embeds the data that'll be pushed in the prediction model
    """
    Pclass : Pclass
    Name : str = Field(..., description="The passenger's Name", example="Braund, Mr. Owen Harris", regex = "([A-Za-z]+),\s(Mr|Mrs|Miss|Master)\.\s([A-Za-z\s]+)")
    Sex : BSex = Field (..., description= "The passenger's Sex", example = "male")
    Age : float = Field(..., description="The passenger's age in years", example = 30)
    SibSp : float = Field (..., description="Number of siblings / spouses aboard the Titanic", example = 3)
    Parch : float = Field (..., description="Number of parents / children aboard the Titanic	", example = 2)
    Ticket : str | None = Field(None, description="The passenger's Ticket number", example = "CA54G", max_length=5)
    Fare : float = Field(..., description="The passenger's ticket fare in british pounds", example = 10)
    Cabin : str | None = Field(None, description="The passenger's Cabin number", example = "C86", max_length=3)
    Embarked : EmbarkationPort = Field(..., description="The passenger's port of Embarkation", example = 'C')

class PredictionResult (float, Enum):
    """The prediction Result of the model.
    In this binary classification case we have two outputs 0 : Died, 1 : Survived
    """
    Died = 0
    Survived = 1

class PredictProbaResult(BaseModel):
    """
    The datamodel of the predictProba class : a predictionResult associated to it's probability
    """
    Died: float
    Survived: float

#Prediction endpoint
@fastapp.post("/predict", response_model = PredictionResult)
def predict(Passenger : Annotated[InputData,
             Body(embed=True,
                  description = "The passengers information")]):
    """
    Predict if the passenger survived or not

    Parameters
    ----------
    Passenger : Annotated[InputData,             
                          Body(embed, optional
        DESCRIPTION. The default is True,                  
        description = "The passengers information")].

    Returns
    -------
    prediction : PredictionResult
        Wether the passenger survuved or not . 0 means death.

    """
    input_data = [Passenger.Pclass.value, Passenger.Name, Passenger.Sex.value, Passenger.Age, Passenger.SibSp, Passenger.Parch, Passenger.Ticket, Passenger.Fare, Passenger.Cabin, Passenger.Embarked.value]
    input_data = pd.DataFrame([input_data], columns=cols)
    processed_input = data_pipeline.preprocess_api(input_data)
    prediction = model.predict(processed_input)
    return prediction

#Predict proba endpoint
@fastapp.post("/predict_proba", response_model = PredictProbaResult)
def predict_proba(Passenger : Annotated[InputData,
             Body(embed=True,
                  description = "The passengers information")]):
    """
    The death and survival probability of the described passenger.    

    Parameters
    ----------
    Passenger : Annotated[InputData,             
                          Body(embed, optional
        DESCRIPTION. The default is True,                  
        description = "The passengers information")].

    Returns
    -------
    PredictProbaResult
        

    """
    input_data = [Passenger.Pclass.value, Passenger.Name, Passenger.Sex.value, Passenger.Age, Passenger.SibSp, Passenger.Parch, Passenger.Ticket, Passenger.Fare, Passenger.Cabin, Passenger.Embarked.value]
    input_data = pd.DataFrame([input_data], columns=cols)
    processed_input = data_pipeline.preprocess_api(input_data)
    # Get prediction probabilities
    prediction_proba = model.predict_proba(processed_input)[0]
    return PredictProbaResult(
        Died=prediction_proba[0],
        Survived=prediction_proba[1]
    )
prediction_list = list(PredictionResult)
#Prediction endpoint
@fastapp.post("/predict/multiple", response_model = list[PredictionResult])
def predict_mult(Passengers : Annotated[list[InputData],
             Body(embed=True,
                  description = "The passengers information")]):
    """
    

    Parameters
    ----------
    Passengers : Annotated[list[InputData],             
                           Body(embed, optional
        DESCRIPTION. The default is True,                  
        description = "The passengers information")].

    Returns
    -------
    predicted_class : list[PredictionResult] the outcome for each passenger
        DESCRIPTION.

    """
    input_data = [[Passenger.Pclass.value, Passenger.Name, Passenger.Sex.value, Passenger.Age, Passenger.SibSp, Passenger.Parch, Passenger.Ticket, Passenger.Fare, Passenger.Cabin, Passenger.Embarked.value] for Passenger in Passengers]
    input_data = pd.DataFrame(input_data, columns=cols)
    processed_input = data_pipeline.preprocess_api(input_data)
    prediction = model.predict(processed_input)
    predicted_class = [prediction_list[int(pred)] for pred in prediction]
    return predicted_class
