# Titanic Survival

This project is part of a series of 100 machine learning projects aimed at building and deploying machine learning models. 
The goal of this project is to predict the survival probability of the Titanic passengers based on there attributes.
In fact, it looks like some sorts of people were most likely to survive than others
The swagger : https://titanic-survival-service-697664343307.europe-west4.run.app/docs
# True Goal
The goal is not the model again, but the structure, coding clarity and robustness, documentation. Also a fast deployment to cloud run

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Model Prototyping](#Model-Prototyping)
5. [Usage](#usage)
6. [Model Performance](#model-performance)
7. [Results](#results)
8. [Conclusions](#conclusions)
9. [Deploy Model](#deploy-model)
10. [Project Structure](#project-structure)


## Introduction
Classic machine learning case from a kaggle challenge to predict wether a passenger of the titanic will survice the shipwreck.

## Dataset
Train.csv contains a subset of passengers on board and reveal if they survived or not(891 rows)
Test.csv is the same to test our model (418 rows)
You can find detailed informations on the data sit on the kaggle challenge site
https://www.kaggle.com/competitions/titanic/data

## Installation

To run this project, you'll need to install the required dependencies. You can install them using the following command:

'''bash'''
pip install -r requirements.txt


- **Installation**: Instructions on how to set up the environment, including installing dependencies. 

#### **Model Prototyping**

## Model prototyping
We used an xgboost with an hyperparametrization we hyperopt.
It sucks that I lost that commit but managed to save the best parameters...
The case was really interesting with the iterative curves but the mac os upgrade got the best of us


#### **Model Performance**
Accuracy of 84%

## Model Performance

The model was evaluated using cross-validation and achieved the following metrics:

- **Accuracy**: 
- **Precision**: 
- **Recall**: 
- **F1-Score**: 
The metrics were lost
## Results

The model successfully classifiesthe survival proba with high accuracy. 

## Conclusions

The classification model performs well on the Titanic dataset, demonstrating the effectiveness of basic machine learning techniques on a well-structured dataset. 
This project serves as a solid foundation for more complex classification problems.

## Deploy Model
### FastAPI
in application/app.py we've built a fast api service to predict classes and render probabilities
To run in the "root" folder 

'''bash

uvicorn application.app:fastapp --reload

### Docker
We've then created a docker image (see Dockerfile) at the root of this project
to build and run the docker image : run at the root of the project

'''bash

make build
make run
check errors : docker ps -a
docker logs <logs>

### Google Cloud Run
https://cloud.google.com/run/docs/configuring/services/containers?hl=fr#gcloud_1
https://medium.com/@saverio3107/deploy-fastapi-with-docker-cloud-run-a-step-by-step-guide-a01c42df0fee
The deployment failed after a lot of attempts due to app failing to listen to port 8000 or 8080

'''bash

make build_gcr
make push_gcr
### Deploy it 
'''bash

make deploy_gcr

## Project Structure

The project is organized into the following structure:

### Explanation:
- **`data/`**: Directory to store dataset files.
- **`notebooks/`**: Contains Jupyter notebooks for Exploratory Data Analysis (EDA) and other prototyping tasks. Includes a notebook 'Fundamentals.ipynb' where theoretical answers are documented.
- **`scripts/`**: Python scripts used for data processing, model training, and evaluation.
- **`models/`**: Directory where serialized models (like '.pkl' or '.h5' files) are stored after training.
- **`app/`**: Directory for application code, such as a Flask or FastAPI app, used to deploy the model.
- **`README.md`**: The main project documentation, providing an overview and instructions.
- **`requirements.txt`**: Lists all the dependencies required to run the project.
- **`Dockerfile`** : The dockerfile to create the image of our application to deploy on GCP