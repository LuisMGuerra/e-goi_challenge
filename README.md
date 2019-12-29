# E-goi Technical Challenge

This small report details both the structure of the scripts I produced to solve the challenge and the decisions I made during coding.

## Overview

The main script were the model was researched and developed is called *model_research.py* and should be run first as it produces the model and data sample necessary to test the deployment REST API.
Once that script is run and the files *final_model.sav* and *request_sample.csv* are created the script *flask_server_side.py* can be run, simulating the server, and the script *flask_client_side.py* can be run to simulate a call to the API where it asks for a prediction for the sample in *request_sample.csv* done by the model *final_model.sav* on the server side.

### Prerequisites

The script was developed on an anaconda environment but should run on any python 3.7 installation provided, numpy, pandas, scikit-learn, matplotlib, seaborn and flask are installed.
Numpy and Pandas were used for general data wrangling, maplotlib and seaborn for visualizations, scikit for modelling and flask to create the deployment REST API.

## Preprocessing

The preprocessing step is rather short. It searches for missing values in the dataset, normalizes all numerical features to [0,1], plots their distributions to help look for outliers and evaluates categorical feature cardinality.

## Feature Importance Analysis

This step plots the correlation matrix for all numerical features searching for multicolinearity and for features that may be strongly correlated to the target feature directly. N1 and N2 seem to be fairly correlated to the output but are also correlated among themselves. N4 and N5 are also correlated among themselves.
The information gain ratio of each categorical feature is computed. None of the features shows high information gain.

## Feature Engineering

Since the cardinality of the categorical features is not too high they can be one-hot encoded so that they are compatible with most models while preserving their characteristics.
A step of PCA is performed between features (N1,N2) and (N4,N5) in order to reduce the overall dimensionality of the dataset.

## Feature Selection

In this step, all features, including the one-hot encodings are selected based on wether or not they correlate with the output variable with a pearson's correlation factor over 0.1 or under -0.1. This extracts value from categorical features that were hindered by uninformative categories and helps future proofing the model in case new categories are added to these features.

## Modelling

Four different models were optimized to this dataset using a 10-fold crossvalidation setup over a training set consisting of 70% of the original data.
The decision tree obtained the lowest results of the group, being surpassed by the logistic regression that, itself was surpassed by both the random forest and the gradient boosting.
Increasing the depth of the decision tree didn't seem to improve overall results.
Logistic regression saw some gains with a slight decrease of its default regularization factor.
Due to the size of the dataset both the random forest and the gradient boosting worked well with a low number of trees. Their performance plateaued at depths similar to that of the decision tree wich leads me to believe their individual predictors are not very dissimilar.

## Model Selection

Once the models are optimized they are trained on the whole training set once, evaluated by the 30% of data put on hold before the modelling step and the model with the higher F1-score is chosen (I opted to use the F1-score as my metric as the classes were heavily imbalanced). This model then trained on the whole dataset once and pickled into a file to be used in production.

A small sample of processed data is also saved to test the REST API.
