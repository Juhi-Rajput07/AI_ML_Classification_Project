# AI-Based Multiclass Wine Classification

## Project Overview

This project implements an end-to-end machine learning workflow for multiclass classification using the Wine dataset from Scikit-learn.

The project covers:

- Dataset loading and inspection
- Data quality checking
- Exploratory Data Analysis
- Data preprocessing
- Random Forest classification
- Deep Neural Network classification
- Model evaluation
- Model comparison
- Streamlit-based prediction application

## Dataset

The Wine dataset contains:

- 178 observations
- 13 numerical features
- 3 target classes

The dataset contains no missing values or duplicate rows.

## Models

### Random Forest

A Random Forest classifier with:

- 200 decision trees
- random_state = 42

### Neural Network

A basic Deep Neural Network implemented using TensorFlow/Keras.

Architecture:

- Input layer
- Dense layer with 64 neurons and ReLU
- Dropout
- Dense layer with 32 neurons and ReLU
- Output layer with 3 neurons and softmax

## Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results

On the project's 80/20 stratified test split:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Neural Network | 0.9444 | 0.9466 | 0.9444 | 0.9443 |

These results describe the specific experimental split used in this project and should not be interpreted as universal performance.

## Application

A Streamlit application is included in `app.py`.

The application accepts the 13 Wine dataset features and predicts one of the three wine classes using the trained Random Forest model.

## Technologies

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow/Keras
- Streamlit
- Joblib