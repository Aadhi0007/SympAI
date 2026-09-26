# SympAI — Symptom-Based Disease Prediction

SympAI is a machine learning web application that predicts possible diseases from user-provided symptoms.

The project combines a trained machine learning model with a web interface to provide an interactive symptom-based prediction system.

## 🧠 Project Overview

SympAI follows a machine learning pipeline in which symptom information is processed and transformed into features before being passed to a trained classification model.

The application consists of:

- Symptom-based input
- Data preprocessing
- Feature transformation
- Machine learning prediction
- Web-based user interface
- Prediction output

## 🚀 Key Features

### Symptom-Based Prediction

Users can provide symptoms through the web interface and receive a predicted disease based on the trained model.

### Machine Learning Pipeline

The project includes preprocessing components such as:

- Feature transformation
- Label encoding
- Feature scaling
- Vectorization

### Web Application

The prediction system is connected to a web interface using **Flask** and HTML.

### Saved ML Components

The repository contains trained and serialized components used during prediction, including:

```text
disease_model.h5
label_encoder.pkl
label_encoders.pkl
scaler.pkl
vectorizer.pkl
X_columns.pkl
