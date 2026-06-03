# Employee Attrition Prediction

## Overview

This project aims to predict employee attrition using machine learning techniques on the IBM HR Analytics Employee Attrition dataset. The objective is to identify employees who are likely to leave the organization, enabling proactive retention strategies.

## Problem Statement

Employee attrition can result in significant costs related to recruitment, training, and productivity loss. The goal of this project is to build a classification model capable of identifying employees at risk of leaving the company.

## Dataset

Dataset: IBM HR Analytics Employee Attrition Dataset

Key Features:

* Age
* Monthly Income
* Overtime Status
* Job Level
* Stock Option Level
* Years at Company
* Total Working Years
* Distance From Home
* Job Role
* Marital Status
* Department
* and other HR-related attributes

Target Variable:

* Attrition (Yes / No)

## Data Preprocessing

The following preprocessing steps were performed:

* Removal of target variable from feature set
* One-Hot Encoding for categorical features
* Standardization using StandardScaler
* Train-Test Split
* Class imbalance handling using class_weight='balanced'

## Models Implemented

### 1. Logistic Regression

Configuration:

* Class Weight Balancing
* Hyperparameter Tuning (C)
* Probability Threshold Analysis
* Feature Scaling

### 2. Random Forest Classifier

Configuration:

* Balanced Class Weights
* Multiple Tree Ensemble
* Feature Importance Analysis

## Model Evaluation

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Best Logistic Regression Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.78  |
| Precision | 0.34  |
| Recall    | 0.72  |
| F1 Score  | 0.46  |

### Random Forest Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.85  |
| Precision | 0.38  |
| Recall    | 0.23  |
| F1 Score  | 0.29  |

## Key Findings

* Logistic Regression outperformed Random Forest on the minority attrition class.
* Class balancing significantly improved recall.
* One-Hot Encoding produced better results than Label Encoding.
* Hyperparameter tuning demonstrated the impact of regularization strength on model performance.
* Overtime, Monthly Income, Stock Option Level, Age, and Total Working Years emerged as influential features.

## Top Important Features

1. OverTime
2. Monthly Income
3. Stock Option Level
4. Age
5. Total Working Years

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Jupyter Notebook

## Future Improvements

* Cross-Validation
* XGBoost Implementation
* SHAP Explainability
* Hyperparameter Optimization using GridSearchCV
* Model Deployment using Flask or Streamlit

## Project Structure

```text
AttritionClassifier/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebooks/
│   └── AttritionClassifier.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Conclusion

This project demonstrates an end-to-end machine learning workflow including data preprocessing, feature engineering, model training, evaluation, and interpretation. Logistic Regression achieved the best balance between recall and overall classification performance for employee attrition prediction.


Link to dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
