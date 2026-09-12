[IT3091_Assignment.pdf](https://github.com/user-attachments/files/31911280/IT3091_Assignment_Descriptor_V1.pdf)

# Student Performance Prediction (Regression Lens)

## Overview
This branch contains the **Regression component** of our IT3091 Machine Learning group assignment.

The objective is to predict students' final grades (`G3`) using demographic, social, and academic attributes. The predictions can help schools identify students who may need early academic support and intervention.

## Dataset
* **Source:** Student Performance Dataset (`student-mat.csv`).
* **Scope:** This project utilizes **only the Math dataset** ($n=395$ students) to maintain analytical focus and capture high behavioral variance.
* **Target Variable:** `G3` (Final grade ranging from 0 to 20).
* **Features:** Includes weekly study time, past failures, family background, alcohol consumption, and term progression grades (`G1`, `G2`).







# Performance Prediction (Regression Lens)

What it is: Predicting a student's exact numerical final grade (G3, scored from 0 to 20) based on academic history, demographics, and lifestyle factors.

ML Task & Models: Regression algorithms such as Linear Regression, Ridge/Lasso, Random Forest Regressor, and XGBoost.

Evaluation Metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ score.

Business Value: Enables teachers to forecast a student's final trajectory well in advance, allowing for general academic planning and continuous monitoring.


| Model              | CV RMSE | Test RMSE | Test MAE | R2 Score |
|--------------------|---------|-----------|----------|----------|
| Baseline (Mean)    | 4.575   | 4.550     | 3.646    | -0.010   |
| Linear Regression  | 1.895   | 2.378     | 1.647    | 0.724    |
| Random Forest      | 1.397   | 2.009     | 1.214    | 0.803    |
| XGBoost            | 1.575   | 2.160     | 1.192    | 0.773    |



This means Random Forest explains approximately 80.3% of the variation in final student grades, with an average prediction error of about 1.21 grade points.





### Requirements

pandas>=2.0.0

numpy>=1.24.0

scikit-learn>=1.2.0

matplotlib>=3.7.0

seaborn>=0.12.0

xgboost>=1.7.0
